"""Unit tests for payment handling."""

import json
import socket
import threading
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import requests
from web3.exceptions import RequestTimedOut, TransactionNotFound, Web3RPCError

from x402gate.core.config import NetworkConfig
from x402gate.core.payment import (
    InFlightAmounts,
    PaymentClaims,
    PaymentHandler,
    _ResyncingWeb3Signer,
    _send_outcome,
    _settlement_failure_reason,
    payment_key,
    settlement_state,
)


@pytest.fixture
def handler() -> PaymentHandler:
    """Create a PaymentHandler for testing (EVM only, mocked signer)."""
    networks = {
        "base": NetworkConfig(
            type="evm",
            network="eip155:8453",
            pay_to="0x1234567890abcdef1234567890abcdef12345678",
            token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            token_name="USD Coin",
            token_version="2",
            rpc_url="https://mainnet.base.org",
            facilitator_key="0x0000000000000000000000000000000000000000000000000000000000000001",
        ),
    }
    return PaymentHandler(networks=networks, max_timeout=3600)


class TestCreatePaymentRequired:
    """Tests for 402 response generation."""

    def test_status_code_402(self, handler: PaymentHandler):
        """Response has status code 402."""
        response = handler.create_payment_required(Decimal("0.003150"))
        assert response.status_code == 402

    def test_response_contains_accepts(self, handler: PaymentHandler):
        """Response body contains 'accepts' with payment details."""

        response = handler.create_payment_required(Decimal("0.003150"))
        body = json.loads(response.body.decode())
        assert "accepts" in body
        assert len(body["accepts"]) == 1

    def test_payment_details_fields(self, handler: PaymentHandler):
        """Payment details contain all required fields."""

        response = handler.create_payment_required(Decimal("0.003150"))
        body = json.loads(response.body.decode())
        details = body["accepts"][0]

        assert details["scheme"] == "exact"
        assert details["network"] == "eip155:8453"
        assert details["payTo"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert details["price"] == "$0.003150"
        assert details["asset"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert details["amount"] == "3150"
        assert details["maxTimeoutSeconds"] == 3600
        assert details["extra"] == {"name": "USD Coin", "version": "2"}

    def test_error_message(self, handler: PaymentHandler):
        """Response contains an error message."""

        response = handler.create_payment_required(Decimal("0.003150"))
        body = json.loads(response.body.decode())
        assert body["error"] == "Payment Required"


class TestPaymentRequirements:
    """Tests for payment requirements dict building."""

    def test_requirements_format(self, handler: PaymentHandler):
        """get_all_payment_requirements returns correct format."""
        reqs_list = handler.get_all_payment_requirements(Decimal("0.005000"))
        assert len(reqs_list) == 1
        reqs = reqs_list[0]
        assert reqs["scheme"] == "exact"
        assert reqs["network"] == "eip155:8453"
        assert reqs["payTo"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert reqs["price"] == "$0.005000"
        assert reqs["asset"] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert reqs["amount"] == "5000"
        assert reqs["maxTimeoutSeconds"] == 3600
        assert reqs["extra"] == {"name": "USD Coin", "version": "2"}


class TestSettlementFailureReason:
    """Tests for readable settlement failure text."""

    def test_ignores_empty_error_message(self):
        """A None error_message must not be logged as the text 'None'."""
        result = SimpleNamespace(
            error_reason="transaction_simulation_failed",
            error_message=None,
        )
        assert _settlement_failure_reason(result) == "transaction_simulation_failed"

    def test_includes_simulation_text(self):
        """Simulation detail is appended when the SDK provides it."""
        result = SimpleNamespace(
            error_reason="transaction_simulation_failed",
            error_message=None,
            invalid_message="Blockhash not found",
        )
        assert (
            _settlement_failure_reason(result)
            == "transaction_simulation_failed: Blockhash not found"
        )

    def test_prefers_error_message(self):
        """error_message wins over invalid_message when both are set."""
        result = SimpleNamespace(
            error_reason="transaction_simulation_failed",
            error_message="Simulation failed",
            invalid_message="ignored",
        )
        assert _settlement_failure_reason(result) == (
            "transaction_simulation_failed: Simulation failed"
        )


class TestEvmBalanceCheck:
    """Tests for the on-chain USDC balance check."""

    async def test_rpc_error_fails_closed(self, handler: PaymentHandler):
        """A failing balance RPC is retried once and then reported as unavailable."""
        ns = handler._schemes["eip155:8453"]
        ns.w3.eth.call = Mock(side_effect=ConnectionError("rpc down"))
        result = await handler.check_evm_balance(
            "eip155:8453",
            "0x1234567890abcdef1234567890abcdef12345678",
            Decimal("1"),
        )
        assert result is None
        assert ns.w3.eth.call.call_count == 2

    async def test_insufficient_balance_is_false(self, handler: PaymentHandler):
        ns = handler._schemes["eip155:8453"]
        ns.w3.eth.call = Mock(return_value=(0).to_bytes(32, "big"))
        result = await handler.check_evm_balance(
            "eip155:8453",
            "0x1234567890abcdef1234567890abcdef12345678",
            Decimal("1"),
        )
        assert result is False


def _encoded(payload: dict) -> str:
    """Base64-encode an x402 payment payload the way clients send it."""
    import base64

    return base64.b64encode(json.dumps(payload).encode()).decode()


class TestPaymentKey:
    """One payment maps to one id however it is encoded."""

    def test_evm_key_uses_payer_and_nonce(self):
        auth = {"from": "0xAbC", "nonce": "0xDEAD", "value": "1"}
        first = _encoded(
            {"accepted": {"network": "eip155:8453"}, "payload": {"authorization": auth}}
        )
        auth_lower = {"from": "0xabc", "nonce": "0xdead", "value": "1"}
        second = _encoded(
            {
                "payload": {"authorization": auth_lower, "signature": "0x1"},
                "accepted": {"network": "eip155:8453"},
            }
        )
        assert payment_key(first) == payment_key(second) == "eip155:8453:0xabc:dead"

    def test_solana_key_hashes_the_transaction(self):
        import base64

        tx = base64.b64encode(b"tx-bytes").decode()
        sig = _encoded({"accepted": {"network": "solana:x"}, "payload": {"transaction": tx}})
        other = _encoded({"accepted": {"network": "solana:x"}, "payload": {"transaction": "eHg="}})
        assert payment_key(sig).startswith("solana:x:")
        assert payment_key(sig) != payment_key(other)

    def test_unreadable_header_still_gets_a_key(self):
        assert payment_key("not base64 at all").startswith("raw:")


class TestPaymentClaims:
    """A payment can be claimed by one request at a time."""

    def test_second_claim_fails_until_release(self):
        claims = PaymentClaims()
        assert claims.claim("k")
        assert not claims.claim("k")
        claims.release("k")
        assert claims.claim("k")

    def test_hold_blocks_until_it_expires(self, monkeypatch):
        import x402gate.core.payment as payment

        now = [100.0]
        monkeypatch.setattr(payment.time, "monotonic", lambda: now[0])
        claims = PaymentClaims()
        assert claims.claim("k")
        claims.release("k", hold_s=60)
        now[0] = 159.0
        assert not claims.claim("k")
        now[0] = 161.0
        assert claims.claim("k")


class TestPaymentKeyNonceSpelling:
    """The SDK reads the nonce with bytes.fromhex, so spellings are one nonce."""

    def test_nonce_spellings_share_a_key(self):
        def key(nonce: str) -> str:
            auth = {"from": "0xAbC", "nonce": nonce}
            return payment_key(
                _encoded(
                    {"accepted": {"network": "eip155:8453"}, "payload": {"authorization": auth}}
                )
            )

        assert key("0xABCD") == key("abcd") == key("ab cd") == "eip155:8453:0xabc:abcd"


class TestSettlementState:
    """Settle results are ok, failed, or unknown when they may still land."""

    def test_states(self):
        assert settlement_state({"success": True}) == "ok"
        assert settlement_state({"success": False}) == "failed"
        assert settlement_state({"success": False, "pending": True}) == "unknown"
        assert settlement_state(None) == "unknown"


class TestInFlightAmounts:
    """Promised Base amounts add up per payer, case-insensitively."""

    def test_add_and_remove(self):
        amounts = InFlightAmounts()
        assert amounts.add("0xAB", Decimal("1")) == Decimal("1")
        assert amounts.add("0xab", Decimal("2")) == Decimal("3")
        amounts.remove("0xAB", Decimal("1"))
        assert amounts.add("0xab", Decimal("0")) == Decimal("2")
        amounts.remove("0xab", Decimal("5"))
        assert amounts.add("0xab", Decimal("0")) == Decimal("0")


class TestPriceFetchBackoff:
    """A failed CoinGecko fetch is not retried on every settlement."""

    @pytest.mark.asyncio
    async def test_failure_is_not_retried_at_once(self, monkeypatch):
        import x402gate.core.payment as payment

        calls = []

        async def boom():
            calls.append(1)
            raise RuntimeError("down")

        monkeypatch.setattr(payment, "_fetch_prices", boom)
        monkeypatch.setitem(payment._price_cache, "ethereum", 0.0)
        monkeypatch.setitem(payment._price_cache, "solana", 0.0)
        monkeypatch.setitem(payment._price_cache, "failed_at", 0.0)
        await payment._refresh_prices()
        await payment._refresh_prices()
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_empty_answer_counts_as_failure(self, monkeypatch):
        import x402gate.core.payment as payment

        async def rate_limited():
            return {"status": {"error_code": 429}}

        monkeypatch.setattr(payment, "_fetch_prices", rate_limited)
        monkeypatch.setitem(payment._price_cache, "ethereum", 0.0)
        monkeypatch.setitem(payment._price_cache, "solana", 0.0)
        monkeypatch.setitem(payment._price_cache, "failed_at", 0.0)
        await payment._refresh_prices()
        assert payment._price_cache["failed_at"] > 0


@pytest.fixture
def stale_prices(monkeypatch):
    """A price cache old enough to need a refresh, holding SOL at $100."""
    import x402gate.core.payment as payment

    monkeypatch.setitem(payment._price_cache, "ethereum", 2000.0)
    monkeypatch.setitem(payment._price_cache, "solana", 100.0)
    monkeypatch.setitem(payment._price_cache, "ts", 0.0)
    monkeypatch.setitem(payment._price_cache, "failed_at", 0.0)
    monkeypatch.setattr(payment, "_price_refresh", None)
    return payment


@pytest.fixture
def fresh_prices(monkeypatch):
    """A fresh price cache holding ETH at $2000, so nothing is fetched."""
    import x402gate.core.payment as payment

    monkeypatch.setitem(payment._price_cache, "ethereum", 2000.0)
    monkeypatch.setitem(payment._price_cache, "ts", time.time())
    return payment


async def _drain_background(payment) -> None:
    import asyncio

    while payment._background_tasks:
        await asyncio.gather(*payment._background_tasks, return_exceptions=True)


class TestPriceFetch:
    """CoinGecko goes through the pooled client and never blocks a settlement."""

    async def test_fetch_uses_the_shared_client(self, monkeypatch):
        import httpx

        import x402gate.core.payment as payment

        response = Mock()
        response.json.return_value = {"solana": {"usd": 150.0}}
        client = Mock()
        client.get = AsyncMock(return_value=response)
        monkeypatch.setattr(payment.http, "shared_client", lambda: client)

        assert await payment._fetch_prices() == {"solana": {"usd": 150.0}}
        kwargs = client.get.await_args.kwargs
        assert kwargs["params"] == {"ids": "ethereum,solana", "vs_currencies": "usd"}
        assert isinstance(kwargs["timeout"], httpx.Timeout)
        assert kwargs["timeout"].read == payment._PRICE_TIMEOUT_S

    async def test_stale_cache_refreshes_once_in_the_background(self, stale_prices, monkeypatch):
        payment = stale_prices
        calls = []

        async def fetch():
            calls.append(1)
            return {"ethereum": {"usd": 2500.0}, "solana": {"usd": 120.0}}

        monkeypatch.setattr(payment, "_fetch_prices", fetch)
        # Returns the stale value at once; one refresh runs behind it.
        assert payment._cached_price("solana") == 100.0
        assert payment._cached_price("solana") == 100.0
        await _drain_background(payment)
        assert calls == [1]
        assert payment._price_cache["solana"] == 120.0

    async def test_background_failure_is_logged_not_raised(self, caplog):
        import x402gate.core.payment as payment

        async def boom():
            raise RuntimeError("bookkeeping broke")

        task = payment._spawn(boom())
        assert task in payment._background_tasks
        await _drain_background(payment)
        assert task not in payment._background_tasks
        assert "bookkeeping broke" in caplog.text

    async def test_verify_starts_the_price_refresh(self, stale_prices, handler, monkeypatch):
        payment = stale_prices
        calls = []

        async def fetch():
            calls.append(1)
            return {"ethereum": {"usd": 2500.0}}

        monkeypatch.setattr(payment, "_fetch_prices", fetch)
        await handler.verify("not a payment", Decimal("1"))
        await _drain_background(payment)
        assert calls == [1]


_SOLANA = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
_USDC_MINT = "EPjFWDvL5JKX8YQjKdL4o2G1uR7oAPpqCq2QvZJ3vJhW"


def _solana_payment(fee_payer) -> tuple[str, dict]:
    """A payment header carrying a real client transaction (2 signatures)."""
    import base64

    from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
    from solders.hash import Hash
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.system_program import TransferParams, transfer
    from solders.transaction import VersionedTransaction

    client = Keypair()
    message = MessageV0.try_compile(
        payer=fee_payer.pubkey(),
        instructions=[
            set_compute_unit_limit(20_000),
            set_compute_unit_price(1),
            transfer(
                TransferParams(from_pubkey=client.pubkey(), to_pubkey=client.pubkey(), lamports=1)
            ),
        ],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.default(),
    )
    tx = base64.b64encode(bytes(VersionedTransaction(message, [fee_payer, client]))).decode()
    accepted = {
        "scheme": "exact",
        "network": _SOLANA,
        "asset": _USDC_MINT,
        "amount": "1000000",
        "payTo": str(Keypair().pubkey()),
        "maxTimeoutSeconds": 3600,
        "extra": {"feePayer": str(fee_payer.pubkey())},
    }
    header = _encoded({"x402Version": 2, "accepted": accepted, "payload": {"transaction": tx}})
    return header, accepted


class TestSolanaSettleGas:
    """Settle reports gas from the transaction itself, without waiting on the network."""

    def _handler(self, fee_payer, accepted, settle_result) -> PaymentHandler:
        from x402gate.core.payment import _NetworkScheme
        from x402gate.core.svm_signer import SolanaSigner

        handler = PaymentHandler(networks={})
        cfg = NetworkConfig(
            type="svm",
            network=_SOLANA,
            pay_to=accepted["payTo"],
            token_address=_USDC_MINT,
            rpc_url="http://127.0.0.1:9",
            facilitator_key="unused",
        )
        scheme = Mock()
        scheme.get_extra.return_value = {"feePayer": str(fee_payer.pubkey())}
        scheme.settle.return_value = settle_result
        signer = SolanaSigner.__new__(SolanaSigner)
        handler._schemes[_SOLANA] = _NetworkScheme(cfg, scheme, svm_signer=signer)
        return handler

    async def test_gas_fields_are_ready_without_rpc_or_price_wait(self, stale_prices, monkeypatch):
        import asyncio

        from solders.keypair import Keypair
        from x402 import SettleResponse

        payment = stale_prices
        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler = self._handler(
            fee_payer,
            accepted,
            SettleResponse(success=True, transaction="sig", network=_SOLANA, payer="p"),
        )
        release = asyncio.Event()

        async def slow_fetch():
            await release.wait()
            return {"solana": {"usd": 120.0}}

        monkeypatch.setattr(payment, "_fetch_prices", slow_fetch)
        settled = await asyncio.wait_for(
            handler.settle(header, Decimal("1"), _SOLANA, {"provider_name": "x"}), 1
        )
        # The price fetch is still hanging; settle did not wait for it.
        assert payment._background_tasks
        release.set()
        await _drain_background(payment)

        assert settled["success"] is True
        assert settled["amount_usdc"] == 1.0
        assert settled["network_type"] == "svm"
        assert settled["gas_label"] == "SOL"
        assert settled["gas_cost_native"] == pytest.approx(10_001 / 1e9)
        assert settled["gas_cost_usd"] == pytest.approx(10_001 / 1e9 * 100.0)

    async def test_failed_settle_is_unchanged(self, stale_prices):
        from solders.keypair import Keypair
        from x402 import SettleResponse

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler = self._handler(
            fee_payer,
            accepted,
            SettleResponse(
                success=False,
                error_reason="transaction_failed",
                error_message="Simulation failed",
                network=_SOLANA,
                payer="p",
                transaction="",
            ),
        )
        settled = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert settlement_state(settled) == "failed"
        assert settled["error_reason"] == "transaction_failed: Simulation failed"
        assert "gas_cost_usd" not in settled


def _pending(transaction: str = "sig"):
    from x402 import SettleResponse

    return SettleResponse(
        success=False,
        error_reason="settlement_pending",
        error_message="confirm failed",
        network=_SOLANA,
        payer="p",
        transaction=transaction,
    )


class TestSolanaKnownFailure:
    """A Solana transfer that landed with an error or expired is failed, not unknown."""

    def _handler(self, fee_payer, accepted, results, terminal=None):  # noqa: ANN001
        handler = TestSolanaSettleGas()._handler(fee_payer, accepted, None)
        ns = handler._schemes[_SOLANA]
        ns.scheme.settle.side_effect = results
        terminal = terminal if terminal is not None else {}
        ns.svm_signer.terminal_failure = Mock(side_effect=lambda sig: terminal.get(sig))
        return handler, ns, terminal

    async def test_landed_error_skips_the_second_settle(self, stale_prices):
        from solders.keypair import Keypair

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler, ns, _ = self._handler(
            fee_payer,
            accepted,
            [_pending()],
            terminal={"sig": "Transaction failed: InsufficientFunds"},
        )
        settled = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert ns.scheme.settle.call_count == 1
        assert settlement_state(settled) == "failed"
        assert settled["pending"] is False
        ns.scheme._pending_store.delete.assert_called_once()
        assert (
            settled["error_reason"] == "transaction_failed: Transaction failed: InsufficientFunds"
        )

    async def test_failure_seen_by_the_second_settle(self, stale_prices):
        """The first confirm timed out; the retry saw the error at confirmed."""
        from solders.keypair import Keypair

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        terminal: dict[str, str] = {}
        calls = []

        def settle(*args):  # noqa: ANN002
            calls.append(args)
            if len(calls) == 2:
                terminal["sig"] = "Transaction expired (blockhash no longer valid)"
            return _pending()

        handler, ns, _ = self._handler(fee_payer, accepted, settle, terminal=terminal)
        settled = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert len(calls) == 2
        assert settlement_state(settled) == "failed"

    async def test_unknown_outcome_stays_unknown(self, stale_prices):
        from solders.keypair import Keypair

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler, ns, _ = self._handler(fee_payer, accepted, [_pending(), _pending()])
        settled = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert ns.scheme.settle.call_count == 2
        assert settlement_state(settled) == "unknown"
        # The pending entry stays, so a retry waits on the same transaction.
        ns.scheme._pending_store.delete.assert_not_called()


class TestSolanaKnownFailureWithSdk:
    """End to end through the SDK's ExactSvmScheme and its pending store."""

    async def test_failed_landing_is_failed_and_retry_does_not_poll(self, stale_prices):
        import threading

        from solders.keypair import Keypair
        from x402.mechanisms.svm.exact.facilitator import ExactSvmScheme

        from x402gate.core.svm_signer import SolanaSigner

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler = TestSolanaSettleGas()._handler(fee_payer, accepted, None)
        ns = handler._schemes[_SOLANA]
        signer = SolanaSigner.__new__(SolanaSigner)
        signer._terminal = {}
        signer._terminal_lock = threading.Lock()
        signer._inflight = {}
        signer._keypair = fee_payer
        confirms = []

        def confirm(signature, network):  # noqa: ANN001
            _ = network
            confirms.append(signature)
            known = signer.terminal_failure(signature)
            if known:
                raise RuntimeError(known)
            signer._record_terminal(signature, "Transaction failed: InstructionError")
            raise RuntimeError("Transaction failed: InstructionError")

        signer.confirm_transaction = confirm
        signer.sign_transaction = lambda tx, fee_payer, network: tx
        signer.send_transaction = lambda tx, network: "sig1"
        scheme = ExactSvmScheme(signer)
        scheme.verify = Mock(return_value=SimpleNamespace(is_valid=True, payer="p"))
        ns.scheme = scheme
        ns.svm_signer = signer

        settled = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert settlement_state(settled) == "failed"
        assert confirms == ["sig1"]
        assert scheme._pending_store.entries == {}

    async def test_replay_after_the_record_ages_out_is_verified_again(self, stale_prices):
        """A replayed expired payload goes through verify, not a 120 s wait on the old one."""
        import threading

        from solders.keypair import Keypair
        from x402.mechanisms.svm.exact.facilitator import ExactSvmScheme

        from x402gate.core.svm_signer import _EXPIRED_MESSAGE, _TERMINAL_TTL_S, SolanaSigner

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler = TestSolanaSettleGas()._handler(fee_payer, accepted, None)
        ns = handler._schemes[_SOLANA]
        signer = SolanaSigner.__new__(SolanaSigner)
        signer._terminal = {}
        signer._terminal_lock = threading.Lock()
        signer._inflight = {}
        signer._keypair = fee_payer
        confirms = []

        def confirm(signature, network):  # noqa: ANN001
            _ = network
            confirms.append(signature)
            if not signer.terminal_failure(signature) and len(confirms) == 1:
                signer._record_terminal(signature, _EXPIRED_MESSAGE)
            raise RuntimeError(signer.terminal_failure(signature) or "not confirmed within 120s")

        signer.confirm_transaction = confirm
        signer.sign_transaction = lambda tx, fee_payer, network: tx
        signer.send_transaction = lambda tx, network: "sig1"
        scheme = ExactSvmScheme(signer)
        scheme.verify = Mock(return_value=SimpleNamespace(is_valid=True, payer="p"))
        ns.scheme = scheme
        ns.svm_signer = signer

        first = await handler.settle(header, Decimal("1"), _SOLANA)
        assert settlement_state(first) == "failed"

        # The record ages out and another failure prunes it.
        signer._terminal["sig1"] = (_EXPIRED_MESSAGE, time.monotonic() - _TERMINAL_TTL_S - 1)
        signer._record_terminal("other", "Transaction failed: x")
        assert signer.terminal_failure("sig1") is None
        scheme.verify.return_value = SimpleNamespace(
            is_valid=False, invalid_reason="transaction_simulation_failed", payer="p"
        )

        replay = await handler.settle(header, Decimal("1"), _SOLANA)
        await _drain_background(stale_prices)

        assert settlement_state(replay) == "failed"
        assert confirms == ["sig1"]


class TestPermit2Refused:
    """Permit2 is never advertised, and payment_key cannot identify it."""

    def _header(self, handler: PaymentHandler) -> str:
        accepted = handler.get_payment_requirements_for_network(Decimal("1"), "eip155:8453")
        return _encoded(
            {
                "x402Version": 2,
                "accepted": accepted,
                "payload": {
                    "signature": "0x" + "11" * 65,
                    "permit2Authorization": {"from": "0xabc", "nonce": "1"},
                    "authorization": {"from": "0xabc", "nonce": "0x01"},
                },
            }
        )

    async def test_verify_refuses_without_calling_the_sdk(self, handler: PaymentHandler):
        ns = handler._schemes["eip155:8453"]
        ns.scheme = Mock()
        ok, network, payer, reason = await handler.verify(self._header(handler), Decimal("1"))

        assert ok is False
        assert network == "eip155:8453"
        assert payer == ""
        assert "Permit2" in reason
        ns.scheme.verify.assert_not_called()

    async def test_settle_fails_without_calling_the_sdk(self, handler: PaymentHandler):
        ns = handler._schemes["eip155:8453"]
        ns.scheme = Mock()
        settled = await handler.settle(self._header(handler), Decimal("1"), "eip155:8453")

        assert settlement_state(settled) == "failed"
        ns.scheme.settle.assert_not_called()


class TestChainExecutors:
    """Blocking SDK calls run on a pool per chain, not asyncio's default one."""

    async def test_base_verify_runs_on_the_base_pool(self, handler: PaymentHandler, fresh_prices):
        # Fresh prices: a refresh would leave the shared client on this test's loop.
        import threading

        ns = handler._schemes["eip155:8453"]
        seen = []

        def verify(*_args):  # noqa: ANN002
            seen.append(threading.current_thread().name)
            return SimpleNamespace(is_valid=False, payer="", invalid_reason="x")

        ns.scheme = Mock()
        ns.scheme.verify.side_effect = verify
        accepted = handler.get_payment_requirements_for_network(Decimal("1"), "eip155:8453")
        header = _encoded(
            {
                "x402Version": 2,
                "accepted": accepted,
                "payload": {"signature": "0x1", "authorization": {"from": "0xabc", "nonce": "1"}},
            }
        )
        await handler.verify(header, Decimal("1"))

        assert seen and seen[0].startswith("x402-base")

    async def test_solana_settle_runs_on_the_solana_pool(self, fresh_prices):
        import threading

        from solders.keypair import Keypair
        from x402 import SettleResponse

        fee_payer = Keypair()
        header, accepted = _solana_payment(fee_payer)
        handler = TestSolanaSettleGas()._handler(fee_payer, accepted, None)
        ns = handler._schemes[_SOLANA]
        seen = []

        def settle(*_args):  # noqa: ANN002
            seen.append(threading.current_thread().name)
            return SettleResponse(success=True, transaction="sig", network=_SOLANA, payer="p")

        ns.scheme.settle.side_effect = settle
        await handler.settle(header, Decimal("1"), _SOLANA)

        assert seen and seen[0].startswith("x402-solana")


# Errors a Base replica returns when its forward to the sequencer failed,
# possibly after the sequencer took the transaction.
_AMBIGUOUS_SEND_ERRORS = [
    "{'code': -32000, 'message': 'Post \"https://sequencer\": context deadline exceeded'}",
    "{'code': -32000, 'message': 'Post \"https://sequencer\": EOF'}",
    "{'code': -32603, 'message': 'Internal error'}",
    "upstream request timeout",
]
# Errors the node gives when it turned a transaction away before taking it.
_REFUSED_SEND_ERRORS = [
    "insufficient funds for gas * price + value",
    "transaction underpriced",
    "intrinsic gas too low",
    "exceeds block gas limit",
    "gas limit reached",
    "max fee per gas less than block base fee",
    "fee cap less than block base fee",
    "invalid sender",
    "invalid chain id",
    "only replay-protected (EIP-155) transactions allowed over RPC",
    "oversized data",
    "transaction type not supported",
    "tx fee (1.10 ether) exceeds the configured cap (1.00 ether)",
]


class _FakeNode:
    """A geth-like node for signed bytes b"raw<nonce>": a pool by nonce, and a
    pending count that stops at the first nonce it lacks, so a hole stalls the
    ones queued behind it. Each send takes the next of ``replies`` ("accept",
    "lost" or "refuse"), then accepts once they run out."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.pool: dict[int, bytes] = {}
        self.mined: dict[int, bytes] = {}

    @staticmethod
    def _hash(nonce: int) -> bytes:
        return bytes([nonce]) * 32

    def get_transaction_count(self, address, block) -> int:  # noqa: ANN001
        n = 0
        while n in self.mined or n in self.pool:
            n += 1
        return n

    def send_raw_transaction(self, raw: bytes) -> bytes:
        nonce = int(raw.removeprefix(b"raw"))
        reply = self.replies.pop(0) if self.replies else "accept"
        if reply == "lost":
            raise requests.ReadTimeout("read timed out")
        if reply == "refuse":
            raise Web3RPCError("insufficient funds for gas * price + value")
        if nonce in self.mined:
            raise Web3RPCError("nonce too low")
        if nonce in self.pool:
            raise Web3RPCError("already known")
        self.pool[nonce] = raw
        return self._hash(nonce)

    def get_transaction(self, tx_hash: bytes) -> dict:
        for nonce in (*self.pool, *self.mined):
            if self._hash(nonce) == bytes(tx_hash):
                return {"nonce": nonce}
        raise TransactionNotFound("not found")

    def mine(self) -> None:
        n = len(self.mined)
        while n in self.pool:
            self.mined[n] = self.pool.pop(n)
            n += 1


class TestNonceResync:
    """A failed send must not leave a nonce gap, nor hand one nonce out twice."""

    _TO = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    def _signer(self, handler: PaymentHandler, pending: int = 7):
        signer = handler._schemes["eip155:8453"].scheme._signer
        nonces: list[int] = []
        w3 = Mock()
        w3.eth.get_transaction_count.return_value = pending
        w3.eth.gas_price = 1
        w3.eth.send_raw_transaction.return_value = b"\x11" * 32
        w3.eth.get_transaction.side_effect = TransactionNotFound("not found")
        signer._w3 = w3
        signer._send_w3 = w3
        account = Mock(address=signer.address)
        account.sign_transaction.side_effect = lambda tx: (
            nonces.append(tx["nonce"]) or SimpleNamespace(raw_transaction=b"raw", hash=b"h")
        )
        signer._account = account
        return signer, w3, nonces

    def test_failed_gas_price_read_does_not_burn_the_nonce(self, handler: PaymentHandler):
        from unittest.mock import PropertyMock

        signer, w3, nonces = self._signer(handler)
        type(w3.eth).gas_price = PropertyMock(side_effect=[ConnectionError("429"), 1, 1])

        with pytest.raises(ConnectionError):
            signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 8]

    def test_refused_send_returns_the_nonce(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = [
            Web3RPCError("{'code': -32000, 'message': 'insufficient funds for gas * price'}"),
            b"\x11" * 32,
        ]

        with pytest.raises(Web3RPCError):
            signer.send_transaction(self._TO, b"")
        assert (signer._next_nonce, signer._free_nonces) == (7, set())
        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 7]

    def test_overlapping_failure_never_reissues_a_nonce(self, handler: PaymentHandler):
        """A=7 and B=8 are reserved, B is sent and queued, A fails before sending."""
        signer, w3, nonces = self._signer(handler)
        a = signer._reserve_nonce()
        b = signer._reserve_nonce()
        assert (a, b) == (7, 8)

        # B is in the mempool behind the gap, so the pending count stays 7.
        signer._release_nonce(a)
        signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 9]

    def test_returned_nonce_the_chain_used_is_skipped(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        a = signer._reserve_nonce()
        signer._reserve_nonce()
        signer._release_nonce(a)
        w3.eth.get_transaction_count.return_value = 9

        signer.send_transaction(self._TO, b"")

        assert nonces == [9]

    def test_refusal_while_the_node_has_it_keeps_the_nonce(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = [
            Web3RPCError("insufficient funds for gas * price + value"),
            b"\x11" * 32,
        ]
        w3.eth.get_transaction.side_effect = None
        w3.eth.get_transaction.return_value = {"nonce": 7}

        # The node took it, so the receipt wait decides instead of a failure.
        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()
        assert signer._unconfirmed == {}
        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 8]

    def test_send_and_lookup_both_unanswered_keep_the_nonce(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = ConnectionError("reset")
        w3.eth.get_transaction.side_effect = ConnectionError("reset")

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        assert signer._next_nonce == 8
        assert signer._unconfirmed == {7: b"raw"}

    def test_unconfirmed_transaction_is_resent_before_the_next_nonce(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = [ConnectionError("reset"), b"\x22", b"\x33"]
        w3.eth.get_transaction.side_effect = ConnectionError("reset")
        signer.send_transaction(self._TO, b"")

        # The chain still counts 7 as next, so 7 never arrived: send it again.
        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 8]
        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw"] * 3
        assert signer._unconfirmed == {}

    def test_refused_resend_gives_its_nonce_to_the_next_send(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = [
            ConnectionError("reset"),
            b"\x22",  # nonce 8
            Web3RPCError("insufficient funds"),  # resending 7
            b"\x33",
        ]
        w3.eth.get_transaction.side_effect = ConnectionError("reset")
        signer.send_transaction(self._TO, b"")
        signer._unconfirmed.clear()  # nonce 8 goes out normally
        w3.eth.get_transaction.side_effect = TransactionNotFound("not found")
        signer.send_transaction(self._TO, b"")
        signer._unconfirmed[7] = b"raw7"

        signer.send_transaction(self._TO, b"")

        assert nonces == [7, 8, 7]
        assert signer._next_nonce == 9
        assert signer._unconfirmed == {}

    def test_unconfirmed_transaction_the_chain_counted_is_dropped(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.get_transaction_count.return_value = 8

        signer.send_transaction(self._TO, b"")

        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw"]
        assert nonces == [8]
        assert signer._unconfirmed == {}

    def test_unanswered_resend_is_kept_for_the_next_send(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.send_raw_transaction.side_effect = [ConnectionError("reset"), b"\x22"]

        signer.send_transaction(self._TO, b"")

        assert nonces == [8]
        assert signer._unconfirmed == {7: b"raw7"}

    def test_sends_are_serial_so_a_failure_leaves_no_gap(self, handler: PaymentHandler):
        """B waits for A's reply; A's refusal gives 7 back and B sends 7, never 8 alone."""
        import threading

        signer, w3, nonces = self._signer(handler)
        a_sending = threading.Event()
        a_may_fail = threading.Event()

        def send(raw):  # noqa: ANN001, ANN202
            if len(nonces) == 1:
                a_sending.set()
                a_may_fail.wait(5)
                raise Web3RPCError("insufficient funds for gas * price")
            return b"\x22" * 32

        w3.eth.send_raw_transaction.side_effect = send
        errors = []
        a = threading.Thread(target=lambda: _record(errors, signer.send_transaction, self._TO))
        a.start()
        assert a_sending.wait(5)
        b = threading.Thread(target=signer.send_transaction, args=(self._TO, b""))
        b.start()
        b.join(0.2)
        assert b.is_alive() and nonces == [7]  # B waits on the send lock

        a_may_fail.set()
        a.join(5)
        b.join(5)

        assert nonces == [7, 7]
        assert [type(e) for e in errors] == [Web3RPCError]
        assert (signer._next_nonce, signer._free_nonces) == (8, set())

    def test_rpc_calls_are_bounded(self, handler: PaymentHandler):
        signer = handler._schemes["eip155:8453"].scheme._signer
        assert signer._w3.provider._request_kwargs == {"timeout": 10}
        assert signer._w3.provider.endpoint_uri == "https://mainnet.base.org"
        assert signer._send_w3.provider._request_kwargs == {"timeout": 10}
        assert signer._send_w3.provider.endpoint_uri == "https://mainnet.base.org"
        # Calls under the send lock are tried once; reads get one retry.
        assert signer._send_w3.provider.exception_retry_configuration is None
        assert signer._w3.provider.exception_retry_configuration.retries == 2

    @pytest.mark.parametrize(
        "error",
        [
            requests.ReadTimeout("read timed out"),
            requests.HTTPError("504 Server Error: Gateway Timeout"),
            requests.ConnectionError("connection reset by peer"),
            RequestTimedOut("{'code': -32000, 'message': 'request timed out'}"),
        ],
        ids=["read-timeout", "http-504", "reset", "node-timeout"],
    )
    def test_lost_send_reply_keeps_the_nonce_even_if_the_lookup_misses(
        self, handler: PaymentHandler, error
    ):
        """A load-balanced replica may not have a delivered transaction yet."""
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = error

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        assert signer._unconfirmed == {7: b"raw"}
        assert signer._reserve_nonce() == 8
        # The lookup cannot prove it never arrived, so it is not asked.
        w3.eth.get_transaction.assert_not_called()

    @pytest.mark.parametrize(
        "message",
        ["already known", "AlreadyKnown", "Known transaction: 0xabc", "already imported"],
    )
    def test_reply_that_the_node_knows_the_transaction_keeps_it(
        self, handler: PaymentHandler, message
    ):
        """The node has this very transaction (same hash): the receipt wait decides."""
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError(
            f"{{'code': -32000, 'message': '{message}'}}"
        )

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        assert signer._unconfirmed == {}
        assert signer._reserve_nonce() == 8
        w3.eth.get_transaction.assert_not_called()

    def test_refusal_the_node_cannot_check_keeps_the_nonce(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError("insufficient funds")
        w3.eth.get_transaction.side_effect = requests.ReadTimeout("read timed out")

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        assert signer._unconfirmed == {7: b"raw"}
        assert signer._reserve_nonce() == 8

    def test_insufficient_funds_refusal_gives_the_nonce_back(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError(
            "{'code': -32000, 'message': 'insufficient funds for gas * price + value'}"
        )

        with pytest.raises(Web3RPCError):
            signer.send_transaction(self._TO, b"")

        assert signer._unconfirmed == {}
        assert signer._reserve_nonce() == 7

    @pytest.mark.parametrize(
        "message", ["already known", "nonce too low", "replacement transaction underpriced"]
    )
    def test_resend_the_node_holds_keeps_its_nonce(self, handler: PaymentHandler, message):
        """A lagging count still says 7, but the node has 7: the next send must not reuse it."""
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.send_raw_transaction.side_effect = [Web3RPCError(message), b"\x22"]

        signer.send_transaction(self._TO, b"")

        assert nonces == [8]
        assert signer._unconfirmed == {}
        assert signer._free_nonces == set()

    def test_refused_resend_the_node_cannot_check_is_kept(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.send_raw_transaction.side_effect = [Web3RPCError("insufficient funds"), b"\x22"]
        w3.eth.get_transaction.side_effect = requests.ConnectionError("reset")

        signer.send_transaction(self._TO, b"")

        assert nonces == [8]
        assert signer._unconfirmed == {7: b"raw7"}

    @pytest.mark.parametrize("message", _AMBIGUOUS_SEND_ERRORS)
    def test_ambiguous_resend_error_is_kept_even_if_the_lookup_misses(
        self, handler: PaymentHandler, message
    ):
        """A replica's failed forward may come after the sequencer took the resend."""
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 9
        signer._unconfirmed.update({7: b"raw7", 8: b"raw8"})
        w3.eth.send_raw_transaction.side_effect = [Web3RPCError(message), b"\x22"]

        signer.send_transaction(self._TO, b"")

        assert nonces == [9]
        assert signer._unconfirmed == {7: b"raw7", 8: b"raw8"}
        assert signer._free_nonces == set()
        w3.eth.get_transaction.assert_not_called()

    def test_resend_refusal_the_node_has_is_dropped_and_keeps_its_nonce(
        self, handler: PaymentHandler
    ):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.send_raw_transaction.side_effect = [Web3RPCError("transaction underpriced"), b"\x22"]
        w3.eth.get_transaction.side_effect = None
        w3.eth.get_transaction.return_value = {"nonce": 7}

        signer.send_transaction(self._TO, b"")

        assert nonces == [8]
        assert signer._unconfirmed == {}
        assert signer._free_nonces == set()

    def test_unanswered_resend_stops_the_resend_loop(self, handler: PaymentHandler):
        """Each unanswered resend holds the send lock, so the rest wait for the next send."""
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 9
        signer._unconfirmed.update({7: b"raw7", 8: b"raw8"})
        w3.eth.send_raw_transaction.side_effect = [requests.ReadTimeout("timed out"), b"\x22"]

        signer.send_transaction(self._TO, b"")

        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw7", b"raw"]
        assert nonces == [9]
        assert signer._unconfirmed == {7: b"raw7", 8: b"raw8"}

    def test_resend_node_timeout_is_kept(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.send_raw_transaction.side_effect = [RequestTimedOut("request timed out"), b"\x22"]

        signer.send_transaction(self._TO, b"")

        assert nonces == [8]
        assert signer._unconfirmed == {7: b"raw7"}

    def test_failed_count_skips_the_resend_and_still_sends(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        w3.eth.get_transaction_count.side_effect = [requests.ReadTimeout("read timed out"), 7]

        signer.send_transaction(self._TO, b"")

        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw"]
        assert nonces == [8]
        assert signer._unconfirmed == {7: b"raw7"}

    @pytest.mark.parametrize("message", _REFUSED_SEND_ERRORS)
    def test_each_refusal_gives_the_nonce_back(self, handler: PaymentHandler, message):
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError(
            f"{{'code': -32000, 'message': '{message}'}}"
        )

        with pytest.raises(Web3RPCError):
            signer.send_transaction(self._TO, b"")

        assert signer._unconfirmed == {}
        assert signer._reserve_nonce() == 7

    @pytest.mark.parametrize("message", _AMBIGUOUS_SEND_ERRORS)
    def test_ambiguous_send_error_keeps_the_nonce_even_if_the_lookup_misses(
        self, handler: PaymentHandler, message
    ):
        """A replica's failed forward is not a refusal: the sequencer may have the transfer."""
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError(message)

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        assert signer._unconfirmed == {7: b"raw"}
        assert signer._reserve_nonce() == 8
        w3.eth.get_transaction.assert_not_called()

    @staticmethod
    def _per_nonce_hashes(signer, nonces: list[int]) -> None:  # noqa: ANN001
        """Sign each nonce to its own raw bytes and hash, recording it in ``nonces``."""
        signer._account.sign_transaction.side_effect = lambda tx: (
            nonces.append(tx["nonce"])
            or SimpleNamespace(
                raw_transaction=b"raw%d" % tx["nonce"], hash=bytes([tx["nonce"]]) * 32
            )
        )

    @pytest.mark.parametrize(
        "message",
        ["nonce too low: next nonce 8, tx nonce 7", "replacement transaction underpriced"],
    )
    def test_first_send_whose_nonce_is_taken_is_not_signed_again(
        self, handler: PaymentHandler, message
    ):
        """An upstream retry may have delivered it; a second copy of the authorization reverts."""
        signer, w3, nonces = self._signer(handler)
        self._per_nonce_hashes(signer, nonces)
        w3.eth.send_raw_transaction.side_effect = [Web3RPCError(message), b"\x08" * 32]

        tx_hash = signer.send_transaction(self._TO, b"")

        # Its own hash: the receipt wait, then the reconcile, decide.
        assert tx_hash == "0x" + "07" * 32
        assert nonces == [7]
        assert w3.eth.send_raw_transaction.call_count == 1
        # Some transaction holds 7, so resending it cannot help.
        assert signer._unconfirmed == {}
        assert signer._free_nonces == set()
        assert signer._reserve_nonce() == 8

    def test_first_send_whose_nonce_the_node_holds_for_it_returns_its_hash(
        self, handler: PaymentHandler
    ):
        signer, w3, nonces = self._signer(handler)
        self._per_nonce_hashes(signer, nonces)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError("nonce too low")
        w3.eth.get_transaction.side_effect = None
        w3.eth.get_transaction.return_value = {"nonce": 7}

        assert signer.send_transaction(self._TO, b"") == "0x" + "07" * 32

        assert nonces == [7]
        assert signer._unconfirmed == {}
        assert signer._reserve_nonce() == 8

    def test_taken_nonce_the_node_cannot_check_is_kept_unconfirmed(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        self._per_nonce_hashes(signer, nonces)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError("nonce too low")
        w3.eth.get_transaction.side_effect = requests.ReadTimeout("read timed out")

        assert signer.send_transaction(self._TO, b"") == "0x" + "07" * 32

        assert nonces == [7]
        assert signer._unconfirmed == {7: b"raw7"}
        assert signer._reserve_nonce() == 8

    def test_refusal_after_the_lock_budget_skips_the_lookup_and_keeps_the_nonce(
        self, handler: PaymentHandler, monkeypatch
    ):
        monkeypatch.setattr("x402gate.core.payment._EVM_SEND_LOCK_BUDGET_S", 0)
        signer, w3, nonces = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = Web3RPCError("insufficient funds")

        assert signer.send_transaction(self._TO, b"") == "0x" + b"h".hex()

        w3.eth.get_transaction.assert_not_called()
        assert signer._unconfirmed == {7: b"raw"}
        assert signer._reserve_nonce() == 8

    def test_resends_stop_at_the_lock_budget_and_keep_the_rest(
        self, handler: PaymentHandler, monkeypatch
    ):
        monkeypatch.setattr("x402gate.core.payment._EVM_SEND_LOCK_BUDGET_S", 0.05)
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 10
        signer._unconfirmed.update({7: b"raw7", 8: b"raw8", 9: b"raw9"})

        def send(raw):  # noqa: ANN001, ANN202
            time.sleep(0.06)
            raise Web3RPCError("already known")

        w3.eth.send_raw_transaction.side_effect = send

        signer.send_transaction(self._TO, b"")

        # 7 went out and the node had it; 8 and 9 wait for the next call.
        sent = [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list]
        assert sent == [b"raw7", b"raw"]
        assert signer._unconfirmed == {8: b"raw8", 9: b"raw9"}
        assert nonces == [10]

    def test_lock_is_held_for_the_budget_plus_a_count_and_a_send(
        self, handler: PaymentHandler, monkeypatch
    ):
        """A slow RPC no longer holds the lock for count, resend, lookup, count, send, lookup."""
        delay = 0.05
        monkeypatch.setattr("x402gate.core.payment._EVM_SEND_LOCK_BUDGET_S", 2.5 * delay)
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        calls: list[str] = []

        def slow(name, result=None, error=None):  # noqa: ANN001, ANN202
            def call(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                calls.append(name)
                time.sleep(delay)
                if error is not None:
                    raise error
                return result

            return call

        w3.eth.get_transaction_count.side_effect = slow("count", 7)
        w3.eth.send_raw_transaction.side_effect = slow(
            "send", error=Web3RPCError("insufficient funds")
        )
        w3.eth.get_transaction.side_effect = slow("lookup", error=requests.ReadTimeout("t"))

        start = time.monotonic()
        signer.send_transaction(self._TO, b"")
        elapsed = time.monotonic() - start

        assert calls == ["count", "send", "lookup", "count", "send"]
        # Budget, the lookup in progress at it, then the count and the send.
        assert elapsed < 2.5 * delay + 3 * delay + 0.1
        assert set(signer._unconfirmed) == {7, 8}

    def test_public_resend_sends_unconfirmed_and_drops_what_the_chain_counted(
        self, handler: PaymentHandler
    ):
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 9
        signer._unconfirmed.update({7: b"raw7", 8: b"raw8"})
        w3.eth.get_transaction_count.return_value = 8

        signer.resend_unconfirmed()

        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw8"]
        assert signer._unconfirmed == {}
        assert nonces == []

    def test_public_resend_with_nothing_unconfirmed_calls_no_rpc(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)

        signer.resend_unconfirmed()

        w3.eth.get_transaction_count.assert_not_called()

    def _on_node(self, handler: PaymentHandler, node: _FakeNode):
        signer, w3, nonces = self._signer(handler)
        self._per_nonce_hashes(signer, nonces)
        signer._send_w3 = SimpleNamespace(eth=node)
        return signer, nonces

    def test_refused_public_resend_behind_a_queued_send_gets_its_gap_filled(
        self, handler: PaymentHandler
    ):
        """S1's 0 never arrives, S2's 1 queues behind it, then the node refuses 0."""
        node = _FakeNode("lost", "lost", "accept", "refuse", "refuse")
        signer, nonces = self._on_node(handler, node)
        signer.send_transaction(self._TO, b"")  # S1 at 0: lost
        signer.send_transaction(self._TO, b"")  # S1's resend lost, S2 at 1: queued
        assert sorted(node.pool) == [1] and node.get_transaction_count(None, "pending") == 0

        # Settle's pending path: the resend of 0 is refused, and so is the gap fill.
        signer.resend_unconfirmed()
        assert (signer._unconfirmed, signer._free_nonces, signer._next_nonce) == ({}, {0}, 2)

        # The next reconcile poll fills 0, so S2's transfer can be mined.
        signer.resend_unconfirmed()

        assert node.get_transaction_count(None, "pending") == 2
        assert signer._free_nonces == set()
        assert nonces == [0, 1, 0, 0]
        fill = signer._account.sign_transaction.call_args.args[0]
        assert (fill["to"], fill["value"], fill["gas"]) == (signer.address, 0, 21_000)
        node.mine()
        assert sorted(node.mined) == [0, 1]

    def test_gap_fill_that_gets_no_answer_is_resent_later(self, handler: PaymentHandler):
        node = _FakeNode("lost")
        signer, nonces = self._on_node(handler, node)
        node.pool[1] = b"raw1"
        signer._next_nonce = 2
        signer._free_nonces = {0}

        signer.resend_unconfirmed()
        assert signer._unconfirmed == {0: b"raw0"} and signer._free_nonces == set()
        signer.resend_unconfirmed()

        assert node.get_transaction_count(None, "pending") == 2
        assert signer._unconfirmed == {}
        assert nonces == [0]

    def test_gap_fill_skips_given_back_nonces_the_chain_used(self, handler: PaymentHandler):
        """A stale given-back nonce must not make the fill burn a fresh one."""
        signer, w3, nonces = self._signer(handler)
        signer._next_nonce = 9
        signer._free_nonces = {7}
        w3.eth.get_transaction_count.return_value = 9

        signer.resend_unconfirmed()

        assert nonces == []
        w3.eth.send_raw_transaction.assert_not_called()
        assert (signer._free_nonces, signer._next_nonce) == (set(), 9)

    def test_failed_gas_price_read_still_resends_and_leaves_the_gap(self, handler: PaymentHandler):
        from unittest.mock import PropertyMock

        signer, w3, nonces = self._signer(handler)
        type(w3.eth).gas_price = PropertyMock(side_effect=ConnectionError("429"))
        signer._next_nonce = 9
        signer._free_nonces = {7}
        signer._unconfirmed[8] = b"raw8"

        signer.resend_unconfirmed()

        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw8"]
        assert nonces == []
        assert signer._free_nonces == {7}

    def test_build_under_the_lock_needs_no_chain_id_call(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        signer._chain_id = 8453
        func = Mock()
        func.return_value.build_transaction.side_effect = lambda f: {**f, "data": "0xab"}
        w3.eth.contract.return_value.functions.transferWithAuthorization = func

        signer.write_contract(self._TO, [], "transferWithAuthorization")

        assert func.return_value.build_transaction.call_args.args[0]["chainId"] == 8453

    def test_write_contract_returns_the_nonce_on_a_failed_build(self, handler: PaymentHandler):
        signer, w3, nonces = self._signer(handler)
        func = Mock()
        func.return_value.build_transaction.side_effect = [
            ConnectionError("chain id"),
            {"data": "0xab", "nonce": 7},
        ]
        w3.eth.contract.return_value.functions.transferWithAuthorization = func

        with pytest.raises(ConnectionError):
            signer.write_contract(self._TO, [], "transferWithAuthorization")
        tx_hash = signer.write_contract(
            self._TO, [], "transferWithAuthorization", data_suffix="0xcd"
        )

        assert func.return_value.build_transaction.call_args_list[1].args[0]["nonce"] == 7
        assert signer._account.sign_transaction.call_args.args[0]["data"].startswith("0xab")
        assert tx_hash == "0x" + "11" * 32

    def test_receipt_timeout_keeps_the_cursor(self, handler: PaymentHandler):
        """A slow transaction is still pending; resetting would reuse its nonce."""
        signer = handler._schemes["eip155:8453"].scheme._signer
        signer._next_nonce = 9
        signer._w3 = Mock()
        signer._w3.eth.wait_for_transaction_receipt.side_effect = TimeoutError("no receipt")

        with pytest.raises(TimeoutError):
            signer.wait_for_transaction_receipt("0x" + "11" * 32)

        assert signer._next_nonce == 9

    def test_receipt_wait_is_bounded(self, handler: PaymentHandler):
        signer = handler._schemes["eip155:8453"].scheme._signer
        assert signer._confirmation_timeout_seconds == 30


def _signed_chain_id(raw: bytes) -> int | None:
    """The EIP-155 chain id a signed legacy transaction carries, None when unprotected."""
    import rlp

    v = int.from_bytes(rlp.decode(raw)[6], "big")
    return None if v in (27, 28) else (v - 35) // 2


def _signed_nonce(raw: bytes) -> int:
    import rlp

    return int.from_bytes(rlp.decode(raw)[0], "big")


def _evm_handler(network: str) -> PaymentHandler:
    cfg = NetworkConfig(
        type="evm",
        network=network,
        pay_to="0x1234567890abcdef1234567890abcdef12345678",
        token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        token_name="USD Coin",
        token_version="2",
        rpc_url="https://sepolia.base.org",
        facilitator_key="0x" + "00" * 31 + "01",
    )
    return PaymentHandler(networks={"base": cfg})


class TestEip155ChainId:
    """Every Base transaction is signed with its chain id, so it cannot be replayed elsewhere."""

    _TO = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

    def _signer(self, handler: PaymentHandler, network: str = "eip155:8453"):
        """The real signer and key, with a mocked node that records each raw send."""
        from unittest.mock import PropertyMock

        signer = handler._schemes[network].scheme._signer
        w3 = Mock()
        w3.eth.get_transaction_count.return_value = 7
        w3.eth.gas_price = 1
        w3.eth.send_raw_transaction.return_value = b"\x11" * 32
        w3.eth.get_transaction.side_effect = TransactionNotFound("not found")
        # The chain id must come from the network, not a node read.
        type(w3.eth).chain_id = PropertyMock(side_effect=AssertionError("chain id read"))
        signer._w3 = w3
        signer._send_w3 = w3
        return signer, w3

    @staticmethod
    def _sent(w3: Mock) -> list[bytes]:
        return [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list]

    def test_sdk_send_transaction_would_sign_without_a_chain_id(self):
        """The gap this closes: eth_account signs a dict without chainId as unprotected."""
        from eth_account import Account

        account = Account.from_key("0x" + "00" * 31 + "01")
        signed = account.sign_transaction(
            {"to": self._TO, "data": b"", "nonce": 0, "gas": 21000, "gasPrice": 1, "value": 0}
        )
        assert _signed_chain_id(signed.raw_transaction) is None

    def test_send_transaction_signs_base_mainnet_chain_id(self, handler: PaymentHandler):
        signer, w3 = self._signer(handler)

        signer.send_transaction(self._TO, b"\x01")

        (raw,) = self._sent(w3)
        assert _signed_chain_id(raw) == 8453

    def test_write_contract_signs_base_mainnet_chain_id(self, handler: PaymentHandler):
        signer, w3 = self._signer(handler)
        func = Mock()
        func.return_value.build_transaction.side_effect = lambda f: {
            **f,
            "to": self._TO,
            "data": "0xab",
            "value": 0,
        }
        w3.eth.contract.return_value.functions.transferWithAuthorization = func

        signer.write_contract(self._TO, [], "transferWithAuthorization", data_suffix="0xcd")

        (raw,) = self._sent(w3)
        assert _signed_chain_id(raw) == 8453

    def test_resend_after_a_lost_reply_keeps_the_chain_id(self, handler: PaymentHandler):
        signer, w3 = self._signer(handler)
        w3.eth.send_raw_transaction.side_effect = [
            RequestTimedOut("timed out"),
            b"\x11" * 32,
            b"\x22" * 32,
        ]

        signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")

        first, resent, second = self._sent(w3)
        assert resent == first
        assert [_signed_chain_id(r) for r in (first, resent, second)] == [8453] * 3
        assert [_signed_nonce(r) for r in (first, second)] == [7, 8]

    def test_base_sepolia_signs_its_own_chain_id(self):
        handler = _evm_handler("eip155:84532")
        signer, w3 = self._signer(handler, "eip155:84532")

        signer.send_transaction(self._TO, b"")

        assert signer.get_chain_id() == 84532
        assert _signed_chain_id(self._sent(w3)[0]) == 84532

    def test_without_a_configured_id_the_node_is_read_once(self):
        from unittest.mock import PropertyMock

        signer = _ResyncingWeb3Signer(
            private_key="0x" + "00" * 31 + "01",
            rpc_url="http://127.0.0.1:1",
            confirmation_timeout_seconds=1,
        )
        w3 = Mock()
        w3.eth.get_transaction_count.return_value = 7
        w3.eth.gas_price = 1
        w3.eth.send_raw_transaction.return_value = b"\x11" * 32
        chain_id = PropertyMock(return_value=8453)
        type(w3.eth).chain_id = chain_id
        signer._w3 = w3
        signer._send_w3 = w3

        signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")

        assert chain_id.call_count == 1
        assert [_signed_chain_id(r) for r in self._sent(w3)] == [8453, 8453]

    def test_failed_chain_id_read_reserves_no_nonce(self):
        from unittest.mock import PropertyMock

        signer = _ResyncingWeb3Signer(
            private_key="0x" + "00" * 31 + "01",
            rpc_url="http://127.0.0.1:1",
            confirmation_timeout_seconds=1,
        )
        w3 = Mock()
        w3.eth.get_transaction_count.return_value = 7
        w3.eth.gas_price = 1
        w3.eth.send_raw_transaction.return_value = b"\x11" * 32
        type(w3.eth).chain_id = PropertyMock(side_effect=[ConnectionError("429"), 8453])
        signer._w3 = w3
        signer._send_w3 = w3

        with pytest.raises(ConnectionError):
            signer.send_transaction(self._TO, b"")
        signer.send_transaction(self._TO, b"")

        (raw,) = self._sent(w3)
        assert _signed_nonce(raw) == 7
        assert _signed_chain_id(raw) == 8453

    def test_network_chain_id_parsing(self):
        from x402gate.core.payment import _evm_chain_id

        assert _evm_chain_id("eip155:8453") == 8453
        assert _evm_chain_id("eip155:84532") == 84532
        assert _evm_chain_id("solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp") is None
        assert _evm_chain_id("eip155:") is None
        assert _evm_chain_id("base") is None


def _record(errors: list, fn, *args) -> None:  # noqa: ANN001
    try:
        fn(*args, b"")
    except Exception as e:
        errors.append(e)


class TestBaseSendLostReply:
    """A send whose reply was lost after the node took it is awaited, not failed."""

    async def _settle(self, handler: PaymentHandler, receipt, send_error=None, lookup=None):  # noqa: ANN001
        from unittest.mock import patch

        from x402.mechanisms.evm.exact.eip3009_utils import EIP3009SignatureClassification

        ns = handler._schemes["eip155:8453"]
        ns.w3 = None  # no gas cost lookup against the real RPC
        scheme = ns.scheme
        signer = scheme._signer
        w3 = Mock()
        w3.eth.get_transaction_count.return_value = 7
        w3.eth.gas_price = 1
        w3.eth.send_raw_transaction.side_effect = send_error or TimeoutError("read timeout")
        if lookup is None:
            w3.eth.get_transaction.return_value = {"nonce": 7}
        else:
            w3.eth.get_transaction.side_effect = lookup
        func = Mock()
        func.return_value.build_transaction.side_effect = lambda f: {**f, "data": "0xab"}
        w3.eth.contract.return_value.functions.transferWithAuthorization = func
        if isinstance(receipt, (Exception, list)):
            w3.eth.wait_for_transaction_receipt.side_effect = receipt
        else:
            w3.eth.wait_for_transaction_receipt.return_value = receipt
        signer._w3 = w3
        signer._send_w3 = w3
        signer._account = Mock(address=signer.address)
        signer._account.sign_transaction.return_value = SimpleNamespace(
            raw_transaction=b"raw", hash=b"\x12" * 32
        )

        accepted = handler.get_payment_requirements_for_network(Decimal("1"), "eip155:8453")
        auth = {
            "from": "0x" + "ab" * 20,
            "to": accepted["payTo"],
            "value": accepted["amount"],
            "validAfter": "0",
            "validBefore": str(int(time.time()) + 600),
            "nonce": "0x" + "01" * 32,
        }
        header = _encoded(
            {
                "x402Version": 2,
                "accepted": accepted,
                "payload": {"signature": "0x" + "11" * 65, "authorization": auth},
            }
        )
        classification = EIP3009SignatureClassification(
            valid=True,
            is_smart_wallet=False,
            is_undeployed=False,
            sig_data=SimpleNamespace(
                inner_signature=b"\x11" * 65, factory=b"", factory_calldata=b""
            ),
        )
        verified = SimpleNamespace(is_valid=True, payer=auth["from"], invalid_reason="")
        with (
            patch.object(type(scheme), "_verify", return_value=(verified, classification)),
            patch("x402.mechanisms.evm.exact.facilitator.has_deployment_info", return_value=False),
            # The receipt's Transfer log is the SDK's own check, not under test here.
            patch(
                "x402.mechanisms.evm.exact.facilitator.verify_eip3009_transfer_event",
                return_value=True,
            ),
        ):
            settled = await handler.settle(header, Decimal("1"), "eip155:8453")
        return settled, signer

    async def test_mined_transfer_is_ok_and_keeps_the_nonce(self, handler: PaymentHandler):
        receipt = {"status": 1, "blockNumber": 1, "logs": []}

        settled, signer = await self._settle(handler, receipt)

        assert settlement_state(settled) == "ok"
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._next_nonce == 8

    async def test_unanswered_receipt_is_unknown_not_failed(self, handler: PaymentHandler):
        settled, signer = await self._settle(handler, TimeoutError("no receipt"))

        assert settlement_state(settled) == "unknown"
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._next_nonce == 8

    @pytest.mark.parametrize(
        "send_error",
        [requests.ReadTimeout("read timed out"), Web3RPCError("already known")],
        ids=["timeout", "already-known"],
    )
    async def test_lookup_miss_after_a_lost_reply_is_unknown_not_failed(
        self, handler: PaymentHandler, send_error
    ):
        """The replica the lookup reached has not seen the delivered transaction yet."""
        settled, signer = await self._settle(
            handler,
            TimeoutError("no receipt"),
            send_error=send_error,
            lookup=TransactionNotFound("not found"),
        )

        assert settlement_state(settled) == "unknown"
        assert settled["pending"] is True
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._next_nonce == 8

    @pytest.mark.parametrize("message", _AMBIGUOUS_SEND_ERRORS)
    async def test_failed_forward_with_a_lookup_miss_is_unknown_not_failed(
        self, handler: PaymentHandler, message
    ):
        """The replica's forward failed, maybe after the sequencer took it: it may still land."""
        settled, signer = await self._settle(
            handler,
            TimeoutError("no receipt"),
            send_error=Web3RPCError(message),
            lookup=TransactionNotFound("not found"),
        )

        assert settlement_state(settled) == "unknown"
        assert settled["pending"] is True
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._unconfirmed == {7: b"raw"}
        assert signer._reserve_nonce() == 8

    @pytest.mark.parametrize(
        "send_error",
        [
            requests.ConnectTimeout("connect timed out"),
            requests.ConnectionError("connection refused"),
            requests.HTTPError("429 Too Many Requests"),
        ],
        ids=["connect-timeout", "refused", "http-429"],
    )
    async def test_unsent_transfer_is_sent_again_before_the_pending_wait(
        self, handler: PaymentHandler, send_error
    ):
        """The pending retry only waits, so the transfer goes out again before it."""
        settled, signer = await self._settle(
            handler, TimeoutError("no receipt"), send_error=send_error
        )

        assert settlement_state(settled) == "unknown"
        w3 = signer._send_w3
        assert w3.eth.send_raw_transaction.call_count == 2
        assert [c.args[0] for c in w3.eth.send_raw_transaction.call_args_list] == [b"raw"] * 2
        # Still no answer: kept for the reconcile's polls.
        assert signer._unconfirmed == {7: b"raw"}

    async def test_resent_transfer_that_lands_in_the_pending_wait_is_ok(
        self, handler: PaymentHandler
    ):
        receipt = {"status": 1, "blockNumber": 1, "logs": []}
        settled, signer = await self._settle(
            handler,
            [TimeoutError("no receipt"), receipt],
            send_error=[requests.ConnectTimeout("connect timed out"), b"\x12" * 32],
        )

        assert settlement_state(settled) == "ok"
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._unconfirmed == {}

    @pytest.mark.parametrize(
        ("receipt", "state"),
        [({"status": 1, "blockNumber": 1, "logs": []}, "ok"), (TimeoutError("none"), "unknown")],
        ids=["mined", "no-receipt"],
    )
    async def test_taken_nonce_with_a_lookup_miss_is_awaited_not_signed_again(
        self, handler: PaymentHandler, receipt, state
    ):
        """An upstream retry hit "nonce too low" after the sequencer took the transfer.

        A copy at a new nonce would revert on the used authorization and read
        as a failed payment while the USDC moved.
        """
        settled, signer = await self._settle(
            handler,
            receipt,
            send_error=Web3RPCError("nonce too low: next nonce 8, tx nonce 7"),
            lookup=TransactionNotFound("not found"),
        )

        assert settlement_state(settled) == state
        assert settled["transaction"] == "0x" + "12" * 32
        assert signer._account.sign_transaction.call_count == 1
        assert signer._send_w3.eth.send_raw_transaction.call_count == 1
        assert signer._unconfirmed == {}
        assert signer._reserve_nonce() == 8

    async def test_refused_send_is_a_known_failure(self, handler: PaymentHandler):
        settled, signer = await self._settle(
            handler,
            TimeoutError("no receipt"),
            send_error=Web3RPCError("insufficient funds for gas * price + value"),
            lookup=TransactionNotFound("not found"),
        )

        assert settlement_state(settled) == "failed"
        assert signer._reserve_nonce() == 7


class TestSendOutcome:
    @pytest.mark.parametrize(
        ("error", "outcome"),
        [
            (requests.ReadTimeout("x"), "unknown"),
            (requests.ConnectTimeout("x"), "unknown"),
            (requests.HTTPError("502"), "unknown"),
            (TimeoutError("x"), "unknown"),
            (RequestTimedOut("request timed out"), "unknown"),
            (Web3RPCError("{'code': -32000, 'message': 'already known'}"), "known"),
            (Web3RPCError("transaction already imported"), "known"),
            (Web3RPCError("nonce too low"), "taken"),
            (Web3RPCError("replacement transaction underpriced"), "taken"),
            (Web3RPCError("transaction underpriced"), "refused"),
            (Web3RPCError("insufficient funds for gas * price + value"), "refused"),
            (Web3RPCError("intrinsic gas too low"), "refused"),
            (Web3RPCError("internal error"), "unknown"),
            (Web3RPCError("{'code': -32000, 'message': 'something new'}"), "unknown"),
        ],
    )
    def test_classification(self, error, outcome):
        assert _send_outcome(error) == outcome

    @pytest.mark.parametrize("message", _REFUSED_SEND_ERRORS)
    def test_refusals_before_the_node_takes_it(self, message):
        assert _send_outcome(Web3RPCError(f"{{'code': -32000, 'message': '{message}'}}")) == (
            "refused"
        )

    @pytest.mark.parametrize("message", _AMBIGUOUS_SEND_ERRORS)
    def test_failed_forwards_are_unknown(self, message):
        assert _send_outcome(Web3RPCError(message)) == "unknown"


@pytest.fixture
def hung_rpc():
    """An endpoint that accepts connections and never replies."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    conns: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                conns.append(server.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}"
    stop.set()
    thread.join(1)
    for c in conns:
        c.close()
    server.close()


class TestBaseSendLockIsBounded:
    """A hung RPC holds the send lock for one timeout per call, not web3's 5 tries."""

    _T = 0.5

    def _signer(self, monkeypatch, url: str) -> _ResyncingWeb3Signer:
        monkeypatch.setattr("x402gate.core.payment._EVM_RPC_TIMEOUT_S", self._T)
        return _ResyncingWeb3Signer(
            private_key="0x" + "00" * 31 + "01",
            rpc_url=url,
            confirmation_timeout_seconds=1,
            chain_id=8453,
        )

    def test_send_and_lookup_give_up_after_one_timeout(self, monkeypatch, hung_rpc):
        signer = self._signer(monkeypatch, hung_rpc)

        start = time.monotonic()
        with pytest.raises(requests.Timeout):
            signer._send_w3.eth.send_raw_transaction(b"\x01")
        assert time.monotonic() - start < 2 * self._T

        start = time.monotonic()
        assert signer._node_has(b"\x11" * 32) is None
        assert time.monotonic() - start < 2 * self._T

    def test_concurrent_sends_queue_for_about_one_timeout_per_call(self, monkeypatch, hung_rpc):
        signer = self._signer(monkeypatch, hung_rpc)
        signer._w3 = Mock()
        signer._w3.eth.gas_price = 1
        reserved = iter(range(7, 100))
        monkeypatch.setattr(signer, "_reserve_nonce", lambda: next(reserved))
        hashes: list[str] = []

        def send() -> None:
            hashes.append(signer.send_transaction("0x" + "22" * 20, b""))

        start = time.monotonic()
        threads = [threading.Thread(target=send) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        elapsed = time.monotonic() - start

        # First: send (T). Second: resend count (T, skipped) + send (T).
        assert elapsed < 4 * self._T
        assert len(hashes) == 2
        assert set(signer._unconfirmed) == {7, 8}

    def test_reads_retry_a_dropped_connection_once(self, monkeypatch):
        signer = self._signer(monkeypatch, "http://127.0.0.1:1")
        ok = json.dumps({"jsonrpc": "2.0", "id": 0, "result": "0x2105"}).encode()
        read = Mock(side_effect=[requests.ConnectionError("reset"), ok])
        monkeypatch.setattr(signer._w3.provider._request_session_manager, "make_post_request", read)

        assert signer._w3.eth.chain_id == 8453
        assert read.call_count == 2

        send = Mock(side_effect=[requests.ConnectionError("reset"), ok])
        monkeypatch.setattr(
            signer._send_w3.provider._request_session_manager, "make_post_request", send
        )
        with pytest.raises(requests.ConnectionError):
            signer._send_w3.eth.get_transaction_count(signer.address, "pending")
        assert send.call_count == 1


_BASE = "eip155:8453"
_PAYER = "0x" + "ab" * 20
_PAY_TO = "0x1234567890abcdef1234567890abcdef12345678"


def _transfer_log(
    *,
    token: str = "0x833589FCD6EDB6E08F4C7C32D4F71B54BDA02913",
    sender: str = _PAYER,
    to: str = _PAY_TO,
    value: int = 1_000_000,
) -> dict:
    """A USDC Transfer log as web3 returns it."""
    from x402gate.core.payment import _TRANSFER_TOPIC

    def topic(address: str) -> bytes:
        return bytes(12) + bytes.fromhex(address[2:])

    return {
        "address": token,
        "topics": [_TRANSFER_TOPIC, topic(sender), topic(to)],
        "data": value.to_bytes(32, "big"),
    }


def _receipt(status: int = 1, logs: list | None = None) -> dict:
    return {
        "status": status,
        "logs": [_transfer_log()] if logs is None else logs,
        "gasUsed": 50_000,
        "effectiveGasPrice": 10**9,
    }


class TestSettlementReceipt:
    """A late receipt counts only a transfer of the payer's USDC to the gateway."""

    async def _lookup(self, handler: PaymentHandler, receipt, payer: str = _PAYER):  # noqa: ANN001
        ns = handler._schemes[_BASE]
        ns.w3 = Mock()
        if isinstance(receipt, Exception):
            ns.w3.eth.get_transaction_receipt.side_effect = receipt
        else:
            ns.w3.eth.get_transaction_receipt.return_value = receipt
        return await handler.settlement_receipt(_BASE, "0xaa", payer, Decimal("1"))

    async def test_matching_transfer_is_a_success_with_gas_in_usd(self, handler, fresh_prices):
        result = await self._lookup(handler, _receipt(), payer=_PAYER.upper().replace("0X", "0x"))

        assert settlement_state(result) == "ok"
        assert result["transaction"] == "0xaa"
        assert result["amount_usdc"] == 1.0
        assert result["gas_label"] == "ETH"
        assert result["gas_cost_native"] == pytest.approx(0.00005)
        # Priced like settle's gas, not left at 0.
        assert result["gas_cost_usd"] == pytest.approx(0.1)

    async def test_without_an_eth_price_the_gas_is_zero_usd(
        self, handler, fresh_prices, monkeypatch
    ):
        monkeypatch.setitem(fresh_prices._price_cache, "ethereum", 0.0)
        monkeypatch.setitem(fresh_prices._price_cache, "failed_at", time.time())
        result = await self._lookup(handler, _receipt())

        assert settlement_state(result) == "ok"
        assert result["gas_cost_usd"] == 0.0

    async def test_reverted_transaction_is_a_failure(self, handler, fresh_prices):
        result = await self._lookup(handler, _receipt(status=0, logs=[]))
        assert settlement_state(result) == "failed"

    @pytest.mark.parametrize(
        "log",
        [
            _transfer_log(sender="0x" + "cd" * 20),
            _transfer_log(to="0x" + "ef" * 20),
            _transfer_log(value=999_999),
            _transfer_log(token="0x" + "11" * 20),
        ],
    )
    async def test_other_transfer_is_a_failure(self, handler, fresh_prices, log):
        result = await self._lookup(handler, _receipt(logs=[log]))
        assert settlement_state(result) == "failed"

    @pytest.mark.parametrize("receipt", [None, TransactionNotFound("no tx"), TimeoutError("rpc")])
    async def test_no_receipt_is_unresolved(self, handler, receipt):
        assert await self._lookup(handler, receipt) is None

    async def test_unknown_network_is_unresolved(self, handler):
        assert await handler.settlement_receipt("eip155:1", "0xaa", _PAYER, Decimal("1")) is None

    def _unconfirmed_signer(self, handler: PaymentHandler):
        signer = handler._schemes[_BASE].evm_signer
        send_w3 = Mock()
        send_w3.eth.get_transaction_count.return_value = 7
        send_w3.eth.send_raw_transaction.return_value = b"\x11" * 32
        signer._send_w3 = send_w3
        signer._next_nonce = 8
        signer._unconfirmed[7] = b"raw7"
        return signer, send_w3

    @pytest.mark.parametrize("receipt", [None, TransactionNotFound("no tx")])
    async def test_poll_without_a_receipt_sends_unconfirmed_again(self, handler, receipt):
        """A reconcile keeps sending a transfer that may never have reached the node."""
        signer, send_w3 = self._unconfirmed_signer(handler)

        assert await self._lookup(handler, receipt) is None

        assert [c.args[0] for c in send_w3.eth.send_raw_transaction.call_args_list] == [b"raw7"]
        assert signer._unconfirmed == {}

    async def test_poll_whose_resend_gets_no_answer_keeps_it(self, handler):
        signer, send_w3 = self._unconfirmed_signer(handler)
        send_w3.eth.send_raw_transaction.side_effect = requests.ConnectTimeout("connect")

        assert await self._lookup(handler, None) is None

        assert signer._unconfirmed == {7: b"raw7"}

    async def test_poll_that_finds_the_receipt_sends_nothing(self, handler, fresh_prices):
        signer, send_w3 = self._unconfirmed_signer(handler)

        assert settlement_state(await self._lookup(handler, _receipt())) == "ok"

        send_w3.eth.send_raw_transaction.assert_not_called()

    async def test_lookup_runs_on_the_base_pool(self, handler, fresh_prices):
        import threading

        seen = []

        def get_receipt(tx_hash):  # noqa: ANN001
            seen.append(threading.current_thread().name)
            return _receipt()

        ns = handler._schemes[_BASE]
        ns.w3 = Mock()
        ns.w3.eth.get_transaction_receipt.side_effect = get_receipt
        await handler.settlement_receipt(_BASE, "0xaa", _PAYER, Decimal("1"))

        assert seen and seen[0].startswith("x402-base")


class TestEvmSettleGas:
    """settle prices Base gas from the receipt at the cached ETH price."""

    async def test_gas_in_eth_and_usd(self, handler: PaymentHandler, fresh_prices):
        from x402 import SettleResponse

        ns = handler._schemes[_BASE]
        ns.scheme = Mock()
        ns.scheme.settle.return_value = SettleResponse(
            success=True, transaction="0xaa", network=_BASE, payer=_PAYER
        )
        ns.w3 = Mock()
        ns.w3.eth.get_transaction_receipt.return_value = _receipt()
        accepted = handler.get_payment_requirements_for_network(Decimal("1"), _BASE)
        header = _encoded(
            {
                "x402Version": 2,
                "accepted": accepted,
                "payload": {"signature": "0x1", "authorization": {"from": _PAYER, "nonce": "1"}},
            }
        )
        settled = await handler.settle(header, Decimal("1"), _BASE)

        assert settlement_state(settled) == "ok"
        assert settled["gas_cost_native"] == pytest.approx(0.00005)
        assert settled["gas_cost_usd"] == pytest.approx(0.1)
        assert settled["gas_label"] == "ETH"

    async def test_gas_comes_from_the_receipt_the_confirmation_wait_fetched(
        self, handler: PaymentHandler, fresh_prices
    ):
        from x402 import SettleResponse

        ns = handler._schemes[_BASE]
        signer = ns.evm_signer
        signer._w3 = Mock()
        signer._w3.eth.wait_for_transaction_receipt.return_value = {
            **_receipt(),
            "blockNumber": 1,
        }
        tx_hash = "0x" + "AA" * 32

        def settle(payload, requirements):  # noqa: ANN001, ANN202
            assert signer.wait_for_transaction_receipt(tx_hash).status == 1
            return SettleResponse(success=True, transaction=tx_hash, network=_BASE, payer=_PAYER)

        ns.scheme = Mock()
        ns.scheme.settle.side_effect = settle
        ns.w3 = Mock()
        ns.w3.eth.get_transaction_receipt.side_effect = AssertionError("second receipt fetch")
        accepted = handler.get_payment_requirements_for_network(Decimal("1"), _BASE)
        header = _encoded(
            {
                "x402Version": 2,
                "accepted": accepted,
                "payload": {"signature": "0x1", "authorization": {"from": _PAYER, "nonce": "1"}},
            }
        )
        settled = await handler.settle(header, Decimal("1"), _BASE)

        assert settlement_state(settled) == "ok"
        assert settled["gas_cost_native"] == pytest.approx(0.00005)
        assert signer._receipts == {}

    def test_receipt_cache_is_bounded(self, handler: PaymentHandler, monkeypatch):
        monkeypatch.setattr("x402gate.core.payment._RECEIPT_CACHE_SIZE", 2)
        signer = handler._schemes[_BASE].evm_signer
        signer._w3 = Mock()
        signer._w3.eth.wait_for_transaction_receipt.return_value = {**_receipt(), "blockNumber": 1}

        for h in ("0x01", "0x02", "0x03"):
            signer.wait_for_transaction_receipt(h)

        assert signer.take_receipt("0x01") is None
        assert signer.take_receipt("0x03")["gasUsed"] == 50_000
        assert signer.take_receipt("0x03") is None


class TestBaseReadsAreBounded:
    """Receipt polls, the gas lookup and the balance check use the signer's bounded reads."""

    def test_reads_share_the_signer_provider(self, handler: PaymentHandler):
        ns = handler._schemes[_BASE]
        provider = ns.w3.provider
        assert ns.w3 is ns.evm_signer._w3
        assert provider._request_kwargs == {"timeout": 10}
        assert requests.Timeout not in provider.exception_retry_configuration.errors
        assert requests.ReadTimeout not in provider.exception_retry_configuration.errors

    async def test_receipt_poll_on_a_hung_rpc_gives_up_after_one_timeout(
        self, monkeypatch, hung_rpc
    ):
        t = 0.3
        monkeypatch.setattr("x402gate.core.payment._EVM_RPC_TIMEOUT_S", t)
        cfg = NetworkConfig(
            type="evm",
            network=_BASE,
            pay_to=_PAY_TO,
            token_address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            token_name="USD Coin",
            token_version="2",
            rpc_url=hung_rpc,
            facilitator_key="0x" + "00" * 31 + "01",
        )
        handler = PaymentHandler(networks={"base": cfg})

        start = time.monotonic()
        assert (
            await handler.settlement_receipt(_BASE, "0x" + "aa" * 32, _PAYER, Decimal("1")) is None
        )
        assert time.monotonic() - start < 2 * t

        start = time.monotonic()
        assert await handler.check_evm_balance(_BASE, _PAYER, Decimal("1")) is None
        # Its own two attempts, each tried once.
        assert time.monotonic() - start < 3 * t
