"""Unit tests for payment handling."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from x402gate.core.config import NetworkConfig
from x402gate.core.payment import (
    InFlightAmounts,
    PaymentClaims,
    PaymentHandler,
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
