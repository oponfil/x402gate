"""Solana signer fee guard, rebroadcast and blockhash expiry."""

import base64
from types import SimpleNamespace

import pytest
from solders.commitment_config import CommitmentLevel
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.rpc.config import RpcContextConfig
from solders.rpc.requests import IsBlockhashValid
from solders.rpc.responses import IsBlockhashValidResp
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from x402gate.core.http import KEEPALIVE_EXPIRY_S
from x402gate.core.svm_signer import (
    _BLOCKHASH_CHECK_INTERVAL_S,
    _CONFIRM_RUN_TIMEOUT_S,
    _CONFIRM_TIMEOUT_S,
    _FAST_POLL_WINDOW_S,
    _INFLIGHT_TTL_S,
    _MAX_COMPUTE_UNIT_LIMIT,
    _MAX_COMPUTE_UNIT_PRICE_MICROLAMPORTS,
    _POLL_INTERVAL_S,
    _REBROADCAST_INTERVAL_S,
    _SIMULATION_REUSE_S,
    _SLOW_POLL_INTERVAL_S,
    _STATUS_RETRY_DELAY_S,
    SolanaSigner,
)


def _signed_transfer() -> tuple[bytes, str, str]:
    """A real signed transaction, so blockhash decoding matches production."""
    payer = Keypair()
    instruction = transfer(
        TransferParams(from_pubkey=payer.pubkey(), to_pubkey=payer.pubkey(), lamports=1)
    )
    message = MessageV0.try_compile(
        payer=payer.pubkey(),
        instructions=[instruction],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.default(),
    )
    transaction = VersionedTransaction(message, [payer])
    raw = bytes(transaction)
    encoded = base64.b64encode(raw).decode()
    return raw, encoded, str(transaction.signatures[0])


_SDK_CU_LIMIT = 20_000
_SDK_CU_PRICE = 1


def _client_payment(
    fee_payer: Keypair,
    *,
    cu_limit: int | None = _SDK_CU_LIMIT,
    cu_price: int = _SDK_CU_PRICE,
    extra_signer: Keypair | None = None,
    extra_instruction: Instruction | None = None,
) -> str:
    """A client-built payment with the SDK client's compute-budget layout.

    The fee payer is account 0 and the client signs the transfer, so the
    gateway's signature fills slot 0 just as in a real settle.
    """
    client = Keypair()
    instructions = []
    if cu_limit is not None:
        instructions.append(set_compute_unit_limit(cu_limit))
    instructions.append(set_compute_unit_price(cu_price))
    instructions.append(
        transfer(TransferParams(from_pubkey=client.pubkey(), to_pubkey=client.pubkey(), lamports=1))
    )
    signers = [fee_payer, client]
    if extra_signer is not None:
        # A memo that demands one more signature, which the fee payer pays for.
        instructions.append(
            Instruction(
                Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"),
                b"x",
                [AccountMeta(extra_signer.pubkey(), is_signer=True, is_writable=False)],
            )
        )
        signers.append(extra_signer)
    if extra_instruction is not None:
        instructions.append(extra_instruction)
    message = MessageV0.try_compile(
        payer=fee_payer.pubkey(),
        instructions=instructions,
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.default(),
    )
    return base64.b64encode(bytes(VersionedTransaction(message, signers))).decode()


def _fee_signer(fee_payer: Keypair) -> SolanaSigner:
    signer = SolanaSigner.__new__(SolanaSigner)
    signer._keypair = fee_payer
    return signer


class TestFeeGuard:
    """The gateway pays the fee, so it refuses transactions that inflate it."""

    def _sign(self, fee_payer: Keypair, encoded: str) -> str:
        return _fee_signer(fee_payer).sign_transaction(
            encoded, str(fee_payer.pubkey()), "solana:mainnet"
        )

    def test_sdk_client_transaction_is_signed(self) -> None:
        fee_payer = Keypair()
        signed = self._sign(fee_payer, _client_payment(fee_payer))

        tx = VersionedTransaction.from_bytes(base64.b64decode(signed))
        assert tx.signatures[0] == fee_payer.sign_message(bytes([0x80]) + bytes(tx.message))

    def test_caps_are_accepted(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(
            fee_payer,
            cu_limit=_MAX_COMPUTE_UNIT_LIMIT,
            cu_price=_MAX_COMPUTE_UNIT_PRICE_MICROLAMPORTS,
        )

        self._sign(fee_payer, encoded)

    def test_worst_case_priority_fee_stays_below_the_base_fee(self) -> None:
        priority_lamports = _MAX_COMPUTE_UNIT_LIMIT * _MAX_COMPUTE_UNIT_PRICE_MICROLAMPORTS
        assert priority_lamports // 1_000_000 < 10_000
        assert _MAX_COMPUTE_UNIT_LIMIT >= 2 * _SDK_CU_LIMIT

    def test_huge_compute_unit_limit_is_refused(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(fee_payer, cu_limit=1_400_000)

        with pytest.raises(ValueError, match="Compute unit limit 1400000"):
            self._sign(fee_payer, encoded)

    def test_huge_compute_unit_price_is_refused(self) -> None:
        fee_payer = Keypair()
        # The SDK verifier's own cap; still far above the gateway's.
        encoded = _client_payment(fee_payer, cu_price=5_000_000)

        with pytest.raises(ValueError, match="Compute unit price 5000000"):
            self._sign(fee_payer, encoded)

    def test_missing_compute_unit_limit_is_refused(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(fee_payer, cu_limit=None)

        with pytest.raises(ValueError, match="no SetComputeUnitLimit"):
            self._sign(fee_payer, encoded)

    def test_extra_signature_is_refused(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(fee_payer, extra_signer=Keypair())

        with pytest.raises(ValueError, match="requires 3 signatures"):
            self._sign(fee_payer, encoded)

    def test_fee_payer_in_optional_instruction_is_refused(self) -> None:
        fee_payer = Keypair()
        # The SDK verifier allows Lighthouse here without checking accounts;
        # with the fee payer as a writable signer it could pay rent from it.
        lighthouse = Instruction(
            Pubkey.from_string("L2TExMFKdjpN9kozasaurPirfHy9P8sbXoAN1qA3S95"),
            b"\x00",
            [
                AccountMeta(fee_payer.pubkey(), is_signer=True, is_writable=True),
                AccountMeta(Keypair().pubkey(), is_signer=False, is_writable=True),
            ],
        )
        encoded = _client_payment(fee_payer, extra_instruction=lighthouse)

        with pytest.raises(ValueError, match="Fee payer must not appear"):
            self._sign(fee_payer, encoded)

    def test_fee_payer_as_readonly_memo_account_is_refused(self) -> None:
        fee_payer = Keypair()
        memo = Instruction(
            Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"),
            b"x",
            [AccountMeta(fee_payer.pubkey(), is_signer=False, is_writable=False)],
        )
        encoded = _client_payment(fee_payer, extra_instruction=memo)

        with pytest.raises(ValueError, match="Fee payer must not appear"):
            self._sign(fee_payer, encoded)


def _status(confirmation_status=None, err=None):  # noqa: ANN001
    return SimpleNamespace(confirmation_status=confirmation_status, err=err)


class _Rpc:
    """Async stand-in for the Solana client methods the signer calls."""

    def __init__(self, statuses: list, blockhash_result=True, send_error: Exception | None = None):
        self._statuses = list(statuses)
        self._blockhash_result = blockhash_result
        self._send_error = send_error
        self.events: list = []
        self._provider = self

    async def send_transaction(self, transaction, opts=None):  # noqa: ANN001
        self.events.append(("first", opts))
        return SimpleNamespace(value=str(transaction.signatures[0]))

    async def get_signature_statuses(self, signatures, search_transaction_history=True):  # noqa: ANN001
        _ = signatures, search_transaction_history
        item = self._statuses.pop(0) if self._statuses else None
        self.events.append("status")
        # An exception in the script stands for one failed RPC call.
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(value=[item])

    async def send_raw_transaction(self, txn, opts=None):  # noqa: ANN001
        self.events.append(("send", bytes(txn), opts))
        if self._send_error is not None:
            raise self._send_error

    async def make_request(self, body, parser):  # noqa: ANN001
        self.events.append(("check", body, parser))
        if isinstance(self._blockhash_result, Exception):
            raise self._blockhash_result
        return SimpleNamespace(value=self._blockhash_result)


def _signer(rpc: _Rpc) -> SolanaSigner:
    signer = SolanaSigner.__new__(SolanaSigner)
    signer._inflight = {}
    signer._simulated = {}
    signer._client = rpc
    return signer


def _read_times(count: int = 200) -> list[float]:
    """When the loop reads the status for a send at t=0: fast, then slow."""
    times = [0.0]
    while len(times) < count:
        last = times[-1]
        step = _POLL_INTERVAL_S if last < _FAST_POLL_WINDOW_S else _SLOW_POLL_INTERVAL_S
        times.append(last + step)
    return times


def _reads_until(t: float) -> int:
    """The status reads up to and including the first one at or after t."""
    return next(i for i, at in enumerate(_read_times()) if at >= t) + 1


# The status reads up to and including the first blockhash check, and up to
# the first resend.
_READS_TO_CHECK = _reads_until(_BLOCKHASH_CHECK_INTERVAL_S)
_READS_TO_RESEND = _reads_until(_REBROADCAST_INTERVAL_S)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Advance the confirm loop's clock inside sleep, so a 60s wait is instant."""
    state = {"t": 0.0}

    def monotonic() -> float:
        return state["t"]

    async def sleep(delay: float) -> None:
        state["t"] += delay

    monkeypatch.setattr("x402gate.core.svm_signer.time.monotonic", monotonic)
    monkeypatch.setattr("x402gate.core.svm_signer.asyncio.sleep", sleep)
    return state


def _sends(rpc: _Rpc) -> list:
    return [event for event in rpc.events if isinstance(event, tuple) and event[0] == "send"]


class TestSolanaConfirm:
    """Rebroadcast the original bytes, and stop once the blockhash is dead."""

    async def test_rebroadcasts_the_same_bytes_until_confirmed(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([None] * (2 * _READS_TO_RESEND) + [_status(confirmation_status="confirmed")])
        signer = _signer(rpc)

        assert await signer._send_async(encoded) == signature
        await signer._confirm_async(signature)

        sends = _sends(rpc)
        assert len(sends) >= 2
        assert all(payload == raw for _, payload, _ in sends)
        opts = sends[0][2]
        assert opts.skip_preflight is True
        assert opts.skip_confirmation is True
        assert opts.max_retries == 0
        first_opts = next(event[1] for event in rpc.events if event[0] == "first")
        assert first_opts.skip_preflight is False
        assert first_opts.max_retries == 3
        assert signer._inflight == {}
        assert clock["t"] < 60

    async def test_expired_blockhash_stops_before_the_timeout(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([], blockhash_result=False)
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="blockhash no longer valid"):
            await signer._confirm_async(signature)

        assert clock["t"] < 60
        sends = _sends(rpc)
        assert len(sends) == 2
        assert all(payload == raw for _, payload, _ in sends)
        check_at = next(
            index
            for index, event in enumerate(rpc.events)
            if isinstance(event, tuple) and event[0] == "check"
        )
        assert not any(
            isinstance(event, tuple) and event[0] == "send" for event in rpc.events[check_at + 1 :]
        )
        check = rpc.events[check_at]
        assert isinstance(check[1], IsBlockhashValid)
        assert check[1].blockhash == VersionedTransaction.from_bytes(raw).message.recent_blockhash
        assert check[1].config == RpcContextConfig(commitment=CommitmentLevel.Confirmed)
        assert '"commitment":"confirmed"' in check[1].to_json()
        assert check[2] is IsBlockhashValidResp
        assert signer._inflight == {}

    async def test_landed_status_on_the_expiry_recheck_is_kept(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc(
            [None] * _READS_TO_CHECK + [_status(confirmation_status="confirmed")],
            blockhash_result=False,
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        sends = _sends(rpc)
        assert len(sends) == 2
        assert all(payload == raw for _, payload, _ in sends)
        check_at = next(
            index
            for index, event in enumerate(rpc.events)
            if isinstance(event, tuple) and event[0] == "check"
        )
        assert not any(
            isinstance(event, tuple) and event[0] == "send" for event in rpc.events[check_at + 1 :]
        )
        assert signer._inflight == {}
        assert clock["t"] < 60

    async def test_transaction_error_does_not_rebroadcast(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([_status(err="InstructionError")])
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="Transaction failed: InstructionError"):
            await signer._confirm_async(signature)

        assert _sends(rpc) == []
        assert signer._inflight == {}
        assert clock["t"] == 0

    async def test_duplicate_rebroadcast_error_still_confirms(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc(
            [None] * _READS_TO_RESEND + [_status(confirmation_status="confirmed")],
            send_error=RuntimeError("This transaction has already been processed"),
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        sends = _sends(rpc)
        assert len(sends) == 1
        assert sends[0][1] == raw
        assert signer._inflight == {}
        assert clock["t"] < 60

    async def test_timeout_keeps_bytes_for_a_later_confirm(self, clock: dict[str, float]) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([], blockhash_result=True)
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="not confirmed within 120s"):
            await signer._confirm_async(signature)

        assert clock["t"] >= _CONFIRM_TIMEOUT_S
        assert signer._inflight[signature][0] == raw

        rpc.events.clear()
        rpc._statuses = [None, _status(confirmation_status="finalized")]
        await signer._confirm_async(signature)

        sends = _sends(rpc)
        assert sends
        assert sends[0][1] == raw
        assert signer._inflight == {}

    async def test_blockhash_rpc_error_is_not_expiry(self, clock: dict[str, float]) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([], blockhash_result=ConnectionError("rpc down"))
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="not confirmed within 120s") as raised:
            await signer._confirm_async(signature)

        assert "expired" not in str(raised.value)
        assert signer._inflight[signature][0] == raw
        assert clock["t"] >= _CONFIRM_TIMEOUT_S

    async def test_processed_status_waits_for_confirmed(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc(
            [
                _status(confirmation_status="processed"),
                _status(confirmation_status="confirmed"),
            ]
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        assert _sends(rpc) == []
        assert signer._inflight == {}
        assert clock["t"] < 60


class TestTransientStatusErrors:
    """A failed status read is retried, since the broadcast usually lands."""

    async def test_status_rpc_error_is_retried(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc(
            [
                ConnectionError("rpc down"),
                ConnectionError("rpc down"),
                _status(confirmation_status="confirmed"),
            ]
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        assert rpc.events.count("status") == 3
        assert signer._inflight == {}
        assert clock["t"] < 60

    async def test_status_rpc_error_on_expiry_recheck_is_not_expiry(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        # The first blockhash check finds it dead, but the re-read fails. The
        # next poll then sees the transaction confirmed.
        rpc = _Rpc(
            [None] * _READS_TO_CHECK
            + [ConnectionError("rpc down"), _status(confirmation_status="confirmed")],
            blockhash_result=False,
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        check_at = next(
            index
            for index, event in enumerate(rpc.events)
            if isinstance(event, tuple) and event[0] == "check"
        )
        assert not any(
            isinstance(event, tuple) and event[0] == "send" for event in rpc.events[check_at + 1 :]
        )
        assert all(payload == raw for _, payload, _ in _sends(rpc))
        assert signer._inflight == {}
        assert clock["t"] < 60

    async def test_status_rpc_down_until_timeout_is_not_expiry(
        self, clock: dict[str, float]
    ) -> None:
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([ConnectionError("rpc down")] * 100, blockhash_result=False)
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="not confirmed within 120s") as raised:
            await signer._confirm_async(signature)

        assert "expired" not in str(raised.value)
        assert signer._inflight[signature][0] == raw
        assert clock["t"] >= _CONFIRM_TIMEOUT_S

    async def test_failed_transaction_still_fails_after_a_read_error(
        self, clock: dict[str, float]
    ) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([ConnectionError("rpc down"), _status(err="InstructionError")])
        signer = _signer(rpc)
        await signer._send_async(encoded)

        with pytest.raises(RuntimeError, match="Transaction failed: InstructionError"):
            await signer._confirm_async(signature)

        assert signer._inflight == {}
        assert clock["t"] < 60


def test_confirm_wait_outlives_the_poll_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thread helper must not cut off a confirm that is still inside its loop."""
    captured: dict[str, float] = {}

    def _run(self, coro, *, timeout: float = 90):  # noqa: ANN001
        _ = self
        captured["timeout"] = timeout
        coro.close()

    monkeypatch.setattr(SolanaSigner, "_run_async", _run)
    signer = SolanaSigner.__new__(SolanaSigner)
    signer.confirm_transaction("sig", "solana:mainnet")

    assert captured["timeout"] == _CONFIRM_RUN_TIMEOUT_S
    assert _CONFIRM_RUN_TIMEOUT_S > _CONFIRM_TIMEOUT_S


class TestInflightCleanup:
    """Bytes from a transaction that is never retried do not stay forever."""

    async def test_entries_older_than_two_minutes_are_dropped(
        self, clock: dict[str, float]
    ) -> None:
        _ = clock
        raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([_status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        signer._inflight["old"] = (b"stale", -(_INFLIGHT_TTL_S + 1.0))

        await signer._send_async(encoded)

        assert "old" not in signer._inflight
        assert signer._inflight[signature][0] == raw


class TestFasterConfirm:
    """Status polls run every 0.5 s for 3 s; resends and expiry checks keep their pace."""

    async def test_first_poll_has_no_fixed_delay(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([None, _status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        # The old loop saw the confirmation only after a 2 s sleep.
        assert clock["t"] == _POLL_INTERVAL_S
        assert _POLL_INTERVAL_S <= 0.5
        assert _sends(rpc) == []

    async def test_resends_stay_two_seconds_apart(self, clock: dict[str, float]) -> None:
        raw, encoded, signature = _signed_transfer()
        # Reads at t = 0, 0.5, ..., 2.5 inside the fast window, then confirmed.
        reads = _reads_until(_FAST_POLL_WINDOW_S) - 1
        rpc = _Rpc([None] * reads + [_status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        sent_at: list[float] = []
        send_raw = rpc.send_raw_transaction

        async def timed_send(txn, opts=None):  # noqa: ANN001
            sent_at.append(clock["t"])
            await send_raw(txn, opts)

        rpc.send_raw_transaction = timed_send
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        assert rpc.events.count("status") == reads + 1
        # Six status reads in 2.5 s, but one resend, at 2 s.
        assert sent_at == [_REBROADCAST_INTERVAL_S]
        assert all(payload == raw for _, payload, _ in _sends(rpc))

    async def test_polls_slow_down_after_the_fast_window(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([None] * 12 + [_status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        read_at: list[float] = []
        read_status = rpc.get_signature_statuses

        async def timed_read(signatures, search_transaction_history=True):  # noqa: ANN001
            read_at.append(clock["t"])
            return await read_status(signatures, search_transaction_history)

        rpc.get_signature_statuses = timed_read
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        fast = [t for t in read_at if t < _FAST_POLL_WINDOW_S]
        assert fast == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        # A confirm still waiting after 3 s reads every 2 s, as before.
        slow = read_at[len(fast) :]
        assert slow[0] == _FAST_POLL_WINDOW_S
        gaps = [later - earlier for earlier, later in zip(slow, slow[1:], strict=False)]
        assert gaps and all(gap == _SLOW_POLL_INTERVAL_S for gap in gaps)
        assert _SLOW_POLL_INTERVAL_S == 2.0

    async def test_later_confirm_polls_at_the_slow_pace(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc([None, None, _status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        await signer._send_async(encoded)
        # A retry of a confirm that timed out, long after the send.
        clock["t"] = _CONFIRM_TIMEOUT_S

        await signer._confirm_async(signature)

        assert clock["t"] == _CONFIRM_TIMEOUT_S + 2 * _SLOW_POLL_INTERVAL_S

    async def test_failed_read_waits_the_retry_delay(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        rpc = _Rpc(
            [ConnectionError("429 Too Many Requests"), _status(confirmation_status="confirmed")]
        )
        signer = _signer(rpc)
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        assert clock["t"] == _STATUS_RETRY_DELAY_S
        assert _STATUS_RETRY_DELAY_S > _POLL_INTERVAL_S

    async def test_blockhash_is_checked_every_six_seconds(self, clock: dict[str, float]) -> None:
        _raw, encoded, signature = _signed_transfer()
        times = _read_times()
        first_check = times[_READS_TO_CHECK - 1]
        reads = _reads_until(first_check + _BLOCKHASH_CHECK_INTERVAL_S)
        rpc = _Rpc([None] * reads + [_status(confirmation_status="confirmed")])
        signer = _signer(rpc)
        checked_at: list[float] = []
        check = rpc.make_request

        async def timed_check(body, parser):  # noqa: ANN001
            checked_at.append(clock["t"])
            return await check(body, parser)

        rpc.make_request = timed_check
        await signer._send_async(encoded)

        await signer._confirm_async(signature)

        assert len(checked_at) == 2
        assert checked_at[1] - checked_at[0] >= _BLOCKHASH_CHECK_INTERVAL_S


class _SimRpc:
    """Counts simulateTransaction calls and fails them when asked."""

    def __init__(self, err=None) -> None:  # noqa: ANN001
        self.err = err
        self.calls = 0

    async def simulate_transaction(self, transaction):  # noqa: ANN001
        _ = transaction
        self.calls += 1
        return SimpleNamespace(value=SimpleNamespace(err=self.err))


class TestSimulationReuse:
    """Settle's re-verify reuses the verify simulation of the same bytes once."""

    async def test_same_bytes_simulate_once_then_again(self, clock: dict[str, float]) -> None:
        _ = clock
        _raw, encoded, _signature = _signed_transfer()
        rpc = _SimRpc()
        signer = _signer(rpc)

        await signer._simulate_async(encoded)  # the request's verify
        await signer._simulate_async(encoded)  # settle's verify, inside the SDK
        assert rpc.calls == 1

        # Reused once only: any further check goes to the RPC again.
        await signer._simulate_async(encoded)
        assert rpc.calls == 2

    async def test_failed_simulation_is_not_reused(self, clock: dict[str, float]) -> None:
        _ = clock
        _raw, encoded, _signature = _signed_transfer()
        rpc = _SimRpc(err="InsufficientFunds")
        signer = _signer(rpc)

        for _attempt in range(2):
            with pytest.raises(RuntimeError, match="Simulation failed"):
                await signer._simulate_async(encoded)
        assert rpc.calls == 2

    async def test_old_simulation_is_not_reused(self, clock: dict[str, float]) -> None:
        _raw, encoded, _signature = _signed_transfer()
        rpc = _SimRpc()
        signer = _signer(rpc)

        await signer._simulate_async(encoded)
        clock["t"] += _SIMULATION_REUSE_S
        await signer._simulate_async(encoded)
        assert rpc.calls == 2
        assert len(signer._simulated) == 1

    async def test_other_bytes_are_simulated(self, clock: dict[str, float]) -> None:
        _ = clock
        rpc = _SimRpc()
        signer = _signer(rpc)

        await signer._simulate_async(_signed_transfer()[1])
        await signer._simulate_async(_signed_transfer()[1])
        assert rpc.calls == 2

    async def test_send_keeps_the_preflight(self, clock: dict[str, float]) -> None:
        """The preflight is the check at broadcast the reuse relies on."""
        _ = clock
        _raw, encoded, _signature = _signed_transfer()
        rpc = _Rpc([])
        await _signer(rpc)._send_async(encoded)

        first_opts = next(event[1] for event in rpc.events if event[0] == "first")
        assert first_opts.skip_preflight is False


class TestFeeLamports:
    """The gas figure comes from the message, with no getTransaction call."""

    def test_sdk_client_fee(self) -> None:
        fee_payer = Keypair()
        # Two signatures, plus 20,000 CU x 1 microlamport rounded up.
        assert SolanaSigner.fee_lamports(_client_payment(fee_payer)) == 10_001

    def test_priority_fee_rounds_up(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(fee_payer, cu_limit=20_000, cu_price=51)
        assert SolanaSigner.fee_lamports(encoded) == 10_002

    def test_fee_at_the_caps(self) -> None:
        fee_payer = Keypair()
        encoded = _client_payment(
            fee_payer,
            cu_limit=_MAX_COMPUTE_UNIT_LIMIT,
            cu_price=_MAX_COMPUTE_UNIT_PRICE_MICROLAMPORTS,
        )
        assert SolanaSigner.fee_lamports(encoded) == 14_000


def test_rpc_connections_stay_open_between_payments() -> None:
    """One RPC client per signer, keeping idle connections like the shared pool."""
    signer = SolanaSigner(str(Keypair()), "http://127.0.0.1:9")
    try:
        pool = signer._client._provider.session._transport._pool
        assert pool._keepalive_expiry == KEEPALIVE_EXPIRY_S
    finally:
        signer.close()
