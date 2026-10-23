"""Solana signer rebroadcast and blockhash expiry."""

import base64
from types import SimpleNamespace

import pytest
from solders.commitment_config import CommitmentLevel
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.rpc.config import RpcContextConfig
from solders.rpc.requests import IsBlockhashValid
from solders.rpc.responses import IsBlockhashValidResp
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from x402gate.core.svm_signer import (
    _CONFIRM_RUN_TIMEOUT_S,
    _CONFIRM_TIMEOUT_S,
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
    signer._client = rpc
    return signer


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
        rpc = _Rpc([None, None, _status(confirmation_status="confirmed")])
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
            [None, None, None, _status(confirmation_status="confirmed")],
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
            [None, _status(confirmation_status="confirmed")],
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
        signer._inflight["old"] = (b"stale", -121.0)

        await signer._send_async(encoded)

        assert "old" not in signer._inflight
        assert signer._inflight[signature][0] == raw
