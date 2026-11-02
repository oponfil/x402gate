"""x402 payment handling for x402gate.

Supports both EVM (Base) and SVM (Solana) networks via the x402 SDK.
Verifies and settles payments directly on-chain using local facilitator
signers — no external facilitator service required.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

import requests
from fastapi import Request
from fastapi.responses import JSONResponse
from web3 import Web3
from web3.exceptions import RequestTimedOut, TransactionNotFound, Web3RPCError
from web3.providers.rpc.utils import ExceptionRetryConfiguration
from x402 import PaymentPayload, PaymentRequirements
from x402.mechanisms.evm.constants import TX_STATUS_SUCCESS
from x402.mechanisms.evm.data_suffix import append_data_suffix
from x402.mechanisms.evm.exact.facilitator import ExactEvmScheme
from x402.mechanisms.evm.signers import FacilitatorWeb3Signer, _hex_with_0x_prefix
from x402.mechanisms.evm.types import TransactionReceipt
from x402.mechanisms.svm.exact.facilitator import ExactSvmScheme
from x402.mechanisms.svm.types import ExactSvmPayload
from x402.mechanisms.svm.utils import decode_transaction_from_payload, transaction_message_hash

from x402gate.core import http
from x402gate.core.config import NetworkConfig
from x402gate.core.pricing import format_price_for_x402
from x402gate.core.svm_signer import SolanaSigner

logger = logging.getLogger(__name__)

# Cached token prices for gas cost estimation
_price_cache: dict[str, float] = {"ethereum": 0.0, "solana": 0.0, "ts": 0.0, "failed_at": 0.0}
_PRICE_TTL = 300  # 5 minutes
# A failed fetch waits this long before the next try.
_PRICE_RETRY_AFTER_FAILURE = 60
_PRICE_TIMEOUT_S = 5.0
_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
# Two signatures, the fee of a payment without a priority fee.
_SVM_FALLBACK_FEE_LAMPORTS = 10_000
# The SDK waits this long for a Base receipt (its default is 120 s), and the
# pending retry waits once more. Base makes a block every 2 s.
_EVM_CONFIRMATION_TIMEOUT_S = 30
# Base sends are serial, so one hung RPC call must not hold back every other
# settlement for web3's default 30 s. Calls made under the send lock are tried
# once, so each holds the lock for at most this long; web3 otherwise tries a
# timed-out call 5 times. Reads and receipt polls run outside the lock.
_EVM_RPC_TIMEOUT_S = 10
# Attempts per Base read: one retry rides out a dropped connection or a 5xx.
# A read timeout is not retried.
_EVM_READ_ATTEMPTS = 2
# Calls under the send lock stop starting lookups and resends once one send
# has held it this long: a slow but answering RPC must not hold it for every
# call's timeout in turn. The reservation's count read and the send itself
# still run, so the lock is held for at most this, the call in progress at
# the deadline, a count read and a send.
_EVM_SEND_LOCK_BUDGET_S = 10
# A send error with one of these means the node already has this very
# transaction (same hash).
_TX_KNOWN_ERRORS = (
    "already known",
    "alreadyknown",
    "known transaction",
    "already imported",
)
# A send error with one of these means some transaction holds the nonce: our
# own earlier copy when this is a resend. On a first send it may also be ours:
# web3 never retries the send here, but a load balancer in front of the node
# may, after the sequencer took the first try.
_NONCE_TAKEN_ERRORS = (
    "nonce too low",
    "replacement transaction underpriced",
)
# The only send errors that mean the node checked the transaction and turned
# it away before taking it. Any other error, including a node that could not
# forward it to the sequencer ('Post "...": context deadline exceeded', EOF,
# "internal error"), may come after the sequencer took it, so it is unknown.
_SEND_REFUSED_ERRORS = (
    "insufficient funds",
    "transaction underpriced",
    "intrinsic gas too low",
    "exceeds block gas limit",
    "gas limit reached",
    "max fee per gas less than block base fee",
    "fee cap less than block base fee",
    "invalid sender",
    "invalid chain id",
    "only replay-protected",
    "oversized data",
    "transaction type not supported",
    "tx type not supported",
    "exceeds the configured cap",
)
# Gas for the 0-value transfer to itself that fills a given-back nonce.
_GAP_FILL_GAS = 21_000
# Receipts the signer's own wait fetched, kept by hash for settle's gas cost
# so it needs no second fetch. Each settle takes its own, so few are kept.
_RECEIPT_CACHE_SIZE = 256
# keccak("Transfer(address,address,uint256)"), the ERC-20 Transfer event.
_TRANSFER_TOPIC = bytes.fromhex("ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")

# Verify, settle and receipt waits block a thread for up to a confirmation
# timeout, so each chain has its own pool: a slow chain must not fill asyncio's
# default pool, which also serves request hashing and other short work, nor
# starve the other chain. The threads mostly wait on RPC replies.
_CHAIN_WORKERS = 64
_EXECUTORS = {
    "evm": ThreadPoolExecutor(max_workers=_CHAIN_WORKERS, thread_name_prefix="x402-base"),
    "svm": ThreadPoolExecutor(max_workers=_CHAIN_WORKERS, thread_name_prefix="x402-solana"),
}

# The price only converts gas to USD for the dashboard, so settlement never
# waits for it: a stale price is refreshed in the background, and a running
# refresh is held here so it is not garbage-collected.
_background_tasks: set[asyncio.Task] = set()
_price_refresh: asyncio.Task | None = None


async def _fetch_prices() -> dict:
    """Fetch ETH and SOL prices from CoinGecko."""
    r = await http.shared_client().get(
        _COINGECKO_URL,
        params={"ids": "ethereum,solana", "vs_currencies": "usd"},
        timeout=http.timeout(_PRICE_TIMEOUT_S),
    )
    return r.json()


def _prices_fresh(now: float) -> bool:
    """True when the cache needs no fetch: fresh, or a fetch failed recently."""
    if now - _price_cache["ts"] < _PRICE_TTL and (
        _price_cache["ethereum"] > 0 or _price_cache["solana"] > 0
    ):
        return True
    return now - _price_cache["failed_at"] < _PRICE_RETRY_AFTER_FAILURE


async def _refresh_prices() -> None:
    """Refresh cached ETH and SOL prices from CoinGecko. Never raises."""
    now = time.time()
    if _prices_fresh(now):
        return
    try:
        data = await _fetch_prices()
        if not isinstance(data, dict) or not (
            data.get("ethereum", {}).get("usd") or data.get("solana", {}).get("usd")
        ):
            raise ValueError(f"no prices in response: {str(data)[:200]}")
        _price_cache["ethereum"] = data.get("ethereum", {}).get("usd", _price_cache["ethereum"])
        _price_cache["solana"] = data.get("solana", {}).get("usd", _price_cache["solana"])
        _price_cache["ts"] = now
        logger.info(
            "Prices updated: ETH=$%.2f, SOL=$%.2f", _price_cache["ethereum"], _price_cache["solana"]
        )
    except Exception as e:
        _price_cache["failed_at"] = now
        logger.warning("Failed to fetch prices from CoinGecko: %s", e)


def _spawn(coro: Any) -> asyncio.Task:
    """Run ``coro`` in the background, keeping a reference until it ends."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_done)
    return task


def _background_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background payment task failed: %s", task.exception())


def _schedule_price_refresh() -> None:
    """Start a background price refresh when the cache is stale."""
    global _price_refresh
    if _prices_fresh(time.time()):
        return
    running = _price_refresh
    if (
        running is not None
        and not running.done()
        and running.get_loop() is asyncio.get_running_loop()
    ):
        return
    _price_refresh = _spawn(_refresh_prices())


def _cached_price(coin: str) -> float:
    """Return the cached USD price of ``coin`` (0 if never fetched) without waiting."""
    _schedule_price_refresh()
    return _price_cache[coin]


def _evm_gas_cost(receipt: Any) -> tuple[float, float]:
    """Gas a Base receipt paid, in ETH and in USD at the cached ETH price (0 if none)."""
    gas_cost_native = receipt["gasUsed"] * receipt["effectiveGasPrice"] / 1e18
    eth_price = _cached_price("ethereum")
    return gas_cost_native, gas_cost_native * eth_price if eth_price else 0.0


def _topic_address(topic: Any) -> str:
    """The address in an indexed event topic, as lowercase hex without 0x."""
    return bytes(topic)[-20:].hex()


def _settlement_failure_reason(result: Any) -> str:
    """Build a readable reason from a failed settle result.

    The SDK often sets ``error_message`` to None, which used to log as the
    text "None". Prefer the first real detail attached to the result.
    """
    error_reason = getattr(result, "error_reason", None) or "unknown"
    detail = getattr(result, "error_message", None) or getattr(result, "invalid_message", None)
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or str(detail)
    if detail:
        return f"{error_reason}: {detail}"
    return str(error_reason)


def _canonical_nonce(nonce: str) -> str:
    """Return the nonce as plain lowercase hex.

    The SDK reads it with ``bytes.fromhex``, which accepts an optional 0x,
    either case and spaces, so all of those spellings are one nonce.
    """
    try:
        return bytes.fromhex(nonce.strip().removeprefix("0x").removeprefix("0X")).hex()
    except ValueError:
        return nonce.lower()


_PERMIT2_REFUSED = "unsupported_payload: Permit2 is not accepted"


def _is_permit2(payload_dict: Any) -> bool:
    """True for a Permit2 payload, which the gateway never advertises.

    The SDK routes any payload carrying ``permit2Authorization`` to its Permit2
    path, and payment_key cannot identify such a payment by payer and nonce.
    """
    inner = payload_dict.get("payload") if isinstance(payload_dict, dict) else None
    return isinstance(inner, dict) and "permit2Authorization" in inner


def payment_key(payment_signature: str) -> str:
    """Return a stable id for one payment.

    A Base authorization is identified by its payer and nonce, so re-encoding
    the same signed authorization does not produce a new id. A Solana
    transaction is identified by a hash of its bytes, anything else by a hash
    of its inner payload, and an unreadable header by a hash of the header.
    """
    try:
        payload_dict = json.loads(base64.b64decode(payment_signature))
        network = str(payload_dict.get("accepted", {}).get("network", ""))
        inner = payload_dict.get("payload", {})
        authorization = inner.get("authorization") if isinstance(inner, dict) else None
        if isinstance(authorization, dict) and authorization.get("nonce"):
            payer = str(authorization.get("from", "")).lower()
            return f"{network}:{payer}:{_canonical_nonce(str(authorization['nonce']))}"
        transaction = inner.get("transaction") if isinstance(inner, dict) else None
        if isinstance(transaction, str):
            digest = hashlib.sha256(base64.b64decode(transaction)).hexdigest()
        else:
            digest = hashlib.sha256(json.dumps(inner, sort_keys=True).encode()).hexdigest()
        return f"{network}:{digest}"
    except Exception:
        return "raw:" + hashlib.sha256(payment_signature.encode()).hexdigest()


class PaymentClaims:
    """Payments that a request is using right now, or used a moment ago.

    One signed payment must buy one generation. Without this, the same
    payment sent in parallel requests passes verify in all of them, because
    nothing has landed on-chain yet. A claim lasts until the request's
    settlement finishes. After a successful settle the claim is kept for
    ``hold_s`` more, so an RPC node that has not seen the transaction yet
    cannot let a replay through.
    """

    def __init__(self) -> None:
        self._claims: dict[str, float] = {}

    def claim(self, key: str) -> bool:
        """Take the payment for this request. False if another request has it."""
        now = time.monotonic()
        expired = [k for k, until in self._claims.items() if until <= now]
        for k in expired:
            del self._claims[k]
        if key in self._claims:
            return False
        self._claims[key] = float("inf")
        return True

    def release(self, key: str, hold_s: float = 0.0) -> None:
        """Free the payment, keeping it blocked for ``hold_s`` seconds."""
        if hold_s > 0:
            self._claims[key] = time.monotonic() + hold_s
        else:
            self._claims.pop(key, None)

    def clear(self) -> None:
        """Forget every claim."""
        self._claims.clear()


class InFlightAmounts:
    """USDC that each Base payer has promised to requests still settling.

    Each authorization has its own nonce, so the payment claim cannot see
    two of them drawing on one balance. Without this, a wallet holding the
    price of one job could start many paid provider calls at once; only
    one settlement would land.
    """

    def __init__(self) -> None:
        self._amounts: dict[str, Decimal] = {}

    def add(self, payer: str, amount: Decimal) -> Decimal:
        """Promise ``amount`` and return the payer's total in flight."""
        key = payer.lower()
        total = self._amounts.get(key, Decimal("0")) + amount
        self._amounts[key] = total
        return total

    def remove(self, payer: str, amount: Decimal) -> None:
        """Drop a promise once its settlement has finished."""
        key = payer.lower()
        left = self._amounts.get(key, Decimal("0")) - amount
        if left > 0:
            self._amounts[key] = left
        else:
            self._amounts.pop(key, None)

    def clear(self) -> None:
        """Forget every promise."""
        self._amounts.clear()


def settlement_state(settlement: dict[str, Any] | None) -> str:
    """Classify a settle result as "ok", "failed" or "unknown".

    Unknown means the transaction may still land: settle crashed (None), or
    it was broadcast and its confirmation wait failed (settlement_pending).
    """
    if settlement is None or settlement.get("pending"):
        return "unknown"
    return "ok" if settlement.get("success") else "failed"


class _NetworkScheme:
    """Holds a network's scheme, signer, and config for verify/settle."""

    def __init__(
        self,
        net_cfg: NetworkConfig,
        scheme: Any,
        w3: Any = None,
        svm_signer: Any = None,
        evm_signer: Any = None,
    ) -> None:
        self.config = net_cfg
        self.scheme = scheme
        self.w3 = w3  # Web3 instance (EVM only, for gas queries)
        self.svm_signer = svm_signer  # SolanaSigner instance (SVM only, for tx fee queries)
        self.evm_signer = evm_signer  # _ResyncingWeb3Signer (EVM only, for resends)

    async def run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking call on this chain's thread pool."""
        loop = asyncio.get_running_loop()
        executor = _EXECUTORS.get(self.config.type)
        return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))

    def resend_unconfirmed(self) -> None:
        """Send again the Base transactions whose send had no clear answer. Never raises.

        Blocking: run it on the chain's pool.
        """
        if self.evm_signer is None:
            return
        try:
            self.evm_signer.resend_unconfirmed()
        except Exception as e:
            logger.warning("Resend of unconfirmed Base sends failed: %s", e)

    def receipt_for_gas(self, tx_hash: str) -> Any:
        """The receipt of a confirmed Base transaction, for its gas cost.

        The SDK's confirmation wait already fetched it, so it comes from the
        signer; only a receipt it does not hold costs one RPC call. Blocking.
        """
        if self.evm_signer is not None:
            receipt = self.evm_signer.take_receipt(tx_hash)
            if receipt is not None:
                return receipt
        return self.w3.eth.get_transaction_receipt(tx_hash)

    def terminal_failure(self, result: Any) -> str | None:
        """Return why a pending Solana settle failed for certain, if it did.

        The SDK reports every confirm error as settlement_pending, including
        a transaction that landed with an error or expired unlanded. The
        signer records those, so they are a known failure, not an unknown one.
        """
        if self.svm_signer is None or getattr(result, "error_reason", "") != "settlement_pending":
            return None
        signature = getattr(result, "transaction", "") or ""
        return self.svm_signer.terminal_failure(signature) if signature else None

    def forget_pending(self, payload: PaymentPayload) -> None:
        """Drop the SDK's pending entry for a Solana payment known to have failed.

        Otherwise a replay of the same payload skips verify and only waits on
        the old signature again, and once the signer's record of the failure
        is gone that wait times out as unknown. Without the entry, the replay
        is verified again and refused (expired blockhash, or already processed).
        """
        store = getattr(self.scheme, "_pending_store", None)
        if store is None:
            return
        try:
            tx = decode_transaction_from_payload(ExactSvmPayload.from_dict(payload.payload))
            store.delete(transaction_message_hash(tx))
        except Exception as e:
            logger.warning("Could not drop the pending Solana settlement: %s", e)


def _send_outcome(error: Exception) -> str:
    """Sort a failed Base send: "known", "taken", "refused" or "unknown".

    "known": the node already has this transaction. "taken": some transaction
    holds the nonce. "refused": the node checked the transaction and turned it
    away (only the errors in _SEND_REFUSED_ERRORS). "unknown": anything else,
    such as a timeout, a dropped connection, a 5xx from a load balancer, or a
    JSON-RPC error from a replica whose forward to the sequencer failed; the
    transaction may have arrived.
    """
    if not isinstance(error, Web3RPCError) or isinstance(error, RequestTimedOut):
        return "unknown"
    message = str(error).lower()
    if any(s in message for s in _TX_KNOWN_ERRORS):
        return "known"
    # Before the refusals: "replacement transaction underpriced" contains
    # "transaction underpriced".
    if any(s in message for s in _NONCE_TAKEN_ERRORS):
        return "taken"
    if any(s in message for s in _SEND_REFUSED_ERRORS):
        return "refused"
    return "unknown"


def _evm_chain_id(network: str) -> int | None:
    """The chain id in a CAIP-2 "eip155:<id>" network, or None for any other form."""
    namespace, _, reference = network.partition(":")
    if namespace != "eip155" or not reference.isdigit():
        return None
    return int(reference)


class _ResyncingWeb3Signer(FacilitatorWeb3Signer):
    """FacilitatorWeb3Signer that leaves no nonce gap and reports no sent transfer as failed.

    The SDK reserves a nonce before it reads the gas price and sends, and
    resyncs only when the chain's pending count is above its own, so one failed
    send left a gap that every later settlement queued behind. Here the gas
    price and chain id are read first, and sends are serial from the
    reservation to the node's reply, so a nonce the node refused is given back
    while no later one is out: the cursor steps back and nothing queues behind
    it. Only a refusal the node gives before taking a transaction (insufficient
    funds, underpriced and the like), with the node also not having it, gives
    the nonce back and fails the send. Any other error, such as a timeout, a
    dropped connection, a 5xx or a replica's failed forward to the sequencer,
    keeps the nonce and returns the hash, and the receipt wait decides: a
    transfer that may land is not reported as failed. Such a transaction is
    sent again before the next nonce, before a pending settle's second wait,
    and on each reconcile poll that finds no receipt, which fills its nonce if
    it never arrived. If such a resend is refused, its nonce is given back;
    when a later nonce is already out, the resend outside a send fills it with
    a 0-value transfer to the facilitator's own address, so the later ones do
    not queue behind it until the next settlement. A send told that another
    transaction holds its nonce is never signed again at a new one: the holder
    may be this very transfer, and a second copy of the authorization would
    revert and read as a failed payment. Calls under the send lock are tried
    once, and lookups and resends stop after _EVM_SEND_LOCK_BUDGET_S, so one
    send holds the lock for a few RPC timeouts at most.

    Every transaction is signed with a chain id (EIP-155), so it cannot be
    replayed on another chain: the SDK's send_transaction signs a bare legacy
    transaction without one. The id comes from the configured network, or is
    read from the node once and cached when none is given.
    """

    def __init__(self, *args: Any, chain_id: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The SDK's get_chain_id cache: seeded from the network, it needs no RPC.
        if chain_id is not None:
            self._chain_id = chain_id
        endpoint = self._w3.provider.endpoint_uri
        self._w3.provider = Web3.HTTPProvider(
            endpoint,
            request_kwargs={"timeout": _EVM_RPC_TIMEOUT_S},
            exception_retry_configuration=ExceptionRetryConfiguration(
                errors=(ConnectionError, requests.ConnectionError, requests.HTTPError),
                retries=_EVM_READ_ATTEMPTS,
            ),
        )
        # Every call made under the send lock goes here and is tried once.
        self._send_w3 = Web3(
            Web3.HTTPProvider(
                endpoint,
                request_kwargs={"timeout": _EVM_RPC_TIMEOUT_S},
                exception_retry_configuration=None,
            )
        )
        # Held from the nonce reservation to the node's reply.
        self._send_lock = threading.Lock()
        # Nonces handed out and given back while a later one was in use.
        self._free_nonces: set[int] = set()
        # Signed transactions the node could not confirm it took, by nonce.
        self._unconfirmed: dict[int, bytes] = {}
        # Raw receipts of confirmed transactions, by hash, for settle's gas cost.
        self._receipts: dict[str, Any] = {}
        self._receipts_lock = threading.Lock()

    def _reserve_nonce(self, gap_only: bool = False) -> int | None:
        """The next nonce to send at; with ``gap_only``, only a given-back one, else None."""
        with self._nonce_lock:
            pending = self._send_w3.eth.get_transaction_count(self._account.address, "pending")
            # A returned nonce the chain has used since is not reused.
            self._free_nonces = {n for n in self._free_nonces if n >= pending}
            if self._free_nonces:
                nonce = min(self._free_nonces)
                self._free_nonces.discard(nonce)
                return nonce
            if gap_only:
                return None
            if self._next_nonce is None or pending > self._next_nonce:
                self._next_nonce = pending
            nonce = self._next_nonce
            self._next_nonce = nonce + 1
            return nonce

    def _release_nonce(self, nonce: int) -> None:
        with self._nonce_lock:
            self._free_nonces.add(nonce)
            # Step the cursor back over returned nonces at the top.
            while self._next_nonce is not None and self._next_nonce - 1 in self._free_nonces:
                self._next_nonce -= 1
                self._free_nonces.discard(self._next_nonce)

    def write_contract(
        self,
        address: str,
        abi: list[dict[str, Any]],
        function_name: str,
        *args: Any,
        data_suffix: str | None = None,
    ) -> str:
        contract = self._w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)
        func = getattr(contract.functions, function_name)
        # Read (and cached) here, so building under the send lock calls no RPC.
        chain_id = self.get_chain_id()

        def build(fields: dict[str, Any]) -> dict[str, Any]:
            tx = func(*args).build_transaction({**fields, "chainId": chain_id})
            if data_suffix:
                calldata = tx["data"]
                if isinstance(calldata, (bytes, bytearray)):
                    calldata = "0x" + bytes(calldata).hex()
                tx["data"] = append_data_suffix(calldata, data_suffix)
            return tx

        return self._sign_and_send(build)

    def send_transaction(self, to: str, data: bytes) -> str:
        # Read (and cached) here, so building under the send lock calls no RPC.
        chain_id = self.get_chain_id()
        to = Web3.to_checksum_address(to)
        return self._sign_and_send(
            lambda fields: {**fields, "to": to, "data": data, "chainId": chain_id}
        )

    def _sign_and_send(self, build: Any) -> str:
        """Build, sign and send one transaction; give its nonce back if it was not sent."""
        gas_price = self._w3.eth.gas_price
        with self._send_lock:
            deadline = time.monotonic() + _EVM_SEND_LOCK_BUDGET_S
            self._resend_unconfirmed(deadline)
            return self._send_at(self._reserve_nonce(), build, gas_price, deadline)

    def _send_at(self, nonce: int, build: Any, gas_price: int, deadline: float) -> str:
        """Sign and send at a reserved nonce; give it back if the node refused the send.

        Called under the send lock. Returns the transaction hash whenever the
        transaction may land, and the receipt wait decides.
        """
        fields = {
            "from": self._account.address,
            "nonce": nonce,
            "gas": self._gas_limit,
            "gasPrice": gas_price,
        }
        try:
            signed_tx = self._account.sign_transaction(build(fields))
        except Exception:
            self._release_nonce(nonce)
            raise
        try:
            tx_hash = self._send_w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        except Exception as e:
            outcome = _send_outcome(e)
            if outcome in ("refused", "taken"):
                found = self._node_has(signed_tx.hash, deadline)
                if found is False and outcome == "refused":
                    # The node turned it away and does not have it.
                    self._release_nonce(nonce)
                    raise
                if found is None:
                    outcome = "unknown"
                # found is False with the nonce taken: the holder may be this
                # very transaction, delivered by an upstream retry of the send
                # and missed by a lagging replica's lookup. It is not signed
                # again at a new nonce, since a second copy of the same
                # authorization would revert and read as a failed payment. The
                # nonce stays used, and resending cannot help.
            if outcome == "unknown":
                # It may have arrived: keep the nonce and send it again
                # before the next nonce, which fills it if it never did.
                self._unconfirmed[nonce] = bytes(signed_tx.raw_transaction)
            # Only the reply may be lost: the receipt wait decides.
            logger.warning("Base send of nonce %d got no clear answer (%s), awaiting it", nonce, e)
            return _hex_with_0x_prefix(signed_tx.hash)
        return _hex_with_0x_prefix(tx_hash)

    def resend_unconfirmed(self) -> None:
        """Send again each transaction whose send had no clear answer, then fill any gap.

        Without this, such a transaction went out again only before the next
        settlement's send, so on a quiet gateway it was never sent. It reuses
        the signed bytes and nonce, so it cannot pay twice. A resend the node
        refuses gives its nonce back, and with a later nonce already out
        nothing but a new send would fill it: the later transfers would queue
        behind it until their authorizations expire. So the lowest given-back
        nonce gets a 0-value transfer to the facilitator's own address.
        """
        # A given-back nonce below the cursor means a later one was reserved:
        # _release_nonce steps the cursor back over given-back nonces at the top.
        if not self._unconfirmed and not self._free_nonces:
            return
        try:
            # Read before the lock, as for a send.
            gas_price = self._w3.eth.gas_price
            chain_id = self.get_chain_id()
        except Exception as e:
            logger.warning("Base gas price read failed, gap fill skipped: %s", e)
            gas_price = None
        with self._send_lock:
            deadline = time.monotonic() + _EVM_SEND_LOCK_BUDGET_S
            self._resend_unconfirmed(deadline)
            if gas_price is None or not self._free_nonces or time.monotonic() >= deadline:
                return
            nonce = self._reserve_nonce(gap_only=True)
            if nonce is None:
                return
            to = self._account.address
            try:
                tx_hash = self._send_at(
                    nonce,
                    lambda fields: {
                        **fields,
                        "gas": _GAP_FILL_GAS,
                        "to": to,
                        "value": 0,
                        "chainId": chain_id,
                    },
                    gas_price,
                    deadline,
                )
            except Exception as e:
                # Given back: the next call or send tries again.
                logger.warning("Base nonce %d gap fill failed: %s", nonce, e)
                return
            logger.warning("Base nonce %d gap filled by transaction %s", nonce, tx_hash)

    def _resend_unconfirmed(self, deadline: float) -> None:
        """Send again each unconfirmed transaction the chain has not counted yet.

        Called under the send lock. After ``deadline`` no resend or lookup
        starts: the rest are kept for the next call.
        """
        if not self._unconfirmed:
            return
        try:
            pending = self._send_w3.eth.get_transaction_count(self._account.address, "pending")
        except Exception as e:
            # Keep them for the next send; this one still reserves a nonce.
            logger.warning("Base nonce read failed, resend of unconfirmed sends skipped: %s", e)
            return
        for nonce, raw in sorted(self._unconfirmed.items()):
            if nonce >= pending:
                if time.monotonic() >= deadline:
                    break
                try:
                    self._send_w3.eth.send_raw_transaction(raw)
                except Exception as e:
                    outcome = _send_outcome(e)
                    if outcome == "refused":
                        found = self._node_has(Web3.keccak(raw), deadline)
                        if found is False:
                            # Refused: the nonce is free, and the next send or
                            # resend_unconfirmed's gap fill takes it.
                            self._release_nonce(nonce)
                        elif found is None:
                            break
                    elif outcome == "unknown":
                        # Still no answer: keep this and the later ones for the
                        # next send rather than wait on each under the lock.
                        break
                    # "known" or "taken": the node or the chain holds the nonce.
            del self._unconfirmed[nonce]

    def wait_for_transaction_receipt(self, tx_hash: str) -> TransactionReceipt:
        """The SDK's receipt wait, keeping the raw receipt for settle's gas cost."""
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash
        receipt = self._w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=self._confirmation_timeout_seconds
        )
        with self._receipts_lock:
            self._receipts[tx_hash.lower()] = receipt
            while len(self._receipts) > _RECEIPT_CACHE_SIZE:
                del self._receipts[next(iter(self._receipts))]
        return TransactionReceipt(
            status=TX_STATUS_SUCCESS if receipt["status"] == 1 else 0,
            block_number=receipt["blockNumber"],
            tx_hash=tx_hash,
            logs=list(receipt.get("logs") or []),
        )

    def take_receipt(self, tx_hash: str) -> Any:
        """The raw receipt the receipt wait fetched for ``tx_hash`` (removed), or None."""
        key = tx_hash.lower() if tx_hash.startswith("0x") else "0x" + tx_hash.lower()
        with self._receipts_lock:
            return self._receipts.pop(key, None)

    def _node_has(self, tx_hash: Any, deadline: float | None = None) -> bool | None:
        """Whether the node has the transaction, or None when it did not answer.

        Also None, with no call, once ``deadline`` (a monotonic time) has passed.
        """
        if deadline is not None and time.monotonic() >= deadline:
            return None
        try:
            self._send_w3.eth.get_transaction(tx_hash)
        except TransactionNotFound:
            return False
        except Exception:
            return None
        return True


class PaymentHandler:
    """Handles x402 payment verification and settlement.

    Supports multiple networks (EVM + SVM). Each network is configured
    independently with its own scheme, signer, and RPC endpoint.
    """

    def __init__(
        self,
        networks: dict[str, NetworkConfig],
        max_timeout: int = 3600,
    ) -> None:
        self._max_timeout = max_timeout
        self._schemes: dict[str, _NetworkScheme] = {}

        for name, net_cfg in networks.items():
            try:
                if net_cfg.type == "evm":
                    self._init_evm(name, net_cfg)
                elif net_cfg.type == "svm":
                    self._init_svm(name, net_cfg)
                else:
                    logger.warning(
                        "Unknown network type '%s' for '%s', skipping", net_cfg.type, name
                    )
            except Exception as e:
                logger.warning(
                    "Failed to initialize network '%s' (%s): %s — skipping",
                    name,
                    net_cfg.type,
                    e,
                )

    def _init_evm(self, name: str, cfg: NetworkConfig) -> None:
        """Initialize an EVM network (Base, Ethereum, etc.)."""

        signer = _ResyncingWeb3Signer(
            private_key=cfg.facilitator_key,
            rpc_url=cfg.rpc_url,
            confirmation_timeout_seconds=_EVM_CONFIRMATION_TIMEOUT_S,
            chain_id=_evm_chain_id(cfg.network),
        )
        scheme = ExactEvmScheme(signer)

        # Reads share the signer's bounded provider: web3's default tries a
        # timed-out call 5 times at 30 s each, which held a Base pool thread
        # about 150 s per receipt poll or balance check on a hung RPC.
        self._schemes[cfg.network] = _NetworkScheme(cfg, scheme, w3=signer._w3, evm_signer=signer)
        logger.info(
            "EVM network '%s' initialized (address=%s, network=%s)",
            name,
            signer.address,
            cfg.network,
        )

    def _init_svm(self, name: str, cfg: NetworkConfig) -> None:
        """Initialize an SVM network (Solana)."""

        signer = SolanaSigner(
            private_key=cfg.facilitator_key,
            rpc_url=cfg.rpc_url,
        )
        scheme = ExactSvmScheme(signer)

        self._schemes[cfg.network] = _NetworkScheme(cfg, scheme, svm_signer=signer)
        logger.info(
            "SVM network '%s' initialized (address=%s, network=%s)",
            name,
            signer.address,
            cfg.network,
        )

    def get_all_payment_requirements(self, price: Decimal) -> list[dict[str, Any]]:
        """Build payment requirements for ALL configured networks.

        The client chooses which network to pay on.

        Args:
            price: Final price in USD (after commission).

        Returns:
            List of payment requirements dicts — one per network.
        """
        requirements = []
        for _network_id, ns in self._schemes.items():
            req = self._build_requirements(price, ns)
            requirements.append(req)
        return requirements

    def _build_requirements(self, price: Decimal, ns: _NetworkScheme) -> dict[str, Any]:
        """Build payment requirements for a single network."""
        cfg = ns.config
        price_str = format_price_for_x402(price)

        # USDC has 6 decimals on both EVM and Solana
        amount = str(int(price * 1_000_000))

        req: dict[str, Any] = {
            "scheme": "exact",
            "network": cfg.network,
            "payTo": cfg.pay_to,
            "price": price_str,
            "asset": cfg.token_address,
            "amount": amount,
            "maxTimeoutSeconds": self._max_timeout,
        }

        if cfg.type == "evm":
            req["extra"] = {
                "name": cfg.token_name,
                "version": cfg.token_version,
            }
        elif cfg.type == "svm":
            # SVM requires feePayer in extra
            req["extra"] = ns.scheme.get_extra(cfg.network)

        return req

    def create_payment_required(self, price: Decimal) -> JSONResponse:
        """Build an HTTP 402 response with x402 payment requirements.

        Returns requirements for all configured networks so the client
        can choose which chain to pay on.
        """
        accepts = self.get_all_payment_requirements(price)
        body = {
            "error": "Payment Required",
            "accepts": accepts,
        }
        return JSONResponse(
            status_code=402,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    def _detect_network(self, payment_signature: str) -> tuple[str, dict]:
        """Detect which network a payment signature is for.

        Args:
            payment_signature: Base64-encoded payment payload.

        Returns:
            Tuple of (network_id, payload_dict).
        """
        payload_json = base64.b64decode(payment_signature)
        payload_dict = json.loads(payload_json)

        # x402 v2 payload has 'accepted.network'
        network = payload_dict.get("accepted", {}).get("network", "")
        return network, payload_dict

    def get_payment_requirements_for_network(self, price: Decimal, network: str) -> dict[str, Any]:
        """Get requirements for a specific network."""
        ns = self._schemes.get(network)
        if not ns:
            raise ValueError(f"Unknown network: {network}")
        return self._build_requirements(price, ns)

    @staticmethod
    def extract_payment_signature(request: Request) -> str | None:
        """Extract the PAYMENT-SIGNATURE header from a request."""
        return request.headers.get("payment-signature")

    async def verify(self, payment_signature: str, price: Decimal) -> tuple[bool, str, str, str]:
        """Verify a payment signature on the correct network.

        Args:
            payment_signature: Base64-encoded payment payload.
            price: Expected payment amount.

        Returns:
            Tuple of (is_valid, network_id, payer_address, invalid_reason).
        """
        # Settle follows a passed verify, so the gas price is fetched now and
        # is in the cache by then.
        _schedule_price_refresh()
        try:
            network, payload_dict = self._detect_network(payment_signature)
            ns = self._schemes.get(network)
            if not ns:
                logger.warning("Payment on unsupported network: %s", network)
                return False, network, "", "unsupported_network"

            if _is_permit2(payload_dict):
                logger.warning("Payment rejected on %s: Permit2 is not accepted", network)
                return False, network, "", _PERMIT2_REFUSED

            payload = PaymentPayload.model_validate(payload_dict)
            requirements_dict = self._build_requirements(price, ns)
            requirements = PaymentRequirements.model_validate(requirements_dict)

            result = await ns.run(ns.scheme.verify, payload, requirements)
            amount_usdc = int(requirements.amount) / 1_000_000
            payer = getattr(result, "payer", "") or ""
            invalid_reason = getattr(result, "invalid_reason", "") or ""
            invalid_message = getattr(result, "invalid_message", "") or ""
            if result.is_valid:
                logger.info(
                    "Payment verified: $%g USDC from %s on %s (valid=True)",
                    amount_usdc,
                    payer[:10] + "…" if payer else "",
                    network,
                )
            else:
                logger.warning(
                    "Payment rejected: $%g USDC from %s on %s — reason=%s: %s",
                    amount_usdc,
                    payer[:10] + "…" if payer else "",
                    network,
                    invalid_reason,
                    invalid_message,
                )
            reason = f"{invalid_reason}: {invalid_message}" if invalid_message else invalid_reason
            return result.is_valid, network, payer, reason

        except Exception:
            logger.exception("Payment verification failed")
            return False, "", "", "internal_error"

    async def check_evm_balance(self, network: str, payer: str, amount: Decimal) -> bool | None:
        """Check if payer has sufficient USDC balance on EVM.

        EVM verify() only checks the ECDSA signature — not the on-chain balance.
        This method prevents free generation from clients with insufficient funds.
        An RPC failure is not treated as a success: the caller must stop the
        request, because Base settlement happens after the result is returned.

        Args:
            network: Network ID (e.g. "eip155:8453").
            payer: Payer EVM address (0x...).
            amount: Required amount in USD.

        Returns:
            True if balance >= required amount, False if the balance is too
            small, None if the RPC check could not be completed.
        """
        ns = self._schemes.get(network)
        if not ns or not ns.w3:
            return True  # non-EVM or no w3 — skip check

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                bal_sig = ns.w3.keccak(text="balanceOf(address)")[:4]
                data = bal_sig + bytes(12) + bytes.fromhex(payer[2:])
                raw = await ns.run(ns.w3.eth.call, {"to": ns.config.token_address, "data": data})
                balance = int.from_bytes(raw, "big")
                required = int(amount * 1_000_000)
                if balance < required:
                    logger.warning(
                        "Insufficient USDC: %s has %d, need %d",
                        payer[:10] + "…",
                        balance,
                        required,
                    )
                return balance >= required
            except Exception as e:
                last_error = e
                logger.warning(
                    "EVM balance check failed for %s (attempt %d/2): %s",
                    payer[:10] + "…",
                    attempt + 1,
                    e,
                )
        logger.error(
            "EVM balance check unavailable for %s: %s",
            payer[:10] + "…",
            last_error,
        )
        return None

    async def settle(
        self,
        payment_signature: str,
        price: Decimal,
        network: str,
        extra_context: dict | None = None,
    ) -> dict[str, Any] | None:
        """Settle a verified payment on-chain.

        Args:
            payment_signature: Base64-encoded payment payload.
            price: Payment amount.
            network: Network ID (from verify result).

        Returns:
            Settlement response dict on success, None on failure.
        """
        try:
            _, payload_dict = self._detect_network(payment_signature)
            ns = self._schemes.get(network)
            if not ns:
                logger.error("Cannot settle — unknown network: %s", network)
                return None

            if _is_permit2(payload_dict):
                # verify refuses these first; a None here would read as unknown.
                logger.error("Cannot settle — Permit2 payload on %s", network)
                return {"success": False, "error_reason": _PERMIT2_REFUSED, "pending": False}

            payload = PaymentPayload.model_validate(payload_dict)
            requirements_dict = self._build_requirements(price, ns)
            requirements = PaymentRequirements.model_validate(requirements_dict)

            result = await ns.run(ns.scheme.settle, payload, requirements)
            known_failure = ns.terminal_failure(result)
            if (
                known_failure is None
                and not result.success
                and getattr(result, "error_reason", "") == "settlement_pending"
            ):
                # Broadcast, but the confirmation wait failed. The SDK keeps the
                # transaction, and settling the same payload again waits for it
                # instead of sending a new one.
                logger.warning("Settlement pending on %s; waiting for it once more", network)
                # A send that may never have reached the node goes out again
                # first, so it can still land within this wait.
                await ns.run(ns.resend_unconfirmed)
                result = await ns.run(ns.scheme.settle, payload, requirements)
                known_failure = ns.terminal_failure(result)
            if known_failure is not None:
                ns.forget_pending(payload)
            amount_usdc = int(requirements.amount) / 1_000_000
            settle_data = result.__dict__.copy()

            if result.success:
                # Gas cost reporting
                gas_cost_native = 0.0
                gas_cost_usd = 0.0
                gas_label = ""

                if ns.config.type == "evm" and ns.w3:
                    try:
                        receipt = await ns.run(ns.receipt_for_gas, result.transaction)
                        gas_cost_native, gas_cost_usd = _evm_gas_cost(receipt)
                        gas_label = "ETH"
                    except Exception as e:
                        logger.warning("EVM gas cost lookup failed: %s", e)
                        gas_label = "ETH"
                elif ns.config.type == "svm" and ns.svm_signer:
                    # The fee follows from the signed message, so no RPC call:
                    # a getTransaction here delayed every Solana job.
                    try:
                        fee_lamports = ns.svm_signer.fee_lamports(payload.payload["transaction"])
                    except Exception as e:
                        logger.warning("SVM gas cost lookup failed: %s", e)
                        fee_lamports = _SVM_FALLBACK_FEE_LAMPORTS
                    gas_cost_native = fee_lamports / 1e9
                    sol_price = _cached_price("solana")
                    gas_cost_usd = gas_cost_native * sol_price if sol_price else 0
                    gas_label = "SOL"

                settle_data["gas_cost_native"] = gas_cost_native
                settle_data["gas_cost_usd"] = gas_cost_usd
                settle_data["gas_label"] = gas_label
                settle_data["amount_usdc"] = amount_usdc
                settle_data["network_type"] = ns.config.type

                # Compact single-line transaction summary
                provider_cost = 0.0
                generation_s = 0.0
                t_client = 0.0
                provider_label = "?"
                profit = amount_usdc - gas_cost_usd

                if extra_context:
                    provider_cost = extra_context.get("provider_cost", 0)
                    generation_s = extra_context.get("generation_s", 0)
                    provider_label = extra_context.get("provider_name", "?")
                    t_client = extra_context.get("t_client", generation_s)
                    profit = amount_usdc - provider_cost - gas_cost_usd

                def _fmt(v: float) -> str:
                    """Format USD with meaningful precision, no trailing zeros."""
                    return f"{v:.6f}".rstrip("0").rstrip(".")

                logger.info(
                    "TX %s settled: $%s -> cost=$%s gas=$%s %s profit=$%s (%.1fs -> %.1fs)",
                    provider_label,
                    _fmt(amount_usdc),
                    _fmt(provider_cost),
                    _fmt(gas_cost_usd),
                    gas_label,
                    _fmt(profit),
                    generation_s,
                    t_client,
                )
            else:
                reason = _settlement_failure_reason(result)
                if known_failure is not None:
                    # Landed with an error or expired: no payment was taken,
                    # and a later confirm of it fails at once.
                    reason = f"transaction_failed: {known_failure}"
                logger.warning(
                    "Settlement failed: $%.6f USDC from %s on %s -- %s",
                    amount_usdc,
                    getattr(result, "payer", "") or "",
                    network,
                    reason,
                )
                settle_data["error_reason"] = reason
                settle_data["pending"] = (
                    known_failure is None
                    and getattr(result, "error_reason", "") == "settlement_pending"
                )
            return settle_data

        except Exception:
            logger.exception("Payment settlement failed")
            return None

    async def settlement_receipt(
        self, network: str, tx_hash: str, payer: str, amount: Decimal
    ) -> dict[str, Any] | None:
        """Look up a broadcast Base settlement by its transaction hash.

        Returns a settlement dict shaped like :meth:`settle`'s. It is a
        success only when the receipt shows ``amount`` of USDC moving from
        ``payer`` to the gateway, and a failure when the transaction reverted
        or moved anything else. None while no receipt is found: not mined
        yet, or the RPC failed. The RPC calls run on the Base pool.

        With no receipt found, the signer's transactions whose send had no
        clear answer are sent again, so a reconcile keeps sending a transfer
        that may never have reached the node until it lands or expires.
        """
        ns = self._schemes.get(network)
        if ns is None or ns.w3 is None:
            return None
        try:
            receipt = await ns.run(ns.w3.eth.get_transaction_receipt, tx_hash)
        except Exception as e:  # TransactionNotFound, or an RPC error
            logger.debug("No receipt yet for Base tx %s: %s", tx_hash, e)
            receipt = None
        if receipt is None:
            await ns.run(ns.resend_unconfirmed)
            return None
        payer_hex = payer.lower().removeprefix("0x")
        pay_to_hex = ns.config.pay_to.lower().removeprefix("0x")
        token = ns.config.token_address.lower()
        atomic = int(amount * 1_000_000)
        landed = receipt["status"] == 1 and any(
            str(log["address"]).lower() == token
            and len(log["topics"]) == 3
            and bytes(log["topics"][0]) == _TRANSFER_TOPIC
            and _topic_address(log["topics"][1]) == payer_hex
            and _topic_address(log["topics"][2]) == pay_to_hex
            and int.from_bytes(bytes(log["data"]), "big") == atomic
            for log in receipt["logs"]
        )
        gas_cost_native, gas_cost_usd = _evm_gas_cost(receipt)
        return {
            "success": landed,
            "transaction": tx_hash,
            "error_reason": None if landed else "transaction_failed: no matching transfer",
            "pending": False,
            "amount_usdc": float(amount),
            "gas_cost_native": gas_cost_native,
            "gas_cost_usd": gas_cost_usd,
            "gas_label": "ETH",
            "network_type": ns.config.type,
        }
