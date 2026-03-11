"""Solana (SVM) facilitator signer for x402gate.

Implements the FacilitatorSvmSigner protocol from x402 SDK,
handling transaction signing, simulation, sending, and confirmation
on Solana mainnet.

Uses AsyncClient on a dedicated background event loop for non-blocking,
concurrent RPC calls.  The protocol interface stays synchronous because
the x402 SDK calls these methods from sync code; coroutines are submitted
via ``asyncio.run_coroutine_threadsafe`` and block the caller thread only
until the result is ready — without preventing other coroutines from
running concurrently on the same loop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
import time

from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solders.keypair import Keypair
from solders.signature import Signature
from solders.transaction import VersionedTransaction

logger = logging.getLogger(__name__)


class SolanaSigner:
    """Facilitator signer for Solana (SVM) networks.

    Implements the FacilitatorSvmSigner protocol required by
    x402's ExactSvmScheme.

    A dedicated background thread runs a persistent event loop with a
    single ``AsyncClient``.  All RPC calls are submitted as coroutines
    via ``run_coroutine_threadsafe``, allowing multiple verify/settle
    operations to proceed **concurrently** without blocking each other.
    """

    def __init__(self, private_key: str, rpc_url: str) -> None:
        """Initialize with a Solana keypair.

        Args:
            private_key: Base58-encoded Solana private key (64-byte keypair).
            rpc_url: Solana RPC endpoint URL.
        """
        self._keypair = Keypair.from_base58_string(private_key)
        self._rpc_url = rpc_url

        # Persistent background event loop + async client
        self._loop = asyncio.new_event_loop()
        self._client = AsyncClient(rpc_url)
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="solana-rpc-loop",
        )
        self._thread.start()

        logger.info(
            "Solana facilitator initialized (address=%s)",
            str(self._keypair.pubkey()),
        )

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------

    def _run_async(self, coro, *, timeout: float = 90):  # noqa: ANN001
        """Submit a coroutine to the background loop and wait for the result.

        Multiple callers (from different ``asyncio.to_thread`` workers)
        can submit coroutines concurrently — the background loop runs
        them all in parallel via cooperative multitasking.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Protocol: properties
    # ------------------------------------------------------------------

    @property
    def address(self) -> str:
        """The facilitator's Solana address (base58)."""
        return str(self._keypair.pubkey())

    def get_addresses(self) -> list[str]:
        """Get all fee payer addresses managed by this signer."""
        return [str(self._keypair.pubkey())]

    # ------------------------------------------------------------------
    # Protocol: sign_transaction  (pure crypto, no RPC)
    # ------------------------------------------------------------------

    def sign_transaction(
        self,
        tx_base64: str,
        fee_payer: str,
        network: str,
    ) -> str:
        """Sign a partially-signed transaction with the facilitator keypair.

        Supports both legacy and versioned (v0) transactions:
        - Legacy transactions: Sign message bytes directly
        - Versioned (v0) transactions: Sign with 0x80 version prefix

        Args:
            tx_base64: Base64-encoded partially-signed transaction.
            fee_payer: Fee payer address (must match our keypair).
            network: CAIP-2 network identifier.

        Returns:
            Base64-encoded fully-signed transaction.
        """
        _ = network
        if fee_payer != str(self._keypair.pubkey()):
            raise ValueError(f"Unknown fee payer: {fee_payer}")

        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)

        message = tx.message
        is_versioned = self._is_versioned_transaction(tx_bytes)

        msg_to_sign = bytes([0x80]) + bytes(message) if is_versioned else bytes(message)

        sig = self._keypair.sign_message(msg_to_sign)

        new_sigs = list(tx.signatures)
        new_sigs[0] = sig

        signed_tx = VersionedTransaction.populate(message, new_sigs)
        return base64.b64encode(bytes(signed_tx)).decode()

    @staticmethod
    def _is_versioned_transaction(tx_bytes: bytes) -> bool:
        """Determine if tx bytes are versioned (v0) or legacy.

        Versioned transactions have a version byte >= 128 (0x80) at the
        start of the message portion.
        """
        offset = 0
        first_byte = tx_bytes[offset]
        if first_byte < 0x80:
            num_signatures = first_byte
            offset += 1
        else:
            num_signatures = (first_byte & 0x7F) | ((tx_bytes[offset + 1] & 0x7F) << 7)
            offset += 2

        offset += num_signatures * 64

        if offset < len(tx_bytes):
            return tx_bytes[offset] >= 0x80
        return False

    # ------------------------------------------------------------------
    # Protocol: simulate_transaction
    # ------------------------------------------------------------------

    def simulate_transaction(self, tx_base64: str, network: str) -> None:
        """Simulate a signed transaction to verify it would succeed."""
        _ = network
        self._run_async(self._simulate_async(tx_base64))

    async def _simulate_async(self, tx_base64: str) -> None:
        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        result = await self._client.simulate_transaction(tx)
        if result.value.err:
            raise RuntimeError(f"Simulation failed: {result.value.err}")

    # ------------------------------------------------------------------
    # Protocol: send_transaction
    # ------------------------------------------------------------------

    def send_transaction(self, tx_base64: str, network: str) -> str:
        """Send a signed transaction to the Solana network."""
        _ = network
        return self._run_async(self._send_async(tx_base64))

    async def _send_async(self, tx_base64: str) -> str:

        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)
        opts = TxOpts(skip_preflight=False, max_retries=3)
        result = await self._client.send_transaction(tx, opts=opts)

        sig = getattr(result, "value", None)
        if sig:
            logger.info("Transaction sent: %s", str(sig)[:16] + "...")
            return str(sig)
        raise RuntimeError(f"Transaction send failed: {result}")

    # ------------------------------------------------------------------
    # Protocol: confirm_transaction
    # ------------------------------------------------------------------

    def confirm_transaction(self, signature: str, network: str) -> None:
        """Wait for transaction confirmation on Solana."""
        _ = network
        self._run_async(self._confirm_async(signature))

    async def _confirm_async(self, signature: str) -> None:

        sig = Signature.from_string(signature)
        timeout = 60
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            resp = await self._client.get_signature_statuses(
                [sig],
                search_transaction_history=True,
            )
            if resp.value and resp.value[0]:
                status = resp.value[0]
                if status.err:
                    raise RuntimeError(f"Transaction failed: {status.err}")
                if status.confirmation_status is not None:
                    conf = str(status.confirmation_status).lower()
                    if "confirmed" in conf or "finalized" in conf:
                        elapsed = time.monotonic() - start
                        logger.info(
                            "Transaction %s %s (%.1fs)",
                            signature[:16] + "...",
                            conf,
                            elapsed,
                        )
                        return
            # NOTE: This polling interval does NOT affect client latency.
            # Settlement runs as a background task (asyncio.create_task in app.py),
            # so the HTTP response is already returned to the client before this runs.
            await asyncio.sleep(2)

        raise RuntimeError(f"Transaction {signature} not confirmed within {timeout}s")

    # ------------------------------------------------------------------
    # Extra: async get_transaction (used by payment.py for gas reporting)
    # ------------------------------------------------------------------

    async def get_transaction_async(
        self,
        tx_sig,  # noqa: ANN001
        *,
        commitment: str = "confirmed",
        max_supported_transaction_version: int = 0,
    ):
        """Fetch a confirmed transaction (async, called from payment.py).

        Creates a temporary AsyncClient because this method runs on the
        main event loop (not the signer's background loop).
        """
        async with AsyncClient(self._rpc_url) as client:
            return await client.get_transaction(
                tx_sig,
                commitment=commitment,
                max_supported_transaction_version=max_supported_transaction_version,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the background event loop and client."""

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
