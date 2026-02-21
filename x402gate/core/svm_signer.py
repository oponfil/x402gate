"""Solana (SVM) facilitator signer for x402gate.

Implements the FacilitatorSvmSigner protocol from x402 SDK,
handling transaction signing, simulation, sending, and confirmation
on Solana mainnet.
"""

from __future__ import annotations

import base64
import logging
import time

from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.signature import Signature
from solders.transaction import VersionedTransaction

logger = logging.getLogger(__name__)


class SolanaSigner:
    """Facilitator signer for Solana (SVM) networks.

    Implements the FacilitatorSvmSigner protocol required by
    x402's ExactSvmScheme.
    """

    def __init__(self, private_key: str, rpc_url: str) -> None:
        """Initialize with a Solana keypair.

        Args:
            private_key: Base58-encoded Solana private key (64-byte keypair).
            rpc_url: Solana RPC endpoint URL.
        """
        self._keypair = Keypair.from_base58_string(private_key)
        self._rpc_url = rpc_url
        self._client = SolanaClient(rpc_url)
        logger.info(
            "Solana facilitator initialized (address=%s)",
            str(self._keypair.pubkey()),
        )

    @property
    def address(self) -> str:
        """The facilitator's Solana address (base58)."""
        return str(self._keypair.pubkey())

    def get_addresses(self) -> list[str]:
        """Get all fee payer addresses managed by this signer."""
        return [str(self._keypair.pubkey())]

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

        # Get the message and existing signatures
        message = tx.message

        # Detect versioned vs legacy transaction
        is_versioned = self._is_versioned_transaction(tx_bytes)

        # Versioned (MessageV0): prepend 0x80; Legacy: sign directly
        msg_to_sign = bytes([0x80]) + bytes(message) if is_versioned else bytes(message)

        sig = self._keypair.sign_message(msg_to_sign)

        # Build new signatures list — fee payer is always index 0
        new_sigs = list(tx.signatures)
        new_sigs[0] = sig

        signed_tx = VersionedTransaction.populate(message, new_sigs)
        signed_bytes = bytes(signed_tx)
        return base64.b64encode(signed_bytes).decode()

    @staticmethod
    def _is_versioned_transaction(tx_bytes: bytes) -> bool:
        """Determine if transaction bytes represent a versioned (v0) or legacy transaction.

        Versioned transactions have a version byte >= 128 (0x80) at the start
        of the message portion. Legacy transactions have numRequiredSignatures < 128.
        """
        offset = 0

        # Read compact u16 for signature count
        first_byte = tx_bytes[offset]
        if first_byte < 0x80:
            num_signatures = first_byte
            offset += 1
        else:
            num_signatures = (first_byte & 0x7F) | ((tx_bytes[offset + 1] & 0x7F) << 7)
            offset += 2

        # Skip signatures (64 bytes each)
        offset += num_signatures * 64

        # First byte of message: >= 128 means versioned, < 128 means legacy
        if offset < len(tx_bytes):
            return tx_bytes[offset] >= 0x80

        return False

    def simulate_transaction(self, tx_base64: str, network: str) -> None:
        """Simulate a signed transaction to verify it would succeed.

        Args:
            tx_base64: Base64-encoded fully-signed transaction.
            network: CAIP-2 network identifier.

        Raises:
            RuntimeError: If simulation fails.
        """
        _ = network
        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)

        result = self._client.simulate_transaction(tx)
        if result.value.err:
            raise RuntimeError(f"Simulation failed: {result.value.err}")

    def send_transaction(self, tx_base64: str, network: str) -> str:
        """Send a signed transaction to the Solana network.

        Args:
            tx_base64: Base64-encoded fully-signed transaction.
            network: CAIP-2 network identifier.

        Returns:
            Transaction signature (base58-encoded).

        Raises:
            RuntimeError: If send fails.
        """
        from solana.rpc.types import TxOpts

        _ = network
        tx_bytes = base64.b64decode(tx_base64)
        tx = VersionedTransaction.from_bytes(tx_bytes)

        opts = TxOpts(skip_preflight=False, max_retries=3)
        result = self._client.send_transaction(tx, opts=opts)
        sig = getattr(result, "value", None)
        if sig:
            logger.info("Transaction sent: %s", sig)
            return str(sig)
        raise RuntimeError(f"Transaction send failed: {result}")

    def confirm_transaction(self, signature: str, network: str) -> None:
        """Wait for transaction confirmation on Solana.

        Args:
            signature: Transaction signature (base58-encoded).
            network: CAIP-2 network identifier.

        Raises:
            RuntimeError: If confirmation fails or times out.
        """
        _ = network
        sig = Signature.from_string(signature)
        timeout = 60  # seconds
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            resp = self._client.get_signature_statuses(
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
            time.sleep(2)

        raise RuntimeError(f"Transaction {signature} not confirmed within {timeout}s")
