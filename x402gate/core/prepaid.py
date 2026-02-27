"""Prepaid balance management for x402gate.

In-memory wallet-bound balances: users top-up once via Solana transaction,
then spend from the balance on subsequent requests using Ed25519 signatures
(no on-chain transactions per request).

WARNING: Balances are stored in RAM only and will be lost on server restart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal

from solders.pubkey import Pubkey
from solders.signature import Signature

logger = logging.getLogger(__name__)

# Global in-memory balances: {solana_pubkey_base58: Decimal(USD)}
_balances: dict[str, Decimal] = {}
_lock = asyncio.Lock()

# Signature timestamp validity window (seconds)
TIMESTAMP_WINDOW = 60


async def deposit(pubkey: str, amount: Decimal) -> Decimal:
    """Add funds to a wallet's prepaid balance.

    Args:
        pubkey: Solana public key (base58).
        amount: Amount in USD to add (after commission deduction).

    Returns:
        New total balance.
    """
    async with _lock:
        current = _balances.get(pubkey, Decimal("0"))
        new_balance = current + amount
        _balances[pubkey] = new_balance
        logger.info(
            "Prepaid deposit: +$%s to %s (balance: $%s)",
            amount, pubkey[:12] + "…", new_balance,
        )
        return new_balance


async def deduct(pubkey: str, amount: Decimal) -> bool:
    """Deduct funds from a wallet's prepaid balance.

    Args:
        pubkey: Solana public key (base58).
        amount: Amount in USD to deduct.

    Returns:
        True if deduction succeeded, False if insufficient funds.
    """
    async with _lock:
        current = _balances.get(pubkey, Decimal("0"))
        if current < amount:
            return False
        _balances[pubkey] = current - amount
        logger.info(
            "Prepaid deduct: -$%s from %s (remaining: $%s)",
            amount, pubkey[:12] + "…", _balances[pubkey],
        )
        return True


def get_balance(pubkey: str) -> Decimal:
    """Get current prepaid balance for a wallet.

    Args:
        pubkey: Solana public key (base58).

    Returns:
        Current balance in USD (0 if unknown).
    """
    return _balances.get(pubkey, Decimal("0"))


def verify_wallet_signature(
    pubkey_str: str,
    signature_b58: str,
    message: bytes,
) -> bool:
    """Verify an Ed25519 signature proves wallet ownership.

    Args:
        pubkey_str: Solana public key (base58).
        signature_b58: Ed25519 signature (base58-encoded).
        message: The signed message bytes.

    Returns:
        True if the signature is valid for the given pubkey and message.
    """
    try:
        pubkey = Pubkey.from_string(pubkey_str)
        sig = Signature.from_string(signature_b58)
        return sig.verify(pubkey, message)
    except Exception:
        logger.debug("Signature verification failed for %s", pubkey_str[:12] + "…")
        return False


def build_signing_message(path: str, timestamp: int) -> bytes:
    """Build the canonical message that clients must sign.

    Format: "x402gate:{path}:{timestamp}" encoded as UTF-8.

    Args:
        path: The API path (e.g. "openrouter/google/gemini-2.0-flash-001").
        timestamp: Unix timestamp (integer seconds).

    Returns:
        Message bytes for Ed25519 signing.
    """
    return f"x402gate:{path}:{timestamp}".encode()


def validate_timestamp(timestamp: int) -> bool:
    """Check if a timestamp is within the allowed window.

    Args:
        timestamp: Unix timestamp from the client.

    Returns:
        True if within TIMESTAMP_WINDOW seconds of current time.
    """
    now = int(time.time())
    return abs(now - timestamp) <= TIMESTAMP_WINDOW


def reset() -> None:
    """Clear all balances (for testing)."""
    _balances.clear()
