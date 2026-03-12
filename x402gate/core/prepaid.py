"""Prepaid balance management for x402gate.

In-memory wallet-bound balances: users top-up once via USDC transaction
(Solana or Base), then spend from the balance on subsequent requests
using wallet signatures (Ed25519 for Solana, EIP-191 for EVM).

WARNING: Balances are stored in RAM only and will be lost on server restart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal

from eth_account import Account
from eth_account.messages import encode_defunct
from solders.pubkey import Pubkey
from solders.signature import Signature

logger = logging.getLogger(__name__)

# Global in-memory balances: {solana_pubkey_base58: Decimal(USD)}
_balances: dict[str, Decimal] = {}
_lock = asyncio.Lock()

# Signature timestamp validity window (seconds), set by init()
_timestamp_window: int = 300  # default, overridden from config


async def deposit(pubkey: str, amount: Decimal) -> Decimal:
    """Add funds to a wallet's prepaid balance.

    Args:
        pubkey: Wallet address (Solana base58 or EVM hex).
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
            amount,
            pubkey[:12] + "…",
            new_balance,
        )
        return new_balance


async def deduct(pubkey: str, amount: Decimal) -> bool:
    """Deduct funds from a wallet's prepaid balance.

    Args:
        pubkey: Wallet address (Solana base58 or EVM hex).
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
            amount,
            pubkey[:12] + "…",
            _balances[pubkey],
        )
        return True


def get_balance(pubkey: str) -> Decimal:
    """Get current prepaid balance for a wallet.

    Args:
        pubkey: Wallet address (Solana base58 or EVM hex).

    Returns:
        Current balance in USD (0 if unknown).
    """
    return _balances.get(pubkey, Decimal("0"))


def get_total_balance() -> Decimal:
    """Return the sum of all active prepaid balances."""
    return sum(_balances.values(), Decimal("0"))


def verify_wallet_signature(
    pubkey_str: str,
    signature_str: str,
    message: bytes,
) -> bool:
    """Verify a wallet signature proves ownership.

    Auto-detects wallet type by address format:
    - 0x... (42 chars) → EVM: EIP-191 personal_sign recovery
    - Otherwise → Solana: Ed25519 signature verification

    Args:
        pubkey_str: Wallet address (EVM hex or Solana base58).
        signature_str: Signature string (hex for EVM, base58 for Solana).
        message: The signed message bytes.

    Returns:
        True if the signature is valid for the given address and message.
    """
    if pubkey_str.startswith("0x") and len(pubkey_str) == 42:
        return _verify_evm_signature(pubkey_str, signature_str, message)
    return _verify_solana_signature(pubkey_str, signature_str, message)


def _verify_solana_signature(
    pubkey_str: str,
    signature_b58: str,
    message: bytes,
) -> bool:
    """Verify a Solana Ed25519 signature."""
    try:
        pubkey = Pubkey.from_string(pubkey_str)
        sig = Signature.from_string(signature_b58)
        return sig.verify(pubkey, message)
    except Exception:
        logger.debug("Solana signature verification failed for %s", pubkey_str[:12] + "…")
        return False


def _verify_evm_signature(
    address: str,
    signature_hex: str,
    message: bytes,
) -> bool:
    """Verify an EVM EIP-191 personal_sign signature."""
    try:
        msg = encode_defunct(message)
        recovered = Account.recover_message(msg, signature=signature_hex)
        return recovered.lower() == address.lower()
    except Exception:
        logger.debug("EVM signature verification failed for %s", address[:12] + "…")
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
    return abs(now - timestamp) <= _timestamp_window


def init(*, timestamp_window: int = 300) -> None:
    """Initialize prepaid module settings from config."""
    global _timestamp_window  # noqa: PLW0603
    _timestamp_window = timestamp_window


def reset() -> None:
    """Clear all balances (for testing)."""
    _balances.clear()
