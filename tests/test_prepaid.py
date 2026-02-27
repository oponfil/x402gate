"""Unit tests for prepaid balance management."""
import time
from decimal import Decimal

import pytest
from solders.keypair import Keypair

from x402gate.core.prepaid import (
    build_signing_message,
    deduct,
    deposit,
    get_balance,
    reset,
    validate_timestamp,
    verify_wallet_signature,
)


@pytest.fixture(autouse=True)
def _clean_balances():
    """Reset balances before each test."""
    reset()
    yield
    reset()


# --- deposit / deduct / get_balance ---


class TestBalanceOperations:
    """Tests for deposit, deduct, get_balance."""

    async def test_deposit_creates_balance(self):
        new_bal = await deposit("ABC123", Decimal("1.50"))
        assert new_bal == Decimal("1.50")
        assert get_balance("ABC123") == Decimal("1.50")

    async def test_deposit_adds_to_existing(self):
        await deposit("ABC123", Decimal("1.00"))
        new_bal = await deposit("ABC123", Decimal("0.50"))
        assert new_bal == Decimal("1.50")

    async def test_deduct_reduces_balance(self):
        await deposit("ABC123", Decimal("2.00"))
        ok = await deduct("ABC123", Decimal("0.50"))
        assert ok is True
        assert get_balance("ABC123") == Decimal("1.50")

    async def test_deduct_insufficient_funds(self):
        await deposit("ABC123", Decimal("0.10"))
        ok = await deduct("ABC123", Decimal("0.50"))
        assert ok is False
        # Balance should be unchanged
        assert get_balance("ABC123") == Decimal("0.10")

    async def test_deduct_exact_balance(self):
        await deposit("ABC123", Decimal("1.00"))
        ok = await deduct("ABC123", Decimal("1.00"))
        assert ok is True
        assert get_balance("ABC123") == Decimal("0")

    async def test_get_balance_unknown_pubkey(self):
        assert get_balance("UNKNOWN") == Decimal("0")


# --- Ed25519 signature verification ---


class TestSignatureVerification:
    """Tests for verify_wallet_signature using real Ed25519 keys."""

    def test_verify_valid_signature(self):
        kp = Keypair()
        pubkey_str = str(kp.pubkey())
        message = b"x402gate:openrouter/test:1234567890"
        sig = kp.sign_message(message)
        sig_str = str(sig)

        assert verify_wallet_signature(pubkey_str, sig_str, message) is True

    def test_verify_invalid_signature(self):
        kp = Keypair()
        pubkey_str = str(kp.pubkey())
        message = b"x402gate:openrouter/test:1234567890"
        wrong_message = b"x402gate:openrouter/test:9999999999"
        sig = kp.sign_message(wrong_message)
        sig_str = str(sig)

        assert verify_wallet_signature(pubkey_str, sig_str, message) is False

    def test_verify_wrong_pubkey(self):
        kp1 = Keypair()
        kp2 = Keypair()
        message = b"x402gate:openrouter/test:1234567890"
        sig = kp1.sign_message(message)
        sig_str = str(sig)

        # Verify with wrong pubkey
        assert verify_wallet_signature(str(kp2.pubkey()), sig_str, message) is False

    def test_verify_garbage_signature(self):
        kp = Keypair()
        assert verify_wallet_signature(str(kp.pubkey()), "garbage", b"test") is False

    def test_verify_garbage_pubkey(self):
        assert verify_wallet_signature("not_a_pubkey", "not_a_sig", b"test") is False


# --- Signing message & timestamp ---


class TestSigningMessage:
    """Tests for build_signing_message and validate_timestamp."""

    def test_build_signing_message(self):
        msg = build_signing_message("openrouter/google/gemini-2.0-flash-001", 1234567890)
        assert msg == b"x402gate:openrouter/google/gemini-2.0-flash-001:1234567890"

    def test_validate_timestamp_current(self):

        now = int(time.time())
        assert validate_timestamp(now) is True

    def test_validate_timestamp_expired(self):

        old = int(time.time()) - 120  # 2 minutes ago
        assert validate_timestamp(old) is False

    def test_validate_timestamp_future(self):

        future = int(time.time()) + 120  # 2 minutes from now
        assert validate_timestamp(future) is False

    def test_roundtrip_sign_verify(self):
        """Full roundtrip: build message → sign → verify."""

        kp = Keypair()
        ts = int(time.time())
        msg = build_signing_message("openrouter/test-model", ts)
        sig = kp.sign_message(msg)

        assert verify_wallet_signature(str(kp.pubkey()), str(sig), msg) is True
        assert validate_timestamp(ts) is True
