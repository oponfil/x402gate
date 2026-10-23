"""Unit tests for payment handling."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from x402gate.core.config import NetworkConfig
from x402gate.core.payment import PaymentHandler, _settlement_failure_reason


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
