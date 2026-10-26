"""E2E test: Base (EVM) payment -> SpicyAPI image generation."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_spicyapi(gateway_process, base_chain):
    """Pay USDC on Base, generate an image with Z-Image Spicy, verify balances."""
    api_key = os.environ.get("SPICYAPI_API_KEY")
    if not api_key or api_key.startswith("sk-spicy-your"):
        pytest.skip("SPICYAPI_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_spicyapi_test_client.py",
        chain=base_chain,
        provider_name="spicyapi",
        label="Base -> SpicyAPI",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
    assert diff.client_paid >= 10000, (
        f"Client should pay at least $0.01, paid {diff.client_paid / 1e6:.6f}"
    )
    assert diff.gas_spent > 0, "Gas should have been spent for settlement"
