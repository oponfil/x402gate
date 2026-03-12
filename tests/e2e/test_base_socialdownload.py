"""E2E test: Base (EVM) payment -> SocialDownload media download."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_socialdownload(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> SocialDownload -> verify balances."""
    rapi_key = os.environ.get("RAPIDAPI_KEY")
    if not rapi_key or rapi_key == "placeholder":
        pytest.skip("RAPIDAPI_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_socialdownload_test_client.py",
        chain=base_chain,
        provider_name="socialdownload",
        label="Base -> SocialDownload",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
