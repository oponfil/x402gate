"""E2E test: Base (EVM) payment -> Fish Audio TTS."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_fishaudio(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> Fish Audio TTS -> verify balances."""
    api_key = os.environ.get("FISHAUDIO_API_KEY")
    if not api_key or api_key == "placeholder":
        pytest.skip("FISHAUDIO_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_fishaudio_test_client.py",
        chain=base_chain,
        provider_name="fishaudio",
        label="Base -> FishAudio",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
