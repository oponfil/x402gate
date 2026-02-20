"""E2E test: Base (EVM) payment -> OpenRouter chat completion."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_openrouter(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> OpenRouter LLM chat -> verify balances."""
    if not os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_openrouter_test_client.py",
        chain=base_chain,
        label="Base -> OpenRouter",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
