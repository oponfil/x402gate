"""E2E test: Base (EVM) payment -> OpenRouter embedding."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_embedding(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> OpenRouter embedding -> verify balances."""
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key or or_key == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_embedding_test_client.py",
        chain=base_chain,
        provider_name="openrouter",
        label="Base -> OpenRouter Embedding",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
