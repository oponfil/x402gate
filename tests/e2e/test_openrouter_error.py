"""E2E test: verify that provider errors do NOT settle payment.

Sends a paid request with an invalid model. The gateway should return an
error and NOT settle — client USDC balance stays unchanged.
"""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
@pytest.mark.order("first")
async def test_error_no_settlement(gateway_process, base_chain):
    """Provider error should NOT settle payment — client keeps USDC."""
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key or or_key == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_openrouter_error_client.py",
        chain=base_chain,
        provider_name="openrouter",
        label="Base -> OpenRouter (ERROR)",
    )

    assert diff.client_paid == 0, (
        f"Client should NOT pay on error, but paid {diff.client_paid / 1e6:.6f} USDC"
    )
    assert diff.payto_received == 0, (
        f"PayTo should NOT receive on error, but got {diff.payto_received / 1e6:.6f} USDC"
    )
