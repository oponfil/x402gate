"""E2E test: Base (EVM) payment -> WaveSpeed generation."""

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_wavespeed(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> generate image -> verify balances."""
    diff = run_e2e_client(
        "tests/e2e/x402_test_client.py",
        chain=base_chain,
        provider_name="wavespeed",
        label="Base -> WaveSpeed",
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
    assert diff.client_paid >= 5000, (
        f"Client should pay at least $0.005 (base price), paid {diff.client_paid / 1e6:.6f}"
    )
    assert diff.gas_spent > 0, "Gas should have been spent for settlement"
