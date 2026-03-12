"""E2E test: Base (EVM) payment → CloudConvert file conversion."""

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_cloudconvert(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC → convert HTML to PDF → verify balances."""
    diff = run_e2e_client(
        "tests/e2e/x402_cloudconvert_client.py",
        chain=base_chain,
        provider_name="cloudconvert",
        label="Base -> CloudConvert",
        settle_wait=10,
    )

    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.client_paid == diff.payto_received, (
        f"Client paid {diff.client_paid} != PayTo received {diff.payto_received}"
    )
    # $0.03 base + 4% commission ≈ $0.0312 → 31200 raw units minimum
    assert diff.client_paid >= 30000, (
        f"Client should pay at least $0.03 (base price), paid {diff.client_paid / 1e6:.6f}"
    )
    assert diff.gas_spent > 0, "Gas should have been spent for settlement"
