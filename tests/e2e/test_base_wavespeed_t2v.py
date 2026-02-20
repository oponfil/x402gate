"""E2E test: Base (EVM) payment -> WaveSpeed text-to-video generation."""

import os

import pytest

from tests.e2e.conftest import run_e2e_client


@pytest.mark.asyncio
async def test_base_wavespeed_t2v(gateway_process, base_chain):
    """Full payment flow on Base: pay USDC -> generate 480p video -> verify balances."""
    if not os.environ.get("WAVESPEED_API_KEY"):
        pytest.skip("WAVESPEED_API_KEY not set")

    diff = run_e2e_client(
        "tests/e2e/x402_video_test_client.py",
        chain=base_chain,
        label="Base -> WaveSpeed T2V",
    )

    # Wan 2.2 t2v-480p-ultra-fast costs $0.05, with 5% commission + gas surcharge -> ~$0.0535
    assert diff.client_paid > 0, f"Client should pay USDC, paid {diff.client_paid}"
    assert diff.payto_received > 0, f"PayTo should receive USDC, got {diff.payto_received}"
    assert diff.client_paid >= 50_000, (
        f"Client should pay at least $0.05 (base price), paid {diff.client_paid / 1e6:.6f}"
    )
