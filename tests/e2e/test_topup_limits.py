"""E2E test: Prepaid top-up amount limits (min $0.10, max $10)."""

import base64
import json

import httpx
import pytest


def _fake_payment_sig(usdc_amount_raw: int) -> str:
    """Build a fake PAYMENT-SIGNATURE with a specific USDC amount.

    The server extracts the amount from the payload *before* verifying
    the on-chain signature, so we can test amount validation without
    a real wallet or chain interaction.
    """
    payload = {
        "accepted": {
            "amount": str(usdc_amount_raw),
            "network": "eip155:8453",
        },
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


@pytest.mark.asyncio
async def test_topup_below_minimum(gateway_process):
    """Top-up with $0.05 (below $0.10 minimum) should be rejected."""
    sig = _fake_payment_sig(50_000)  # $0.05 in USDC raw units

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:4022/v1/topup",
            headers={"PAYMENT-SIGNATURE": sig},
            timeout=10.0,
        )

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "below minimum" in data["error"], f"Unexpected error: {data}"
    print(f"Correctly rejected: {data['error']}")


@pytest.mark.asyncio
async def test_topup_above_maximum(gateway_process):
    """Top-up with $15 (above $10 maximum) should be rejected."""
    sig = _fake_payment_sig(15_000_000)  # $15 in USDC raw units

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:4022/v1/topup",
            headers={"PAYMENT-SIGNATURE": sig},
            timeout=10.0,
        )

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "exceeds maximum" in data["error"], f"Unexpected error: {data}"
    print(f"Correctly rejected: {data['error']}")
