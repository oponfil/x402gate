"""E2E test: Prepaid top-up amount limits (min $0.10, max $10)."""

import os

import httpx
import pytest


@pytest.mark.order("first")
@pytest.mark.asyncio
async def test_topup_below_minimum(gateway_process):
    """Top-up with $0.05 (below $0.10 minimum) should be rejected."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{gateway_url}/v1/topup",
            json={"amount": 0.05},
            timeout=10.0,
        )

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "below minimum" in data["error"], f"Unexpected error: {data}"
    print(f"Correctly rejected: {data['error']}")


@pytest.mark.order("first")
@pytest.mark.asyncio
async def test_topup_above_maximum(gateway_process):
    """Top-up with $15 (above $10 maximum) should be rejected."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{gateway_url}/v1/topup",
            json={"amount": 15.0},
            timeout=10.0,
        )

    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "exceeds maximum" in data["error"], f"Unexpected error: {data}"
    print(f"Correctly rejected: {data['error']}")
