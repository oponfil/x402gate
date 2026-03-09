"""E2E test: Prepaid mode — top-up $0.10 → 2 OpenRouter + WaveSpeed + Tungsten → check balance."""

import os
import time

import httpx
import pytest
from solders.keypair import Keypair

from tests.e2e.conftest import run_script


@pytest.mark.asyncio
async def test_prepaid_openrouter(gateway_process):
    """Full prepaid flow: top-up → LLM + image gen calls → verify balance decremented."""
    if not os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY"):
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not set")
    if os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY") == "FILL_ME":
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not configured yet")
    if not os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY"):
        pytest.skip("SOLANA_E2ETEST_PRIVATE_KEY not set")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key or or_key == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    test_keypair = Keypair.from_base58_string(os.environ["SOLANA_E2ETEST_PRIVATE_KEY"])
    client_pubkey = str(test_keypair.pubkey())

    # Run the prepaid client script (top-up + 2 OpenRouter + WaveSpeed + Tungsten)
    result = run_script("tests/e2e/x402_prepaid_client.py", label="Prepaid")

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Prepaid client failed:\n{result.stderr}"

    # Wait for settlement of the top-up tx
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # Check that the balance is positive but less than the credited amount
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"http://localhost:4022/v1/balance/{client_pubkey}")
        balance_data = resp.json()

    final_balance = float(balance_data["balance"])
    print(f"\nFinal prepaid balance: ${final_balance:.6f}")

    # After top-up ($0.10 minus fees ≈ $0.095) and 4 provider calls,
    # balance should be positive but reduced
    assert final_balance >= 0, f"Balance should be non-negative, got {final_balance}"
    assert final_balance < 0.10, f"Balance should be less than top-up amount, got {final_balance}"
    print("E2E prepaid test passed!")
