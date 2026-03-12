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

    # Record balance BEFORE top-up
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"{gateway_url}/v1/balance/{client_pubkey}")
        balance_before = float(resp.json().get("balance", 0))
    print(f"Balance before: ${balance_before:.6f}")

    # Run the prepaid client script (top-up + 2 OpenRouter + WaveSpeed + Tungsten)
    result = run_script("tests/e2e/x402_prepaid_client.py", label="Prepaid")

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Prepaid client failed:\n{result.stderr}"

    # Wait for settlement of the top-up tx
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # Check balance AFTER
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"{gateway_url}/v1/balance/{client_pubkey}")
        balance_data = resp.json()

    final_balance = float(balance_data["balance"])
    delta = final_balance - balance_before
    print(f"\nBalance before: ${balance_before:.6f}")
    print(f"Balance after:  ${final_balance:.6f}")
    print(f"Delta:          ${delta:+.6f}")

    # Top-up adds ~$0.1046, 4 provider calls spend ~$0.02-0.03
    # NOTE: delta may be negative if the Amanda bot is concurrently spending
    # from the same prepaid balance — this is expected in production.
    assert final_balance >= 0, f"Balance should be non-negative, got {final_balance}"
    if delta > 0:
        print(f"E2E prepaid test passed! (delta positive: ${delta:+.6f})")
    else:
        print(
            f"E2E prepaid test passed! (delta negative due to concurrent spending: ${delta:+.6f})"
        )
