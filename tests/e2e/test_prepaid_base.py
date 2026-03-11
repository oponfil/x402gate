"""E2E test: Prepaid mode (Base/EVM) — top-up → 2 OpenRouter + WaveSpeed + Tungsten."""

import os
import time

import httpx
import pytest
from eth_account import Account

from tests.e2e.conftest import run_script


@pytest.mark.asyncio
async def test_prepaid_base(gateway_process):
    """Full prepaid flow with Base (EVM) wallet: top-up → provider calls → verify balance."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key or or_key == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    account = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"])
    evm_address = account.address

    # Record balance BEFORE top-up
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"{gateway_url}/v1/balance/{evm_address}")
        balance_before = float(resp.json().get("balance", 0))
    print(f"Balance before: ${balance_before:.6f}")

    # Run the Base prepaid client script
    result = run_script("tests/e2e/x402_prepaid_base_client.py", label="Prepaid Base")

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Prepaid Base client failed:\n{result.stderr}"

    # Wait for settlement of the top-up tx
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # Check balance AFTER
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"{gateway_url}/v1/balance/{evm_address}")
        balance_data = resp.json()

    final_balance = float(balance_data["balance"])
    delta = final_balance - balance_before
    print(f"\nBalance before: ${balance_before:.6f}")
    print(f"Balance after:  ${final_balance:.6f}")
    print(f"Delta:          ${delta:+.6f}")

    # Top-up adds ~$0.1046, 4 provider calls spend ~$0.02-0.03
    # Net delta should be positive (top-up > spending) and less than top-up amount
    assert final_balance >= 0, f"Balance should be non-negative, got {final_balance}"
    assert delta > 0, f"Balance should have increased (top-up > spending), got delta={delta}"
    assert delta < 0.11, f"Delta should be less than top-up amount, got {delta}"
    print("E2E prepaid Base test passed!")
