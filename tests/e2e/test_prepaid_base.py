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

    # Run the Base prepaid client script
    result = run_script("tests/e2e/x402_prepaid_base_client.py", label="Prepaid Base")

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Prepaid Base client failed:\n{result.stderr}"

    # Wait for settlement of the top-up tx
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # Check that the balance is positive but less than the credited amount
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"http://localhost:4022/v1/balance/{evm_address}")
        balance_data = resp.json()

    final_balance = float(balance_data["balance"])
    print(f"\nFinal prepaid balance (Base): ${final_balance:.6f}")

    # After top-up ($0.10 minus fees) and 4 provider calls,
    # balance should be non-negative and less than top-up amount
    assert final_balance >= 0, f"Balance should be non-negative, got {final_balance}"
    assert final_balance < 0.10, f"Balance should be less than top-up amount, got {final_balance}"
    print("E2E prepaid Base test passed!")
