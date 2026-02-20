"""E2E test: BlockRun passthrough proxy.

Tests the full passthrough flow through a real gateway instance.
Step 1 (always runs): Verify 402 passthrough from BlockRun.
Step 2 (requires funded wallet): Full payment + LLM response.
"""

import os
import subprocess

import httpx
import pytest


@pytest.mark.asyncio
async def test_blockrun_402_passthrough(gateway_process):
    """Request without payment returns BlockRun's 402 (not ours)."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            },
            timeout=15.0,
        )

    print(f"\nStatus: {response.status_code}")
    print(f"Body: {response.text[:500]}")

    assert response.status_code == 402, f"Expected 402 from BlockRun, got {response.status_code}"

    body = response.json()
    # BlockRun 402 format: {price: {amount, currency}, paymentInfo: {network, asset}}
    # (different from x402gate's {accepts: [{scheme, network, price, payTo}]})
    assert "price" in body or "accepts" in body, (
        f"Expected BlockRun payment info in 402, got: {body}"
    )

    # Verify this is BlockRun's 402 (not x402gate's)
    # x402gate 402 uses "accepts" array; BlockRun uses "price" + "paymentInfo"
    if "price" in body:
        print(f"Price: {body['price']}")
        print(f"PaymentInfo: {body.get('paymentInfo', {})}")
        # This confirms it's BlockRun's 402, not ours
    elif "accepts" in body:
        accepts = body["accepts"]
        pay_to = accepts[0].get("payTo", "")
        our_pay_to = os.environ.get("BASE_PAY_TO_ADDRESS", "")
        if our_pay_to:
            assert pay_to != our_pay_to, (
                "402 payTo matches OUR address — not BlockRun's passthrough!"
            )


@pytest.mark.asyncio
async def test_blockrun_full_flow(gateway_process):
    """Full flow: pay BlockRun via passthrough → get LLM response."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")

    from eth_account import Account
    from web3 import Web3

    # --- Setup: derive addresses ---
    private_key = os.environ["BASE_E2ETEST_PRIVATE_KEY"]
    client_account = Account.from_key(private_key)
    CLIENT = client_account.address
    USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"   # USDC on Base

    w3 = Web3(Web3.HTTPProvider("https://mainnet.base.org"))
    usdc = w3.eth.contract(
        address=Web3.to_checksum_address(USDC_ADDRESS),
        abi=[{
            "constant": True,
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
        }],
    )

    def get_usdc_balance(address: str) -> int:
        return usdc.functions.balanceOf(Web3.to_checksum_address(address)).call()

    # --- Record balance BEFORE ---
    client_before = get_usdc_balance(CLIENT)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    # --- Run the BlockRun client script ---
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(
        ["python", "tests/e2e/x402_blockrun_test_client.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"BlockRun client script failed: {result.stderr}"

    # Wait for on-chain settlement (Base ~2s block time)
    import time
    print("Waiting for on-chain settlement...")
    time.sleep(10)

    # --- Record balance AFTER ---
    client_after = get_usdc_balance(CLIENT)
    client_diff = client_before - client_after

    print(f"\n=== [Base] Client USDC AFTER:  {client_after / 1e6:.6f} ===")
    print(f"=== [Base] Client paid:        {client_diff / 1e6:.6f} USDC ===")

    # Verify: client should have paid $0.001
    assert client_diff > 0, f"Client should pay USDC, diff={client_diff}"
    assert client_diff >= 1000, (
        f"Client should pay at least $0.001 (1000 units), paid {client_diff}"
    )

