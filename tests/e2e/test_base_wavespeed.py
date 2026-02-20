"""E2E test: Base (EVM) payment → WaveSpeed generation."""

import os
import subprocess
import time

import pytest


@pytest.mark.asyncio
async def test_base_wavespeed(gateway_process):
    """Full payment flow on Base: pay USDC → generate image → verify balances."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")

    from eth_account import Account
    from web3 import Web3

    from x402gate.core.config import load_config

    cfg = load_config()
    base_cfg = cfg.payment.networks["base"]
    w3 = Web3(Web3.HTTPProvider(base_cfg.rpc_url))
    USDC = base_cfg.token_address
    CLIENT = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"]).address
    PAYTO = base_cfg.pay_to
    bal_sig = w3.keccak(text="balanceOf(address)")[:4]

    def get_usdc(addr):
        data = bal_sig + bytes(12) + bytes.fromhex(addr[2:])
        r = w3.eth.call({"to": USDC, "data": data})
        return int.from_bytes(r, "big")

    # --- Record balances BEFORE ---
    client_before = get_usdc(CLIENT)
    payto_before = get_usdc(PAYTO)
    payto_eth_before = w3.eth.get_balance(PAYTO)

    print("\n=== [Base] Balances BEFORE ===")
    print(f"Client USDC: {client_before / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_before / 1e6:.6f}")
    print(f"PayTo  ETH:  {payto_eth_before / 1e18:.10f}")

    # Run the client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(
        ["python", "tests/e2e/x402_test_client.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, "Client script failed"

    # Wait for background settlement
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(10)

    # --- Record balances AFTER ---
    client_after = get_usdc(CLIENT)
    payto_after = get_usdc(PAYTO)
    payto_eth_after = w3.eth.get_balance(PAYTO)

    client_diff = client_before - client_after
    payto_diff = payto_after - payto_before
    gas_spent = payto_eth_before - payto_eth_after

    print("\n=== [Base] Balances AFTER ===")
    print(f"Client USDC: {client_after / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_after / 1e6:.6f}")
    print(f"PayTo  ETH:  {payto_eth_after / 1e18:.10f}")
    print("\n=== [Base] Balance Changes ===")
    print(f"Client paid:     {client_diff / 1e6:.6f} USDC")
    print(f"PayTo received:  {payto_diff / 1e6:.6f} USDC")
    print(f"Gas spent:       {gas_spent / 1e18:.10f} ETH")

    # Verify amounts: client pays provider_base + commission, PayTo receives full amount
    assert client_diff > 0, f"Client should pay some USDC, paid {client_diff}"
    assert client_diff == payto_diff, f"Client paid {client_diff} != PayTo received {payto_diff}"
    assert client_diff >= 5000, (
        f"Client should pay at least $0.005 (base price), paid {client_diff / 1e6:.6f}"
    )
    assert gas_spent > 0, "Gas should have been spent for settlement"
