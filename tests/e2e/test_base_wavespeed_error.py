"""E2E test: invalid params → provider error → no settlement → client keeps money."""

import os
import subprocess
import time

import pytest


@pytest.mark.asyncio
async def test_bad_params_no_settlement(gateway_process):
    """Send unsupported resolution to WaveSpeed, verify error + no money lost."""
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

    usdc = w3.eth.contract(
        address=Web3.to_checksum_address(USDC),
        abi=[{
            "constant": True,
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
        }],
    )

    def get_usdc(addr):
        return usdc.functions.balanceOf(Web3.to_checksum_address(addr)).call()

    # --- Record client balance BEFORE ---
    client_before = get_usdc(CLIENT)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    # Run the bad-params client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(
        ["python", "tests/e2e/x402_test_client_bad_params.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Client script failed: {result.stderr}"

    # Wait for any potential settlement
    print("Waiting to confirm no settlement...")
    time.sleep(10)

    # --- Record client balance AFTER ---
    client_after = get_usdc(CLIENT)
    client_diff = client_before - client_after

    print(f"\n=== [Base] Client USDC AFTER:  {client_after / 1e6:.6f} ===")
    print(f"=== [Base] Client diff:        {client_diff / 1e6:.6f} USDC ===")

    # ✅ KEY ASSERTION: client should NOT lose money on provider error
    assert client_diff == 0, (
        f"Client lost {client_diff / 1e6:.6f} USDC but should have lost nothing "
        f"(provider rejected the request, settlement should NOT happen)"
    )
    print("OK: Client balance unchanged -- no settlement on provider error")
