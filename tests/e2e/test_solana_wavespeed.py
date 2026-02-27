"""E2E test: Solana (SVM) payment → WaveSpeed generation."""

import os
import subprocess
import time

import pytest
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.keypair import Keypair as _Kp
from solders.pubkey import Pubkey
from solders.pubkey import Pubkey as Pk

from x402gate.core.config import load_config
from x402gate.core.config import load_config as _lc


@pytest.mark.asyncio
async def test_solana_wavespeed(gateway_process):
    """Full payment flow on Solana: pay USDC → generate image → verify balances."""
    if not os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY"):
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not set")
    if os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY") == "FILL_ME":
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not configured yet")

    cfg = load_config()
    sol_cfg = cfg.payment.networks["solana"]
    client = SolanaClient(sol_cfg.rpc_url)

    # Derive client address from E2E test key (Solana keypair)
    # For Solana E2E tests, we'll need a SOLANA_E2ETEST_PRIVATE_KEY
    if not os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY"):
        pytest.skip("SOLANA_E2ETEST_PRIVATE_KEY not set")

    test_keypair = Keypair.from_base58_string(os.environ["SOLANA_E2ETEST_PRIVATE_KEY"])
    client_addr = str(test_keypair.pubkey())
    payto_addr = sol_cfg.pay_to
    usdc_mint = Pubkey.from_string(sol_cfg.token_address)

    def get_spl_balance(owner_str: str) -> int:
        """Get USDC (SPL token) balance for a Solana address."""

        owner = Pk.from_string(owner_str)
        # Derive Associated Token Account (ATA)
        token_program = Pk.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        ata_program = Pk.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
        seeds = [bytes(owner), bytes(token_program), bytes(usdc_mint)]
        ata, _ = Pk.find_program_address(seeds, ata_program)

        resp = client.get_token_account_balance(ata)
        if resp.value:
            return int(resp.value.amount)
        return 0

    def get_sol_balance(addr_str: str) -> int:
        """Get SOL balance in lamports."""
        pk = Pubkey.from_string(addr_str)
        return client.get_balance(pk).value

    # Also track facilitator SOL for gas (facilitator pays gas, not PayTo)

    _cfg = _lc()
    _sol_cfg = _cfg.payment.networks["solana"]
    # Facilitator address derived from its key

    fac_kp = _Kp.from_base58_string(os.environ["SOLANA_FACILITATOR_PRIVATE_KEY"])
    FAC = str(fac_kp.pubkey())  # noqa: N806

    # --- Record balances BEFORE ---
    client_usdc_before = get_spl_balance(client_addr)
    payto_usdc_before = get_spl_balance(payto_addr)
    fac_sol_before = get_sol_balance(FAC)

    print("\n=== [Solana] Balances BEFORE ===")
    print(f"Client USDC: {client_usdc_before / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_before / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_before / 1e9:.9f}")

    # Run the Solana client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "tests/e2e/x402_solana_test_client.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Solana client script failed: {result.stderr}"

    # Wait for background settlement (Solana confirmation takes ~2-5s)
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # --- Record balances AFTER ---
    client_usdc_after = get_spl_balance(client_addr)
    payto_usdc_after = get_spl_balance(payto_addr)
    fac_sol_after = get_sol_balance(FAC)

    client_diff = client_usdc_before - client_usdc_after
    payto_diff = payto_usdc_after - payto_usdc_before
    gas_spent = fac_sol_before - fac_sol_after

    print("\n=== [Solana] Balances AFTER ===")
    print(f"Client USDC: {client_usdc_after / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_after / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_after / 1e9:.9f}")
    print("\n=== [Solana] Balance Changes ===")
    print(f"Client paid:     {client_diff / 1e6:.6f} USDC")
    print(f"PayTo received:  {payto_diff / 1e6:.6f} USDC")
    print(f"Gas spent:       {gas_spent / 1e9:.9f} SOL")

    # Verify amounts: client pays provider_base + commission, PayTo receives full amount
    assert client_diff > 0, f"Client should pay some USDC, paid {client_diff}"
    assert client_diff == payto_diff, f"Client paid {client_diff} != PayTo received {payto_diff}"
    assert client_diff >= 5000, (
        f"Client should pay at least $0.005 (base price), paid {client_diff / 1e6:.6f}"
    )
    assert gas_spent > 0, "Facilitator should spend SOL for gas"
