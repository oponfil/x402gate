"""E2E test: Solana (SVM) payment → OpenRouter chat completion."""
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from x402gate.core.config import load_config
from solders.pubkey import Pubkey as Pk

import os
import subprocess
import time

import pytest


@pytest.mark.asyncio
async def test_solana_openrouter(gateway_process):
    """Full payment flow on Solana: pay USDC → OpenRouter LLM chat → verify balances."""
    if not os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY"):
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not set")
    if os.environ.get("SOLANA_FACILITATOR_PRIVATE_KEY") == "FILL_ME":
        pytest.skip("SOLANA_FACILITATOR_PRIVATE_KEY not configured yet")
    if not os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY"):
        pytest.skip("SOLANA_E2ETEST_PRIVATE_KEY not set")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if not or_key or or_key == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")



    cfg = load_config()
    sol_cfg = cfg.payment.networks["solana"]
    client = SolanaClient(sol_cfg.rpc_url)

    test_keypair = Keypair.from_base58_string(os.environ["SOLANA_E2ETEST_PRIVATE_KEY"])
    client_addr = str(test_keypair.pubkey())
    payto_addr = sol_cfg.pay_to
    usdc_mint = Pubkey.from_string(sol_cfg.token_address)

    def get_spl_balance(owner_str: str) -> int:
        """Get USDC (SPL token) balance for a Solana address."""

        owner = Pk.from_string(owner_str)
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

    # Facilitator address (pays gas)
    fac_kp = Keypair.from_base58_string(os.environ["SOLANA_FACILITATOR_PRIVATE_KEY"])
    FAC = str(fac_kp.pubkey())  # noqa: N806

    # --- Record balances BEFORE ---
    client_usdc_before = get_spl_balance(client_addr)
    payto_usdc_before = get_spl_balance(payto_addr)
    fac_sol_before = get_sol_balance(FAC)

    print("\n=== [Solana -> OpenRouter] Balances BEFORE ===")
    print(f"Client USDC: {client_usdc_before / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_before / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_before / 1e9:.9f}")

    # Run the client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "tests/e2e/x402_solana_openrouter_client.py"],
        env=env,
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Solana OpenRouter client failed:\n{result.stderr}"

    # Wait for settlement
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(15)  # noqa: ASYNC251

    # --- Record balances AFTER ---
    client_usdc_after = get_spl_balance(client_addr)
    payto_usdc_after = get_spl_balance(payto_addr)
    fac_sol_after = get_sol_balance(FAC)

    client_diff = client_usdc_before - client_usdc_after
    payto_diff = payto_usdc_after - payto_usdc_before
    gas_spent = fac_sol_before - fac_sol_after

    print("\n=== [Solana -> OpenRouter] Balances AFTER ===")
    print(f"Client USDC: {client_usdc_after / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_after / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_after / 1e9:.9f}")
    print("\n=== [Solana -> OpenRouter] Balance Changes ===")
    print(f"Client paid:     {client_diff / 1e6:.6f} USDC")
    print(f"PayTo received:  {payto_diff / 1e6:.6f} USDC")
    print(f"Gas spent:       {gas_spent / 1e9:.9f} SOL")

    # Verify
    assert client_diff > 0, f"Client should pay USDC, paid {client_diff}"
    assert client_diff == payto_diff, f"Client paid {client_diff} != PayTo received {payto_diff}"
    assert gas_spent > 0, "Facilitator should spend SOL for gas"
