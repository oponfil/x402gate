"""E2E test: Solana (SVM) payment → OpenRouter chat completion."""

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
    if not os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") == "placeholder":
        pytest.skip("OPENROUTER_API_KEY not set")

    from solana.rpc.api import Client as SolanaClient
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    from x402gate.core.config import load_config

    cfg = load_config()
    sol_cfg = cfg.payment.networks["solana"]
    client = SolanaClient(sol_cfg.rpc_url)

    test_keypair = Keypair.from_base58_string(os.environ["SOLANA_E2ETEST_PRIVATE_KEY"])
    CLIENT = str(test_keypair.pubkey())
    PAYTO = sol_cfg.pay_to
    USDC_MINT = Pubkey.from_string(sol_cfg.token_address)

    def get_spl_balance(owner_str: str) -> int:
        """Get USDC (SPL token) balance for a Solana address."""
        from solders.pubkey import Pubkey as Pk

        owner = Pk.from_string(owner_str)
        TOKEN_PROGRAM = Pk.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        ATA_PROGRAM = Pk.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
        seeds = [bytes(owner), bytes(TOKEN_PROGRAM), bytes(USDC_MINT)]
        ata, _ = Pk.find_program_address(seeds, ATA_PROGRAM)

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
    FAC = str(fac_kp.pubkey())

    # --- Record balances BEFORE ---
    client_usdc_before = get_spl_balance(CLIENT)
    payto_usdc_before = get_spl_balance(PAYTO)
    fac_sol_before = get_sol_balance(FAC)

    print("\n=== [Solana -> OpenRouter] Balances BEFORE ===")
    print(f"Client USDC: {client_usdc_before / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_before / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_before / 1e9:.9f}")

    # Run the client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(
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
    time.sleep(15)

    # --- Record balances AFTER ---
    client_usdc_after = get_spl_balance(CLIENT)
    payto_usdc_after = get_spl_balance(PAYTO)
    fac_sol_after = get_sol_balance(FAC)

    client_diff = client_usdc_before - client_usdc_after
    payto_diff = payto_usdc_after - payto_usdc_before
    gas_spent = fac_sol_before - fac_sol_after

    print(f"\n=== [Solana -> OpenRouter] Balances AFTER ===")
    print(f"Client USDC: {client_usdc_after / 1e6:.6f}")
    print(f"PayTo  USDC: {payto_usdc_after / 1e6:.6f}")
    print(f"Facilitator SOL: {fac_sol_after / 1e9:.9f}")
    print(f"\n=== [Solana -> OpenRouter] Balance Changes ===")
    print(f"Client paid:     {client_diff / 1e6:.6f} USDC")
    print(f"PayTo received:  {payto_diff / 1e6:.6f} USDC")
    print(f"Gas spent:       {gas_spent / 1e9:.9f} SOL")

    # Verify
    assert client_diff > 0, f"Client should pay USDC, paid {client_diff}"
    assert client_diff == payto_diff, (
        f"Client paid {client_diff} != PayTo received {payto_diff}"
    )
    assert gas_spent > 0, "Facilitator should spend SOL for gas"
