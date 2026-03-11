"""E2E test: Solana (SVM) payment → WaveSpeed generation."""

import os
import time

import pytest
from solana.rpc.api import Client as SolanaClient
from solders.keypair import Keypair
from solders.keypair import Keypair as _Kp
from solders.pubkey import Pubkey
from solders.pubkey import Pubkey as Pk

from tests.e2e.conftest import run_script
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
    _t_start = time.monotonic()
    result = run_script("tests/e2e/x402_solana_test_client.py", label="Solana WaveSpeed")
    _client_wait_s = time.monotonic() - _t_start

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

    # --- Timing ---
    from tests.e2e.conftest import _parse_generation_time, _parse_timings

    combined = result.stdout + "\n" + result.stderr
    timings = _parse_timings(result.stdout)
    generation_s = _parse_generation_time(combined)
    client_wait_s = _client_wait_s

    print("\n=== [Solana] Timing ===")
    if timings:
        print(f"Pricing (402):           {timings['pricing']:.1f}s")
        print(f"Signing:                 {timings['signing']:.1f}s")

        sv = timings.get("server_verify")
        sg = timings.get("server_generation")
        if sv is not None and sg is not None:
            network_overhead = timings["paid_request"] - sv - sg
            print(f"Payment verify:          {sv:.1f}s")
            print(f"Generation time:         {sg:.1f}s")
            print(f"Network overhead:        {network_overhead:.1f}s")
        elif generation_s is not None:
            overhead = timings["paid_request"] - generation_s
            print(f"Generation time:         {generation_s:.1f}s")
            print(f"Payment time (overhead): {overhead:.1f}s")
        else:
            print(f"Paid request:            {timings['paid_request']:.1f}s")

        dl = timings.get("download", 0.0)
        if dl > 0:
            print(f"Download:                {dl:.1f}s")

        client_timings = {k: v for k, v in timings.items() if not k.startswith("server_")}
        total_timings = sum(client_timings.values())
        other = client_wait_s - total_timings
        print(f"Other (subprocess):      {other:.1f}s")
    elif generation_s is not None:
        print(f"Generation time:         {generation_s:.1f}s")
    print(f"Total client time:       {client_wait_s:.1f}s")

    # Verify amounts: client pays provider_base + commission, PayTo receives full amount
    assert client_diff > 0, f"Client should pay some USDC, paid {client_diff}"
    assert client_diff == payto_diff, f"Client paid {client_diff} != PayTo received {payto_diff}"
    assert client_diff >= 5000, (
        f"Client should pay at least $0.005 (base price), paid {client_diff / 1e6:.6f}"
    )
    assert gas_spent > 0, "Facilitator should spend SOL for gas"
