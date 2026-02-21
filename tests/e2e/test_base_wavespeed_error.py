"""E2E test: invalid params -> provider error -> no settlement -> client keeps money."""

import os
import subprocess
import time

import pytest


@pytest.mark.asyncio
async def test_bad_params_no_settlement(gateway_process, base_chain):
    """Send unsupported resolution to WaveSpeed, verify error + no money lost."""
    client_before = base_chain.get_usdc(base_chain.client_address)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    # Run the bad-params client script
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022"}
    result = subprocess.run(  # noqa: ASYNC221
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
    time.sleep(10)  # noqa: ASYNC251

    # Verify client balance unchanged
    client_after = base_chain.get_usdc(base_chain.client_address)
    client_diff = client_before - client_after

    print(f"\n=== [Base] Client USDC AFTER:  {client_after / 1e6:.6f} ===")
    print(f"=== [Base] Client diff:        {client_diff / 1e6:.6f} USDC ===")

    assert client_diff == 0, (
        f"Client lost {client_diff / 1e6:.6f} USDC but should have lost nothing "
        f"(provider rejected the request, settlement should NOT happen)"
    )
    print("OK: Client balance unchanged -- no settlement on provider error")
