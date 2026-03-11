"""E2E test: unsupported operation → CloudConvert error → no settlement → client keeps money."""

import time

import pytest

from tests.e2e.conftest import run_script


@pytest.mark.order("first")
@pytest.mark.asyncio
async def test_cloudconvert_bad_operation_no_settlement(gateway_process, base_chain):
    """Send unsupported operation to CloudConvert, verify error + no money lost."""
    client_before = base_chain.get_usdc(base_chain.client_address)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    result = run_script("tests/e2e/x402_cloudconvert_error_client.py", label="CloudConvert Error")

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
