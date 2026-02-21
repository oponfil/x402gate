"""Shared fixtures and helpers for E2E tests."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load .env file manually."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip("'\"")
                    os.environ.setdefault(key.strip(), value)


# ---------------------------------------------------------------------------
# Gateway server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gateway_process(load_env):
    """Start the gateway server in a subprocess.

    Uses a temp file for output instead of a pipe to avoid deadlocks
    on Windows when the pipe buffer fills up (only ~4KB on Windows).
    """
    log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        suffix="_x402gate.log",
        delete=False,
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        ["python", "-u", "-m", "x402gate.main"],
        env={**os.environ, "PORT": "4022", "PYTHONUNBUFFERED": "1"},
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)  # Wait for startup
    yield proc
    time.sleep(12)  # Wait for background settlement to complete
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
    # Read and print server logs
    try:
        with open(log_file.name, encoding="utf-8", errors="replace") as f:
            print("\n=== Server Logs ===")
            print(f.read())
    finally:
        os.unlink(log_file.name)


# ---------------------------------------------------------------------------
# Base (EVM) chain helpers
# ---------------------------------------------------------------------------


class BaseChain:
    """Helper for interacting with Base (EVM) on-chain state."""

    def __init__(self):
        from eth_account import Account
        from web3 import Web3

        from x402gate.core.config import load_config

        cfg = load_config()
        self.base_cfg = cfg.payment.networks["base"]
        self.w3 = Web3(Web3.HTTPProvider(self.base_cfg.rpc_url))
        self.usdc_address = self.base_cfg.token_address
        self.client_address = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"]).address
        self.payto_address = self.base_cfg.pay_to
        self._bal_sig = self.w3.keccak(text="balanceOf(address)")[:4]

    def get_usdc(self, addr: str) -> int:
        """Get USDC balance in raw units (6 decimals)."""
        data = self._bal_sig + bytes(12) + bytes.fromhex(addr[2:])
        r = self.w3.eth.call({"to": self.usdc_address, "data": data})
        return int.from_bytes(r, "big")

    def get_eth(self, addr: str) -> int:
        """Get ETH balance in wei."""
        return self.w3.eth.get_balance(addr)


class BalanceSnapshot(NamedTuple):
    """USDC balances at a point in time."""

    client_usdc: int
    payto_usdc: int
    payto_eth: int


class BalanceDiff(NamedTuple):
    """Balance changes between two snapshots."""

    client_paid: int  # client_before - client_after  (positive = paid)
    payto_received: int  # payto_after - payto_before    (positive = received)
    gas_spent: int  # payto_eth_before - payto_eth_after


@pytest.fixture
def base_chain():
    """Provide a BaseChain helper, skipping if key not set."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")
    return BaseChain()


def run_e2e_client(
    script: str,
    *,
    chain: BaseChain,
    label: str = "Base",
    settle_wait: int = 10,
) -> BalanceDiff:
    """Run an E2E client script and return balance changes.

    1. Records USDC/ETH balances before
    2. Runs the client script
    3. Waits for settlement
    4. Records balances after
    5. Returns the diff

    Args:
        script: Path to the client script (relative to project root).
        chain: BaseChain instance.
        label: Label for log output (e.g., "Base -> OpenRouter").
        settle_wait: Seconds to wait for on-chain settlement.

    Returns:
        BalanceDiff with client_paid, payto_received, gas_spent.

    Raises:
        AssertionError: If the client script exits with non-zero.
    """
    # --- Balances BEFORE ---
    before = BalanceSnapshot(
        client_usdc=chain.get_usdc(chain.client_address),
        payto_usdc=chain.get_usdc(chain.payto_address),
        payto_eth=chain.get_eth(chain.payto_address),
    )

    print(f"\n=== [{label}] Balances BEFORE ===")
    print(f"Client USDC: {before.client_usdc / 1e6:.6f}")
    print(f"PayTo  USDC: {before.payto_usdc / 1e6:.6f}")

    # --- Run client ---
    env = {**os.environ, "GATEWAY_URL": "http://localhost:4022", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["python", script],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Client script failed:\n{result.stderr}"

    # --- Wait for settlement ---
    print("Waiting for settlement to confirm on-chain...")
    time.sleep(settle_wait)

    # --- Balances AFTER ---
    after = BalanceSnapshot(
        client_usdc=chain.get_usdc(chain.client_address),
        payto_usdc=chain.get_usdc(chain.payto_address),
        payto_eth=chain.get_eth(chain.payto_address),
    )

    diff = BalanceDiff(
        client_paid=before.client_usdc - after.client_usdc,
        payto_received=after.payto_usdc - before.payto_usdc,
        gas_spent=before.payto_eth - after.payto_eth,
    )

    print(f"\n=== [{label}] Balances AFTER ===")
    print(f"Client USDC: {after.client_usdc / 1e6:.6f}")
    print(f"PayTo  USDC: {after.payto_usdc / 1e6:.6f}")
    print(f"\n=== [{label}] Balance Changes ===")
    print(f"Client paid:     {diff.client_paid / 1e6:.6f} USDC")
    print(f"PayTo received:  {diff.payto_received / 1e6:.6f} USDC")
    if diff.gas_spent:
        print(f"Gas spent:       {diff.gas_spent / 1e18:.10f} ETH")

    return diff
