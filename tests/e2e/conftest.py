"""Shared fixtures and helpers for E2E tests."""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import pytest
from eth_account import Account
from web3 import Web3

from x402gate.core.config import load_config

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

PROD_GATEWAY_URL = "https://x402gate.io"


def pytest_addoption(parser):
    """Add --prod flag to run E2E tests against production."""
    parser.addoption(
        "--prod",
        action="store_true",
        default=False,
        help=f"Run tests against production gateway ({PROD_GATEWAY_URL})",
    )


@pytest.fixture(scope="session")
def gateway_process(request, load_env):
    """Start the gateway server in a subprocess.

    If ``--prod`` is passed or ``GATEWAY_URL`` is set in the environment,
    the local server is NOT started and the tests run against the
    external gateway.

    Uses a temp file for output instead of a pipe to avoid deadlocks
    on Windows when the pipe buffer fills up (only ~4KB on Windows).
    """
    if request.config.getoption("--prod"):
        os.environ["GATEWAY_URL"] = PROD_GATEWAY_URL

    external_url = os.environ.get("GATEWAY_URL")
    if external_url:
        print(f"\n=== Using external gateway: {external_url} ===")
        yield None
        return

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


def run_script(
    script: str,
    *,
    label: str = "",
    heartbeat: int = 30,
) -> subprocess.CompletedProcess:
    """Run a client script with heartbeat output every N seconds.

    Prevents silent waits by printing elapsed time while the subprocess runs.
    """
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    env = {**os.environ, "GATEWAY_URL": gateway_url, "PYTHONIOENCODING": "utf-8"}
    tag = f"[{label}] " if label else ""

    # Use temp files to capture output (avoids pipe buffer deadlocks on Windows)
    stdout_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        suffix="_stdout.log",
        delete=False,
        encoding="utf-8",
    )
    stderr_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w",
        suffix="_stderr.log",
        delete=False,
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        ["python", script],
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )

    elapsed = 0
    try:
        while proc.poll() is None:
            time.sleep(1)
            elapsed += 1
            if elapsed % heartbeat == 0:
                print(f"{tag}⏳ Waiting... {elapsed}s elapsed", flush=True)
    finally:
        stdout_file.close()
        stderr_file.close()

    with open(stdout_file.name, encoding="utf-8", errors="replace") as f:
        stdout = f.read()
    with open(stderr_file.name, encoding="utf-8", errors="replace") as f:
        stderr = f.read()
    os.unlink(stdout_file.name)
    os.unlink(stderr_file.name)

    if elapsed >= heartbeat:
        print(f"{tag}✅ Script finished after {elapsed}s (exit code {proc.returncode})", flush=True)

    return subprocess.CompletedProcess(
        args=["python", script],
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


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

    # --- Run client (measure wall-clock time) ---
    t_start = time.monotonic()
    result = run_script(script, label=label)
    client_wait_s = time.monotonic() - t_start

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

    print(f"\n=== [{label}] Balances Changes ===")
    print(f"Client USDC: ${after.client_usdc / 1e6:.6f}")
    print(f"PayTo  USDC: ${after.payto_usdc / 1e6:.6f}")
    print(f"Client paid:     ${diff.client_paid / 1e6:.6f} USDC")
    print(f"PayTo received:  ${diff.payto_received / 1e6:.6f} USDC")
    if diff.gas_spent:
        gas_eth = diff.gas_spent / 1e18
        print(f"Gas spent:       {gas_eth:.10f} ETH")

    # --- Profit ---
    gas_usd = diff.gas_spent / 1e18 * _get_eth_price()
    revenue_usd = diff.client_paid / 1e6
    net_profit = revenue_usd - gas_usd
    print(f"Net profit:      ${net_profit:.6f} USDC")

    # --- Timing ---
    combined_output = result.stdout + "\n" + result.stderr
    timings = _parse_timings(result.stdout)
    generation_s = _parse_generation_time(combined_output)

    from tests.e2e.helpers import print_timing_summary

    print_timing_summary(label, timings, generation_s, client_wait_s)

    return diff


def _parse_timings(stdout: str) -> dict[str, float] | None:
    """Parse structured TIMINGS: line from client script output."""
    import re

    m = re.search(r"TIMINGS:([\w=.,]+)", stdout)
    if not m:
        return None
    try:
        return {k: float(v) for k, v in (pair.split("=") for pair in m.group(1).split(","))}
    except (ValueError, IndexError):
        return None


def _parse_generation_time(stdout: str) -> float | None:
    """Try to extract generation/inference time from client script output."""
    import re

    # WaveSpeed: 'executionTime': 10495  (milliseconds)
    m = re.search(r"'executionTime':\s*(\d+)", stdout)
    if m:
        return int(m.group(1)) / 1000.0

    # Tungsten: "completed after 15s"
    m = re.search(r"completed after (\d+)s", stdout)
    if m:
        return float(m.group(1))

    # OpenRouter: "Tokens: prompt=X, completion=Y" — use total client time
    # CloudConvert: "finished after Xs"
    m = re.search(r"finished after (\d+)s", stdout)
    if m:
        return float(m.group(1))

    return None


def _get_eth_price() -> float:
    """Get current ETH price in USD (cached for the session)."""
    if not hasattr(_get_eth_price, "_cached"):
        try:
            import httpx

            resp = httpx.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
                timeout=5,
            )
            _get_eth_price._cached = resp.json()["ethereum"]["usd"]
        except Exception:
            _get_eth_price._cached = 2000.0  # fallback
    return _get_eth_price._cached
