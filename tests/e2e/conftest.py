"""Shared fixtures for E2E tests."""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


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


@pytest.fixture(scope="session")
def gateway_process(load_env):
    """Start the gateway server in a subprocess.

    Uses a temp file for output instead of a pipe to avoid deadlocks
    on Windows when the pipe buffer fills up (only ~4KB on Windows).
    """
    log_file = tempfile.NamedTemporaryFile(
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
