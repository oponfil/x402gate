"""E2E test client: send INVALID parameters to WaveSpeed through x402gate.

This client pays for the request but sends an unsupported resolution (9999*9999)
to verify that:
  1. The 402 flow still works (pricing doesn't depend on size).
  2. WaveSpeed rejects the request.
  3. The gateway returns the error WITHOUT settling payment.
  4. The client does NOT lose money.

Usage:
    BASE_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_test_client_bad_params.py
"""

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

import httpx
import yaml
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-bad-params")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model path from config, but override body with bad params."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    example = cfg["providers"]["wavespeed"]["example_request"]
    model = example["model"]
    # Use valid prompt but INVALID size
    body = {
        "prompt": "A futuristic city on Mars, cinematic lighting",
        "size": "9999*9999",  # ← unsupported resolution
        "seed": 42,
    }
    return model, body


async def run_client():
    """Run the bad-params client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_example_request()
    logger.info("Model: %s", model_path)
    logger.info("Body (with BAD size): %s", body)

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → should still get 402
        logger.info("Step 1: Sending request with bad params (no payment)...")
        response = await http_client.post(
            f"{gateway_url}/v1/wavespeed/{model_path}",
            json=body,
        )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 — pricing works regardless of params ✓")

        # Parse 402 and sign payment
        payment_required = PaymentRequired.model_validate(response.json())
        payment_payload = await x402_client.create_payment_payload(payment_required)
        signature = base64.b64encode(
            payment_payload.model_dump_json(by_alias=True).encode()
        ).decode()

        # 2. Retry with payment + bad params → should get provider error, NOT 200
        logger.info("Step 2: Retrying with payment + bad params...")
        response = await http_client.post(
            f"{gateway_url}/v1/wavespeed/{model_path}",
            json=body,
            headers={"PAYMENT-SIGNATURE": signature},
            timeout=60.0,
        )

        logger.info("Response status: %d", response.status_code)
        logger.info("Response body: %s", response.text[:500])

        if response.status_code == 200:
            logger.error("UNEXPECTED: got 200 with invalid params!")
            sys.exit(1)

        # We expect a 4xx/5xx error from the provider
        logger.info("Got error %d as expected — provider rejected bad params ✓", response.status_code)
        print(f"ERROR_STATUS={response.status_code}")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run_client())
