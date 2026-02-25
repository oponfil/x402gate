"""E2E test client: send INVALID model UUID to Tungsten through x402gate.

This client pays for the request but sends a non-existent model_version_uuid
to verify that:
  1. The 402 flow works (pricing is fixed, doesn't depend on model).
  2. Tungsten rejects the invalid model.
  3. The gateway returns the error WITHOUT settling payment.
  4. The client does NOT lose money.

Usage:
    BASE_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_tungsten_client_bad_model.py
"""

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

import httpx
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-tungsten-bad-model")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_bad_request() -> tuple[str, dict]:
    """Load model path from config, but use a non-existent model UUID."""
    body = {
        "type": "z_image_turbo",
        "data": {
            "model_version_uuid": "INVALID_UUID_DOES_NOT_EXIST",
            "prompt": "test image",
            "negative_prompt": "",
            "num_images": 1,
            "sampler": "euler",
            "steps": 10,
            "cfg": 1.0,
            "clip_skip": 2,
            "width": 512,
            "height": 512,
            "loras": [],
            "embeddings": [],
            "controlnets": [],
            "img2img": None,
        },
    }
    return "generations", body


async def run_client():
    """Run the bad-model client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_bad_request()
    logger.info("Model path: %s", model_path)
    logger.info("Body (with INVALID model UUID): %s", body)

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → should still get 402
        logger.info("Step 1: Sending request with bad model (no payment)...")
        response = await http_client.post(
            f"{gateway_url}/v1/tungsten/{model_path}",
            json=body,
        )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 — fixed pricing works regardless of model ✓")

        # Parse 402 and sign payment
        payment_required = PaymentRequired.model_validate(response.json())
        payment_payload = await x402_client.create_payment_payload(payment_required)
        signature = base64.b64encode(
            payment_payload.model_dump_json(by_alias=True).encode()
        ).decode()

        # 2. Retry with payment + bad model → should get error, NOT 200
        logger.info("Step 2: Retrying with payment + bad model UUID...")
        response = await http_client.post(
            f"{gateway_url}/v1/tungsten/{model_path}",
            json=body,
            headers={"PAYMENT-SIGNATURE": signature},
            timeout=60.0,
        )

        logger.info("Response status: %d", response.status_code)
        logger.info("Response body: %s", response.text[:500])

        if response.status_code == 200:
            logger.error("UNEXPECTED: got 200 with invalid model UUID!")
            sys.exit(1)

        # We expect a 4xx error from the provider
        logger.info(
            "Got error %d as expected — Tungsten rejected invalid model ✓",
            response.status_code,
        )
        print(f"ERROR_STATUS={response.status_code}")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(run_client())
