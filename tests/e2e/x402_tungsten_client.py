"""E2E test client for x402gate → Tungsten image generation.

This client simulates a user:
1. Sends a request to the gateway (Tungsten provider).
2. Receives a 402 Payment Required response.
3. Signs the payment using x402 EVM scheme.
4. Resends with payment → gets generated image as base64.

Usage:
    BASE_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_tungsten_client.py
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
from helpers import Timings, save_images
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-tungsten-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model path and request body from config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    example = cfg["providers"]["tungsten"]["example_request_2"]
    return example["model"], example["body"]


async def run_client():
    """Run the E2E client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_example_request()
    logger.info("Model path: %s", model_path)
    logger.info("Body: %s", body)

    # Initialize x402 client with EVM signer
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)

    timings = Timings()

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → should get 402
        logger.info("Sending initial request (no payment)...")
        with timings.measure("pricing"):
            response = await http_client.post(
                f"{gateway_url}/v1/tungsten/{model_path}",
                json=body,
            )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required ✓")

        # Parse 402 and sign payment
        payment_required = PaymentRequired.model_validate(response.json())
        logger.info("Signing payment...")
        with timings.measure("signing"):
            payment_payload = await x402_client.create_payment_payload(payment_required)
            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()

        # 2. Retry with payment → should get image
        logger.info("Retrying with payment signature...")
        with timings.measure("paid_request"):
            response = await http_client.post(
                f"{gateway_url}/v1/tungsten/{model_path}",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=600.0,  # Tungsten can take a while
            )
        timings.add_server_timings(response)

        if response.status_code == 200:
            result = response.json()
            data = result.get("data", result)
            images = data.get("images", [])
            count = data.get("count", 0)

            logger.info("Success! Generated %d image(s)", count)

            # Save images to disk
            with timings.measure("download"):
                if images:
                    save_images(images, "base_tungsten")
        else:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

    timings.output()


if __name__ == "__main__":
    asyncio.run(run_client())
