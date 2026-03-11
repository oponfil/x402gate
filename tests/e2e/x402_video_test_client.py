"""E2E test client for x402gate -- WaveSpeed video generation (Base EVM).

Generates a short 480p video using Wan 2.2:
1. Sends request -> gets 402 Payment Required.
2. Signs payment with EVM signer.
3. Resends with payment -> gets video URL.

Usage:
    BASE_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_video_test_client.py
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
from helpers import Timings, save_from_urls
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-video-client")

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model and body from config.yaml's wavespeed example_request_2."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    example = cfg["providers"]["wavespeed"]["example_request_2"]
    return example["model"], example["body"]


async def run_client():
    """Run the video generation E2E client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    model, body = _load_example_request()
    logger.info("Model: %s", model)
    logger.info("Body: %s", body)
    logger.info("Client Address: %s", signer.address)
    logger.info("Gateway URL: %s", gateway_url)

    timings = Timings()

    async with httpx.AsyncClient() as http_client:
        logger.info("Sending initial request...")
        with timings.measure("pricing"):
            response = await http_client.post(
                f"{gateway_url}/v1/wavespeed/{model}",
                json=body,
            )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required")
        payment_data = response.json()

        for accept in payment_data.get("accepts", []):
            logger.info(
                "Payment option: %s -- $%.6f",
                accept.get("network", "?"),
                int(accept.get("amount", 0)) / 1e6,
            )

        evm_accepts = [
            a for a in payment_data.get("accepts", []) if "eip155:" in a.get("network", "")
        ]
        if not evm_accepts:
            logger.error("No EVM payment option in 402 response")
            sys.exit(1)

        payment_data["accepts"] = evm_accepts
        payment_required = PaymentRequired.model_validate(payment_data)

        logger.info("Signing payment...")
        with timings.measure("signing"):
            payment_payload = await x402_client.create_payment_payload(payment_required)
            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()

        logger.info("Retrying with payment signature (video may take 30-120s)...")
        with timings.measure("paid_request"):
            response = await http_client.post(
                f"{gateway_url}/v1/wavespeed/{model}",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=300.0,
            )
        timings.add_server_timings(response)

        if response.status_code == 200:
            result = response.json()
            data = result.get("data", result)
            logger.info("Success! Task ID: %s", data.get("id", "?"))

            with timings.measure("download"):
                outputs = data.get("outputs", [])
                if outputs:
                    await save_from_urls(outputs, "video", http_client)
                else:
                    logger.warning("No output URLs in response: %s", data)
        else:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

    timings.output()


if __name__ == "__main__":
    asyncio.run(run_client())
