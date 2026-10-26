"""E2E test client for x402gate → SpicyAPI.

Simulates a user:
1. Sends the SpicyAPI example_request to the gateway.
2. Receives a 402 Payment Required response.
3. Signs the x402 payment (Exact EVM Scheme, Base).
4. Resends with the payment signature and downloads the image.

The model and body come from config.yaml providers.spicyapi.example_request.

Usage:
    python tests/e2e/x402_spicyapi_test_client.py
"""

import asyncio
import base64
import logging
import os
import sys

import httpx
from eth_account import Account
from helpers import Timings, load_config_yaml, save_from_urls
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-spicyapi-client")


def _load_example_request() -> tuple[str, dict]:
    """Load the jobs path and body from the SpicyAPI example."""
    cfg = load_config_yaml()
    example = cfg["providers"]["spicyapi"]["example_request"]
    return example["model"], example["body"]


def _asset_urls(result: dict) -> list[str]:
    """Collect download URLs from a finished SpicyAPI task record."""
    data = result.get("data", result)
    output = data.get("output") if isinstance(data, dict) else None
    assets = output.get("assets") if isinstance(output, dict) else None
    if not isinstance(assets, list):
        return []
    return [asset["url"] for asset in assets if isinstance(asset, dict) and asset.get("url")]


async def run_client():
    """Run the E2E client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_example_request()
    logger.info("Model: %s, Body: %s", model_path, body)

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))
    x402_client.register("eip155:84532", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)

    timings = Timings()

    async with httpx.AsyncClient() as http_client:
        logger.info("Sending initial request...")
        with timings.measure("pricing"):
            response = await http_client.post(
                f"{gateway_url}/v1/spicyapi/{model_path}",
                json=body,
            )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required")
        payment_required = PaymentRequired.model_validate(response.json())

        logger.info("Signing payment...")
        with timings.measure("signing"):
            try:
                payment_payload = await x402_client.create_payment_payload(payment_required)
                signature = base64.b64encode(
                    payment_payload.model_dump_json(by_alias=True).encode()
                ).decode()
            except Exception as e:
                logger.error("Failed to sign payment: %s", e)
                sys.exit(1)

        logger.info("Retrying with signature...")
        with timings.measure("paid_request"):
            response = await http_client.post(
                f"{gateway_url}/v1/spicyapi/{model_path}",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=180.0,
            )
        timings.add_server_timings(response)

        if response.status_code != 200:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

        result = response.json()
        logger.info("Success! Result keys: %s", list(result.get("data", result))[:8])
        urls = _asset_urls(result)
        if not urls:
            logger.error("SpicyAPI response has no output asset URL: %s", result)
            sys.exit(1)
        with timings.measure("download"):
            await save_from_urls(urls, "base_spicyapi", http_client)

    timings.output()


if __name__ == "__main__":
    asyncio.run(run_client())
