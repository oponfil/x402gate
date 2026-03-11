"""E2E test client for x402gate → SocialDownload.

Simulates a user:
1. Sends a social media download request to the gateway.
2. Receives a 402 Payment Required response.
3. Signs the x402 payment (Exact EVM Scheme, Base).
4. Resends with payment signature → gets media download links.

Usage:
    python tests/e2e/x402_socialdownload_test_client.py
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
from helpers import Timings
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-socialdownload-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model and body from config.yaml's socialdownload example_request."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["socialdownload"]["example_request"]
    return ex["model"], dict(ex["body"])


async def run_client():
    """Run the E2E client test for SocialDownload."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model, body = _load_example_request()
    logger.info("Model: %s", model)
    logger.info("URL: %s", body.get("url", "?"))

    # Initialize x402 client with EVM signer
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)

    timings = Timings()

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → expect 402
        logger.info("Sending initial request to SocialDownload via gateway...")
        with timings.measure("pricing"):
            response = await http_client.post(
                f"{gateway_url}/v1/socialdownload/{model}",
                json=body,
                timeout=15.0,
            )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required")
        payment_data = response.json()

        # Log price info
        for accept in payment_data.get("accepts", []):
            logger.info(
                "Payment option: %s — $%s",
                accept.get("network", "?"),
                int(accept.get("amount", 0)) / 1e6,
            )

        # 2. Sign payment
        logger.info("Signing payment...")
        with timings.measure("signing"):
            payment_required = PaymentRequired.model_validate(payment_data)

            # Filter to Base only
            base_accepts = [
                a for a in payment_required.accepts if "eip155:8453" in getattr(a, "network", "")
            ]
            if not base_accepts:
                logger.error("No Base payment option in 402 response")
                sys.exit(1)
            payment_required.accepts = base_accepts

            payment_payload = await x402_client.create_payment_payload(payment_required)
            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()

        # 3. Retry with payment → expect 200 with media links
        logger.info("Retrying with payment signature...")
        with timings.measure("paid_request"):
            response = await http_client.post(
                f"{gateway_url}/v1/socialdownload/{model}",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=60.0,
            )
        timings.add_server_timings(response)

        if response.status_code != 200:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

        result = response.json()
        data = result.get("data", result)

        logger.info("Title: %s", data.get("title", "?"))
        logger.info("Author: %s", data.get("author", "?"))
        logger.info("Source: %s", data.get("source", "?"))
        logger.info("Duration: %s", data.get("duration", "?"))

        best = data.get("best_media")
        if best:
            logger.info(
                "Best media: %s %s (%sx%s)",
                best.get("extension", "?"),
                best.get("quality", "?"),
                best.get("width", "?"),
                best.get("height", "?"),
            )
            logger.info("Download URL: %s", best["url"][:100] + "...")
        else:
            logger.warning("No best_media in response")

        medias = data.get("medias", [])
        logger.info("Total media variants: %d", len(medias))

        logger.info("E2E test passed!")

    timings.output()


if __name__ == "__main__":
    asyncio.run(run_client())
