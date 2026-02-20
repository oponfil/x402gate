"""E2E test client for x402gate — Solana (SVM).

This client simulates a user paying with Solana USDC:
1. Sends a request to the gateway.
2. Receives a 402 Payment Required response (with Solana option).
3. Uses the x402 SDK to sign the payment (Exact SVM Scheme).
4. Resends the request with the payment signature.

Usage:
    SOLANA_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_solana_test_client.py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from solders.keypair import Keypair
from x402 import PaymentRequired, x402Client
from x402.mechanisms.svm.exact.client import ExactSvmScheme
from x402.mechanisms.svm.signers import KeypairSigner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-solana-client")

OUTPUT_DIR = Path(__file__).parent / "output"
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model path and request body from config.yaml."""
    import yaml

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    example = cfg["providers"]["wavespeed"]["example_request"]
    return example["model"], example["body"]


async def run_client():
    """Run the Solana E2E client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("SOLANA_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_example_request()
    logger.info("Model: %s, Body: %s", model_path, body)

    # Initialize x402 client with SVM signer
    keypair = Keypair.from_base58_string(private_key)
    signer = KeypairSigner(keypair)
    x402_client = x402Client()

    # Register the 'exact' scheme for Solana Mainnet
    x402_client.register("solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp", ExactSvmScheme(signer))

    logger.info("Client Address: %s", signer.address)
    logger.info("Gateway URL: %s", gateway_url)

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment
        logger.info("Sending initial request...")
        response = await http_client.post(
            f"{gateway_url}/v1/wavespeed/{model_path}",
            json=body,
        )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required")

        # Find the Solana payment option
        payment_data = response.json()
        solana_accepts = [
            a for a in payment_data.get("accepts", []) if "solana:" in a.get("network", "")
        ]
        if not solana_accepts:
            logger.error("No Solana payment option in 402 response: %s", payment_data)
            sys.exit(1)

        logger.info("Found Solana payment option: %s", solana_accepts[0]["network"])

        # Filter to only include Solana accepts
        payment_data["accepts"] = solana_accepts
        payment_required = PaymentRequired.model_validate(payment_data)

        # 2. Create payment payload (signing)
        logger.info("Signing Solana payment...")
        try:
            payment_payload = await x402_client.create_payment_payload(payment_required)
            import base64

            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()
        except Exception as e:
            logger.error("Failed to sign payment: %s", e)
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # 3. Retry with payment
        logger.info("Retrying with Solana signature...")
        response = await http_client.post(
            f"{gateway_url}/v1/wavespeed/{model_path}",
            json=body,
            headers={"PAYMENT-SIGNATURE": signature},
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("Success! Result: %s", result)

            # Download generated image
            outputs = result.get("data", result).get("outputs", [])
            if outputs:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for i, url in enumerate(outputs):
                    ext = Path(url.split("?")[0]).suffix or ".jpeg"
                    img_path = OUTPUT_DIR / f"solana_{ts}{ext}"
                    img_resp = await http_client.get(url)
                    img_path.write_bytes(img_resp.content)
                    logger.info("Saved image: %s", img_path)
        else:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_client())
