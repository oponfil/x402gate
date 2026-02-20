"""E2E test client for x402gate → OpenRouter on Solana.

Simulates a user paying with Solana USDC:
1. Sends an LLM chat request to the gateway.
2. Receives a 402 Payment Required response.
3. Signs the x402 payment (Exact SVM Scheme, Solana).
4. Resends with payment signature → gets LLM response.

Usage:
    SOLANA_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_solana_openrouter_client.py
"""

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

import httpx
import yaml
from solders.keypair import Keypair
from x402 import PaymentRequired, x402Client
from x402.mechanisms.svm.exact.client import ExactSvmScheme
from x402.mechanisms.svm.signers import KeypairSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-solana-openrouter-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model and body from config.yaml's openrouter example_request."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["openrouter"]["example_request"]
    return ex["model"], dict(ex["body"])


async def run_client():
    """Run the E2E client test for OpenRouter on Solana."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("SOLANA_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model, body = _load_example_request()
    logger.info("Model: %s", model)
    logger.info("Messages: %s", body.get("messages", []))

    # Initialize x402 client with SVM signer
    keypair = Keypair.from_base58_string(private_key)
    signer = KeypairSigner(keypair)
    x402_client = x402Client()
    x402_client.register("solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp", ExactSvmScheme(signer))

    logger.info("Client Address: %s", signer.address)
    logger.info("Gateway URL: %s", gateway_url)

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → expect 402
        logger.info("Sending initial request to OpenRouter via gateway...")
        response = await http_client.post(
            f"{gateway_url}/v1/openrouter/chat/completions",
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

        # 2. Filter to Solana only and sign payment
        solana_accepts = [
            a for a in payment_data.get("accepts", [])
            if "solana:" in a.get("network", "")
        ]
        if not solana_accepts:
            logger.error("No Solana payment option in 402 response")
            sys.exit(1)

        payment_data["accepts"] = solana_accepts
        payment_required = PaymentRequired.model_validate(payment_data)

        logger.info("Signing Solana payment...")
        payment_payload = await x402_client.create_payment_payload(payment_required)
        signature = base64.b64encode(
            payment_payload.model_dump_json(by_alias=True).encode()
        ).decode()

        # 3. Retry with payment → expect 200 with LLM response
        logger.info("Retrying with Solana payment signature...")
        response = await http_client.post(
            f"{gateway_url}/v1/openrouter/chat/completions",
            json=body,
            headers={"PAYMENT-SIGNATURE": signature},
            timeout=60.0,
        )

        if response.status_code != 200:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

        result = response.json()
        data = result.get("data", result)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        print(f"\n{'='*60}")
        print(f"OPENROUTER RESPONSE:\n\n{content}")
        print(f"{'='*60}")
        print(f"Model: {data.get('model', '?')}")
        print(f"Tokens: prompt={usage.get('prompt_tokens')}, "
              f"completion={usage.get('completion_tokens')}, "
              f"total={usage.get('total_tokens')}")
        logger.info("E2E test passed!")


if __name__ == "__main__":
    asyncio.run(run_client())
