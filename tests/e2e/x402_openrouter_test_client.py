"""E2E test client for x402gate → OpenRouter.

Simulates a user:
1. Sends an LLM chat request to the gateway.
2. Receives a 402 Payment Required response.
3. Signs the x402 payment (Exact EVM Scheme, Base).
4. Resends with payment signature → gets LLM response.

Usage:
    python tests/e2e/x402_openrouter_test_client.py
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
logger = logging.getLogger("x402-openrouter-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_example_request() -> tuple[str, dict]:
    """Load model and body from config.yaml's openrouter example_request."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["openrouter"]["example_request"]
    return ex["model"], dict(ex["body"])


async def run_client():
    """Run the E2E client test for OpenRouter."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model, body = _load_example_request()
    logger.info("Model: %s", model)
    logger.info("Messages: %s", body.get("messages", []))

    # Initialize x402 client with EVM signer
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
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

        # 2. Sign payment
        logger.info("Signing payment...")
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

        # 3. Retry with payment → expect 200 with LLM response
        logger.info("Retrying with payment signature...")
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

        logger.info("Response:\n%s", content)
        logger.info("Model: %s", data.get("model", "?"))
        logger.info(
            "Tokens: prompt=%s, completion=%s, total=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        logger.info("E2E test passed!")


if __name__ == "__main__":
    asyncio.run(run_client())
