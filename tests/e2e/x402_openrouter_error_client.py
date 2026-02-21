"""E2E error client: sends a paid request with an invalid model.

The gateway should verify payment, forward to OpenRouter, get an error back,
and return the error WITHOUT settling payment — client keeps USDC.

Usage:
    python tests/e2e/x402_openrouter_error_client.py
"""

import asyncio
import base64
import logging
import os
import sys

import httpx
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-openrouter-error-client")


async def run_client():
    """Run the E2E error client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    # Use a VALID model for the 402 pricing step, then switch to an
    # INVALID model in the paid request so OpenRouter rejects it.
    valid_body = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
    }

    invalid_body = {
        "model": "nonexistent/model-that-doesnt-exist-12345",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
    }

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)

    async with httpx.AsyncClient() as http_client:
        # 1. Get a valid 402 with pricing (using valid model)
        logger.info("Getting 402 with valid model for pricing...")
        response = await http_client.post(
            f"{gateway_url}/v1/openrouter/chat/completions",
            json=valid_body,
            timeout=15.0,
        )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        payment_data = response.json()
        logger.info("Got 402, price options: %d", len(payment_data.get("accepts", [])))

        # 2. Sign payment
        payment_required = PaymentRequired.model_validate(payment_data)
        base_accepts = [
            a for a in payment_required.accepts if "eip155:8453" in getattr(a, "network", "")
        ]
        if not base_accepts:
            logger.error("No Base payment option")
            sys.exit(1)
        payment_required.accepts = base_accepts

        payment_payload = await x402_client.create_payment_payload(payment_required)
        signature = base64.b64encode(
            payment_payload.model_dump_json(by_alias=True).encode()
        ).decode()

        # 3. Send paid request with INVALID model → should get error, no settlement
        logger.info("Sending paid request with invalid model...")
        response = await http_client.post(
            f"{gateway_url}/v1/openrouter/chat/completions",
            json=invalid_body,
            headers={"PAYMENT-SIGNATURE": signature},
            timeout=30.0,
        )

        logger.info("Response status: %d", response.status_code)
        logger.info("Response body: %s", response.text[:500])

        # We expect an error (4xx/5xx), NOT 200
        if response.status_code == 200:
            logger.error("Expected error but got 200 — payment may have been settled!")
            sys.exit(1)

        logger.info("Got expected error %d — payment should NOT be settled", response.status_code)
        logger.info("E2E error test passed!")


if __name__ == "__main__":
    asyncio.run(run_client())
