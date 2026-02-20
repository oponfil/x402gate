"""E2E test client for x402gate — BlockRun passthrough.

This client tests the passthrough proxy flow:
1. Sends a request to the gateway (which forwards to BlockRun).
2. Receives BlockRun's 402 Payment Required response (passthrough).
3. Parses x402 payment requirements from the response header.
4. Uses the x402 SDK to sign payment to BlockRun's wallet.
5. Resends the request with the payment signature (forwarded to BlockRun).

Usage:
    BASE_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_blockrun_test_client.py
"""

import asyncio
import base64
import json
import logging
import os
import sys

import httpx
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-blockrun-client")

MODEL = "openai/gpt-4o-mini"
PROMPT = "Say hello in exactly three words."


def parse_x402_from_response(response: httpx.Response) -> dict:
    """Extract x402 PaymentRequired from response headers.

    BlockRun sends x402 data as base64 JSON in these headers:
    - x-payment-required
    - www-authenticate (X402 requirements="...")
    """
    # Try x-payment-required header first
    header_value = response.headers.get("x-payment-required")
    if header_value:
        return json.loads(base64.b64decode(header_value))

    # Fallback: www-authenticate header
    www_auth = response.headers.get("www-authenticate", "")
    if www_auth.startswith("X402 requirements="):
        b64_data = www_auth.split("=", 1)[1].strip('"')
        return json.loads(base64.b64decode(b64_data))

    # Last fallback: response body
    return response.json()


async def run_client():
    """Run the BlockRun passthrough E2E client test."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    # Initialize x402 client with EVM signer
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()

    # Register the 'exact' scheme for Base Mainnet
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)
    logger.info("Model: %s", MODEL)

    async with httpx.AsyncClient() as http_client:
        # 1. Request without payment → expect BlockRun's 402
        logger.info("Sending initial request (no payment)...")
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 20,
        }
        response = await http_client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
        )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required (passthrough from BlockRun)")

        # Parse x402 payment requirements from header
        payment_data = parse_x402_from_response(response)
        logger.info("x402 version: %s", payment_data.get("x402Version"))

        base_accepts = [
            a for a in payment_data.get("accepts", [])
            if "eip155:8453" in a.get("network", "")
        ]
        if not base_accepts:
            logger.error("No Base (EVM) payment option: %s", payment_data)
            sys.exit(1)

        pay_to = base_accepts[0].get("payTo", "unknown")
        amount = base_accepts[0].get("amount", "?")
        logger.info("PayTo: %s (BlockRun's wallet)", pay_to)
        logger.info("Amount: %s (USDC smallest unit)", amount)

        # Filter to only include Base accepts
        payment_data["accepts"] = base_accepts
        payment_required = PaymentRequired.model_validate(payment_data)

        # 2. Sign payment (to BlockRun's wallet, not ours)
        logger.info("Signing EVM payment to BlockRun...")
        try:
            payment_payload = await x402_client.create_payment_payload(payment_required)
            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()
        except Exception as e:
            logger.error("Failed to sign payment: %s", e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 3. Retry with payment — gateway forwards to BlockRun
        logger.info("Retrying with payment signature (forwarded to BlockRun)...")
        response = await http_client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            headers={"Payment-Signature": signature},
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            logger.info("Success!")
            logger.info("Response: %s", content)
            logger.info("Tokens: prompt=%s, completion=%s, total=%s",
                        usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        usage.get("total_tokens"))
            if "cost" in usage:
                logger.info("Cost: $%.6f", usage["cost"])
        else:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_client())
