"""E2E test client for x402gate → MiniMax TTS.

Simulates a user:
1. Sends a TTS request to the gateway.
2. Receives a 402 Payment Required response.
3. Signs the x402 payment (Exact EVM Scheme, Base).
4. Resends with payment signature → gets audio data.

Tests BOTH example_request and example_request_2 from config.yaml.

Usage:
    python tests/e2e/x402_minimax_test_client.py
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
from helpers import Timings, save_audio
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-minimax-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load_examples() -> list[tuple[str, dict, str]]:
    """Load both example requests from config.yaml.

    Returns list of (model, body, label) tuples.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    provider = cfg["providers"]["minimax"]
    examples = []
    for key in ("example_request", "example_request_2"):
        if key in provider:
            ex = provider[key]
            examples.append((ex["model"], dict(ex["body"]), ex["model"]))
    return examples


async def _run_single(
    http_client: httpx.AsyncClient,
    x402_client: x402Client,
    gateway_url: str,
    model: str,
    body: dict,
    label: str,
) -> None:
    """Run a single TTS request through the full x402 payment flow."""
    logger.info("--- Example: %s ---", label)
    logger.info("Model: %s, Text: %s", model, body.get("text", "?")[:80])

    timings = Timings()

    # 1. Request without payment → expect 402
    logger.info("Sending initial request to MiniMax via gateway...")
    with timings.measure("pricing"):
        response = await http_client.post(
            f"{gateway_url}/v1/minimax/{model}",
            json=body,
            timeout=15.0,
        )

    if response.status_code != 402:
        logger.error("Expected 402, got %d: %s", response.status_code, response.text)
        sys.exit(1)

    logger.info("Got 402 Payment Required")
    payment_data = response.json()

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

    # 3. Retry with payment → expect 200 with audio
    logger.info("Retrying with payment signature...")
    with timings.measure("paid_request"):
        response = await http_client.post(
            f"{gateway_url}/v1/minimax/{model}",
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

    audio_b64 = data.get("audio_base64", "")
    content_type = data.get("content_type", "audio/mpeg")
    chars = data.get("characters", 0)

    if audio_b64:
        saved = save_audio(audio_b64, f"minimax_{label}", content_type)
        logger.info("Audio saved: %s (%d chars)", saved, chars)
    else:
        logger.warning("No audio_base64 in response")

    timings.output()


async def run_client():
    """Run the E2E client test for MiniMax TTS (both examples)."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    examples = _load_examples()
    logger.info("Loaded %d example(s) from config", len(examples))

    # Initialize x402 client with EVM signer
    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)

    async with httpx.AsyncClient() as http_client:
        for model, body, label in examples:
            await _run_single(http_client, x402_client, gateway_url, model, body, label)

    logger.info("E2E test passed! (%d examples)", len(examples))


if __name__ == "__main__":
    asyncio.run(run_client())
