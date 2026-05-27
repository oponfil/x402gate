"""E2E test client for x402gate -> OpenRouter STT (Whisper).

Usage:
    python tests/e2e/x402_stt_test_client.py
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
logger = logging.getLogger("x402-stt-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
FIXTURE_WAV = Path(__file__).parent.parent / "fixtures" / "sample_1s.wav"


def _load_example_request() -> tuple[str, dict]:
    """Load STT path and body from config.yaml example_request_4."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["openrouter"]["example_request_4"]
    body = dict(ex["body"])
    if FIXTURE_WAV.exists():
        body["input_audio"] = {
            "data": base64.b64encode(FIXTURE_WAV.read_bytes()).decode(),
            "format": "wav",
        }
    return ex["model"], body


async def run_client():
    """Run the E2E client test for OpenRouter STT."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("BASE_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("BASE_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    model_path, body = _load_example_request()
    logger.info("Model path: %s", model_path)
    logger.info("STT model: %s", body.get("model"))

    account = Account.from_key(private_key)
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    logger.info("Client Address: %s", account.address)
    logger.info("Gateway URL: %s", gateway_url)

    timings = Timings()

    async with httpx.AsyncClient() as http_client:
        logger.info("Sending initial STT request via gateway...")
        with timings.measure("pricing"):
            response = await http_client.post(
                f"{gateway_url}/v1/openrouter/{model_path}",
                json=body,
                timeout=60.0,
            )

        if response.status_code != 402:
            logger.error("Expected 402, got %d: %s", response.status_code, response.text)
            sys.exit(1)

        logger.info("Got 402 Payment Required")
        payment_data = response.json()

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

        logger.info("Retrying with payment signature...")
        with timings.measure("paid_request"):
            response = await http_client.post(
                f"{gateway_url}/v1/openrouter/{model_path}",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=120.0,
            )
        timings.add_server_timings(response)

        if response.status_code != 200:
            logger.error("Failed: %d %s", response.status_code, response.text)
            sys.exit(1)

        result = response.json()
        data = result.get("data", result)
        text = data.get("text", "")
        usage = data.get("usage", {})

        logger.info("Transcription: %r", text[:200] if text else "(empty)")
        logger.info("Usage: seconds=%s cost=$%s", usage.get("seconds"), usage.get("cost"))
        logger.info("E2E STT test passed!")

    timings.output()


if __name__ == "__main__":
    asyncio.run(run_client())
