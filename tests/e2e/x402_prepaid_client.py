"""E2E test client for x402gate prepaid mode.

Simulates the full prepaid flow:
1. Top-up: Send Solana USDC payment to /v1/topup → get prepaid balance.
2. OpenRouter call #1 and #2: LLM chat with prepaid headers.
3. WaveSpeed call: Image generation with prepaid headers.
4. Tungsten call: Image generation with prepaid headers.
5. Check final balance.

Usage:
    SOLANA_E2ETEST_PRIVATE_KEY=... python tests/e2e/x402_prepaid_client.py
"""

import asyncio
import base64
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import yaml
from helpers import save_from_urls, save_images
from solders.keypair import Keypair
from x402 import PaymentRequired, x402Client
from x402.mechanisms.svm.exact.client import ExactSvmScheme
from x402.mechanisms.svm.signers import KeypairSigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("x402-prepaid-client")

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

# Top-up amount in USDC
TOPUP_AMOUNT = "0.110000"  # $0.11


def _load_config():
    """Load config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def _topup(
    http_client: httpx.AsyncClient,
    gateway_url: str,
    x402_client: x402Client,
) -> dict:
    """Perform a top-up: send 402 → sign → pay → get balance."""
    logger.info("Requesting top-up (%s USDC)...", TOPUP_AMOUNT)
    response = await http_client.post(
        f"{gateway_url}/v1/topup",
        json={"amount": float(TOPUP_AMOUNT)},
        timeout=15.0,
    )

    if response.status_code != 402:
        logger.error("Expected 402, got %d: %s", response.status_code, response.text)
        sys.exit(1)

    payment_data = response.json()

    # Filter to Solana only
    solana_accepts = [
        a for a in payment_data.get("accepts", []) if "solana:" in a.get("network", "")
    ]
    if not solana_accepts:
        logger.error("No Solana payment option in 402 response")
        sys.exit(1)

    payment_data["accepts"] = solana_accepts
    payment_required = PaymentRequired.model_validate(payment_data)

    # Sign and submit payment
    logger.info("Signing Solana payment for top-up...")
    payment_payload = await x402_client.create_payment_payload(payment_required)
    signature = base64.b64encode(payment_payload.model_dump_json(by_alias=True).encode()).decode()

    response = await http_client.post(
        f"{gateway_url}/v1/topup",
        headers={"PAYMENT-SIGNATURE": signature},
        timeout=30.0,
    )

    if response.status_code != 200:
        logger.error("Top-up failed: %d %s", response.status_code, response.text)
        sys.exit(1)

    result = response.json()
    logger.info(
        "Top-up successful! Credited: $%s, Balance: $%s",
        result["credited"],
        result["balance"],
    )
    return result


async def _prepaid_request(
    http_client: httpx.AsyncClient,
    gateway_url: str,
    keypair: Keypair,
    provider: str,
    sub_path: str,
    body: dict,
    label: str,
    request_timeout: float = 60.0,
) -> dict:
    """Make a prepaid API request with Ed25519 signature.

    Args:
        provider: e.g. "openrouter", "wavespeed", "tungsten"
        sub_path: path after /v1/{provider}/, e.g. "chat/completions"
        body: JSON body to send
        label: human-readable label for logging
        request_timeout: request timeout in seconds
    """
    full_path = f"{provider}/{sub_path}"
    ts = int(time.time())
    msg = f"x402gate:{full_path}:{ts}".encode()
    sig = keypair.sign_message(msg)

    logger.info("[%s] Sending prepaid request to /v1/%s ...", label, full_path)
    response = await http_client.post(
        f"{gateway_url}/v1/{full_path}",
        json=body,
        headers={
            "X-PREPAID-PUBKEY": str(keypair.pubkey()),
            "X-PREPAID-SIGNATURE": str(sig),
            "X-PREPAID-TIMESTAMP": str(ts),
        },
        timeout=request_timeout,
    )

    if response.status_code != 200:
        logger.error("[%s] Failed: %d %s", label, response.status_code, response.text[:300])
        sys.exit(1)

    result = response.json()
    data = result.get("data", result)

    # OpenRouter-style response
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})

    # WaveSpeed / Tungsten-style response (image output)
    images = None
    output_urls = None
    if isinstance(data.get("output"), dict):
        images = data["output"].get("images", [])
    elif isinstance(data, dict) and "images" in data:
        images = data["images"]
    # WaveSpeed returns image/video URLs in "outputs"
    if isinstance(data, dict) and "outputs" in data:
        output_urls = data["outputs"]

    print(f"\n{'=' * 60}")
    print(f"[{label}] RESPONSE:")
    if content:
        print(f"\n{content[:200]}")
    if images:
        saved = save_images(images, f"prepaid_{label}")
        print(f"\n  Images returned: {len(images)}, saved: {len(saved)}")
        for p in saved:
            print(f"  -> {p}")
    if output_urls:
        saved = await save_from_urls(output_urls, f"prepaid_{label}", http_client)
        print(f"\n  Outputs downloaded: {len(output_urls)}, saved: {len(saved)}")
        for p in saved:
            print(f"  -> {p}")
    if usage:
        print(
            f"  Tokens: prompt={usage.get('prompt_tokens')}, "
            f"completion={usage.get('completion_tokens')}"
        )
    print(f"{'=' * 60}")

    return result


async def _check_balance(
    http_client: httpx.AsyncClient, gateway_url: str, pubkey: str, label: str
) -> str:
    """Check and log the current prepaid balance."""
    resp = await http_client.get(f"{gateway_url}/v1/balance/{pubkey}")
    balance = resp.json()["balance"]
    logger.info("Balance after %s: $%s", label, balance)
    return balance


async def run_client():
    """Run the full prepaid E2E flow."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4021")
    private_key = os.environ.get("SOLANA_E2ETEST_PRIVATE_KEY")

    if not private_key:
        logger.error("SOLANA_E2ETEST_PRIVATE_KEY env var not set")
        sys.exit(1)

    cfg = _load_config()

    # Initialize x402 client with SVM signer
    keypair = Keypair.from_base58_string(private_key)
    signer = KeypairSigner(keypair)
    x402_client = x402Client()

    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    x402_client.register(
        "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
        ExactSvmScheme(signer, rpc_url=rpc_url),
    )

    pubkey_str = str(keypair.pubkey())
    logger.info("Client wallet: %s", pubkey_str)
    logger.info("Gateway: %s", gateway_url)

    balances = {}

    async with httpx.AsyncClient() as http_client:
        # 1. Top-up
        topup_result = await _topup(http_client, gateway_url, x402_client)
        balances["top-up"] = await _check_balance(http_client, gateway_url, pubkey_str, "top-up")

        # 2. OpenRouter call #1
        or_ex = cfg["providers"]["openrouter"]["example_request"]
        await _prepaid_request(
            http_client,
            gateway_url,
            keypair,
            provider="openrouter",
            sub_path="chat/completions",
            body=or_ex["body"],
            label="OpenRouter #1",
        )
        balances["openrouter-1"] = await _check_balance(
            http_client, gateway_url, pubkey_str, "OpenRouter #1"
        )

        # 3. OpenRouter call #2
        await _prepaid_request(
            http_client,
            gateway_url,
            keypair,
            provider="openrouter",
            sub_path="chat/completions",
            body=or_ex["body"],
            label="OpenRouter #2",
        )
        balances["openrouter-2"] = await _check_balance(
            http_client, gateway_url, pubkey_str, "OpenRouter #2"
        )

        # 4. WaveSpeed call (image generation)
        ws_ex = cfg["providers"]["wavespeed"]["example_request"]
        ws_model = ws_ex["model"]  # e.g. "wavespeed-ai/z-image/turbo"
        await _prepaid_request(
            http_client,
            gateway_url,
            keypair,
            provider="wavespeed",
            sub_path=ws_model,
            body=ws_ex["body"],
            label="WaveSpeed",
            request_timeout=120.0,  # image gen can take a while
        )
        balances["wavespeed"] = await _check_balance(
            http_client, gateway_url, pubkey_str, "WaveSpeed"
        )

        # 5. Tungsten call (image generation)
        tg_ex = cfg["providers"]["tungsten"]["example_request"]
        tg_model = tg_ex["model"]  # e.g. "generations"
        await _prepaid_request(
            http_client,
            gateway_url,
            keypair,
            provider="tungsten",
            sub_path=tg_model,
            body=tg_ex["body"],
            label="Tungsten",
            request_timeout=300.0,  # Tungsten can take 60-120+ seconds
        )
        balances["tungsten"] = await _check_balance(
            http_client, gateway_url, pubkey_str, "Tungsten"
        )

        # Summary
        print(f"\n{'=' * 60}")
        print("PREPAID E2E SUMMARY:")
        print(f"  Top-up credited:          ${topup_result['credited']}")
        for label, bal in balances.items():
            print(f"  Balance after {label:16s} ${bal}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_client())
