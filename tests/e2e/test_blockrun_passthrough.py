"""E2E test: BlockRun passthrough proxy with Live Search.

Tests the full passthrough flow through a real gateway instance:
pays BlockRun via x402, sends a search-enabled request, gets real-time data.
"""

import base64
import json
import os

import httpx
import pytest
import yaml
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact.client import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner


def _load_blockrun_example():
    """Load model and body from config.yaml's blockrun example_request."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["blockrun"]["example_request"]
    return ex["model"], dict(ex["body"])


@pytest.mark.asyncio
async def test_blockrun_live_search(gateway_process):
    """Pay BlockRun via passthrough, get a response with Live Search data."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")

    account = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"])
    signer = EthAccountSigner(account)
    x402_client = x402Client()
    x402_client.register("eip155:8453", ExactEvmScheme(signer))

    _model, body = _load_blockrun_example()

    print(f"\nModel: {_model} (with Live Search)")
    print(f"Question: {body['messages'][0]['content']}")

    async with httpx.AsyncClient() as http:
        # Step 1: Get 402
        r = await http.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            timeout=15.0,
        )
        assert r.status_code == 402, f"Expected 402, got {r.status_code}"

        price = r.json().get("price", {})
        print(f"Price: ${price.get('amount')} {price.get('currency')}")

        # Parse x402 payment from header
        xpr = r.headers.get("x-payment-required", "")
        if xpr:
            payment_data = json.loads(base64.b64decode(xpr))
        else:
            www = r.headers.get("www-authenticate", "")
            b64 = www.split("=", 1)[1].strip('"')
            payment_data = json.loads(base64.b64decode(b64))

        base_accepts = [
            a for a in payment_data.get("accepts", []) if "eip155:8453" in a.get("network", "")
        ]
        assert base_accepts, "No Base payment option"
        payment_data["accepts"] = base_accepts

        amount = int(base_accepts[0]["amount"])
        print(f"Amount: {amount} USDC units (${amount / 1e6:.6f})")

        # Step 2: Sign and pay
        payment_required = PaymentRequired.model_validate(payment_data)
        payment_payload = await x402_client.create_payment_payload(payment_required)
        signature = base64.b64encode(
            payment_payload.model_dump_json(by_alias=True).encode()
        ).decode()

        print("Sending paid request with Live Search...")
        r = await http.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            headers={"Payment-Signature": signature},
            timeout=60.0,
        )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"

    result = r.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})

    print(f"\n{'=' * 60}")
    # Sanitize for Windows cp1252 terminal
    safe_content = content.encode("ascii", errors="replace").decode("ascii")
    print(f"LIVE SEARCH RESPONSE:\n\n{safe_content}")
    print(f"{'=' * 60}")
    print(
        f"Tokens: prompt={usage.get('prompt_tokens')}, "
        f"completion={usage.get('completion_tokens')}, "
        f"total={usage.get('total_tokens')}"
    )
    print(f"Paid: ${amount / 1e6:.6f} USDC")
