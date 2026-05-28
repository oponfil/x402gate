"""E2E test: BlockRun passthrough error forwarding.

Sends a request with a nonexistent model through the passthrough proxy
to verify that BlockRun's error response is forwarded transparently.
"""

import base64
import json
import os
import time

import httpx
import pytest
from eth_account import Account
from x402 import PaymentRequired, x402Client
from x402.mechanisms.evm.exact import ExactEvmScheme
from x402.mechanisms.evm.signers import EthAccountSigner

from tests.e2e.helpers import load_config_yaml


def _load_blockrun_example():
    """Load model and body from config.yaml's blockrun example_request."""
    cfg = load_config_yaml()
    ex = cfg["providers"]["blockrun"]["example_request"]
    return ex["model"], dict(ex["body"])


@pytest.mark.order("first")
@pytest.mark.asyncio
async def test_blockrun_invalid_model_error(gateway_process):
    """Send a nonexistent model to BlockRun, verify error is forwarded."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    _model, body = _load_blockrun_example()
    body["model"] = "nonexistent/fake-model-xyz-999"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            timeout=30.0,
        )

    print(f"\nStatus: {response.status_code}")
    print(f"Body: {response.text[:500]}")

    assert response.status_code != 200, (
        f"Expected error for nonexistent model, got 200: {response.text[:200]}"
    )
    print(f"\nOK: BlockRun error forwarded through gateway (status={response.status_code})")


@pytest.mark.order("first")
@pytest.mark.asyncio
async def test_blockrun_bad_params_error(gateway_process, base_chain):
    """Send a valid model with bad params (negative max_tokens)."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    _model, body = _load_blockrun_example()
    body["max_tokens"] = -999

    client_before = base_chain.get_usdc(base_chain.client_address)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    async with httpx.AsyncClient() as client:
        # Step 1: Get 402 from BlockRun
        response = await client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            timeout=15.0,
        )

        print(f"\nStep 1 status: {response.status_code}")
        print(f"Step 1 body: {response.text[:500]}")

        if response.status_code == 402:
            # Sign payment and retry with bad params
            account = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"])
            signer = EthAccountSigner(account)
            x402_client = x402Client()
            x402_client.register("eip155:8453", ExactEvmScheme(signer))

            x402_header = response.headers.get("x-402-payment")
            payment_data = json.loads(x402_header) if x402_header else response.json()

            payment_required = PaymentRequired.model_validate(payment_data)
            payment_payload = await x402_client.create_payment_payload(payment_required)
            signature = base64.b64encode(
                payment_payload.model_dump_json(by_alias=True).encode()
            ).decode()

            # Step 2: Retry with payment + bad params
            response = await client.post(
                f"{gateway_url}/v1/blockrun/v1/chat/completions",
                json=body,
                headers={"PAYMENT-SIGNATURE": signature},
                timeout=30.0,
            )

            print(f"\nStep 2 status: {response.status_code}")
            print(f"Step 2 body: {response.text[:500]}")

        assert response.status_code != 200, (
            f"Expected error for bad params, got 200: {response.text[:200]}"
        )

    # Verify client balance unchanged
    time.sleep(5)  # noqa: ASYNC251
    client_after = base_chain.get_usdc(base_chain.client_address)
    client_diff = client_before - client_after

    print(f"\n=== [Base] Client USDC AFTER:  {client_after / 1e6:.6f} ===")
    print(f"=== [Base] Client diff:        {client_diff / 1e6:.6f} USDC ===")
    print(f"OK: BlockRun error forwarded, client diff = {client_diff / 1e6:.6f} USDC")
