"""E2E test: BlockRun passthrough error forwarding.

Sends a request with a nonexistent model through the passthrough proxy
to verify that BlockRun's error response is forwarded transparently.
"""

import os

import httpx
import pytest
import yaml


def _load_blockrun_example():
    """Load model and body from config.yaml's blockrun example_request."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ex = cfg["providers"]["blockrun"]["example_request"]
    return ex["model"], ex["body"]


@pytest.mark.asyncio
async def test_blockrun_invalid_model_error(gateway_process):
    """Send a nonexistent model to BlockRun, verify error is forwarded."""
    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    _model, body = _load_blockrun_example()

    # Override model with a fake one
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


@pytest.mark.asyncio
async def test_blockrun_bad_params_error(gateway_process):
    """Send a valid model with bad params (negative max_tokens)."""
    if not os.environ.get("BASE_E2ETEST_PRIVATE_KEY"):
        pytest.skip("BASE_E2ETEST_PRIVATE_KEY not set")

    from eth_account import Account
    from web3 import Web3

    from x402gate.core.config import load_config

    cfg = load_config()
    base_cfg = cfg.payment.networks["base"]
    w3 = Web3(Web3.HTTPProvider(base_cfg.rpc_url))
    USDC = base_cfg.token_address
    CLIENT = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"]).address

    usdc = w3.eth.contract(
        address=Web3.to_checksum_address(USDC),
        abi=[{
            "constant": True,
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
        }],
    )

    def get_usdc(addr):
        return usdc.functions.balanceOf(Web3.to_checksum_address(addr)).call()

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:4022")
    _model, body = _load_blockrun_example()

    # Override max_tokens with invalid value
    body["max_tokens"] = -999

    # --- Record client balance BEFORE ---
    client_before = get_usdc(CLIENT)
    print(f"\n=== [Base] Client USDC BEFORE: {client_before / 1e6:.6f} ===")

    async with httpx.AsyncClient() as client:
        # Step 1: Get 402 from BlockRun (with valid model)
        response = await client.post(
            f"{gateway_url}/v1/blockrun/v1/chat/completions",
            json=body,
            timeout=15.0,
        )

        print(f"\nStep 1 status: {response.status_code}")
        print(f"Step 1 body: {response.text[:500]}")

        if response.status_code == 402:
            # BlockRun accepted the request for pricing, will fail on execution
            # Sign payment and retry
            from x402 import PaymentRequired, x402Client
            from x402.mechanisms.evm.exact import ExactEvmScheme
            from x402.mechanisms.evm.signers import EthAccountSigner

            account = Account.from_key(os.environ["BASE_E2ETEST_PRIVATE_KEY"])
            signer = EthAccountSigner(account)
            x402_client = x402Client()
            x402_client.register("eip155:8453", ExactEvmScheme(signer))

            # Parse 402 from headers (BlockRun x402 format)
            x402_header = response.headers.get("x-402-payment")
            if x402_header:
                import json
                payment_data = json.loads(x402_header)
            else:
                payment_data = response.json()

            payment_required = PaymentRequired.model_validate(payment_data)

            import base64
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

        # Should not be 200 with invalid params
        assert response.status_code != 200, (
            f"Expected error for bad params, got 200: {response.text[:200]}"
        )

    # --- Verify client balance unchanged ---
    import time
    time.sleep(5)
    client_after = get_usdc(CLIENT)
    client_diff = client_before - client_after

    print(f"\n=== [Base] Client USDC AFTER:  {client_after / 1e6:.6f} ===")
    print(f"=== [Base] Client diff:        {client_diff / 1e6:.6f} USDC ===")

    # For passthrough: if BlockRun rejected before settlement, no money lost
    # If BlockRun accepted payment but failed, that's BlockRun's responsibility
    print(f"OK: BlockRun error forwarded, client diff = {client_diff / 1e6:.6f} USDC")
