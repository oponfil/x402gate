"""Integration tests for the full x402gate proxy flow.

Uses respx to mock WaveSpeed and facilitator APIs,
and httpx.AsyncClient with FastAPI's TestClient for request testing.
"""

import asyncio
import os
import time
from decimal import Decimal
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from solders.keypair import Keypair

from x402gate.app import app
from x402gate.core.config import load_config as real_load
from x402gate.core.prepaid import deposit, get_balance, reset

# Set env vars before importing the app
os.environ.setdefault("WAVESPEED_API_KEY", "test-key-12345")
os.environ.setdefault("BASE_PAY_TO_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")
os.environ.setdefault("CLOUDCONVERT_API_KEY", "test-cc-key")


@pytest.fixture
def config_file(tmp_path):
    """Create a test config file."""
    config = tmp_path / "config.yaml"
    config.write_text("""
gateway:
  host: "127.0.0.1"
  port: 4021
  commission: 0.04
  gas_surcharge: 0.001
  price_cache_ttl: 0
  max_upload_mb: 1  # 1 MB limit for testing

payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "0x1234567890abcdef1234567890abcdef12345678"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      token_name: "USD Coin"
      token_version: "2"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"
  max_timeout: 3600

providers:
  wavespeed:
    enabled: true
    base_url: "https://api.wavespeed.ai/api/v3"
    api_key: "test-key-12345"
    poll_interval: 1
    poll_timeout: 10
  blockrun:
    type: passthrough
    enabled: true
    base_url: "https://blockrun.ai/api"
  cloudconvert:
    enabled: true
    base_url: "https://api.cloudconvert.com/v2"
    api_key: "test-cc-key"
    fixed_price_usd: 0.03
    poll_interval: 0
    poll_timeout: 10
""")
    return config


@pytest.fixture
def client(config_file):
    """Create a test client with mocked config path."""
    with patch("x402gate.app.load_config") as mock_load:
        mock_load.return_value = real_load(config_file)

        with TestClient(app) as c:
            yield c


class TestHealthCheck:
    """Tests for the health endpoint."""

    def test_health(self, client):
        """GET /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProvidersList:
    """Tests for the providers list endpoint."""

    def test_list_providers(self, client):
        """GET /v1/providers returns registered providers."""
        response = client.get("/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "wavespeed" in data["providers"]


class TestWaveSpeedProxy:
    """Integration tests for the WaveSpeed proxy endpoint."""

    @respx.mock
    def test_request_without_payment_returns_402(self, client):
        """POST without PAYMENT-SIGNATURE returns 402 with price."""
        # Mock WaveSpeed Pricing API
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(
                200,
                json={"data": {"unit_price": 0.003, "currency": "USD"}},
            )
        )

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat in space"},
        )

        assert response.status_code == 402
        body = response.json()
        assert "accepts" in body
        assert body["accepts"][0]["scheme"] == "exact"
        # 4% of $0.003 = $0.00012 + gas_surcharge $0.001
        # Price: $0.003 base + 4% ($0.00012) + $0.001 gas = $0.004120
        assert body["accepts"][0]["price"] == "$0.004120"

    @respx.mock
    def test_invalid_json_returns_400(self, client):
        """POST with non-JSON body returns 400."""
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    @respx.mock
    def test_pricing_api_error_returns_503(self, client):
        """If WaveSpeed Pricing API is down, return 503."""
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "test"},
        )
        assert response.status_code == 503


class TestBlockRunPassthrough:
    """Integration tests for the BlockRun passthrough proxy."""

    @respx.mock
    def test_402_passthrough(self, client):
        """BlockRun 402 is forwarded to client as-is."""
        blockrun_402 = {
            "x402Version": 1,
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "price": "$0.003000",
                    "payTo": "0xBlockRunWallet",
                }
            ],
        }
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(402, json=blockrun_402)
        )

        response = client.post(
            "/v1/blockrun/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 402
        body = response.json()
        assert body["accepts"][0]["payTo"] == "0xBlockRunWallet"

    @respx.mock
    def test_payment_header_forwarded(self, client):
        """Payment-Signature header is forwarded to BlockRun."""
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Hello!"}}]},
            )
        )

        response = client.post(
            "/v1/blockrun/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Payment-Signature": "test-sig-123"},
        )

        assert response.status_code == 200
        # Verify the header was forwarded to BlockRun
        assert respx.calls[0].request.headers["payment-signature"] == "test-sig-123"

    @respx.mock
    def test_response_body_passthrough(self, client):
        """Upstream response body is returned unchanged."""
        upstream_body = {
            "id": "gen-abc123",
            "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.00002},
        }
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=upstream_body)
        )

        response = client.post(
            "/v1/blockrun/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Payment-Signature": "valid-sig"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "Hi there!"
        assert body["usage"]["cost"] == 0.00002

    def test_list_providers_includes_blockrun(self, client):
        """GET /v1/providers includes blockrun."""
        response = client.get("/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "blockrun" in data["providers"]


class TestPrepaidFlow:
    """Integration tests for prepaid balance flow."""

    def test_balance_unknown_pubkey(self, client):
        """GET /v1/balance/{pubkey} returns 0 for unknown pubkey."""
        response = client.get("/v1/balance/UnknownPubKey123")
        assert response.status_code == 200
        assert response.json()["balance"] == "0"

    def test_balance_after_deposit(self, client):
        """GET /v1/balance after manual deposit shows correct balance."""

        reset()
        asyncio.get_event_loop().run_until_complete(deposit("TestPubKey", Decimal("5.00")))

        response = client.get("/v1/balance/TestPubKey")
        assert response.status_code == 200
        assert response.json()["balance"] == "5.00"
        reset()

    @respx.mock
    def test_prepaid_request_deducts_balance(self, client):
        """POST with prepaid headers deducts base_price from balance."""

        reset()
        kp = Keypair()
        pubkey_str = str(kp.pubkey())
        asyncio.get_event_loop().run_until_complete(deposit(pubkey_str, Decimal("1.00")))

        # Mock WaveSpeed pricing + submit
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post("https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-dev").mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["base64img"]}}})
        )

        ts = int(time.time())
        msg = f"x402gate:wavespeed/wavespeed-ai/flux-dev:{ts}".encode()
        sig = kp.sign_message(msg)

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers={
                "X-PREPAID-PUBKEY": pubkey_str,
                "X-PREPAID-SIGNATURE": str(sig),
                "X-PREPAID-TIMESTAMP": str(ts),
            },
        )

        assert response.status_code == 200
        remaining = get_balance(pubkey_str)
        assert remaining < __import__("decimal").Decimal("1.00")
        reset()

    @respx.mock
    def test_prepaid_insufficient_balance(self, client):
        """POST with prepaid headers and insufficient balance returns 402."""

        reset()
        kp = Keypair()
        pubkey_str = str(kp.pubkey())
        asyncio.get_event_loop().run_until_complete(deposit(pubkey_str, Decimal("0.001")))

        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        ts = int(time.time())
        msg = f"x402gate:wavespeed/wavespeed-ai/flux-dev:{ts}".encode()
        sig = kp.sign_message(msg)

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers={
                "X-PREPAID-PUBKEY": pubkey_str,
                "X-PREPAID-SIGNATURE": str(sig),
                "X-PREPAID-TIMESTAMP": str(ts),
            },
        )

        assert response.status_code == 402
        assert "Insufficient" in response.json()["error"]
        reset()

    @respx.mock
    def test_prepaid_invalid_signature(self, client):
        """POST with invalid prepaid signature returns 401."""

        reset()
        kp = Keypair()

        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers={
                "X-PREPAID-PUBKEY": str(kp.pubkey()),
                "X-PREPAID-SIGNATURE": "invalid_signature",
                "X-PREPAID-TIMESTAMP": str(int(time.time())),
            },
        )

        assert response.status_code == 401
        reset()

    @respx.mock
    def test_prepaid_expired_timestamp(self, client):
        """POST with expired timestamp returns 401."""

        reset()
        kp = Keypair()
        old_ts = int(time.time()) - 400  # exceeds 300s window
        msg = f"x402gate:wavespeed/wavespeed-ai/flux-dev:{old_ts}".encode()
        sig = kp.sign_message(msg)

        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers={
                "X-PREPAID-PUBKEY": str(kp.pubkey()),
                "X-PREPAID-SIGNATURE": str(sig),
                "X-PREPAID-TIMESTAMP": str(old_ts),
            },
        )

        assert response.status_code == 401
        reset()


class TestCloudConvertUploadLimit:
    """Integration tests for file upload size limit."""

    def test_oversized_file_returns_413(self, client):
        """File exceeding max_upload_mb returns 413."""
        # 1 MB limit in test config, send 1.5 MB
        big_file = b"x" * (1_500_000)
        response = client.post(
            "/v1/cloudconvert/convert",
            files={"file": ("big.bin", big_file)},
            data={"output_format": "pdf"},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["error"].lower()

    @respx.mock
    def test_small_file_accepted(self, client):
        """File within limit proceeds to 402 (payment required)."""
        small_file = b"x" * 1000  # 1 KB — well within 1 MB limit
        response = client.post(
            "/v1/cloudconvert/convert",
            files={"file": ("small.txt", small_file)},
            data={"output_format": "pdf"},
        )
        # Should get 402 (no payment), NOT 413
        assert response.status_code == 402


class TestDashboard:
    """Integration tests for the dashboard endpoints."""

    def test_dashboard_returns_html(self, client):
        """GET /dashboard returns 200 with HTML content."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "x402gate" in response.text
        assert "Dashboard" in response.text

    def test_stats_endpoint(self, client):
        """GET /v1/stats returns valid statistics JSON."""
        response = client.get("/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_s" in data
        assert "total_requests" in data
        assert "providers" in data
        assert "total_revenue_usd" in data

    def test_logs_endpoint(self, client):
        """GET /v1/logs returns a list of log entries."""
        response = client.get("/v1/logs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_logs_limit_param(self, client):
        """GET /v1/logs?limit=5 respects limit parameter."""
        response = client.get("/v1/logs?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5
