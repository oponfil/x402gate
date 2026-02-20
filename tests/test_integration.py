"""Integration tests for the full x402gate proxy flow.

Uses respx to mock WaveSpeed and facilitator APIs,
and httpx.AsyncClient with FastAPI's TestClient for request testing.
"""

import os
from unittest.mock import patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

# Set env vars before importing the app
os.environ.setdefault("WAVESPEED_API_KEY", "test-key-12345")
os.environ.setdefault("BASE_PAY_TO_ADDRESS", "0x1234567890abcdef1234567890abcdef12345678")


@pytest.fixture
def config_file(tmp_path):
    """Create a test config file."""
    config = tmp_path / "config.yaml"
    config.write_text("""
gateway:
  host: "127.0.0.1"
  port: 4021
  commission: 0.05
  min_commission: 0.001
  price_cache_ttl: 0

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
""")
    return config


@pytest.fixture
def client(config_file):
    """Create a test client with mocked config path."""
    with patch("x402gate.app.load_config") as mock_load:
        from x402gate.core.config import load_config as real_load

        mock_load.return_value = real_load(config_file)

        from x402gate.app import app

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
        # 5% of $0.003 = $0.00015, but min_commission = $0.001
        # So commission = max($0.00015, $0.001) = $0.001, total = $0.004
        assert body["accepts"][0]["price"] == "$0.004000"

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
