"""Integration tests for the full x402gate proxy flow.

Uses respx to mock WaveSpeed and facilitator APIs,
and httpx.AsyncClient with FastAPI's TestClient for request testing.
"""

import asyncio
import base64
import contextlib
import json
import os
import threading
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from solders.keypair import Keypair

import x402gate.app as gateway
from tests.helpers import make_wav_b64
from x402gate.app import app, providers
from x402gate.core.config import load_config as real_load
from x402gate.core.payment import payment_key
from x402gate.core.prepaid import deposit, get_balance, reset
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError
from x402gate.providers.openrouter import _PricedRates
from x402gate.providers.spicyapi import _PRICE_GRACE_SECONDS

# Set env vars before importing the app
os.environ.setdefault("WAVESPEED_API_KEY", "test-key-12345")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
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
  openrouter:
    enabled: true
    base_url: "https://openrouter.ai/api/v1"
    api_key: "test-openrouter-key"
    poll_timeout: 10
    models_cache_ttl: 86400
    max_stt_audio_mb: 25
  fishaudio:
    enabled: true
    base_url: "https://api.fish.audio"
    api_key: "test-fish-key"
    poll_timeout: 10
  tungsten:
    enabled: false
    base_url: "https://api.tungsten.run/v1"
    jwt_token: "${TUNGSTEN_JWT_TOKEN}"
    cf_clearance: "${TUNGSTEN_CF_CLEARANCE}"
    fixed_price_usd: 0.01
""")
    return config


# Where a Base reconcile looks up its receipt, on whichever handler is current.
_RECEIPT_LOOKUP = "x402gate.core.payment.PaymentHandler.settlement_receipt"


@pytest.fixture
def client(config_file):
    """Create a test client with mocked config path."""
    with (
        patch("x402gate.app.load_config") as mock_load,
        patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={})),
        # Startup warm-up would fetch real catalogs; TestWarmUp covers it.
        patch("x402gate.app._warm_up", new=AsyncMock()),
        # A reconcile left open asks the real handler once more at shutdown.
        patch(_RECEIPT_LOOKUP, new=AsyncMock(return_value=None)),
    ):
        mock_load.return_value = real_load(config_file)
        # Tests reuse the same fake payment header across requests.
        gateway.payment_claims.clear()
        gateway.base_in_flight.clear()

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
        assert "tungsten" not in data["providers"]

    def test_disabled_provider_is_hidden(self, client):
        """A disabled provider has no route and is omitted from discovery."""
        missing = client.post("/v1/tungsten/generations", json={"prompt": "x"})
        assert missing.status_code == 404

        page = client.get("/", headers={"Accept": "text/html"})
        assert page.status_code == 200
        assert "tungsten" not in page.text.lower()

        manifest = client.get("/", headers={"Accept": "application/json"})
        body = manifest.json()
        assert "tungsten" not in body["providers"]
        assert "tungsten" not in body["provider_docs"]

        plugin = client.get("/.well-known/ai-plugin.json")
        assert "tungsten" not in plugin.json()["description_for_model"]


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
        asyncio.run(deposit("TestPubKey", Decimal("5.00")))

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
        asyncio.run(deposit(pubkey_str, Decimal("1.00")))

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
        asyncio.run(deposit(pubkey_str, Decimal("0.001")))

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


class TestOpenRouterSTT:
    """Integration tests for OpenRouter speech-to-text."""

    _STT_BODY = {
        "model": "openai/whisper-1",
        "input_audio": {"data": make_wav_b64(1.0), "format": "wav"},
    }

    _TRANSCRIPTION_CATALOG = {
        "data": [
            {
                "id": "openai/whisper-1",
                "pricing": {"prompt": "0.0001", "completion": "0"},
            }
        ]
    }

    def _mock_transcription_catalog(self) -> None:
        respx.get(
            "https://openrouter.ai/api/v1/models",
            params={"output_modalities": "transcription"},
        ).mock(return_value=httpx.Response(200, json=self._TRANSCRIPTION_CATALOG))

    @respx.mock
    def test_stt_without_payment_returns_402(self, client):
        """POST STT without payment returns 402 with price."""
        self._mock_transcription_catalog()

        response = client.post(
            "/v1/openrouter/audio/transcriptions",
            json=self._STT_BODY,
        )

        assert response.status_code == 402
        body = response.json()
        assert "accepts" in body
        assert body["accepts"][0]["scheme"] == "exact"
        assert body["accepts"][0]["price"].startswith("$")

    @respx.mock
    def test_stt_prepaid_success(self, client):
        """POST STT with prepaid headers returns transcription and deducts balance."""
        reset()
        kp = Keypair()
        pubkey_str = str(kp.pubkey())
        asyncio.run(deposit(pubkey_str, Decimal("1.00")))

        self._mock_transcription_catalog()
        respx.post("https://openrouter.ai/api/v1/audio/transcriptions").mock(
            return_value=httpx.Response(
                200,
                json={"text": "hello", "usage": {"seconds": 1.0, "cost": 0.0001}},
            )
        )

        ts = int(time.time())
        msg = f"x402gate:openrouter/audio/transcriptions:{ts}".encode()
        sig = kp.sign_message(msg)

        response = client.post(
            "/v1/openrouter/audio/transcriptions",
            json=self._STT_BODY,
            headers={
                "X-PREPAID-PUBKEY": pubkey_str,
                "X-PREPAID-SIGNATURE": str(sig),
                "X-PREPAID-TIMESTAMP": str(ts),
            },
        )

        assert response.status_code == 200
        payload = response.json()
        data = payload.get("data", payload)
        assert data["text"] == "hello"
        assert get_balance(pubkey_str) < Decimal("1.00")
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
        assert "low-balance" in response.text
        assert "p.balance" in response.text

    def test_stats_endpoint(self, client):
        """GET /v1/stats returns valid statistics JSON."""
        response = client.get("/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_s" in data
        assert "total_requests" in data
        assert "providers" in data
        assert "total_revenue_usd" in data
        assert data["providers"]["openrouter"]["balance"] is None

    def test_stats_includes_provider_balance(self, client):
        """A fetched upstream balance is attached to that provider card."""
        sample = {
            "openrouter": {
                "amount": "3.00",
                "unit": "usd",
                "label": "$3.00",
                "low": True,
            }
        }
        with patch("x402gate.app.fetch_balances", new=AsyncMock(return_value=sample)):
            response = client.get("/v1/stats")
        assert response.status_code == 200
        balance = response.json()["providers"]["openrouter"]["balance"]
        assert balance["label"] == "$3.00"
        assert balance["low"] is True
        assert response.json()["providers"]["wavespeed"]["balance"] is None

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


class TestPaymentVerificationFailure:
    """Tests that payment rejection reasons are returned in 402 response."""

    @respx.mock
    def test_managed_request_returns_rejection_reason(self, client):
        """POST with invalid payment returns 402 with specific reason."""
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = "fake-sig"
            mock_ph.verify = AsyncMock(
                return_value=(
                    False,
                    "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                    "AakQJK2ssd",
                    "transaction_simulation_failed: Simulation failed: InstructionErrorCustom(1)",
                )
            )

            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 402
        error = response.json()["error"]
        assert "transaction_simulation_failed" in error
        assert "InstructionErrorCustom(1)" in error

    def test_topup_returns_rejection_reason(self, client):
        """POST /v1/topup with invalid payment returns 402 with reason."""
        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = (
                "eyJhY2NlcHRlZCI6eyJhbW91bnQiOiIxMDAwMDAwIn19"  # {"accepted":{"amount":"1000000"}}
            )
            mock_ph.verify = AsyncMock(
                return_value=(
                    False,
                    "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                    "",
                    "transaction_simulation_failed: insufficient funds",
                )
            )

            response = client.post(
                "/v1/topup",
                json={"amount": 1.0},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 402
        error = response.json()["error"]
        assert "transaction_simulation_failed" in error
        assert "insufficient funds" in error

    @respx.mock
    def test_verify_error_without_reason(self, client):
        """POST with failed verify but empty reason returns generic message."""
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = "fake-sig"
            mock_ph.verify = AsyncMock(return_value=(False, "eip155:8453", "", ""))

            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 402
        assert response.json()["error"] == "Payment verification failed"

    @respx.mock
    def test_evm_insufficient_usdc_balance(self, client):
        """POST with valid EVM signature but insufficient USDC returns 402."""
        respx.post("https://api.wavespeed.ai/api/v3/model/pricing").mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = "fake-sig"
            mock_ph.verify = AsyncMock(
                return_value=(
                    True,
                    "eip155:8453",
                    "0xPayer",
                    "",
                )
            )
            mock_ph.check_evm_balance = AsyncMock(return_value=False)

            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 402
        assert "Insufficient USDC balance" in response.json()["error"]


SOLANA_NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
SOL_PAYER = "SolPayer1111"
EVM_NETWORK = "eip155:8453"
WAVESPEED_PRICE = "https://api.wavespeed.ai/api/v3/model/pricing"
WAVESPEED_SUBMIT = "https://api.wavespeed.ai/api/v3/wavespeed-ai/flux-dev"
TOPUP_SIG = "eyJhY2NlcHRlZCI6eyJhbW91bnQiOiIxMDAwMDAwIn19"  # {"accepted":{"amount":"1000000"}}


def _price_ok() -> None:
    """Mock WaveSpeed pricing at $0.003."""
    respx.post(WAVESPEED_PRICE).mock(
        return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
    )


def _paid_handler(mock_ph, network: str, payer: str) -> None:
    """Point a mocked payment handler at a verified payer."""
    mock_ph.extract_payment_signature.return_value = "fake-sig"
    mock_ph.verify = AsyncMock(return_value=(True, network, payer, ""))
    mock_ph.check_evm_balance = AsyncMock(return_value=True)


def _settled_ok(gas_label: str = "SOL") -> dict:
    """Settlement dict the handler returns after a successful settle."""
    return {
        "success": True,
        "amount_usdc": 0.00412,
        "gas_cost_usd": 0.0001,
        "gas_cost_native": 0.00001,
        "gas_label": gas_label,
    }


def _post_wavespeed(client):
    """POST a small WaveSpeed job with a payment header."""
    return client.post(
        "/v1/wavespeed/wavespeed-ai/flux-dev",
        json={"prompt": "a cat"},
        headers={"Payment-Signature": "fake-sig"},
    )


class TestSolanaUpfrontSettlement:
    """Solana payments settle before the provider runs."""

    @respx.mock
    def test_settles_once_before_provider(self, client):
        """Settle is awaited once, and that happens before WaveSpeed is called."""
        reset()
        _price_ok()
        seen: dict[str, int] = {}

        def on_submit(request):
            seen["settle_before_submit"] = mock_ph.settle.await_count
            return httpx.Response(200, json={"outputs": ["ok"]})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert seen["settle_before_submit"] == 1
        assert mock_ph.settle.await_count == 1
        expected = apply_commission(Decimal("0.003"), 0.04, 0.001)
        assert mock_ph.settle.await_args.args[1] == expected
        assert mock_ph.settle.await_args.args[2] == SOLANA_NETWORK
        assert "refunded_to_prepaid" not in response.json()
        assert "x-prepaid-cost" not in response.headers
        assert "prepaid" not in response.json()

    @respx.mock
    def test_failed_settle_does_not_call_provider(self, client):
        """A failed Solana settle is a 402 and the provider is not called."""
        reset()
        _price_ok()

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(
                return_value={
                    "success": False,
                    "error_reason": "transaction_simulation_failed",
                }
            )
            response = _post_wavespeed(client)

        assert response.status_code == 402
        assert "transaction_simulation_failed" in response.json()["error"]
        submit_calls = [
            call for call in respx.calls if str(call.request.url).endswith("wavespeed-ai/flux-dev")
        ]
        assert submit_calls == []
        assert get_balance(SOL_PAYER) == 0
        assert mock_ph.settle.await_count == 1

    @respx.mock
    def test_settle_returning_none_does_not_call_provider(self, client):
        """A settle that returns None is a 402 and the provider is not called."""
        reset()
        _price_ok()

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=None)
            response = _post_wavespeed(client)

        assert response.status_code == 402
        assert response.json()["error"] == "Payment settlement failed: unknown"
        submit_calls = [
            call for call in respx.calls if str(call.request.url).endswith("wavespeed-ai/flux-dev")
        ]
        assert submit_calls == []

    @respx.mock
    def test_provider_500_refunds_prepaid(self, client):
        """A provider 500 after a successful Solana settle credits prepaid."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(500, text="upstream down"))

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            response = _post_wavespeed(client)

        assert response.status_code == 502
        assert "x-prepaid-cost" not in response.headers
        body = response.json()
        assert "prepaid" not in body
        assert Decimal(body["refunded_to_prepaid"]) == get_balance(SOL_PAYER)
        assert Decimal(body["prepaid_balance"]) == get_balance(SOL_PAYER)
        assert mock_ph.settle.await_count == 1

    @respx.mock
    def test_provider_task_failure_refunds_prepaid(self, client):
        """A failed async task after Solana settle credits prepaid."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "task-1", "status": "processing"}},
            )
        )
        respx.get("https://api.wavespeed.ai/api/v3/predictions/task-1/result").mock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "task-1", "status": "failed", "error": "boom"}},
            )
        )

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            response = _post_wavespeed(client)

        assert response.status_code == 502
        body = response.json()
        assert "boom" in body["error"]
        assert Decimal(body["refunded_to_prepaid"]) == get_balance(SOL_PAYER)

    @respx.mock
    def test_provider_timeout_keeps_the_payment(self, client):
        """A poll timeout may still bill us, so the settled Solana payment stays."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "task-1", "status": "processing"}},
            )
        )

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch.object(
                providers["wavespeed"],
                "get_result",
                AsyncMock(side_effect=TaskTimeoutError("task-1", 10)),
            ),
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            response = _post_wavespeed(client)

        assert response.status_code == 504
        body = response.json()
        assert body["task_id"] == "task-1"
        assert body["outcome_unknown"] is True
        assert "refunded_to_prepaid" not in body
        assert get_balance(SOL_PAYER) == 0

    @respx.mock
    def test_long_poll_sends_whitespace_before_json(self, client):
        """A job that outlasts the heartbeat interval keeps the connection busy."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "task-1", "status": "processing"}},
            )
        )

        async def slow_result(task_id: str) -> dict:
            await asyncio.sleep(0.12)
            return {"outputs": ["https://cdn.example/clip.mp4"]}

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04),
            patch.object(providers["wavespeed"], "get_result", side_effect=slow_result),
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert response.content.startswith(b"\n")
        assert response.json()["data"]["outputs"][0].endswith("clip.mp4")


class TestBaseUpfrontSettlement:
    """Base payments settle before the provider runs, like Solana."""

    @respx.mock
    def test_settles_once_before_provider(self, client):
        """Settle is awaited once, and that happens before WaveSpeed is called."""
        reset()
        _price_ok()
        seen: dict[str, int] = {}

        def on_submit(request):
            seen["settle_before_submit"] = mock_ph.settle.await_count
            seen["in_flight"] = _in_flight()
            return httpx.Response(200, json={"outputs": ["ok"]})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert response.json()["data"] == {"outputs": ["ok"]}
        assert seen["settle_before_submit"] == 1
        # The settled amount no longer counts against the wallet.
        assert seen["in_flight"] == 0
        assert mock_ph.settle.await_count == 1
        expected = apply_commission(Decimal("0.003"), 0.04, 0.001)
        assert mock_ph.settle.await_args.args[1] == expected
        assert mock_ph.settle.await_args.args[2] == EVM_NETWORK

    @respx.mock
    def test_failed_settle_does_not_call_provider(self, client):
        """A payer who moved the USDC gets a 402 and the provider is not called."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["secret"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "error_reason": "insufficient_funds"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 402
        assert response.json()["error"] == "Payment settlement failed: insufficient_funds"
        assert not submit.called
        assert get_balance("0xpayer") == 0
        assert _in_flight() == 0
        assert gateway._pending_reconciles == set()

    @respx.mock
    def test_crashed_settle_does_not_call_provider_and_holds_the_payment(self, client):
        """settle returned None, maybe after sending: a 402, the claim and amount stay held."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["secret"]})
        )
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=None)
            response = _post_wavespeed(client)
            replay = _post_wavespeed(client)

        assert response.status_code == 402
        assert response.json()["error"] == "Payment settlement failed: unknown"
        assert not submit.called
        assert replay.status_code == 409
        assert mock_ph.settle.await_count == 1
        # Nothing to look up, so the amount stays counted until the authorization expires.
        assert _in_flight() == price
        assert len(gateway._pending_reconciles) == 1

    @respx.mock
    def test_provider_error_after_settle_refunds_prepaid(self, client):
        """A known provider failure credits the settled Base payment to prepaid."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        expected = apply_commission(Decimal("0.003"), 0.04, 0.001)
        assert response.status_code == 502
        body = response.json()
        assert body["refunded_to_prepaid"] == str(expected)
        assert Decimal(body["prepaid_balance"]) == expected
        assert "settlement_pending" not in body
        assert get_balance("0xpayer") == expected

    @respx.mock
    def test_payment_in_use_is_rejected(self, client):
        """A payment another request is using gets a 409 and no provider call."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        gateway.payment_claims.claim(payment_key("fake-sig"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 409
        assert not submit.called
        assert mock_ph.settle.await_count == 0

    @respx.mock
    def test_slow_settle_is_covered_by_heartbeats(self, client):
        """The heartbeat clock starts at the settle, not at the provider call."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        async def slow_settle(*args, **kwargs):
            await asyncio.sleep(0.15)
            return _settled_ok("ETH")

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04),
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(side_effect=slow_settle)
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert response.content.startswith(b"\n")
        assert response.json()["data"] == {"outputs": ["ok"]}

    @respx.mock
    def test_slow_failed_settle_reports_the_402_in_the_body(self, client):
        """After the first pad the status is 200; the settle failure is in the body."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["secret"]})
        )

        async def slow_failed_settle(*args, **kwargs):
            await asyncio.sleep(0.15)
            return {"success": False, "error_reason": "insufficient_funds"}

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04),
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(side_effect=slow_failed_settle)
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert response.content.startswith(b"\n")
        payload = response.json()
        assert payload["status"] == 402
        assert payload["error"] == "Payment settlement failed: insufficient_funds"
        assert not submit.called
        assert _in_flight() == 0

    @respx.mock
    def test_settled_payment_cannot_be_replayed_at_once(self, client):
        """Right after a settle the same payment is still claimed."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            first = _post_wavespeed(client)
            second = _post_wavespeed(client)

        assert first.status_code == 200
        assert second.status_code == 409
        assert submit.call_count == 1

    @respx.mock
    def test_failed_settle_frees_the_payment(self, client):
        """A payment whose settle clearly failed can be sent again."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                side_effect=[
                    {"success": False, "error_reason": "insufficient_funds"},
                    _settled_ok("ETH"),
                ]
            )
            first = _post_wavespeed(client)
            second = _post_wavespeed(client)

        assert first.status_code == 402
        assert second.status_code == 200

    @respx.mock
    def test_unexpected_error_keeps_the_payment(self, client):
        """A gateway bug after payment is an unknown outcome: no refund."""
        reset()
        _price_ok()

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch.object(providers["wavespeed"], "submit", AsyncMock(side_effect=KeyError("boom"))),
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 502
        assert response.json()["outcome_unknown"] is True
        assert "refunded_to_prepaid" not in response.json()
        assert mock_ph.settle.await_count == 1


class TestBaseSettlementEdgeCases:
    """Pending settlements and many payments drawing on one balance."""

    @respx.mock
    def test_pending_settlement_runs_the_job_and_follows_the_receipt(self, client):
        """A broadcast whose confirmation wait failed most likely lands: the job runs."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = AsyncMock(return_value=None)
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            response = _post_wavespeed(client)
            replay = _post_wavespeed(client)

        assert response.status_code == 200
        assert response.json()["data"] == {"outputs": ["ok"]}
        assert submit.call_count == 1
        assert replay.status_code == 409
        # The reconcile keeps the amount counted until the receipt resolves.
        assert len(gateway._pending_reconciles) == 1
        assert _in_flight() == price

    @respx.mock
    def test_pending_settlement_on_provider_failure_credits_nothing(self, client):
        """An unconfirmed payment is not credited when the provider fails."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = AsyncMock(return_value=None)
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "pending": True, "error_reason": "x"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 502
        body = response.json()
        for field in ("refunded_to_prepaid", "settlement_pending", "pending_refund"):
            assert field not in body
        assert get_balance("0xPayer") == 0

    @respx.mock
    def test_balance_must_cover_payments_still_settling(self, client):
        """A second nonce from the same wallet counts what the first has promised."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)
        gateway.base_in_flight.add("0xPAYER", price)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")

            async def balance(network, payer, amount):
                return amount <= price

            mock_ph.check_evm_balance = AsyncMock(side_effect=balance)
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 402
        assert "still settling" in response.json()["error"]
        assert mock_ph.check_evm_balance.await_args.args[2] == price * 2
        assert not submit.called
        assert mock_ph.settle.await_count == 0
        # The rejected request takes back its own promise only.
        gateway.base_in_flight.remove("0xpayer", price)
        assert gateway.base_in_flight.add("0xpayer", Decimal("0")) == 0

    @pytest.mark.parametrize(
        "settled",
        [_settled_ok("ETH"), {"success": False, "error_reason": "insufficient_funds"}],
        ids=["ok", "failed"],
    )
    @respx.mock
    def test_in_flight_amount_is_released_after_settle(self, client, settled):
        """Once a settle ends with a known outcome, the wallet's promise is gone."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=settled)
            _post_wavespeed(client)

        assert gateway.base_in_flight.add("0xpayer", Decimal("0")) == 0
        assert gateway._pending_reconciles == set()

    @respx.mock
    def test_solana_pending_settlement_runs_the_job(self, client):
        """A Solana transaction that was sent but not confirmed still runs."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "pending": True, "error_reason": "x"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert submit.called
        assert gateway._pending_reconciles == set()


class TestRequestLimits:
    """Bodies are bounded and prices are cached per provider."""

    def test_json_body_over_limit_is_rejected(self, client):
        """A JSON body above max_json_mb is a 413 before pricing."""
        limit = gateway.config.gateway.max_json_mb
        gateway.config.gateway.max_json_mb = 1
        try:
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                content=b'{"prompt": "' + b"a" * (1024 * 1024 + 10) + b'"}',
                headers={"content-type": "application/json"},
            )
        finally:
            gateway.config.gateway.max_json_mb = limit
        assert response.status_code == 413

    def test_json_body_of_tiny_containers_is_rejected(self, client):
        """A small body that would parse into millions of objects is a 413."""
        body = b'{"a": [' + b"{}," * 200_001 + b"{}]}"
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413

    @respx.mock
    @pytest.mark.parametrize(
        "body",
        [
            b'{"prompt": "x", "seed": ' + b"9" * 41 + b"}",
            b'{"prompt": "x", "n": [1, -' + b"9" * 40 + b"]}",
            # Thousands of digits each: seconds of parsing and re-encoding, unpaid.
            b'{"prompt": "x", "a": [' + b",".join([b"9" * 4000] * 500) + b"]}",
        ],
    )
    def test_long_json_integer_is_refused_before_pricing(self, client, body):
        price = respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid JSON body"
        assert not price.called

    @respx.mock
    def test_large_seed_still_prices(self, client):
        """A 64-bit seed and a 40-character integer are ordinary values."""
        price = respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        forty = "-" + "9" * 39
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            content=f'{{"prompt": "x", "seed": {2**63}, "n": {forty}}}'.encode(),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 402
        inputs = json.loads(price.calls.last.request.content)["inputs"]
        assert inputs["seed"] == 2**63
        assert inputs["n"] == int(forty)

    def test_long_topup_integer_is_a_400(self, client):
        response = client.post(
            "/v1/topup",
            content=b'{"amount": ' + b"1" * 41 + b"}",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid JSON body"

    def test_client_cannot_supply_internal_keys(self, client):
        """_file_bytes sent in JSON or as a form text field is dropped, not priced."""
        response = client.post(
            "/v1/cloudconvert/convert",
            json={"output_format": "pdf", "_file_bytes": "fake", "_file_name": "a.txt"},
        )
        assert response.status_code == 400
        assert "No file provided" in response.json()["error"]

        response = client.post(
            "/v1/cloudconvert/convert",
            data={"output_format": "pdf", "_file_bytes": "fake"},
            files={"other": ("x.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    @respx.mock
    def test_price_cache_is_per_provider(self, client):
        """A cached price under the bare path is not used for this provider."""
        _price_ok()
        gateway.price_cache.set("wavespeed-ai/flux-dev", {"prompt": "a cat"}, Decimal("0"))
        gateway.price_cache.set("other/wavespeed-ai/flux-dev", {"prompt": "a cat"}, Decimal("0"))

        response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 402
        assert response.json()["accepts"][0]["price"] == "$0.004120"


class TestTopupSettlesBeforeCredit:
    """Top-up credits a balance only after settlement succeeds."""

    def test_failed_settle_does_not_credit(self, client):
        """A failed top-up settlement returns 402 and leaves the balance at zero."""
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = TOPUP_SIG
            mock_ph.verify = AsyncMock(return_value=(True, SOLANA_NETWORK, SOL_PAYER, ""))
            mock_ph.settle = AsyncMock(
                return_value={
                    "success": False,
                    "error_reason": "transaction_simulation_failed",
                }
            )
            response = client.post(
                "/v1/topup",
                json={"amount": 1.0},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 402
        assert "transaction_simulation_failed" in response.json()["error"]
        assert get_balance(SOL_PAYER) == 0
        assert mock_ph.settle.await_count == 1

    def test_credits_only_after_successful_settle(self, client):
        """The prepaid balance is still zero while settle is in progress."""
        reset()

        async def settle_side(*args, **kwargs):
            assert get_balance(SOL_PAYER) == 0
            return {
                "success": True,
                "amount_usdc": 1.0,
                "gas_cost_usd": 0.0,
                "gas_cost_native": 0.0,
                "gas_label": "SOL",
            }

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = TOPUP_SIG
            mock_ph.verify = AsyncMock(return_value=(True, SOLANA_NETWORK, SOL_PAYER, ""))
            mock_ph.settle = AsyncMock(side_effect=settle_side)
            response = client.post(
                "/v1/topup",
                json={"amount": 1.0},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 200
        body = response.json()
        assert Decimal(body["credited"]) == Decimal("0.959")
        assert get_balance(SOL_PAYER) == Decimal("0.959")

    def test_signed_amount_above_maximum_is_rejected(self, client):
        """A signed top-up above the limit is refused before settlement."""
        reset()
        payload = base64.b64encode(
            json.dumps({"accepted": {"amount": "500000000"}}).encode()
        ).decode()

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = payload
            mock_ph.verify = AsyncMock(return_value=(True, SOLANA_NETWORK, SOL_PAYER, ""))
            mock_ph.settle = AsyncMock(return_value={"success": True, "amount_usdc": 500})
            response = client.post(
                "/v1/topup",
                json={"amount": 1.0},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 400
        assert "between" in response.json()["error"]
        assert get_balance(SOL_PAYER) == 0
        assert mock_ph.settle.await_count == 0
        assert mock_ph.verify.await_count == 0


def _prepaid_headers(kp: Keypair, path: str) -> dict[str, str]:
    """Sign a prepaid request for path with a fresh timestamp."""
    ts = int(time.time())
    message = f"x402gate:{path}:{ts}".encode()
    return {
        "X-PREPAID-PUBKEY": str(kp.pubkey()),
        "X-PREPAID-SIGNATURE": str(kp.sign_message(message)),
        "X-PREPAID-TIMESTAMP": str(ts),
    }


class TestPrepaidReservation:
    """Prepaid funds are held before the provider runs."""

    @respx.mock
    def test_provider_error_restores_reservation(self, client):
        """A provider failure returns the reserved balance in full."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(500, text="upstream down"))

        with patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock):
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )

        assert response.status_code == 502
        assert get_balance(pubkey) == Decimal("1.00")
        assert response.headers["x-prepaid-cost"] == "0"
        assert response.headers["x-prepaid-balance"] == "1"
        assert response.json()["prepaid"] == {"cost": "0", "balance": "1"}

    @respx.mock
    def test_cost_header_is_the_actual_charge(self, client):
        """X-Prepaid-Cost is the actual cost, and the balance drops by exactly that."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )

        async def _cheaper(body, result):
            return Decimal("0.0012")

        providers["wavespeed"].calculate_actual_cost = _cheaper
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
        )

        assert response.status_code == 200
        assert response.headers["x-prepaid-cost"] == "0.0012"
        cost = Decimal(response.headers["x-prepaid-cost"])
        assert Decimal(response.headers["x-prepaid-balance"]) == Decimal("1.00") - cost
        assert get_balance(pubkey) == Decimal("1.00") - cost
        assert response.json()["prepaid"] == {
            "cost": response.headers["x-prepaid-cost"],
            "balance": response.headers["x-prepaid-balance"],
        }

    @respx.mock
    def test_tiny_cost_is_plain_decimal_and_balances_add_up(self, client):
        """Sub-micro amounts never use exponent notation, and old - cost == new exactly."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("0.61609766143")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )

        async def _tiny(body, result):
            return Decimal("5.8653E-7")

        providers["wavespeed"].calculate_actual_cost = _tiny
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
        )

        assert response.status_code == 200
        assert response.headers["x-prepaid-cost"] == "0.00000058"
        assert response.headers["x-prepaid-balance"] == "0.61609708"
        assert Decimal("0.61609766") - Decimal("0.00000058") == Decimal("0.61609708")
        assert get_balance(pubkey) == Decimal("0.61609766143") - Decimal("0.00000058")

    @respx.mock
    def test_cost_header_falls_back_to_estimate(self, client):
        """Without an actual cost, X-Prepaid-Cost is the estimated base price."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )

        async def _unknown(body, result):
            return None

        providers["wavespeed"].calculate_actual_cost = _unknown
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
        )

        assert response.status_code == 200
        assert Decimal(response.headers["x-prepaid-cost"]) == Decimal("0.003")
        assert get_balance(pubkey) == Decimal("1.00") - Decimal("0.003")

    @respx.mock
    def test_actual_cost_above_reservation_does_not_go_negative(self, client):
        """A result that costs more than the hold still charges only the hold."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("0.002")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.002}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )

        async def _expensive(body, result):
            return Decimal("0.01")

        providers["wavespeed"].calculate_actual_cost = _expensive
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
        )

        assert response.status_code == 200
        assert get_balance(pubkey) == Decimal("0")
        assert Decimal(response.headers["x-prepaid-balance"]) == Decimal("0")
        assert Decimal(response.headers["x-prepaid-cost"]) == Decimal("0.002")

    def _post_long_job(self, client, kp, get_result):
        """POST a prepaid WaveSpeed job that outlives the heartbeat interval."""
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(
                200, json={"data": {"id": "task-1", "status": "processing"}}
            )
        )
        providers["wavespeed"].get_result = get_result
        with patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01):
            return client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )

    @respx.mock
    def test_long_job_reports_the_charge_in_the_body(self, client):
        """A heartbeat response has no prepaid headers, so the body carries them."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        async def _slow(task_id):
            await asyncio.sleep(0.1)
            return {"id": task_id, "status": "completed", "outputs": ["ok"]}

        response = self._post_long_job(client, kp, _slow)

        assert response.status_code == 200
        assert response.content.startswith(b"\n")
        assert "x-prepaid-cost" not in response.headers
        assert "x-prepaid-balance" not in response.headers
        assert response.json()["prepaid"] == {"cost": "0.003", "balance": "0.997"}
        assert get_balance(pubkey) == Decimal("0.997")

    @respx.mock
    def test_long_job_failure_reports_zero_cost_in_the_body(self, client):
        """A long job that fails returns the hold, and the body says nothing was charged."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        async def _slow_failure(task_id):
            await asyncio.sleep(0.1)
            raise ProviderError(provider="wavespeed", detail="boom")

        response = self._post_long_job(client, kp, _slow_failure)

        body = response.json()
        assert body["status"] == 502
        assert "boom" in body["error"]
        assert body["prepaid"] == {"cost": "0", "balance": "1"}
        assert get_balance(pubkey) == Decimal("1.00")

    @respx.mock
    def test_concurrent_requests_spend_the_balance_once(self, client):
        """Five parallel requests on a one-request balance produce one success."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("0.003")))
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )
        headers = _prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev")

        async def _fire():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                return await asyncio.gather(
                    *[
                        http.post(
                            "/v1/wavespeed/wavespeed-ai/flux-dev",
                            json={"prompt": "a cat"},
                            headers=headers,
                        )
                        for _ in range(5)
                    ]
                )

        responses = client.portal.call(_fire)
        codes = [response.status_code for response in responses]
        assert codes.count(200) == 1
        assert codes.count(402) == 4
        assert get_balance(pubkey) >= 0
        assert get_balance(pubkey) == Decimal("0")


def _drain(client) -> None:
    """Wait for jobs the gateway finishes in the background."""

    async def _wait() -> None:
        for _ in range(20):
            pending = [task for task in gateway._pending_jobs if not task.done()]
            if not pending:
                return
            await asyncio.wait(pending, timeout=1)

    client.portal.call(_wait)


def _post_and_leave(
    client, path: str, headers: dict[str, str], *, at_once: bool = False
) -> list[dict]:
    """POST a WaveSpeed job over raw ASGI and disconnect after the first heartbeat.

    With ``at_once`` the client is gone as soon as the body is sent.
    """

    async def _run() -> list[dict]:
        raw = json.dumps({"prompt": "a cat"}).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                *[(k.lower().encode(), v.encode()) for k, v in headers.items()],
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("test", 80),
            "state": {},
        }
        left = asyncio.Event()
        if at_once:
            left.set()
        request_sent = False
        messages: list[dict] = []

        async def receive() -> dict:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await left.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            messages.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                left.set()

        await app(scope, receive, send)
        return messages

    return client.portal.call(_run)


def _submitted_task() -> None:
    """Mock WaveSpeed pricing and a submit that returns a polled task."""
    _price_ok()
    respx.post(WAVESPEED_SUBMIT).mock(
        return_value=httpx.Response(200, json={"data": {"id": "task-1", "status": "processing"}})
    )


async def _slow_success(task_id: str) -> dict:
    await asyncio.sleep(0.2)
    return {"id": task_id, "status": "completed", "outputs": ["ok"]}


async def _slow_failure(task_id: str) -> dict:
    await asyncio.sleep(0.2)
    raise ProviderError(provider="wavespeed", detail="task failed: nsfw")


def _body_text(messages: list[dict]) -> bytes:
    return b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")


class TestUnknownOutcome:
    """Funds go back only when the provider reported the failure itself."""

    @respx.mock
    def test_prepaid_poll_timeout_keeps_the_hold(self, client):
        """The task may still finish and bill us, so the estimate is charged."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        _submitted_task()

        with patch.object(
            providers["wavespeed"],
            "get_result",
            AsyncMock(side_effect=TaskTimeoutError("task-1", 10)),
        ):
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )

        assert response.status_code == 504
        body = response.json()
        assert body["outcome_unknown"] is True
        assert body["prepaid"] == {"cost": "0.003", "balance": "0.997"}
        assert response.headers["x-prepaid-cost"] == "0.003"
        assert get_balance(pubkey) == Decimal("0.997")

    @respx.mock
    def test_prepaid_submit_read_timeout_keeps_the_hold(self, client):
        """A submit that timed out after sending may have started the job."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(side_effect=httpx.ReadTimeout("slow"))

        with patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock):
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )

        assert response.json()["outcome_unknown"] is True
        assert response.json()["prepaid"]["cost"] == "0.003"
        assert get_balance(pubkey) == Decimal("0.997")

    @respx.mock
    def test_prepaid_connect_error_returns_the_hold(self, client):
        """A connection that never opened cannot have started anything."""
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(side_effect=httpx.ConnectError("refused"))

        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json={"prompt": "a cat"},
            headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
        )

        assert "outcome_unknown" not in response.json()
        assert response.json()["prepaid"]["cost"] == "0"
        assert get_balance(pubkey) == Decimal("1.00")

    @respx.mock
    def test_base_poll_timeout_settles(self, client):
        """On Base, keeping the payment means settling the authorization."""
        reset()
        _submitted_task()

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch.object(
                providers["wavespeed"],
                "get_result",
                AsyncMock(side_effect=TaskTimeoutError("task-1", 10)),
            ),
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)
            _drain(client)

        assert response.status_code == 504
        assert response.json()["outcome_unknown"] is True
        assert mock_ph.settle.await_count == 1
        assert mock_ph.settle.await_args.args[1] == apply_commission(Decimal("0.003"), 0.04, 0.001)


class TestClientDisconnect:
    """A client that leaves mid-job does not stop the job or its billing."""

    @respx.mock
    def test_prepaid_success_after_leaving_is_charged(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        _submitted_task()

        with (
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01),
            patch.object(providers["wavespeed"], "get_result", side_effect=_slow_success),
        ):
            messages = _post_and_leave(
                client,
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                _prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )
            _drain(client)

        assert messages[0]["status"] == 200
        assert b'"data"' not in _body_text(messages)
        assert get_balance(pubkey) == Decimal("0.997")

    @respx.mock
    def test_prepaid_failure_after_leaving_is_returned(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        _submitted_task()

        with (
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01),
            patch.object(providers["wavespeed"], "get_result", side_effect=_slow_failure),
        ):
            _post_and_leave(
                client,
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                _prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
            )
            _drain(client)

        assert get_balance(pubkey) == Decimal("1.00")

    @respx.mock
    def test_base_success_after_leaving_settles(self, client):
        reset()
        _submitted_task()

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01),
            patch.object(providers["wavespeed"], "get_result", side_effect=_slow_success),
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            _post_and_leave(
                client, "/v1/wavespeed/wavespeed-ai/flux-dev", {"Payment-Signature": "fake-sig"}
            )
            _drain(client)

        assert mock_ph.settle.await_count == 1

    @respx.mock
    def test_base_client_that_leaves_during_settle_still_gets_the_job_run(self, client):
        """Leaving does not stop the settle, and the job runs once it settles."""
        reset()
        _price_ok()
        seen: dict[str, int] = {}

        def on_submit(request):
            seen["settle_before_submit"] = mock_ph.settle.await_count
            return httpx.Response(200, json={"outputs": ["ok"]})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)

        async def slow_settle(*args, **kwargs):
            await asyncio.sleep(0.05)
            return _settled_ok("ETH")

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(side_effect=slow_settle)
            _post_and_leave(
                client,
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                {"Payment-Signature": "fake-sig"},
                at_once=True,
            )
            _drain(client)

        assert seen["settle_before_submit"] == 1
        assert _in_flight() == 0

    @respx.mock
    def test_base_settle_stopped_midway_holds_the_payment(self, client):
        """A cancelled request leaves the settle running; a settle stopped midway holds all."""
        reset()
        _price_ok()
        submit = respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)

        async def _cancel_mid_settle() -> None:
            settling = asyncio.Event()

            async def slow_settle(*args, **kwargs):
                settling.set()
                await asyncio.sleep(30)

            mock_ph.settle = AsyncMock(side_effect=slow_settle)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                request = asyncio.create_task(
                    http.post(
                        "/v1/wavespeed/wavespeed-ai/flux-dev",
                        json={"prompt": "a cat"},
                        headers={"Payment-Signature": "fake-sig"},
                    )
                )
                await settling.wait()
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)
                # The settle runs in the job task, which the request does not own.
                jobs = [task for task in gateway._pending_jobs if not task.done()]
                assert len(jobs) == 1
                # Stopped midway (a crash or a cancel), it may have broadcast.
                jobs[0].cancel()
                await asyncio.gather(*jobs, return_exceptions=True)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            client.portal.call(_cancel_mid_settle)

        assert not submit.called
        # "fake-sig" carries no validBefore, so the configured max_timeout applies.
        assert _held_for("fake-sig") > 3600
        assert _in_flight() == price
        assert len(gateway._pending_reconciles) == 1

    @respx.mock
    def test_solana_failure_after_leaving_is_credited(self, client):
        reset()
        _submitted_task()

        with (
            patch("x402gate.app.payment_handler") as mock_ph,
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01),
            patch.object(providers["wavespeed"], "get_result", side_effect=_slow_failure),
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            mock_ph.settle = AsyncMock(return_value=_settled_ok())
            _post_and_leave(
                client, "/v1/wavespeed/wavespeed-ai/flux-dev", {"Payment-Signature": "fake-sig"}
            )
            _drain(client)

        assert get_balance(SOL_PAYER) == apply_commission(Decimal("0.003"), 0.04, 0.001)


class TestRequestGuards:
    """Cheap request-shape and upstream-failure guards."""

    def test_json_array_is_rejected(self, client):
        response = client.post(
            "/v1/wavespeed/wavespeed-ai/flux-dev",
            json=["not", "an", "object"],
        )
        assert response.status_code == 400
        assert response.json()["error"] == "JSON body must be an object"

    @respx.mock
    def test_passthrough_connection_error_is_502(self, client):
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        response = client.post(
            "/v1/blockrun/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": []},
        )
        assert response.status_code == 502
        assert response.json()["error"] == "Upstream request failed"

    @respx.mock
    def test_passthrough_timeout_is_504(self, client):
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("slow")
        )
        response = client.post(
            "/v1/blockrun/v1/chat/completions",
            json={"model": "openai/gpt-4o", "messages": []},
        )
        assert response.status_code == 504
        assert response.json()["error"] == "Upstream request timed out"

    @respx.mock
    def test_evm_rpc_outage_does_not_call_provider(self, client):
        """A Base balance RPC failure stops the request before the provider."""
        respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            mock_ph.extract_payment_signature.return_value = "fake-sig"
            mock_ph.verify = AsyncMock(return_value=(True, EVM_NETWORK, "0xPayer", ""))
            mock_ph.check_evm_balance = AsyncMock(return_value=None)
            mock_ph.settle = AsyncMock()
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev",
                json={"prompt": "a cat"},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.status_code == 503
        assert response.json()["error"] == "Payment network unavailable, retry"
        submit_calls = [
            call for call in respx.calls if str(call.request.url).endswith("wavespeed-ai/flux-dev")
        ]
        assert submit_calls == []
        assert mock_ph.settle.await_count == 0


class TestWarmUp:
    """Startup loads provider catalogs in the background."""

    def test_managed_providers_warm_up_and_a_failure_is_ignored(self, config_file):
        warmed: list[str] = []

        async def _fake_warm_up(self):
            warmed.append(type(self).__name__)
            if type(self).__name__ == "OpenRouterProvider":
                raise RuntimeError("catalog down")

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("x402gate.app.load_config", return_value=real_load(config_file))
            )
            stack.enter_context(
                patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={}))
            )
            for cls in {*gateway.PROVIDER_REGISTRY.values(), BaseProvider}:
                stack.enter_context(patch.object(cls, "warm_up", _fake_warm_up))
            with TestClient(app) as c:
                deadline = time.monotonic() + 2
                while len(warmed) < 4 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert c.get("/health").status_code == 200

        # Enabled managed providers only; blockrun is a passthrough.
        assert sorted(warmed) == [
            "CloudConvertProvider",
            "FishAudioProvider",
            "OpenRouterProvider",
            "WaveSpeedProvider",
        ]


class TestSharedUpstreamClient:
    """Upstream calls reuse pooled connections instead of a new handshake each time."""

    def test_passthrough_uses_a_pooled_client(self, client):
        upstream = AsyncMock()
        upstream.post.return_value = httpx.Response(200, json={"ok": True})
        with patch("x402gate.app._passthrough_client", return_value=upstream):
            for _ in range(2):
                response = client.post(
                    "/v1/blockrun/v1/chat/completions",
                    json={"model": "openai/gpt-4o", "messages": []},
                    headers={"Payment-Signature": "sig"},
                )
                assert response.status_code == 200
                assert response.json() == {"ok": True}

        assert upstream.post.await_count == 2
        call = upstream.post.await_args
        assert call.args[0] == "https://blockrun.ai/api/v1/chat/completions"
        assert call.kwargs["headers"]["payment-signature"] == "sig"
        # Same 120 s read limit as before; a dead host fails within 10 s.
        assert call.kwargs["timeout"].read == 120.0
        assert call.kwargs["timeout"].connect == 10.0
        # The pool belongs to every caller, so a request never closes it.
        upstream.aclose.assert_not_called()
        upstream.__aexit__.assert_not_called()

    @respx.mock
    def test_passthrough_reuses_one_client_and_closes_it_on_shutdown(self, config_file):
        respx.post("https://blockrun.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        seen: list[httpx.AsyncClient] = []
        real = gateway._passthrough_client

        def _spy() -> httpx.AsyncClient:
            seen.append(real())
            return seen[-1]

        with (
            patch("x402gate.app.load_config", return_value=real_load(config_file)),
            patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={})),
            patch("x402gate.app._warm_up", new=AsyncMock()),
            patch("x402gate.app._passthrough_client", side_effect=_spy),
            TestClient(app) as c,
        ):
            for _ in range(2):
                response = c.post(
                    "/v1/blockrun/v1/chat/completions",
                    json={"model": "openai/gpt-4o", "messages": []},
                )
                assert response.status_code == 200

        assert len(seen) == 2 and seen[0] is seen[1]
        assert seen[0].is_closed

    @respx.mock
    def test_passthrough_does_not_carry_cookies_between_users(self, client):
        """A Set-Cookie for one user must not reach the next user's request."""
        route = respx.post("https://blockrun.ai/api/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json={"ok": 1}, headers={"Set-Cookie": "session=userA; Path=/"}),
            httpx.Response(200, json={"ok": 2}),
        ]

        for sig in ("sig-a", "sig-b"):
            response = client.post(
                "/v1/blockrun/v1/chat/completions",
                json={"model": "openai/gpt-4o", "messages": []},
                headers={"Payment-Signature": sig},
            )
            assert response.status_code == 200

        assert route.call_count == 2
        second = route.calls[1].request
        assert second.headers["payment-signature"] == "sig-b"
        assert "cookie" not in second.headers

    def test_passthrough_pool_is_separate_from_the_managed_pool(self):
        async def _clients():
            from x402gate.core.http import close_shared_client, shared_client

            try:
                return gateway._passthrough_client(), shared_client()
            finally:
                await gateway._close_passthrough_client()
                await close_shared_client()

        passthrough, managed = asyncio.run(_clients())
        # Long passthrough calls cannot use up the managed providers' connections.
        assert passthrough is not managed

    def test_shutdown_waits_for_cancelled_warm_ups_before_closing_pools(self, config_file):
        events: list[str] = []
        started = threading.Event()

        async def _slow_warm_up(self):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                events.append("warm-up unwound")
                raise

        async def _close():
            events.append("pool closed")

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("x402gate.app.load_config", return_value=real_load(config_file))
            )
            stack.enter_context(
                patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={}))
            )
            stack.enter_context(patch("x402gate.app.close_shared_client", new=_close))
            for cls in {*gateway.PROVIDER_REGISTRY.values(), BaseProvider}:
                stack.enter_context(patch.object(cls, "warm_up", _slow_warm_up))
            with TestClient(app):
                assert started.wait(2)

        assert events[-1] == "pool closed"
        assert events.count("warm-up unwound") >= 1
        assert "pool closed" not in events[:-1]


class TestPriceCacheOnRequests:
    """The price cache hashes each body once and serves repeats without a quote."""

    @pytest.fixture
    def cache(self, monkeypatch):
        cache = PriceCache(ttl=60)
        monkeypatch.setattr(gateway, "price_cache", cache)
        return cache

    @respx.mock
    def test_repeat_prepaid_request_skips_the_quote(self, client, cache):
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        price_route = respx.post(WAVESPEED_PRICE).mock(
            return_value=httpx.Response(200, json={"data": {"unit_price": 0.003}})
        )
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"data": {"output": {"images": ["ok"]}}})
        )

        with patch.object(PriceCache, "_make_key", wraps=PriceCache._make_key) as make_key:
            for _ in range(2):
                response = client.post(
                    "/v1/wavespeed/wavespeed-ai/flux-dev",
                    json={"prompt": "a cat"},
                    headers=_prepaid_headers(kp, "wavespeed/wavespeed-ai/flux-dev"),
                )
                assert response.status_code == 200
                assert response.json()["prepaid"]["cost"] == "0.003"

        assert price_route.call_count == 1
        # One hash per request, of the raw JSON bytes.
        assert make_key.call_count == 2
        assert all(isinstance(call.args[2], bytes) for call in make_key.call_args_list)
        assert get_balance(str(kp.pubkey())) == Decimal("0.994")

    @respx.mock
    def test_large_body_is_hashed_off_the_event_loop(self, client, cache, monkeypatch):
        monkeypatch.setattr(gateway, "_HASH_IN_THREAD_BYTES", 10)
        _price_ok()
        hash_threads: list[int] = []
        loop_threads: list[int] = []
        make_key = PriceCache._make_key
        entry_by_key = PriceCache.entry_by_key

        def _make_key(*args):
            hash_threads.append(threading.get_ident())
            return make_key(*args)

        def _entry_by_key(self, key):
            loop_threads.append(threading.get_ident())
            return entry_by_key(self, key)

        with (
            patch.object(PriceCache, "_make_key", side_effect=_make_key),
            patch.object(PriceCache, "entry_by_key", _entry_by_key),
        ):
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat " * 10}
            )

        assert response.status_code == 402
        assert len(hash_threads) == 1 and len(loop_threads) == 1
        assert hash_threads[0] != loop_threads[0]


def _topup(client):
    """POST a $1 top-up with a payment header."""
    return client.post(
        "/v1/topup",
        json={"amount": 1.0},
        headers={"Payment-Signature": "fake-sig"},
    )


def _topup_handler(mock_ph, network: str = SOLANA_NETWORK, payer: str = SOL_PAYER) -> None:
    """Point a mocked payment handler at a verified $1 top-up."""
    mock_ph.extract_payment_signature.return_value = TOPUP_SIG
    mock_ph.verify = AsyncMock(return_value=(True, network, payer, ""))
    mock_ph.check_evm_balance = AsyncMock(return_value=True)


def _held_for(payment_sig: str) -> float:
    """Seconds left on a payment's claim (inf while a request holds it)."""
    until = gateway.payment_claims._claims[payment_key(payment_sig)]
    return until - time.monotonic()


class TestTopupSettlementOutcomes:
    """Top-up claims its payment and never tells a payer who paid to pay again."""

    @pytest.mark.parametrize(
        "raw", [b'{"amount": NaN}', b'{"amount": "sNaN"}', b'{"amount": "NaN"}']
    )
    def test_nan_amount_is_a_400(self, client, raw):
        response = client.post(
            "/v1/topup", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"] == "Invalid amount value"

    @pytest.mark.parametrize("raw", [b'{"amount": Infinity}', b'{"amount": "-Infinity"}'])
    def test_infinite_amount_is_a_400(self, client, raw):
        response = client.post(
            "/v1/topup", content=raw, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400

    def test_pending_settlement_credits_and_holds_the_payment(self, client):
        """A broadcast whose confirmation wait failed most likely lands: credit it."""
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph)
            mock_ph.settle = AsyncMock(
                return_value={
                    "success": False,
                    "pending": True,
                    "error_reason": "settlement_pending",
                    "transaction": "5abc",
                }
            )
            response = _topup(client)
            replay = _topup(client)

        assert response.status_code == 200
        body = response.json()
        assert body["settlement_pending"] is True
        assert Decimal(body["credited"]) == Decimal("0.959")
        assert get_balance(SOL_PAYER) == Decimal("0.959")
        # The same payment cannot credit a second time.
        assert replay.status_code == 409
        assert mock_ph.settle.await_count == 1

    def test_crashed_settle_credits_nothing_and_says_not_to_pay_again(self, client):
        """settle returned None: maybe sent, maybe not. No credit, claim kept."""
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph)
            mock_ph.settle = AsyncMock(return_value=None)
            response = _topup(client)
            replay = _topup(client)

        assert response.status_code == 502
        body = response.json()
        assert body["outcome_unknown"] is True
        assert "Do not sign a new payment" in body["error"]
        assert get_balance(SOL_PAYER) == 0
        assert replay.status_code == 409
        assert mock_ph.settle.await_count == 1

    def test_payment_in_use_is_not_settled_again(self, client):
        """A copy of a payment another request holds gets a 409 before settle."""
        reset()
        gateway.payment_claims.claim(payment_key(TOPUP_SIG))
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _topup(client)

        assert response.status_code == 409
        assert "Sign a new payment" not in response.json()["error"]
        assert mock_ph.settle.await_count == 0
        assert get_balance("0xpayer") == 0

    def test_parallel_copies_settle_once(self, client):
        """Copies sent while the first settle runs never reach settle."""
        reset()
        started = threading.Event()
        finish = threading.Event()
        responses: list[httpx.Response] = []

        async def slow_settle(*args, **kwargs):
            started.set()
            await asyncio.to_thread(finish.wait, 5)
            return _settled_ok("ETH")

        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(side_effect=slow_settle)
            first = threading.Thread(target=lambda: responses.append(_topup(client)))
            first.start()
            assert started.wait(5)
            copy = _topup(client)
            finish.set()
            first.join(5)

        assert copy.status_code == 409
        assert responses[0].status_code == 200
        assert mock_ph.settle.await_count == 1

    def test_settled_topup_holds_the_payment(self, client):
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            first = _topup(client)
            replay = _topup(client)

        assert first.status_code == 200
        assert "settlement_pending" not in first.json()
        assert replay.status_code == 409
        assert 0 < _held_for(TOPUP_SIG) <= gateway._BASE_CLAIM_HOLD_S

    def test_failed_settle_frees_the_payment(self, client):
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph)
            mock_ph.settle = AsyncMock(
                side_effect=[{"success": False, "error_reason": "expired"}, _settled_ok()]
            )
            first = _topup(client)
            second = _topup(client)

        assert first.status_code == 402
        assert second.status_code == 200
        assert get_balance(SOL_PAYER) == Decimal("0.959")


def _evm_topup_sig(nonce: str) -> str:
    """A $1 Base top-up header from 0xPayer, one per nonce."""
    payload = {
        "accepted": {"network": EVM_NETWORK, "amount": "1000000"},
        "payload": {"authorization": {"from": "0xPayer", "nonce": nonce}},
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _pending_settle(tx_hash: str = "0xaa") -> dict:
    """A Base settle whose confirmation wait failed after broadcast."""
    return {
        "success": False,
        "pending": True,
        "error_reason": "settlement_pending",
        "transaction": tx_hash,
    }


def _receipt_settlement(tx_hash: str, landed: bool) -> dict:
    """What PaymentHandler.settlement_receipt returns once a receipt is found."""
    return {
        "success": landed,
        "transaction": tx_hash,
        "error_reason": None if landed else "transaction_failed: no matching transfer",
        "amount_usdc": 1.0,
        "gas_cost_native": 0.00001,
        "gas_cost_usd": 0.0,
        "gas_label": "ETH",
    }


def _in_flight(payer: str = "0xpayer") -> Decimal:
    """USDC base_in_flight counts for a payer."""
    return gateway.base_in_flight.add(payer, Decimal("0"))


def _wait_for_reconciles(timeout: float = 5.0) -> None:
    """Wait until every reconcile task running in the app's loop has finished."""
    deadline = time.monotonic() + timeout
    while gateway._pending_reconciles:
        assert time.monotonic() < deadline, "reconcile did not finish"
        time.sleep(0.01)


class TestUnknownBaseSettlement:
    """A Base settlement with an unknown outcome is credited only once it lands."""

    @pytest.fixture
    def fast_poll(self, monkeypatch):
        monkeypatch.setattr(gateway, "_RECONCILE_FIRST_POLL_S", 0.01)
        monkeypatch.setattr(gateway, "_RECONCILE_MAX_POLL_S", 0.02)

    def test_two_pending_topups_over_one_balance_credit_once(self, client, fast_poll):
        """Both pass the balance check; only the one whose receipt lands is credited."""
        reset()
        answer = threading.Event()
        sigs = iter([_evm_topup_sig("0x01"), _evm_topup_sig("0x02")])

        async def receipt(network, tx_hash, payer, amount):
            await asyncio.to_thread(answer.wait, 5)
            return _receipt_settlement(tx_hash, landed=tx_hash == "0xaa")

        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            lookup = mock_ph.settlement_receipt = AsyncMock(side_effect=receipt)
            mock_ph.extract_payment_signature.side_effect = lambda request: next(sigs)
            mock_ph.settle = AsyncMock(
                side_effect=[_pending_settle("0xaa"), _pending_settle("0xbb")]
            )
            first = _topup(client)
            second = _topup(client)
            # Nothing is credited while the transactions are unconfirmed.
            assert get_balance("0xpayer") == 0
            assert _in_flight() == Decimal("2")
            answer.set()
            _wait_for_reconciles()

        for response, tx_hash in ((first, "0xaa"), (second, "0xbb")):
            assert response.status_code == 202
            body = response.json()
            assert body["settlement_pending"] is True
            assert body["credited"] == "0"
            assert Decimal(body["pending_credit"]) == Decimal("0.959")
            assert body["transaction"] == tx_hash
            assert "Do not sign a new payment" in body["message"]
        assert {call.args[1] for call in lookup.await_args_list} == {"0xaa", "0xbb"}
        # The reverted duplicate never credits.
        assert get_balance("0xpayer") == Decimal("0.959")
        assert _in_flight() == 0

    def test_second_topup_while_first_is_unknown_gets_402(self, client):
        """The unknown amount stays counted, so a new nonce over it fails the check."""
        reset()
        sigs = iter([_evm_topup_sig("0x01"), _evm_topup_sig("0x02")])

        async def balance(network, payer, amount):
            return amount <= Decimal("1")

        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = AsyncMock(return_value=None)
            mock_ph.extract_payment_signature.side_effect = lambda request: next(sigs)
            mock_ph.check_evm_balance = AsyncMock(side_effect=balance)
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            first = _topup(client)
            second = _topup(client)

            assert first.status_code == 202
            assert second.status_code == 402
            assert "still settling" in second.json()["error"]
            assert mock_ph.check_evm_balance.await_args.args[2] == Decimal("2")
            assert mock_ph.settle.await_count == 1
            # The refused top-up takes back only its own amount.
            assert _in_flight() == Decimal("1")
        assert get_balance("0xpayer") == 0

    def test_topup_short_of_usdc_is_402_before_settle(self, client):
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.check_evm_balance = AsyncMock(return_value=False)
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _topup(client)
            retry = _topup(client)

        assert response.status_code == 402
        assert response.json()["error"] == "Insufficient USDC balance"
        assert mock_ph.settle.await_count == 0
        # A refused check frees the payment for a retry.
        assert retry.status_code == 402
        assert _in_flight() == 0

    def test_topup_balance_check_unavailable_is_503(self, client):
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.check_evm_balance = AsyncMock(return_value=None)
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _topup(client)

        assert response.status_code == 503
        assert mock_ph.settle.await_count == 0
        assert _in_flight() == 0

    def test_settled_and_failed_topups_release_the_in_flight_amount(self, client):
        reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.extract_payment_signature.side_effect = iter(
                [_evm_topup_sig("0x01"), _evm_topup_sig("0x02")]
            )
            mock_ph.settle = AsyncMock(
                side_effect=[_settled_ok("ETH"), {"success": False, "error_reason": "x"}]
            )
            settled = _topup(client)
            failed = _topup(client)

        assert settled.status_code == 200
        assert failed.status_code == 402
        assert mock_ph.check_evm_balance.await_args_list[0].args[2] == Decimal("1")
        assert mock_ph.check_evm_balance.await_args_list[1].args[2] == Decimal("1")
        assert _in_flight() == 0
        assert gateway._pending_reconciles == set()

    def test_landed_topup_records_the_settlement(self, client, fast_poll):
        reset()
        gateway.stats.reset()
        landed = _receipt_settlement("0xaa", landed=True)
        landed["gas_cost_usd"] = 0.02
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = AsyncMock(return_value=landed)
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            response = _topup(client)
            _wait_for_reconciles()

        assert response.status_code == 202
        assert get_balance("0xpayer") == Decimal("0.959")
        snapshot = gateway.stats.get_stats()
        assert snapshot["providers"]["topup"]["revenue_usd"] == "1.0000"
        # The late receipt's gas is recorded, not 0.
        assert snapshot["providers"]["topup"]["gas_usd"] == "0.0200"

    @respx.mock
    def test_unknown_job_settlement_stays_in_flight_until_the_receipt(self, client, fast_poll):
        """A later job's balance check counts the unknown amount until it resolves."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)
        landed = threading.Event()
        sigs = iter([_evm_sig(None), "second-sig"])

        async def receipt(network, tx_hash, payer, amount):
            assert amount == price
            return _receipt_settlement(tx_hash, landed=True) if landed.is_set() else None

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = AsyncMock(side_effect=receipt)
            mock_ph.extract_payment_signature.side_effect = lambda request: next(sigs)
            mock_ph.settle = AsyncMock(side_effect=[_pending_settle(), _settled_ok("ETH")])
            first = _post_wavespeed(client)
            assert _in_flight() == price
            second = _post_wavespeed(client)
            assert mock_ph.check_evm_balance.await_args.args[2] == price * 2
            assert _in_flight() == price
            landed.set()
            _wait_for_reconciles()

        assert first.status_code == 200
        assert second.status_code == 200
        assert _in_flight() == 0

    @contextlib.contextmanager
    def _unknown_job(self, client, receipt):
        """POST a WaveSpeed job whose Base settle stays pending; wait for the reconcile on exit."""
        reset()
        gateway.stats.reset()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            lookup = mock_ph.settlement_receipt = AsyncMock(side_effect=receipt)
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            response = _post_wavespeed(client)
            yield SimpleNamespace(response=response, balance=get_balance("0xpayer"), lookup=lookup)
            _wait_for_reconciles()

    @respx.mock
    def test_known_failure_is_credited_when_the_transfer_lands_later(self, client, fast_poll):
        """The response promises the refund; the reconcile pays it once the receipt lands."""
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, json={"m": "bad"}))
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)
        landed = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=True) if landed.is_set() else None

        with self._unknown_job(client, receipt) as job:
            assert _in_flight() == price
            landed.set()
        response = job.response

        assert response.status_code == 502
        body = response.json()
        assert "outcome_unknown" not in body
        assert "refunded_to_prepaid" not in body
        assert body["settlement_pending"] is True
        assert Decimal(body["pending_refund"]) == price
        assert body["transaction"] == "0xaa"
        assert job.balance == 0
        assert get_balance("0xpayer") == price
        assert gateway.stats._stats.total_refund_credit_usd == price
        assert _in_flight() == 0

    @respx.mock
    def test_transfer_that_landed_before_the_failure_is_credited_in_the_response(
        self, client, fast_poll
    ):
        """The reconcile resolves while the job still runs; the error itself refunds."""
        _price_ok()
        price = apply_commission(Decimal("0.003"), 0.04, 0.001)
        lookups: list[str] = []

        async def receipt(network, tx_hash, payer, amount):
            lookups.append(tx_hash)
            return _receipt_settlement(tx_hash, landed=True)

        async def on_submit(request):
            # Fail only after the reconcile has seen the transfer and finished.
            deadline = time.monotonic() + 5
            while not lookups or gateway._pending_reconciles:
                assert time.monotonic() < deadline, "reconcile did not finish"
                await asyncio.sleep(0.01)
            return httpx.Response(400, json={"m": "bad"})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)
        with self._unknown_job(client, receipt) as job:
            pass
        response = job.response

        assert response.status_code == 502
        body = response.json()
        assert Decimal(body["refunded_to_prepaid"]) == price
        assert Decimal(body["prepaid_balance"]) == price
        assert "pending_refund" not in body
        assert job.balance == price
        # Credited exactly once, whichever side came second.
        assert get_balance("0xpayer") == price
        assert gateway.stats._stats.total_refund_credit_usd == price
        assert lookups == ["0xaa"]

    @respx.mock
    def test_known_failure_with_a_reverted_transfer_credits_nothing(self, client, fast_poll):
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, json={"m": "bad"}))

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=False)

        with self._unknown_job(client, receipt) as job:
            pass
        response, lookup = job.response, job.lookup

        assert response.status_code == 502
        assert lookup.await_count >= 1
        assert get_balance("0xpayer") == 0
        assert gateway.stats._stats.total_refund_credit_usd == 0
        assert gateway.stats.get_stats()["providers"]["wavespeed"]["revenue_usd"] == "0.0000"
        assert _in_flight() == 0

    def _fail_after_reconcile(self, lookups: list[str]):
        """A WaveSpeed submit that fails with a 400 only once the reconcile has ended."""

        async def on_submit(request):
            deadline = time.monotonic() + 5
            while not lookups or gateway._pending_reconciles:
                assert time.monotonic() < deadline, "reconcile did not finish"
                await asyncio.sleep(0.01)
            return httpx.Response(400, json={"m": "bad"})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)

    @respx.mock
    def test_transfer_that_reverted_before_the_failure_promises_no_refund(
        self, client, fast_poll, caplog
    ):
        """The reconcile already saw the revert: nothing will follow, so nothing is promised."""
        _price_ok()
        lookups: list[str] = []

        async def receipt(network, tx_hash, payer, amount):
            lookups.append(tx_hash)
            return _receipt_settlement(tx_hash, landed=False)

        self._fail_after_reconcile(lookups)
        with self._unknown_job(client, receipt) as job:
            pass
        response = job.response

        assert response.status_code == 502
        body = response.json()
        assert "settlement_pending" not in body
        assert "pending_refund" not in body
        assert "refunded_to_prepaid" not in body
        assert get_balance("0xpayer") == 0
        assert gateway.stats._stats.total_refund_credit_usd == 0
        assert any("reverted; nothing charged" in r.getMessage() for r in caplog.records)
        assert not any("refund waits" in r.getMessage() for r in caplog.records)

    @respx.mock
    def test_transfer_that_expired_before_the_failure_promises_no_refund(
        self, client, fast_poll, monkeypatch
    ):
        """The reconcile hit its deadline first: the refund would never be paid."""
        _price_ok()
        monkeypatch.setattr(gateway, "_claim_hold_s", lambda *args: 0.05)
        lookups: list[str] = []

        async def receipt(network, tx_hash, payer, amount):
            lookups.append(tx_hash)
            return None

        self._fail_after_reconcile(lookups)
        with self._unknown_job(client, receipt) as job:
            pass
        response = job.response

        assert response.status_code == 502
        body = response.json()
        assert "settlement_pending" not in body
        assert "pending_refund" not in body
        assert get_balance("0xpayer") == 0
        assert _in_flight() == 0

    @respx.mock
    def test_unknown_outcome_keeps_a_payment_that_lands(self, client, fast_poll):
        """A 504 may have run the job, so the landed payment is kept as revenue."""
        _submitted_task()
        landed = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=True) if landed.is_set() else None

        timeout = AsyncMock(side_effect=TaskTimeoutError("task-1", 10))
        with (
            patch.object(providers["wavespeed"], "get_result", timeout),
            self._unknown_job(client, receipt) as job,
        ):
            landed.set()
        response = job.response

        assert response.status_code == 504
        body = response.json()
        assert body["outcome_unknown"] is True
        assert "pending_refund" not in body
        assert get_balance("0xpayer") == 0
        assert gateway.stats._stats.total_refund_credit_usd == 0
        wavespeed = gateway.stats.get_stats()["providers"]["wavespeed"]
        assert wavespeed["revenue_usd"] == "1.0000"

    @respx.mock
    def test_successful_job_is_never_refunded_when_its_payment_lands(self, client, fast_poll):
        """The result went out, so a landed payment is revenue, never a refund."""
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )
        landed = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=True) if landed.is_set() else None

        with self._unknown_job(client, receipt) as job:
            landed.set()
        response = job.response

        assert response.status_code == 200
        assert get_balance("0xpayer") == 0
        assert gateway.stats._stats.total_refund_credit_usd == 0
        wavespeed = gateway.stats.get_stats()["providers"]["wavespeed"]
        assert wavespeed["revenue_usd"] == "1.0000"
        assert wavespeed["cost_usd"] == "0.0030"

    @contextlib.contextmanager
    def _app(self, config_file, lookup):
        """The app started and stopped around the block, its receipt lookups by ``lookup``."""
        with (
            patch("x402gate.app.load_config", return_value=real_load(config_file)),
            patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={})),
            patch("x402gate.app._warm_up", new=AsyncMock()),
            patch(_RECEIPT_LOOKUP, new=lookup),
        ):
            gateway.payment_claims.clear()
            gateway.base_in_flight.clear()
            with TestClient(app) as c:
                yield c

    def _open_topup(self, c) -> None:
        """A Base top-up whose settlement is unknown, left reconciling."""
        with patch("x402gate.app.payment_handler") as mock_ph:
            _topup_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = gateway.PaymentHandler.settlement_receipt
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            response = _topup(c)
        assert response.status_code == 202
        assert len(gateway._pending_reconciles) == 1

    def test_shutdown_stops_open_reconciles_and_logs_them(self, config_file, caplog):
        reset()
        lookup = AsyncMock(return_value=None)
        with self._app(config_file, lookup) as c:
            self._open_topup(c)
            polls = lookup.await_count

        assert gateway._pending_reconciles == set()
        # One last lookup at shutdown.
        assert lookup.await_count == polls + 1
        assert _in_flight() == 0
        assert get_balance("0xpayer") == 0
        assert any("not reconciled" in r.getMessage() for r in caplog.records)

    def test_shutdown_logs_a_transfer_that_landed_as_owed(self, config_file, caplog):
        """The credit dies with the process, so the log names the payer as owed."""
        reset()
        stopping = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=True) if stopping.is_set() else None

        with self._app(config_file, AsyncMock(side_effect=receipt)) as c:
            self._open_topup(c)
            stopping.set()

        assert _in_flight() == 0
        assert get_balance("0xpayer") == 0
        owed = [r.getMessage() for r in caplog.records if "is owed" in r.getMessage()]
        assert owed and "0xaa" in owed[0] and "$1" in owed[0]

    def _open_job(self, c) -> httpx.Response:
        """A WaveSpeed job whose Base settle stays pending, its reconcile left running."""
        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settlement_receipt = gateway.PaymentHandler.settlement_receipt
            mock_ph.settle = AsyncMock(return_value=_pending_settle())
            response = _post_wavespeed(c)
        assert len(gateway._pending_reconciles) == 1
        return response

    @pytest.mark.parametrize(
        ("submit", "status", "is_owed"),
        [
            # Delivered: the landed payment is revenue.
            (httpx.Response(200, json={"outputs": ["ok"]}), 200, False),
            # Unknown outcome: the payment is kept.
            (httpx.ReadTimeout("slow"), 502, False),
            # Known failure: the landed payment is owed back.
            (httpx.Response(400, json={"m": "bad"}), 502, True),
        ],
        ids=["delivered", "outcome-unknown", "known-failure"],
    )
    @respx.mock
    def test_shutdown_logs_a_landed_job_payment_as_owed_only_after_a_known_failure(
        self, config_file, caplog, submit, status, is_owed
    ):
        reset()
        _price_ok()
        if isinstance(submit, Exception):
            respx.post(WAVESPEED_SUBMIT).mock(side_effect=submit)
        else:
            respx.post(WAVESPEED_SUBMIT).mock(return_value=submit)
        stopping = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            return _receipt_settlement(tx_hash, landed=True) if stopping.is_set() else None

        with self._app(config_file, AsyncMock(side_effect=receipt)) as c:
            response = self._open_job(c)
            stopping.set()

        assert response.status_code == status
        if status == 502 and not is_owed:
            assert response.json()["outcome_unknown"] is True
        assert _in_flight() == 0
        messages = [r.getMessage() for r in caplog.records]
        owed = [m for m in messages if "is owed" in m]
        kept = [m for m in messages if "nothing owed" in m and "kept" in m]
        if is_owed:
            assert owed and "0xaa" in owed[0]
            assert not kept
        else:
            assert not owed
            assert kept and "0xaa" in kept[0]

    def test_shutdown_last_lookup_is_bounded(self, config_file, caplog, monkeypatch):
        reset()
        monkeypatch.setattr(gateway, "_RECONCILE_LAST_LOOKUP_S", 0.05)
        stopping = threading.Event()

        async def receipt(network, tx_hash, payer, amount):
            if stopping.is_set():
                await asyncio.sleep(30)
            return None

        started = time.monotonic()
        with self._app(config_file, AsyncMock(side_effect=receipt)) as c:
            self._open_topup(c)
            stopping.set()

        assert time.monotonic() - started < 5
        assert _in_flight() == 0
        assert any("not reconciled" in r.getMessage() for r in caplog.records)


def _transfer_receipt_lookup(receipts):
    """A payment handler whose settlement_receipt answers from ``receipts``."""
    return SimpleNamespace(settlement_receipt=AsyncMock(side_effect=receipts))


class TestReconcileBase:
    """A reconcile holds the in-flight amount until the receipt or the deadline."""

    @pytest.fixture(autouse=True)
    def fast(self, monkeypatch):
        gateway.base_in_flight.clear()
        monkeypatch.setattr(gateway, "_RECONCILE_FIRST_POLL_S", 0.001)
        monkeypatch.setattr(gateway, "_RECONCILE_MAX_POLL_S", 0.002)

    @staticmethod
    async def _reconcile(settlement, on_landed) -> None:  # noqa: ANN001
        """Start a reconcile as the app does and wait for it to end."""
        gateway.base_in_flight.add("0xPayer", Decimal("1"))
        gateway._start_base_reconcile(
            EVM_NETWORK, "0xPayer", "sig", settlement, Decimal("1"), on_landed
        )
        await asyncio.gather(*gateway._pending_reconciles, return_exceptions=True)

    def _run(self, monkeypatch, receipts, settlement=None, hold_s: float = 5.0):
        monkeypatch.setattr(gateway, "_claim_hold_s", lambda *args: hold_s)
        handler = _transfer_receipt_lookup(receipts)
        monkeypatch.setattr(gateway, "payment_handler", handler, raising=False)
        landed = AsyncMock()
        asyncio.run(
            self._reconcile(_pending_settle() if settlement is None else settlement, landed)
        )
        return handler.settlement_receipt, landed

    def test_landed_transfer_runs_on_landed_once(self, monkeypatch):
        ok = _receipt_settlement("0xaa", landed=True)
        lookup, landed = self._run(monkeypatch, [None, None, ok])
        assert lookup.await_count == 3
        assert lookup.await_args.args == (EVM_NETWORK, "0xaa", "0xPayer", Decimal("1"))
        landed.assert_awaited_once_with(ok)
        assert _in_flight() == 0
        assert gateway._pending_reconciles == set()

    def test_reverted_transfer_never_runs_on_landed(self, monkeypatch):
        lookup, landed = self._run(monkeypatch, [_receipt_settlement("0xaa", landed=False)])
        landed.assert_not_awaited()
        assert _in_flight() == 0

    def test_expired_authorization_releases_without_credit(self, monkeypatch):
        lookup, landed = self._run(monkeypatch, lambda *args: None, hold_s=0.02)
        assert lookup.await_count >= 2
        landed.assert_not_awaited()
        assert _in_flight() == 0

    def test_failed_lookup_is_logged_and_polled_again(self, monkeypatch, caplog):
        ok = _receipt_settlement("0xaa", landed=True)
        lookup, landed = self._run(monkeypatch, [KeyError("logs"), ok])
        assert lookup.await_count == 2
        landed.assert_awaited_once_with(ok)
        assert any(
            "Receipt lookup for Base tx 0xaa failed" in r.getMessage() for r in caplog.records
        )
        assert _in_flight() == 0

    def test_crashed_settle_without_hash_holds_until_expiry(self, monkeypatch):
        """settle returned None: nothing to look up, the amount stays until expiry."""
        monkeypatch.setattr(gateway, "_claim_hold_s", lambda *args: 0.02)
        handler = _transfer_receipt_lookup(None)
        monkeypatch.setattr(gateway, "payment_handler", handler, raising=False)

        async def run() -> Decimal:
            gateway.base_in_flight.add("0xPayer", Decimal("1"))
            gateway._start_base_reconcile(
                EVM_NETWORK, "0xPayer", "sig", None, Decimal("1"), AsyncMock()
            )
            await asyncio.sleep(0.005)
            held = _in_flight()
            await asyncio.gather(*gateway._pending_reconciles)
            return held

        assert asyncio.run(run()) == Decimal("1")
        handler.settlement_receipt.assert_not_awaited()
        assert _in_flight() == 0

    def test_on_landed_error_is_logged_not_raised(self, monkeypatch, caplog):
        monkeypatch.setattr(gateway, "_claim_hold_s", lambda *args: 5.0)
        ok = _receipt_settlement("0xaa", landed=True)
        monkeypatch.setattr(
            gateway, "payment_handler", _transfer_receipt_lookup([ok]), raising=False
        )
        asyncio.run(self._reconcile(_pending_settle(), AsyncMock(side_effect=RuntimeError("boom"))))
        assert any("Recording landed Base tx" in r.getMessage() for r in caplog.records)
        assert _in_flight() == 0

    def test_crash_is_logged_and_releases_the_amount(self, monkeypatch, caplog):
        def broken(*args):  # noqa: ANN002
            raise RuntimeError("boom")

        monkeypatch.setattr(gateway, "_claim_hold_s", broken)
        asyncio.run(self._reconcile(_pending_settle(), AsyncMock()))
        assert any(r.getMessage() == "Base reconcile crashed" for r in caplog.records)
        assert _in_flight() == 0
        assert gateway._pending_reconciles == set()

    def _run_unresolved(self, monkeypatch, receipts, hold_s: float = 5.0):
        """Run a reconcile with an ``on_unresolved`` recorder; return what it heard."""
        monkeypatch.setattr(gateway, "_claim_hold_s", lambda *args: hold_s)
        monkeypatch.setattr(
            gateway, "payment_handler", _transfer_receipt_lookup(receipts), raising=False
        )
        heard: list[str] = []
        landed = AsyncMock()

        async def run() -> None:
            gateway.base_in_flight.add("0xPayer", Decimal("1"))
            gateway._start_base_reconcile(
                EVM_NETWORK,
                "0xPayer",
                "sig",
                _pending_settle(),
                Decimal("1"),
                landed,
                on_unresolved=heard.append,
            )
            await asyncio.gather(*gateway._pending_reconciles, return_exceptions=True)

        asyncio.run(run())
        return heard, landed

    def test_on_unresolved_hears_a_revert(self, monkeypatch):
        heard, landed = self._run_unresolved(
            monkeypatch, [_receipt_settlement("0xaa", landed=False)]
        )
        assert heard == ["failed"]
        landed.assert_not_awaited()

    def test_on_unresolved_hears_an_expiry(self, monkeypatch):
        heard, landed = self._run_unresolved(monkeypatch, lambda *args: None, hold_s=0.02)
        assert heard == ["expired"]
        landed.assert_not_awaited()

    def test_on_unresolved_is_not_called_for_a_landing(self, monkeypatch):
        ok = _receipt_settlement("0xaa", landed=True)
        heard, landed = self._run_unresolved(monkeypatch, [ok])
        assert heard == []
        landed.assert_awaited_once_with(ok)

    @pytest.mark.parametrize(
        ("owed", "expect_owed"),
        [(None, True), (lambda: True, True), (lambda: False, False), ("raise", True)],
        ids=["default", "owed", "kept", "check-crashed"],
    )
    def test_shutdown_log_follows_owed(self, monkeypatch, caplog, owed, expect_owed):
        """A landed transfer is logged as owed unless ``owed`` says it was kept."""
        ok = _receipt_settlement("0xaa", landed=True)
        monkeypatch.setattr(
            gateway, "payment_handler", _transfer_receipt_lookup([ok]), raising=False
        )
        if owed == "raise":

            def owed() -> bool:
                raise RuntimeError("boom")

        asyncio.run(gateway._log_unreconciled(EVM_NETWORK, "0xaa", "0xPayer", Decimal("1"), owed))
        messages = [r.getMessage() for r in caplog.records]
        assert any("is owed" in m for m in messages) is expect_owed
        assert any("nothing owed" in m for m in messages) is not expect_owed

    def test_cancel_before_the_first_step_releases_the_amount(self, monkeypatch):
        """The coroutine never runs, so only the task's done callback can release it."""
        monkeypatch.setattr(
            gateway, "payment_handler", _transfer_receipt_lookup(None), raising=False
        )

        async def run() -> None:
            gateway.base_in_flight.add("0xPayer", Decimal("1"))
            gateway._start_base_reconcile(
                EVM_NETWORK, "0xPayer", "sig", _pending_settle(), Decimal("1"), AsyncMock()
            )
            (task,) = gateway._pending_reconciles
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
        assert _in_flight() == 0
        assert gateway._pending_reconciles == set()


def _evm_sig(valid_before: int | str | None) -> str:
    """A Base payment header whose authorization has this validBefore."""
    authorization: dict = {"from": "0xPayer", "nonce": "0x01"}
    if valid_before is not None:
        authorization["validBefore"] = str(valid_before)
    payload = {"accepted": {"network": EVM_NETWORK}, "payload": {"authorization": authorization}}
    return base64.b64encode(json.dumps(payload).encode()).decode()


class TestClaimHold:
    """How long a payment stays claimed after its settle returns."""

    def test_failed_frees_at_once(self, client):
        assert gateway._claim_hold_s(_evm_sig(None), EVM_NETWORK, "failed") == 0.0
        assert gateway._claim_hold_s("x", SOLANA_NETWORK, "failed") == 0.0

    def test_settled_keeps_the_short_hold(self, client):
        sig = _evm_sig(int(time.time()) + 3600)
        assert gateway._claim_hold_s(sig, EVM_NETWORK, "ok") == gateway._BASE_CLAIM_HOLD_S
        assert gateway._claim_hold_s(sig, SOLANA_NETWORK, "ok") == gateway._SOLANA_CLAIM_HOLD_S
        assert gateway._claim_hold_s(sig, SOLANA_NETWORK, "unknown") == gateway._SOLANA_CLAIM_HOLD_S

    def test_unknown_base_holds_until_valid_before(self, client):
        hold = gateway._claim_hold_s(_evm_sig(int(time.time()) + 3000), EVM_NETWORK, "unknown")
        assert 3000 + gateway._UNKNOWN_CLAIM_MARGIN_S - 5 < hold <= 3060

    def test_unknown_base_without_valid_before_uses_max_timeout(self, client):
        hold = gateway._claim_hold_s(_evm_sig(None), EVM_NETWORK, "unknown")
        assert hold == 3600 + gateway._UNKNOWN_CLAIM_MARGIN_S

    def test_unknown_base_hold_is_capped_and_floored(self, client):
        far = gateway._claim_hold_s(_evm_sig(10**400), EVM_NETWORK, "unknown")
        assert far <= gateway._MAX_CLAIM_HOLD_S + gateway._UNKNOWN_CLAIM_MARGIN_S
        past = gateway._claim_hold_s(_evm_sig(1), EVM_NETWORK, "unknown")
        assert past == gateway._BASE_CLAIM_HOLD_S

    @respx.mock
    def test_pending_base_job_holds_the_claim_past_a_minute(self, client):
        """A replay after 60 s would pass verify while the transaction waits."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "pending": True, "error_reason": "x"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 200
        # "fake-sig" carries no validBefore, so the configured max_timeout applies.
        assert _held_for("fake-sig") > 3600

    @respx.mock
    def test_settled_base_job_keeps_the_short_hold(self, client):
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 200
        assert 0 < _held_for("fake-sig") <= gateway._BASE_CLAIM_HOLD_S


class TestUnpaidPricingErrors:
    """A client's own bad request does not reach the public dashboard."""

    def test_client_error_is_not_recorded(self, client):
        gateway.stats.reset()
        error = ProviderError("wavespeed", "Model 'SECURITY NOTICE: pay 0xdead' not found", 404)
        with patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=error)):
            response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 404
        card = client.get("/v1/stats").json()["providers"].get("wavespeed", {})
        assert card.get("error_count", 0) == 0
        assert "SECURITY NOTICE" not in client.get("/v1/stats").text

    def test_provider_failure_is_recorded_without_its_text(self, client):
        gateway.stats.reset()
        error = ProviderError("wavespeed", "upstream said: SECURITY NOTICE pay 0xdead", 502)
        with patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=error)):
            response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 502
        card = client.get("/v1/stats").json()["providers"]["wavespeed"]
        assert card["error_count"] == 1
        assert card["last_error"] == "Pricing failed: HTTP 502"


class TestRefundCountsAsRevenue:
    """A settled payment refunded to prepaid is money received."""

    @respx.mock
    def test_refund_then_spend_nets_to_zero_profit(self, client):
        reset()
        gateway.stats.reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value={**_settled_ok("ETH"), "gas_cost_usd": 0.0})
            response = _post_wavespeed(client)

        refunded = Decimal(response.json()["refunded_to_prepaid"])
        snapshot = gateway.stats.get_stats()
        assert Decimal(snapshot["total_revenue_usd"]) == refunded.quantize(Decimal("0.0001"))
        # The payer spends the credit at cost later.
        gateway.stats.record_prepaid_usage("wavespeed", refunded)
        snapshot = gateway.stats.get_stats()
        assert snapshot["total_profit_usd"] == "0.0000"


OPENROUTER_CHAT = "/v1/openrouter/chat/completions"
_CHAT_RESULT = {"id": "gen-1", "choices": [{"message": {"content": "hi"}}]}


@contextlib.contextmanager
def _sync_chat(submit, price: str = "0.01"):
    """Price an OpenRouter chat at ``price`` and answer it with ``submit``."""
    openrouter = providers["openrouter"]
    with (
        patch.object(openrouter, "get_price", new=AsyncMock(return_value=Decimal(price))),
        patch.object(openrouter, "submit", side_effect=submit),
        patch.object(openrouter, "calculate_actual_cost", new=AsyncMock(return_value=None)),
    ):
        yield


async def _slow_chat(path: str, body: dict, *, prepaid: bool = False) -> dict:
    await asyncio.sleep(0.2)
    return _CHAT_RESULT


async def _slow_chat_failure(path: str, body: dict, *, prepaid: bool = False) -> dict:
    await asyncio.sleep(0.2)
    raise ProviderError(provider="openrouter", detail="model overloaded", status_code=503)


class TestSyncCallHeartbeat:
    """A long synchronous call sends heartbeats, so Railway keeps the connection."""

    def test_slow_call_sends_whitespace_before_json(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        with _sync_chat(_slow_chat), patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04):
            response = client.post(
                OPENROUTER_CHAT,
                json={"model": "m", "messages": []},
                headers=_prepaid_headers(kp, "openrouter/chat/completions"),
            )

        assert response.status_code == 200
        assert response.content.startswith(b"\n")
        payload = response.json()
        assert payload["data"]["choices"][0]["message"]["content"] == "hi"
        assert payload["prepaid"]["cost"] == "0.01"
        assert get_balance(pubkey) == Decimal("0.99")

    def test_fast_call_has_no_heartbeat_and_keeps_its_status(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        async def rejected(path: str, body: dict, *, prepaid: bool = False) -> dict:
            raise ProviderError(provider="openrouter", detail="bad model", status_code=400)

        with _sync_chat(rejected):
            response = client.post(
                OPENROUTER_CHAT,
                json={"model": "m", "messages": []},
                headers=_prepaid_headers(kp, "openrouter/chat/completions"),
            )

        assert response.status_code == 400
        assert not response.content.startswith(b"\n")
        assert get_balance(pubkey) == Decimal("1.00")

    def test_slow_failure_reports_the_error_in_the_body(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        with _sync_chat(_slow_chat_failure), patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04):
            response = client.post(
                OPENROUTER_CHAT,
                json={"model": "m", "messages": []},
                headers=_prepaid_headers(kp, "openrouter/chat/completions"),
            )

        # The 200 went out with the first pad; the error is in the body.
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == 503
        assert payload["prepaid"]["cost"] == "0"
        assert get_balance(pubkey) == Decimal("1.00")

    def test_client_that_leaves_a_slow_call_is_still_charged(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))

        with _sync_chat(_slow_chat), patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.01):
            messages = _post_and_leave(
                client, OPENROUTER_CHAT, _prepaid_headers(kp, "openrouter/chat/completions")
            )
            _drain(client)

        assert messages[0]["status"] == 200
        assert b'"choices"' not in _body_text(messages)
        assert get_balance(pubkey) == Decimal("0.99")

    def test_slow_base_call_settles_and_returns_the_result(self, client):
        reset()
        with (
            _sync_chat(_slow_chat),
            patch("x402gate.app._HEARTBEAT_INTERVAL_S", 0.04),
            patch("x402gate.app.payment_handler") as mock_ph,
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = client.post(
                OPENROUTER_CHAT,
                json={"model": "m", "messages": []},
                headers={"Payment-Signature": "fake-sig"},
            )

        assert response.content.startswith(b"\n")
        assert response.json()["data"]["choices"][0]["message"]["content"] == "hi"
        assert mock_ph.settle.await_count == 1


_BOUNDARY = "x402gateboundary"


def _multipart_chunks(file_size: int, chunk: int = 64 * 1024) -> list[bytes]:
    """A CloudConvert multipart body with a ``file_size`` file, in chunks."""
    head = (
        f"--{_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="output_format"\r\n\r\npdf\r\n'
        f"--{_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{_BOUNDARY}--\r\n".encode()
    data = b"x" * file_size
    return [head, *(data[i : i + chunk] for i in range(0, file_size, chunk)), tail]


def _post_chunked(
    client,
    chunks: list[bytes],
    *,
    finish: bool = True,
    path: str = "/v1/cloudconvert/convert",
) -> tuple[int, int]:
    """POST a multipart body over raw ASGI with no Content-Length.

    With ``finish`` False the client stalls after the last chunk instead of
    ending the body. Returns the status and how many body bytes the gateway read.
    """

    async def _run() -> tuple[int, int]:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"transfer-encoding", b"chunked"),
                (b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode()),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("test", 80),
            "state": {},
        }
        pending = list(chunks)
        read = 0
        status = 0

        async def receive() -> dict:
            nonlocal read
            if not pending:
                await asyncio.sleep(3600)
            part = pending.pop(0)
            read += len(part)
            return {"type": "http.request", "body": part, "more_body": bool(pending) or not finish}

        async def send(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]

        await app(scope, receive, send)
        return status, read

    return client.portal.call(_run)


class TestMultipartUploadCaps:
    """Upload limits hold while the body is read, before any payment."""

    def test_chunked_upload_stops_at_the_limit(self, client):
        # 1 MB limit in test config; the body would be 8 MB.
        status, read = _post_chunked(client, _multipart_chunks(8 * 1024 * 1024))

        assert status == 413
        limit = 1024 * 1024 + gateway._MULTIPART_OVERHEAD_BYTES
        assert read <= limit + 64 * 1024

    @respx.mock
    def test_small_chunked_upload_gets_a_quote(self, client):
        status, _ = _post_chunked(client, _multipart_chunks(1000))

        assert status == 402
        assert gateway._unpaid_upload_bytes == 0

    def test_second_file_is_rejected(self, client):
        response = client.post(
            "/v1/cloudconvert/convert",
            files=[("file", ("a.txt", b"a")), ("other", ("b.txt", b"b"))],
            data={"output_format": "pdf"},
        )

        assert response.status_code == 400
        assert "Too many files" in response.json()["error"]

    def test_unpaid_uploads_over_the_memory_budget_get_503(self, client):
        full = gateway._unpaid_upload_budget(1024 * 1024)
        with patch("x402gate.app._unpaid_upload_bytes", full):
            status, read = _post_chunked(client, _multipart_chunks(900 * 1024))
            assert gateway._unpaid_upload_bytes == full

        # Refused at the first chunk, before the body is spooled.
        assert status == 503
        assert read <= 64 * 1024

    def test_budget_is_taken_while_the_body_is_read(self, client):
        # Room for about 300 KB of body (each byte counts twice).
        held = gateway._unpaid_upload_budget(1024 * 1024) - 2 * 300 * 1024
        with patch("x402gate.app._unpaid_upload_bytes", held):
            status, read = _post_chunked(client, _multipart_chunks(900 * 1024))
            assert gateway._unpaid_upload_bytes == held

        assert status == 503
        assert read <= 300 * 1024 + 64 * 1024

    def test_budget_full_response_has_retry_after(self, client):
        full = gateway._unpaid_upload_budget(1024 * 1024)
        with patch("x402gate.app._unpaid_upload_bytes", full):
            response = client.post(
                "/v1/cloudconvert/convert",
                files={"file": ("small.txt", b"x" * 1000)},
                data={"output_format": "pdf"},
            )

        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"

    def test_stalled_upload_times_out_and_frees_its_bytes(self, client):
        with patch("x402gate.app._UPLOAD_READ_TIMEOUT_S", 0.2):
            status, _ = _post_chunked(client, _multipart_chunks(100 * 1024), finish=False)

        assert status == 408
        assert gateway._unpaid_upload_bytes == 0

    @respx.mock
    def test_only_the_read_copy_is_held_after_parsing(self, client):
        seen: list[int] = []
        cloudconvert = providers["cloudconvert"]
        real_get_price = cloudconvert.get_price

        async def get_price(path: str, inputs: dict) -> Decimal:
            seen.append(gateway._unpaid_upload_bytes)
            return await real_get_price(path, inputs)

        with patch.object(cloudconvert, "get_price", side_effect=get_price):
            response = client.post(
                "/v1/cloudconvert/convert",
                files={"file": ("small.txt", b"x" * 1000)},
                data={"output_format": "pdf"},
            )

        assert response.status_code == 402
        assert seen == [1000]
        assert gateway._unpaid_upload_bytes == 0

    @respx.mock
    def test_unpaid_upload_bytes_are_freed_after_the_quote(self, client):
        response = client.post(
            "/v1/cloudconvert/convert",
            files={"file": ("small.txt", b"x" * 1000)},
            data={"output_format": "pdf"},
        )

        assert response.status_code == 402
        assert gateway._unpaid_upload_bytes == 0

    def test_paid_upload_frees_its_unpaid_bytes_before_the_job(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        seen: list[int] = []

        async def submit(path: str, body: dict, *, prepaid: bool = False) -> dict:
            seen.append(gateway._unpaid_upload_bytes)
            return {"data": {"result": "ok"}}

        cloudconvert = providers["cloudconvert"]
        with (
            patch.object(cloudconvert, "submit", side_effect=submit),
            patch.object(cloudconvert, "calculate_actual_cost", new=AsyncMock(return_value=None)),
        ):
            response = client.post(
                "/v1/cloudconvert/convert",
                files={"file": ("small.txt", b"x" * 1000)},
                data={"output_format": "pdf"},
                headers=_prepaid_headers(kp, "cloudconvert/convert"),
            )

        assert response.status_code == 200
        assert seen == [0]
        assert gateway._unpaid_upload_bytes == 0


class TestMultipartOnlyForUploadProviders:
    """A provider that takes JSON refuses multipart before the body is read."""

    def test_json_provider_refuses_multipart_unread(self, client):
        wavespeed = providers["wavespeed"]
        with patch.object(wavespeed, "get_price", new=AsyncMock()) as get_price:
            status, read = _post_chunked(
                client,
                _multipart_chunks(900 * 1024),
                path="/v1/wavespeed/wavespeed-ai/flux-dev",
            )

        assert status == 400
        assert read == 0
        get_price.assert_not_awaited()
        assert gateway._unpaid_upload_bytes == 0

    def test_refusal_names_json(self, client):
        response = client.post(
            "/v1/openrouter/audio/transcriptions",
            files={"file": ("a.wav", b"RIFF")},
            data={"model": "openai/whisper-1"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == (
            "openrouter takes a JSON body. Multipart uploads are not accepted."
        )
        assert response.json()["provider"] == "openrouter"

    @respx.mock
    def test_cloudconvert_still_takes_multipart(self, client):
        status, _ = _post_chunked(client, _multipart_chunks(1000))

        assert status == 402


class TestPricingFailures:
    """Pricing errors answer the client without charging anything."""

    def test_pricing_crash_is_a_400_and_not_a_provider_failure(self, client, caplog):
        gateway.stats.reset()
        boom = RuntimeError("bug in pricing")
        with patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=boom)):
            response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 400
        assert response.json()["error"] == "Could not price this request"
        card = client.get("/v1/stats").json()["providers"].get("wavespeed", {})
        assert card.get("error_count", 0) == 0
        logged = [r for r in caplog.records if "pricing crashed" in r.getMessage()]
        assert logged and logged[0].exc_info[1] is boom

    def test_pricing_crash_with_payment_does_not_verify(self, client):
        boom = RuntimeError("bug in pricing")
        with (
            patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=boom)),
            patch("x402gate.app.payment_handler") as mock_ph,
        ):
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            response = _post_wavespeed(client)

        assert response.status_code == 400
        mock_ph.verify.assert_not_awaited()

    def test_pricing_429_has_retry_after(self, client):
        busy = ProviderError("wavespeed", "Too many audio files are being priced", 429)
        with patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=busy)):
            response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 429
        assert response.headers["retry-after"] == gateway._PRICING_RETRY_AFTER_S

    @respx.mock
    def test_busy_audio_parse_slots_get_429_with_retry_after(self, client):
        TestOpenRouterSTT()._mock_transcription_catalog()

        class _Busy:
            def acquire(self, timeout: float) -> bool:
                return False

        with patch("x402gate.core.audio_duration._PARSE_SLOTS", _Busy()):
            response = client.post(
                "/v1/openrouter/audio/transcriptions", json=TestOpenRouterSTT._STT_BODY
            )

        assert response.status_code == 429
        assert "retry shortly" in response.json()["error"]
        assert response.headers["retry-after"] == gateway._PRICING_RETRY_AFTER_S

    def test_other_pricing_errors_have_no_retry_after(self, client):
        error = ProviderError("wavespeed", "bad prompt", 400)
        with patch.object(providers["wavespeed"], "get_price", AsyncMock(side_effect=error)):
            response = client.post("/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat"})

        assert response.status_code == 400
        assert "retry-after" not in response.headers


class TestPricedRatesSurviveTheCache:
    """Rates get_price leaves in the body reach submit on a cached price too."""

    @pytest.fixture
    def cache(self, monkeypatch):
        cache = PriceCache(ttl=60)
        monkeypatch.setattr(gateway, "price_cache", cache)
        return cache

    def test_cached_price_restores_the_priced_rates(self, client, cache):
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        rates = _PricedRates({"prompt": Decimal("1"), "completion": Decimal("2")})
        seen_at_pricing: list[bool] = []
        seen_at_submit: list[object] = []

        async def get_price(path: str, inputs: dict) -> Decimal:
            seen_at_pricing.append("_priced_rates" in inputs)
            inputs["_priced_rates"] = rates
            return Decimal("0.01")

        async def submit(path: str, body: dict, *, prepaid: bool = False) -> dict:
            seen_at_submit.append(body.get("_priced_rates"))
            return _CHAT_RESULT

        openrouter = providers["openrouter"]
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        with (
            patch.object(openrouter, "get_price", side_effect=get_price) as priced,
            patch.object(openrouter, "submit", side_effect=submit),
            patch.object(openrouter, "calculate_actual_cost", new=AsyncMock(return_value=None)),
        ):
            for _ in range(2):
                response = client.post(
                    OPENROUTER_CHAT,
                    json=body,
                    headers=_prepaid_headers(kp, "openrouter/chat/completions"),
                )
                assert response.status_code == 200

        assert priced.call_count == 1
        assert seen_at_pricing == [False]
        assert len(seen_at_submit) == 2
        assert all(isinstance(r, _PricedRates) and r == rates for r in seen_at_submit)

    def test_client_supplied_priced_rates_are_stripped_before_pricing(self, client):
        seen: list[dict] = []

        async def get_price(path: str, inputs: dict) -> Decimal:
            seen.append(dict(inputs))
            return Decimal("0.01")

        with patch.object(providers["openrouter"], "get_price", side_effect=get_price):
            response = client.post(
                OPENROUTER_CHAT,
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "_priced_rates": {"prompt": 1000, "completion": 1000, "request": 1000},
                },
            )

        assert response.status_code == 402
        assert "_priced_rates" not in seen[0]


@pytest.fixture
def spicy_client(config_file):
    """A client whose config also enables SpicyAPI."""
    config_file.write_text(
        config_file.read_text()
        + """  spicyapi:
    enabled: true
    base_url: "https://api.spicyapi.ai"
    api_key: "test-spicy-key"
    poll_interval: 0
    poll_timeout: 10
"""
    )
    with (
        patch("x402gate.app.load_config") as mock_load,
        patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={})),
        patch("x402gate.app._warm_up", new=AsyncMock()),
    ):
        mock_load.return_value = real_load(config_file)
        gateway.payment_claims.clear()
        gateway.base_in_flight.clear()
        with TestClient(app) as c:
            yield c
    providers.pop("spicyapi", None)


class TestSpicyApiPaidPrice:
    """SpicyAPI's submit learns the price the request was charged at."""

    def test_price_window_follows_the_gateway_price_cache(self, spicy_client):
        # The test config turns the gateway price cache off (TTL 0).
        assert providers["spicyapi"]._price_window == _PRICE_GRACE_SECONDS

    @pytest.mark.parametrize("prepaid", [True, False])
    def test_submit_gets_the_charged_base_price(self, spicy_client, prepaid):
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        calls: list[dict] = []

        async def submit(path: str, body: dict, **kwargs) -> dict:
            calls.append(kwargs)
            return {"data": {"images": ["ok"]}}

        spicy = providers["spicyapi"]
        with (
            patch.object(spicy, "get_price", new=AsyncMock(return_value=Decimal("0.05"))),
            patch.object(spicy, "submit", side_effect=submit),
            patch.object(spicy, "calculate_actual_cost", new=AsyncMock(return_value=None)),
            patch("x402gate.app.payment_handler") as mock_ph,
        ):
            if prepaid:
                mock_ph.extract_payment_signature.return_value = None
                headers = _prepaid_headers(kp, "spicyapi/jobs")
            else:
                _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
                mock_ph.settle = AsyncMock(return_value=_settled_ok())
                headers = {"Payment-Signature": "fake-sig"}
            response = spicy_client.post(
                "/v1/spicyapi/jobs", json={"model": "m", "input": {}}, headers=headers
            )

        assert response.status_code == 200
        assert calls == [{"prepaid": prepaid, "paid_price": Decimal("0.05")}]


class TestVideoNeedsPrepaid:
    """An OpenRouter video request is held at the context window and paid only from prepaid."""

    MODELS_URL = "https://openrouter.ai/api/v1/models"
    CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
    CATALOG = {
        "data": [
            {
                "id": "qwen/omni",
                "context_length": 100_000,
                "pricing": {"prompt": "0.0000002", "completion": "0.000001"},
            }
        ]
    }
    BODY = {
        "model": "qwen/omni",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                ],
            }
        ],
    }

    @staticmethod
    def _hold() -> Decimal:
        """100k window tokens at the input rate plus the default output ceiling."""
        output = providers["openrouter"]._default_max_tokens
        return Decimal(100_000) * Decimal("0.0000002") + Decimal(output) * Decimal("0.000001")

    def _mock_catalog(self) -> None:
        respx.get(self.MODELS_URL).mock(return_value=httpx.Response(200, json=self.CATALOG))

    def _mock_chat(self, cost: float = 0.0012) -> respx.Route:
        return respx.post(self.CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "gen-1",
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 3000, "completion_tokens": 10, "cost": cost},
                },
            )
        )

    @respx.mock
    def test_unpaid_video_gets_400_not_a_payment_offer(self, client):
        self._mock_catalog()
        response = client.post(OPENROUTER_CHAT, json=self.BODY)
        assert response.status_code == 400
        assert "prepaid balance" in response.json()["error"]
        assert "accepts" not in response.json()

    @respx.mock
    def test_x402_payment_is_refused_before_it_is_verified(self, client):
        self._mock_catalog()
        chat = self._mock_chat()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            response = client.post(
                OPENROUTER_CHAT, json=self.BODY, headers={"Payment-Signature": "fake-sig"}
            )
        assert response.status_code == 400
        assert "prepaid balance" in response.json()["error"]
        mock_ph.verify.assert_not_called()
        mock_ph.settle.assert_not_called()
        assert not chat.called
        assert gateway.payment_claims.claim(payment_key("fake-sig"))

    @respx.mock
    def test_x402_payment_beside_prepaid_headers_is_refused(self, client):
        """With both, the payment header wins in the gateway, so the request is refused."""
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        self._mock_catalog()
        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            response = client.post(
                OPENROUTER_CHAT,
                json=self.BODY,
                headers={
                    "Payment-Signature": "fake-sig",
                    **_prepaid_headers(kp, "openrouter/chat/completions"),
                },
            )
        assert response.status_code == 400
        mock_ph.verify.assert_not_called()
        assert get_balance(str(kp.pubkey())) == Decimal("1.00")

    @respx.mock
    def test_prepaid_holds_the_window_and_returns_the_rest(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        asyncio.run(deposit(pubkey, Decimal("1.00")))
        self._mock_catalog()
        chat = self._mock_chat(cost=0.0012)
        response = client.post(
            OPENROUTER_CHAT,
            json=self.BODY,
            headers=_prepaid_headers(kp, "openrouter/chat/completions"),
        )
        assert response.status_code == 200
        assert response.headers["x-prepaid-cost"] == "0.0012"
        assert get_balance(pubkey) == Decimal("1.00") - Decimal("0.0012")
        sent = json.loads(chat.calls.last.request.content)
        assert sent["messages"][0]["content"][1]["type"] == "video_url"
        assert not any(key.startswith("_") for key in sent)

    @respx.mock
    def test_prepaid_balance_below_the_hold_gets_402(self, client):
        reset()
        kp = Keypair()
        pubkey = str(kp.pubkey())
        hold = self._hold()
        asyncio.run(deposit(pubkey, hold - Decimal("0.000001")))
        self._mock_catalog()
        chat = self._mock_chat()
        response = client.post(
            OPENROUTER_CHAT,
            json=self.BODY,
            headers=_prepaid_headers(kp, "openrouter/chat/completions"),
        )
        assert response.status_code == 402
        assert f"< ${hold}" in response.json()["error"]
        assert not chat.called

    @respx.mock
    def test_cached_price_still_refuses_x402(self, client, monkeypatch):
        monkeypatch.setattr(gateway, "price_cache", PriceCache(ttl=60))
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        self._mock_catalog()
        self._mock_chat()
        first = client.post(
            OPENROUTER_CHAT,
            json=self.BODY,
            headers=_prepaid_headers(kp, "openrouter/chat/completions"),
        )
        assert first.status_code == 200
        with (
            patch.object(providers["openrouter"], "get_price") as priced,
            patch("x402gate.app.payment_handler") as mock_ph,
        ):
            _paid_handler(mock_ph, SOLANA_NETWORK, SOL_PAYER)
            second = client.post(
                OPENROUTER_CHAT, json=self.BODY, headers={"Payment-Signature": "fake-sig"}
            )
        priced.assert_not_called()
        assert second.status_code == 400
        mock_ph.verify.assert_not_called()

    def test_client_cannot_mark_its_own_request_prepaid_only(self, client):
        seen: list[dict] = []

        async def get_price(path: str, inputs: dict) -> Decimal:
            seen.append(dict(inputs))
            return Decimal("0.01")

        with patch.object(providers["openrouter"], "get_price", side_effect=get_price):
            response = client.post(
                OPENROUTER_CHAT,
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "_prepaid_only": "forged",
                },
            )
        assert response.status_code == 402
        assert "_prepaid_only" not in seen[0]


class TestFishAudioExpressiveness:
    """Fish Audio temperature and top_p are checked before payment and forwarded."""

    TTS_URL = "https://api.fish.audio/v1/tts"

    def test_unpaid_request_with_temperature_is_priced(self, client):
        response = client.post("/v1/fishaudio/tts", json={"text": "hi", "temperature": 0.9})
        assert response.status_code == 402

    def test_temperature_above_one_gets_400_before_payment(self, client):
        response = client.post("/v1/fishaudio/tts", json={"text": "hi", "temperature": 2})
        assert response.status_code == 400
        assert "temperature" in response.json()["error"]

    @respx.mock
    def test_prepaid_request_forwards_temperature_and_top_p(self, client):
        reset()
        kp = Keypair()
        asyncio.run(deposit(str(kp.pubkey()), Decimal("1.00")))
        route = respx.post(self.TTS_URL).mock(
            return_value=httpx.Response(200, content=b"a", headers={"content-type": "audio/mpeg"})
        )
        response = client.post(
            "/v1/fishaudio/tts",
            json={"text": "hi", "temperature": 0.9, "top_p": 0.6},
            headers=_prepaid_headers(kp, "fishaudio/tts"),
        )
        assert response.status_code == 200
        sent = json.loads(route.calls.last.request.content)
        assert sent["temperature"] == 0.9
        assert sent["top_p"] == 0.6
