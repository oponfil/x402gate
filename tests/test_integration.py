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
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from solders.keypair import Keypair

import x402gate.app as gateway
from tests.helpers import make_wav_b64
from x402gate.app import _pending_settlements, app, providers
from x402gate.core.config import load_config as real_load
from x402gate.core.payment import payment_key
from x402gate.core.prepaid import deposit, get_balance, reset
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError

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
  tungsten:
    enabled: false
    base_url: "https://api.tungsten.run/v1"
    jwt_token: "${TUNGSTEN_JWT_TOKEN}"
    cf_clearance: "${TUNGSTEN_CF_CLEARANCE}"
    fixed_price_usd: 0.01
""")
    return config


@pytest.fixture
def client(config_file):
    """Create a test client with mocked config path."""
    with (
        patch("x402gate.app.load_config") as mock_load,
        patch("x402gate.app.fetch_balances", new=AsyncMock(return_value={})),
        # Startup warm-up would fetch real catalogs; TestWarmUp covers it.
        patch("x402gate.app._warm_up", new=AsyncMock()),
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


class TestBaseParallelSettlement:
    """Base settles while the provider runs, and the result waits for it."""

    @respx.mock
    def test_settles_once_and_returns_result(self, client):
        """Settle starts with the provider call; a settled job returns 200."""
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
        assert response.json()["data"] == {"outputs": ["ok"]}
        assert mock_ph.settle.await_count == 1
        expected = apply_commission(Decimal("0.003"), 0.04, 0.001)
        assert mock_ph.settle.await_args.args[1] == expected
        assert mock_ph.settle.await_args.args[2] == EVM_NETWORK

    @respx.mock
    def test_failed_settle_withholds_result(self, client):
        """A payer who moved the USDC during the job gets a 402, not the result."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["secret"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "error_reason": "insufficient_funds"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 402
        assert "insufficient_funds" in response.json()["error"]
        assert "secret" not in response.text
        assert "data" not in response.json()

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
        assert response.json()["refunded_to_prepaid"] == str(expected)
        assert get_balance("0xpayer") == expected

    @respx.mock
    def test_provider_error_with_failed_settle_refunds_nothing(self, client):
        """No settlement, no refund: the payer kept the USDC."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "error_reason": "invalid_signature"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 502
        assert "refunded_to_prepaid" not in response.json()
        assert get_balance("0xPayer") == 0

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
    def test_pending_settlement_still_returns_result(self, client):
        """A broadcast whose confirmation wait failed most likely lands."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["ok"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={
                    "success": False,
                    "pending": True,
                    "error_reason": "settlement_pending",
                }
            )
            response = _post_wavespeed(client)
            replay = _post_wavespeed(client)

        assert response.status_code == 200
        assert replay.status_code == 409

    @respx.mock
    def test_crashed_settle_withholds_result_and_keeps_claim(self, client):
        """A settle that crashed (None) withholds the result but holds the payment."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(
            return_value=httpx.Response(200, json={"outputs": ["secret"]})
        )

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=None)
            response = _post_wavespeed(client)
            replay = _post_wavespeed(client)

        assert response.status_code == 402
        assert "secret" not in response.text
        assert replay.status_code == 409

    @respx.mock
    def test_pending_settlement_on_provider_failure_credits_nothing(self, client):
        """An unconfirmed payment is not credited when the provider fails."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(
                return_value={"success": False, "pending": True, "error_reason": "x"}
            )
            response = _post_wavespeed(client)

        assert response.status_code == 502
        assert "refunded_to_prepaid" not in response.json()
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
        # The rejected request takes back its own promise only.
        gateway.base_in_flight.remove("0xpayer", price)
        assert gateway.base_in_flight.add("0xpayer", Decimal("0")) == 0

    @respx.mock
    def test_in_flight_amount_is_released_after_settle(self, client):
        """Once a settle finishes, the wallet's promise is gone."""
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
        assert gateway.base_in_flight.add("0xpayer", Decimal("0")) == 0

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
    """Wait for settlements and jobs the gateway finishes in the background."""

    async def _wait() -> None:
        for _ in range(20):
            pending = [
                task
                for task in (*_pending_settlements, *getattr(gateway, "_pending_jobs", ()))
                if not task.done()
            ]
            if not pending:
                return
            await asyncio.wait(pending, timeout=1)

    client.portal.call(_wait)


def _post_and_leave(client, path: str, headers: dict[str, str]) -> list[dict]:
    """POST a WaveSpeed job over raw ASGI and disconnect after the first heartbeat."""

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
                while len(warmed) < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert c.get("/health").status_code == 200

        # Enabled managed providers only; blockrun is a passthrough.
        assert sorted(warmed) == [
            "CloudConvertProvider",
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
        get_by_key = PriceCache.get_by_key

        def _make_key(*args):
            hash_threads.append(threading.get_ident())
            return make_key(*args)

        def _get_by_key(self, key):
            loop_threads.append(threading.get_ident())
            return get_by_key(self, key)

        with (
            patch.object(PriceCache, "_make_key", side_effect=_make_key),
            patch.object(PriceCache, "get_by_key", _get_by_key),
        ):
            response = client.post(
                "/v1/wavespeed/wavespeed-ai/flux-dev", json={"prompt": "a cat " * 10}
            )

        assert response.status_code == 402
        assert len(hash_threads) == 1 and len(loop_threads) == 1
        assert hash_threads[0] != loop_threads[0]
