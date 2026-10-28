"""Integration tests for the full x402gate proxy flow.

Uses respx to mock WaveSpeed and facilitator APIs,
and httpx.AsyncClient with FastAPI's TestClient for request testing.
"""

import asyncio
import base64
import json
import os
import time
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from solders.keypair import Keypair

from tests.helpers import make_wav_b64
from x402gate.app import _pending_settlements, app, providers
from x402gate.core.config import load_config as real_load
from x402gate.core.prepaid import deposit, get_balance, reset
from x402gate.core.pricing import apply_commission
from x402gate.core.proxy import TaskTimeoutError

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
    ):
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
        body = response.json()
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
    def test_provider_timeout_refunds_prepaid(self, client):
        """A task timeout after Solana settle credits prepaid."""
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
        assert Decimal(body["refunded_to_prepaid"]) == get_balance(SOL_PAYER)
        assert Decimal(body["prepaid_balance"]) == get_balance(SOL_PAYER)

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

    @respx.mock
    def test_evm_provider_runs_before_settle(self, client):
        """Base still calls the provider before any settlement."""
        reset()
        _price_ok()
        seen: dict[str, int] = {}

        def on_submit(request):
            seen["settle_at_submit"] = mock_ph.settle.await_count
            return httpx.Response(200, json={"outputs": ["ok"]})

        respx.post(WAVESPEED_SUBMIT).mock(side_effect=on_submit)
        before = set(_pending_settlements)

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

            async def _drain() -> None:
                pending = [task for task in list(_pending_settlements) if not task.done()]
                if pending:
                    await asyncio.wait(pending, timeout=2)

            portal = getattr(client, "portal", None)
            if portal is not None:
                portal.call(_drain)
            new_tasks = _pending_settlements - before

        assert response.status_code == 200
        assert seen["settle_at_submit"] == 0
        assert "refunded_to_prepaid" not in response.json()
        assert mock_ph.settle.await_count == 1 or new_tasks

    @respx.mock
    def test_evm_provider_error_does_not_settle_or_refund(self, client):
        """A Base provider error leaves the authorization unsettled."""
        reset()
        _price_ok()
        respx.post(WAVESPEED_SUBMIT).mock(return_value=httpx.Response(400, text="bad prompt"))

        with patch("x402gate.app.payment_handler") as mock_ph:
            _paid_handler(mock_ph, EVM_NETWORK, "0xPayer")
            mock_ph.settle = AsyncMock(return_value=_settled_ok("ETH"))
            response = _post_wavespeed(client)

        assert response.status_code == 502
        assert "refunded_to_prepaid" not in response.json()
        assert mock_ph.settle.await_count == 0
        assert get_balance("0xPayer") == 0


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
