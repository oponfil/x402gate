"""Unit tests for the ElevenLabs TTS provider."""

from __future__ import annotations

import base64
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.elevenlabs import ElevenLabsProvider, _is_turbo_model


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for ElevenLabs tests."""
    defaults = {
        "base_url": "https://api.elevenlabs.io",
        "api_key": "test_elevenlabs_key",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> ElevenLabsProvider:
    """Create an ElevenLabsProvider for tests."""
    config = _make_config(**overrides)
    return ElevenLabsProvider(config=config)


# ---------------------------------------------------------------------------
# _is_turbo_model
# ---------------------------------------------------------------------------


class TestIsTurboModel:
    """Tests for _is_turbo_model()."""

    def test_turbo_model(self):
        assert _is_turbo_model("eleven_turbo_v2") is True

    def test_flash_model(self):
        assert _is_turbo_model("eleven_flash_v2_5") is True

    def test_standard_model(self):
        assert _is_turbo_model("eleven_multilingual_v2") is False

    def test_empty(self):
        assert _is_turbo_model("") is False


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------


class TestGetPrice:
    """Tests for ElevenLabsProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_standard_model_pricing(self):
        """Standard model: $0.30 per 1000 characters."""
        provider = _make_provider()
        price = await provider.get_price(
            "voice123",
            {"text": "a" * 1000, "model_id": "eleven_multilingual_v2"},
        )
        assert price == Decimal("0.30")

    @pytest.mark.asyncio
    async def test_turbo_model_pricing(self):
        """Turbo model: $0.15 per 1000 characters."""
        provider = _make_provider()
        price = await provider.get_price(
            "voice123",
            {"text": "a" * 1000, "model_id": "eleven_turbo_v2"},
        )
        assert price == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_flash_model_pricing(self):
        """Flash model: $0.15 per 1000 characters."""
        provider = _make_provider()
        price = await provider.get_price(
            "voice123",
            {"text": "a" * 2000, "model_id": "eleven_flash_v2_5"},
        )
        assert price == Decimal("0.30")

    @pytest.mark.asyncio
    async def test_cyrillic_counts_as_characters(self):
        """Cyrillic characters are counted as characters (not bytes)."""
        provider = _make_provider()
        text = "Привет" * 100  # 600 chars
        price = await provider.get_price("voice123", {"text": text})
        expected = Decimal("600") / Decimal("1000") * Decimal("0.30")
        assert price == expected

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        """Raises ProviderError for empty text."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.get_price("voice123", {"text": ""})


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for ElevenLabsProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful TTS returns base64 audio."""
        provider = _make_provider()
        fake_audio = b"\xff\xfb\x90\x00" * 100
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio
        mock_response.headers = {"content-type": "audio/mpeg"}

        with patch("x402gate.providers.elevenlabs.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            result = await provider.submit(
                "21m00Tcm4TlvDq8ikWAM",
                {"text": "Hello world", "model_id": "eleven_multilingual_v2"},
            )

        assert "audio_base64" in result
        assert result["content_type"] == "audio/mpeg"
        assert result["characters"] == len("Hello world")
        decoded = base64.b64decode(result["audio_base64"])
        assert decoded == fake_audio

    @pytest.mark.asyncio
    async def test_submit_missing_text(self):
        """Raises ProviderError when text is missing."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.submit("voice123", {})

    @pytest.mark.asyncio
    async def test_submit_api_error(self):
        """Raises ProviderError on API error."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("x402gate.providers.elevenlabs.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            with pytest.raises(Exception, match="ElevenLabs error"):
                await provider.submit("voice123", {"text": "Hello"})


# ---------------------------------------------------------------------------
# voice_id validation
# ---------------------------------------------------------------------------

# FastAPI decodes the path, so "..%2fsound-generation" arrives as
# "../sound-generation"; both spellings are covered.
_BAD_VOICE_IDS = [
    "..%2fsound-generation",
    "../sound-generation",
    "%2e%2e%2fsound-generation",
    "..",
    "abc?x=1",
    "abc/def",
    "abc#frag",
    "abc def",
    "",
    "a" * 65,
]


class TestVoiceIdValidation:
    """The voice_id path must not reach any other ElevenLabs endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("voice_id", _BAD_VOICE_IDS)
    async def test_get_price_rejects_bad_voice_id(self, voice_id):
        """Rejected before payment with a 400."""
        provider = _make_provider()
        with pytest.raises(ProviderError) as exc_info:
            await provider.get_price(voice_id, {"text": "Hello"})
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("voice_id", _BAD_VOICE_IDS)
    async def test_submit_rejects_bad_voice_id(self, voice_id):
        """submit() never calls upstream with a bad voice_id."""
        provider = _make_provider()
        with patch("x402gate.providers.elevenlabs.shared_client") as mock_shared:
            with pytest.raises(ProviderError) as exc_info:
                await provider.submit(voice_id, {"text": "Hello"})
            mock_shared.assert_not_called()
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_normal_voice_id_reaches_tts_url(self):
        """A plain id is priced and sent to /v1/text-to-speech/{voice_id}."""
        provider = _make_provider()
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        assert await provider.get_price(voice_id, {"text": "a" * 1000}) == Decimal("0.30")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio"
        mock_response.headers = {"content-type": "audio/mpeg"}
        with patch("x402gate.providers.elevenlabs.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client
            await provider.submit(voice_id, {"text": "Hello"})

        url = mock_client.post.call_args.args[0]
        assert url == f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:
    """Tests for ElevenLabsProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Sync provider does not support polling."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="synchronous"):
            await provider.get_result("any_task_id")


# ---------------------------------------------------------------------------
# pooled connection
# ---------------------------------------------------------------------------


_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/voice123"


@pytest.fixture
async def pooled():
    """Run against the real shared client and close it afterwards."""
    yield
    await http.close_shared_client()


class TestPooledClient:
    """Calls go through the shared pooled client, not a new client per call."""

    @respx.mock
    async def test_requests_reuse_one_client(self, pooled):
        """Two calls build one client, so the second skips the TCP/TLS handshake."""
        provider = _make_provider()
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, content=b"audio", headers={"content-type": "audio/mpeg"}
            )
        )
        with patch("x402gate.core.http.httpx.AsyncClient", wraps=httpx.AsyncClient) as ctor:
            await provider.submit("voice123", {"text": "Hello"})
            await provider.submit("voice123", {"text": "Hello"})
        assert ctor.call_count == 1
        assert route.call_count == 2

    @respx.mock
    async def test_keeps_the_read_timeout_with_a_short_connect_limit(self, pooled):
        """The old per-call timeout still bounds the read; connecting fails fast."""
        provider = _make_provider()
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, content=b"audio", headers={"content-type": "audio/mpeg"}
            )
        )
        await provider.submit("voice123", {"text": "Hello"})
        limits = route.calls.last.request.extensions["timeout"]
        assert limits["read"] == 600
        assert limits["connect"] == http.CONNECT_TIMEOUT_S

    @respx.mock
    async def test_read_timeout_keeps_the_payment(self, pooled):
        """The request was sent and may have run: outcome unknown, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("voice123", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    async def test_connection_never_opened_is_refunded(self, pooled, error):
        """Nothing was sent, so the client is refunded, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=error("down"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("voice123", {"text": "Hello"})
        assert outcome_unknown(caught.value) is False


class TestUpstreamStatus:
    """How ElevenLabs' error statuses are billed and shown to the client."""

    @respx.mock
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_hides_the_body(self, pooled, status):
        """Our remaining credits stay private, and a 402 is not an x402 request."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                status, text='{"detail":{"status":"quota_exceeded","message":"120 credits"}}'
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("voice123", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @respx.mock
    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_gateway_timeout_keeps_the_payment(self, pooled, status):
        """A proxy gave up after forwarding: the audio may be made and billed."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="timeout"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("voice123", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("status", [422, 500, 503])
    async def test_reported_error_is_refunded(self, pooled, status):
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="bad"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("voice123", {"text": "Hello"})
        assert caught.value.status_code == status
        assert outcome_unknown(caught.value) is False

    @respx.mock
    async def test_redirect_is_refunded(self, pooled):
        """httpx does not follow redirects, so a 3xx never reached the API."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(307, text=""))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("voice123", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False


class TestInputTypes:
    """Wrong types are a 400 before payment, not a 500."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [["x" * 5000], 5, {"a": "b"}, "   "])
    async def test_bad_text_is_rejected(self, text):
        """A list must not be priced by its element count."""
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("voice123", {"text": text})
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_id", [5, ["eleven_turbo_v2"], {"a": 1}])
    async def test_non_string_model_id_is_rejected(self, model_id):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("voice123", {"text": "hi", "model_id": model_id})
        assert caught.value.status_code == 400
        assert "model_id" in caught.value.detail

    @pytest.mark.asyncio
    async def test_null_model_id_is_the_standard_rate(self):
        provider = _make_provider()
        price = await provider.get_price("voice123", {"text": "a" * 1000, "model_id": None})
        assert price == Decimal("0.30")
