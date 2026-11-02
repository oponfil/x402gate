"""Unit tests for the FishAudio TTS provider."""

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
from x402gate.providers.fishaudio import FishAudioProvider


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for FishAudio tests."""
    defaults = {
        "base_url": "https://api.fish.audio",
        "api_key": "test_fishaudio_key",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> FishAudioProvider:
    """Create a FishAudioProvider for tests."""
    config = _make_config(**overrides)
    return FishAudioProvider(config=config)


# ---------------------------------------------------------------------------
# get_price — UTF-8 byte-based pricing
# ---------------------------------------------------------------------------


class TestGetPrice:
    """Tests for FishAudioProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_ascii_pricing(self):
        """ASCII: 1 byte per char → 1000 bytes for 1000 chars."""
        provider = _make_provider()
        text = "a" * 1000
        price = await provider.get_price("tts", {"text": text})
        expected = Decimal("1000") / Decimal("1000") * Decimal("0.015")
        assert price == expected

    @pytest.mark.asyncio
    async def test_cyrillic_pricing(self):
        """Cyrillic: 2 bytes per char in UTF-8."""
        provider = _make_provider()
        text = "Привет"  # 6 chars, 12 UTF-8 bytes
        assert len(text.encode("utf-8")) == 12
        price = await provider.get_price("tts", {"text": text})
        expected = Decimal("12") / Decimal("1000") * Decimal("0.015")
        assert price == expected

    @pytest.mark.asyncio
    async def test_emoji_pricing(self):
        """Emoji: 4 bytes per char in UTF-8."""
        provider = _make_provider()
        text = "😀" * 10  # 10 emoji, 40 bytes
        assert len(text.encode("utf-8")) == 40
        price = await provider.get_price("tts", {"text": text})
        expected = Decimal("40") / Decimal("1000") * Decimal("0.015")
        assert price == expected

    @pytest.mark.asyncio
    async def test_mixed_script_pricing(self):
        """Mixed: ASCII + Cyrillic + Emoji."""
        provider = _make_provider()
        text = "Hi Привет 😀"  # 3+1+12+1+4 = 21 bytes
        utf8_len = len(text.encode("utf-8"))
        price = await provider.get_price("tts", {"text": text})
        expected = Decimal(str(utf8_len)) / Decimal("1000") * Decimal("0.015")
        assert price == expected

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        """Raises ProviderError for empty text."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.get_price("tts", {"text": ""})


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for FishAudioProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful TTS returns base64 audio with UTF-8 byte count."""
        provider = _make_provider()
        fake_audio = b"\xff\xfb\x90\x00" * 100
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio
        mock_response.headers = {"content-type": "audio/mpeg"}

        with patch("x402gate.providers.fishaudio.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            result = await provider.submit(
                "tts",
                {"text": "Hello world", "format": "mp3"},
            )

        assert "audio_base64" in result
        assert result["content_type"] == "audio/mpeg"
        assert result["characters"] == len("Hello world")
        assert result["utf8_bytes"] == len(b"Hello world")
        decoded = base64.b64decode(result["audio_base64"])
        assert decoded == fake_audio

    @pytest.mark.asyncio
    async def test_submit_cyrillic_returns_utf8_bytes(self):
        """Cyrillic text correctly reports UTF-8 byte count."""
        provider = _make_provider()
        text = "Привет мир"
        fake_audio = b"\x00" * 50
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio
        mock_response.headers = {"content-type": "audio/mpeg"}

        with patch("x402gate.providers.fishaudio.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            result = await provider.submit("tts", {"text": text})

        assert result["characters"] == 10  # 10 chars
        assert result["utf8_bytes"] == 19  # 9 cyrillic * 2 + 1 space

    @pytest.mark.asyncio
    async def test_submit_missing_text(self):
        """Raises ProviderError when text is missing."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.submit("tts", {})

    @pytest.mark.asyncio
    async def test_submit_api_error(self):
        """Raises ProviderError on API error."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("x402gate.providers.fishaudio.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            with pytest.raises(Exception, match="FishAudio error"):
                await provider.submit("tts", {"text": "Hello"})


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:
    """Tests for FishAudioProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Sync provider does not support polling."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="synchronous"):
            await provider.get_result("any_task_id")


# ---------------------------------------------------------------------------
# pooled connection
# ---------------------------------------------------------------------------


_TTS_URL = "https://api.fish.audio/v1/tts"


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
            await provider.submit("tts", {"text": "Hello"})
            await provider.submit("tts", {"text": "Hello"})
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
        await provider.submit("tts", {"text": "Hello"})
        limits = route.calls.last.request.extensions["timeout"]
        assert limits["read"] == 600
        assert limits["connect"] == http.CONNECT_TIMEOUT_S

    @respx.mock
    async def test_read_timeout_keeps_the_payment(self, pooled):
        """The request was sent and may have run: outcome unknown, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("tts", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    async def test_connection_never_opened_is_refunded(self, pooled, error):
        """Nothing was sent, so the client is refunded, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=error("down"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("tts", {"text": "Hello"})
        assert outcome_unknown(caught.value) is False


class TestUpstreamStatus:
    """How Fish Audio's error statuses are billed and shown to the client."""

    @respx.mock
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_hides_the_body(self, pooled, status):
        """Our account state stays private, and a 402 is not an x402 request."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(status, text="balance: 0.12 credits left")
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("tts", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @respx.mock
    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_gateway_timeout_keeps_the_payment(self, pooled, status):
        """Cloudflare gave up after forwarding: the audio may be made and billed."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="timeout"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("tts", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("status", [400, 500, 503])
    async def test_reported_error_is_refunded(self, pooled, status):
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="bad"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("tts", {"text": "Hello"})
        assert caught.value.status_code == status
        assert outcome_unknown(caught.value) is False

    @respx.mock
    async def test_redirect_is_refunded(self, pooled):
        """httpx does not follow redirects, so a 3xx never reached the API."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(307, text=""))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("tts", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False


class TestInputTypes:
    """Wrong types are a 400 before payment, not a 500."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [["hi"], 5, {"a": "b"}, "   "])
    async def test_bad_text_is_rejected(self, text):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("tts", {"text": text})
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["model", "reference_id"])
    @pytest.mark.parametrize("value", [5, ["s1"], {"a": 1}])
    async def test_non_string_model_or_reference_is_rejected(self, key, value):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("tts", {"text": "hi", key: value})
        assert caught.value.status_code == 400
        assert key in caught.value.detail

    @respx.mock
    async def test_null_model_is_left_out(self, pooled):
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(200, content=b"a", headers={"content-type": "audio/mpeg"})
        )
        body = {"text": "hi", "model": None, "reference_id": None}
        await _make_provider().get_price("tts", body)
        await _make_provider().submit("tts", body)
        request = route.calls.last.request
        assert "model" not in request.headers
        assert b"reference_id" not in request.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model", ["modèle", "a\r\nb", "a\r\nX-Evil: 1", "a\nb", "ok\x00", " lead", "s1\n", "m" * 65]
    )
    async def test_model_that_cannot_be_a_header_is_rejected(self, model):
        """httpx would fail on this header only after payment, so it is a 400 first."""
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("tts", {"text": "hi", "model": model})
        assert caught.value.status_code == 400
        assert "model" in caught.value.detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["s1", "speech-1.6", "s1_mini", "m" * 64])
    async def test_plain_model_name_is_priced(self, model):
        price = await _make_provider().get_price("tts", {"text": "hi", "model": model})
        assert price == Decimal("0.00003")

    @respx.mock
    async def test_bad_model_is_rejected_in_submit_without_a_request(self, pooled):
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(200, content=b"a", headers={"content-type": "audio/mpeg"})
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("tts", {"text": "hi", "model": "modèle"})
        assert caught.value.status_code == 400
        assert outcome_unknown(caught.value) is False
        assert not route.called

    @respx.mock
    async def test_plain_model_is_sent_as_a_header(self, pooled):
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(200, content=b"a", headers={"content-type": "audio/mpeg"})
        )
        await _make_provider().submit("tts", {"text": "hi", "model": "speech-1.6"})
        assert route.calls.last.request.headers["model"] == "speech-1.6"
