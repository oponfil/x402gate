"""Unit tests for the FishAudio TTS provider."""

from __future__ import annotations

import base64
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
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

        with patch("x402gate.providers.fishaudio.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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

        with patch("x402gate.providers.fishaudio.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("x402gate.providers.fishaudio.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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
