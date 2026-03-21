"""Unit tests for the ElevenLabs TTS provider."""

from __future__ import annotations

import base64
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
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

        with patch("x402gate.providers.elevenlabs.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("x402gate.providers.elevenlabs.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(Exception, match="ElevenLabs error"):
                await provider.submit("voice123", {"text": "Hello"})


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
