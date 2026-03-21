"""Unit tests for the MiniMax TTS provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
from x402gate.providers.minimax import MinimaxProvider, _rate_for_model


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for MiniMax tests."""
    defaults = {
        "base_url": "https://api.minimaxi.com",
        "api_key": "test_minimax_key",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> MinimaxProvider:
    """Create a MinimaxProvider for tests."""
    config = _make_config(**overrides)
    return MinimaxProvider(config=config)


# ---------------------------------------------------------------------------
# _rate_for_model
# ---------------------------------------------------------------------------


class TestRateForModel:
    """Tests for _rate_for_model()."""

    def test_turbo_model(self):
        assert _rate_for_model("speech-02-turbo") == Decimal("0.06")

    def test_hd_model(self):
        assert _rate_for_model("speech-02-hd") == Decimal("0.10")

    def test_hd_in_name(self):
        assert _rate_for_model("speech-2.8-hd") == Decimal("0.10")

    def test_default_is_turbo(self):
        """Unknown model defaults to turbo pricing."""
        assert _rate_for_model("speech-02") == Decimal("0.06")


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------


class TestGetPrice:
    """Tests for MinimaxProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_turbo_pricing(self):
        """Turbo: $0.06 per 1k chars."""
        provider = _make_provider()
        price = await provider.get_price(
            "speech-02-turbo",
            {"text": "a" * 1000, "model": "speech-02-turbo"},
        )
        expected = Decimal("1000") / Decimal("1000") * Decimal("0.06")
        assert price == expected

    @pytest.mark.asyncio
    async def test_hd_pricing(self):
        """HD: $0.10 per 1k chars."""
        provider = _make_provider()
        price = await provider.get_price(
            "speech-02-hd",
            {"text": "a" * 1000, "model": "speech-02-hd"},
        )
        expected = Decimal("1000") / Decimal("1000") * Decimal("0.10")
        assert price == expected

    @pytest.mark.asyncio
    async def test_cyrillic_counts_as_characters(self):
        """Cyrillic = characters, not bytes."""
        provider = _make_provider()
        text = "Привет мир"  # 10 chars
        price = await provider.get_price("speech-02-turbo", {"text": text})
        expected = Decimal("10") / Decimal("1000") * Decimal("0.06")
        assert price == expected

    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        """Raises ProviderError for empty text."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.get_price("speech-02-turbo", {"text": ""})


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for MinimaxProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful TTS returns base64 audio."""
        provider = _make_provider()
        fake_audio_hex = b"\xff\xfb\x90\x00".hex()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": fake_audio_hex},
        }

        with patch("x402gate.providers.minimax.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.submit(
                "speech-02-turbo",
                {"text": "Hello world", "model": "speech-02-turbo"},
            )

        assert "audio_base64" in result
        assert result["content_type"] == "audio/mpeg"
        assert result["characters"] == len("Hello world")

    @pytest.mark.asyncio
    async def test_submit_api_error_status(self):
        """Raises on non-zero base_resp.status_code."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 1001, "status_msg": "invalid token"},
            "data": {},
        }

        with patch("x402gate.providers.minimax.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(Exception, match="MiniMax API error"):
                await provider.submit(
                    "speech-02-turbo",
                    {"text": "Hello", "model": "speech-02-turbo"},
                )

    @pytest.mark.asyncio
    async def test_submit_missing_text(self):
        """Raises ProviderError when text is missing."""
        provider = _make_provider()
        with pytest.raises(Exception, match="non-empty 'text'"):
            await provider.submit("speech-02-turbo", {})

    @pytest.mark.asyncio
    async def test_submit_http_error(self):
        """Raises ProviderError on HTTP error."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("x402gate.providers.minimax.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(Exception, match="MiniMax error"):
                await provider.submit(
                    "speech-02-turbo",
                    {"text": "Hello", "model": "speech-02-turbo"},
                )


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:
    """Tests for MinimaxProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Sync provider does not support polling."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="synchronous"):
            await provider.get_result("any_task_id")
