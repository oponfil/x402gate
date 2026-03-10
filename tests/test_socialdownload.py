"""Unit tests for the SocialDownload provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
from x402gate.providers.socialdownload import (
    SocialDownloadProvider,
    _select_best_media,
)


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for SocialDownload tests."""
    defaults = {
        "base_url": "https://social-download-all-in-one.p.rapidapi.com",
        "api_key": "test_rapidapi_key",
        "fixed_price_usd": 0.005,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> SocialDownloadProvider:
    """Create a SocialDownloadProvider for tests."""
    config = _make_config(**overrides)
    return SocialDownloadProvider(config=config)


# ---------------------------------------------------------------------------
# _select_best_media
# ---------------------------------------------------------------------------


class TestSelectBestMedia:
    """Tests for _select_best_media()."""

    def test_picks_combined_mp4_with_highest_resolution(self):
        """Prefers combined (audio+video) MP4 with highest resolution."""
        api_response = [
            {
                "medias": [
                    {
                        "type": "video",
                        "extension": "mp4",
                        "url": "https://low.mp4",
                        "width": 640,
                        "height": 360,
                        "is_audio": True,
                    },
                    {
                        "type": "video",
                        "extension": "mp4",
                        "url": "https://high.mp4",
                        "width": 1280,
                        "height": 720,
                        "is_audio": True,
                    },
                    {
                        "type": "video",
                        "extension": "mp4",
                        "url": "https://video-only.mp4",
                        "width": 1920,
                        "height": 1080,
                        "is_audio": False,
                    },
                ]
            }
        ]
        best = _select_best_media(api_response)
        assert best is not None
        assert best["url"] == "https://high.mp4"
        assert best["width"] == 1280

    def test_falls_back_to_non_combined_mp4(self):
        """Falls back to non-combined MP4 when no combined available."""
        api_response = [
            {
                "medias": [
                    {
                        "type": "video",
                        "extension": "mp4",
                        "url": "https://video.mp4",
                        "width": 1920,
                        "height": 1080,
                    },
                ]
            }
        ]
        best = _select_best_media(api_response)
        assert best is not None
        assert best["url"] == "https://video.mp4"

    def test_returns_none_for_empty_response(self):
        """Returns None for empty or invalid API response."""
        assert _select_best_media([]) is None
        assert _select_best_media(None) is None
        assert _select_best_media([{}]) is None
        assert _select_best_media([{"medias": []}]) is None

    def test_returns_none_for_error_response(self):
        """Returns None when API response contains an error."""
        assert _select_best_media([{"error": "URL not supported"}]) is None

    def test_fallback_to_any_media(self):
        """Falls back to first media with URL when no MP4 videos available."""
        api_response = [
            {
                "medias": [
                    {
                        "type": "audio",
                        "extension": "mp3",
                        "url": "https://audio.mp3",
                    },
                ]
            }
        ]
        best = _select_best_media(api_response)
        assert best is not None
        assert best["url"] == "https://audio.mp3"
        assert best["type"] == "audio"


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------


class TestGetPrice:
    """Tests for SocialDownloadProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_returns_fixed_price(self):
        """Returns the configured fixed price."""
        provider = _make_provider(fixed_price_usd=0.005)
        price = await provider.get_price("download", {})
        assert price == Decimal("0.005")

    @pytest.mark.asyncio
    async def test_raises_if_price_not_configured(self):
        """Raises ProviderError if fixed_price_usd is 0."""
        provider = _make_provider(fixed_price_usd=0.0)
        with pytest.raises(Exception, match="fixed_price_usd is not configured"):
            await provider.get_price("download", {})


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for SocialDownloadProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful submission returns media data."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "title": "Test Video",
                "author": "Test Author",
                "source": "youtube",
                "duration": 120,
                "thumbnail": "https://thumb.jpg",
                "medias": [
                    {
                        "type": "video",
                        "extension": "mp4",
                        "url": "https://download.mp4",
                        "width": 1280,
                        "height": 720,
                        "is_audio": True,
                    }
                ],
            }
        ]

        with patch("x402gate.providers.socialdownload.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.submit(
                "download", {"url": "https://www.youtube.com/watch?v=test"}
            )

        assert result["title"] == "Test Video"
        assert result["source"] == "youtube"
        assert result["best_media"]["url"] == "https://download.mp4"
        assert result["best_media"]["width"] == 1280

    @pytest.mark.asyncio
    async def test_submit_missing_url(self):
        """Raises ProviderError when URL field is missing."""
        provider = _make_provider()
        with pytest.raises(Exception, match="must contain 'url' field"):
            await provider.submit("download", {})

    @pytest.mark.asyncio
    async def test_submit_api_error(self):
        """Raises ProviderError on RapidAPI error response."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too Many Requests"

        with patch("x402gate.providers.socialdownload.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(Exception, match="RapidAPI error"):
                await provider.submit("download", {"url": "https://www.youtube.com/watch?v=test"})

    @pytest.mark.asyncio
    async def test_submit_no_suitable_media(self):
        """Returns result without best_media when no suitable media found."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "title": "No Media",
                "source": "tiktok",
                "medias": [],
            }
        ]

        with patch("x402gate.providers.socialdownload.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.submit(
                "download", {"url": "https://www.tiktok.com/@user/video/123"}
            )

        assert "best_media" not in result


# ---------------------------------------------------------------------------
# get_result
# ---------------------------------------------------------------------------


class TestGetResult:
    """Tests for SocialDownloadProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self):
        """Sync provider does not support polling."""
        provider = _make_provider()
        with pytest.raises(NotImplementedError, match="synchronous"):
            await provider.get_result("any_task_id")
