"""Unit tests for the Tungsten provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
from x402gate.providers.tungsten import TungstenProvider


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for Tungsten tests."""
    defaults = {
        "base_url": "https://api.tungsten.run/v1",
        "jwt_token": "test_jwt_token",
        "cf_clearance": "test_cf_clearance",
        "fixed_price_usd": 0.01,
        "poll_interval": 1,
        "poll_timeout": 10,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> TungstenProvider:
    """Create a TungstenProvider for tests."""
    config = _make_config(**overrides)
    return TungstenProvider(config=config)


class TestGetPrice:
    """Tests for TungstenProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_returns_fixed_price(self):
        """Returns the configured fixed price."""
        provider = _make_provider(fixed_price_usd=0.01)
        price = await provider.get_price("generations", {})
        assert price == Decimal("0.01")

    @pytest.mark.asyncio
    async def test_returns_custom_price(self):
        """Returns a custom fixed price."""
        provider = _make_provider(fixed_price_usd=0.05)
        price = await provider.get_price("any/model", {"prompt": "test"})
        assert price == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_raises_if_price_not_configured(self):
        """Raises ProviderError if fixed_price_usd is 0."""
        provider = _make_provider(fixed_price_usd=0.0)
        with pytest.raises(Exception, match="fixed_price_usd is not configured"):
            await provider.get_price("generations", {})


class TestSubmit:
    """Tests for TungstenProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful submission returns x402gate-compatible async task format."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"uuid": "img_abc123", "generation": {"uuid": "gen_abc123"}, "status": "pending"}
        ]

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            result = await provider.submit("generations", {"type": "sdxl", "data": {}})

        assert result["data"]["id"] == "gen_abc123"
        assert result["data"]["status"] == "processing"

    @pytest.mark.asyncio
    async def test_submit_auth_failure(self):
        """401 response raises ProviderError about expired cookies."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            with pytest.raises(Exception, match="authentication failed"):
                await provider.submit("generations", {})

    @pytest.mark.asyncio
    async def test_submit_bad_request(self):
        """400 response raises ProviderError with detail."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = [{"code": 0, "detail": "Some model versions are missing"}]

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            with pytest.raises(Exception, match="model versions are missing"):
                await provider.submit("generations", {})

    @pytest.mark.asyncio
    async def test_submit_prompt_blocked(self):
        """Safety-filtered prompt returns descriptive error."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = [
            {
                "code": 30100,
                "detail_struct": {"blocked": ["word1", "word2"], "prompt": "test"},
            }
        ]

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get.return_value = mock_client

            with pytest.raises(Exception, match="Prompt blocked"):
                await provider.submit("generations", {})


class TestGetResult:
    """Tests for TungstenProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_poll_until_completed(self):
        """Polls until status is 'completed' and downloads images."""
        provider = _make_provider(poll_interval=0, poll_timeout=10)

        # First poll: pending; second poll: completed with image_upload_uuids
        track_pending = MagicMock()
        track_pending.status_code = 200
        track_pending.json.return_value = [{"uuid": "gen_123", "status": "pending"}]

        track_completed = MagicMock()
        track_completed.status_code = 200
        track_completed.json.return_value = [
            {
                "uuid": "gen_123",
                "status": "success",
                "image_upload_uuids": {"img_uuid_001": "upload_uuid_001"},
            }
        ]

        # Image download response
        download_response = MagicMock()
        download_response.status_code = 200
        download_response.content = b"fake_png_data"
        download_response.raise_for_status = MagicMock()

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [track_pending, track_completed]
            mock_client.get.return_value = download_response
            mock_get.return_value = mock_client

            result = await provider.get_result("gen_123")

        assert result["count"] == 1
        assert len(result["images"]) == 1
        assert result["images"][0]["base64_png"]  # non-empty base64

    @pytest.mark.asyncio
    async def test_poll_generation_failed(self):
        """Raises ProviderError if generation fails."""
        provider = _make_provider(poll_interval=0, poll_timeout=10)

        track_failed = MagicMock()
        track_failed.status_code = 200
        track_failed.json.return_value = [
            {
                "uuid": "gen_fail",
                "status": "failed",
                "failure_reason": "NSFW content detected",
            }
        ]

        with patch.object(provider, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = track_failed
            mock_get.return_value = mock_client

            with pytest.raises(Exception, match="Generation failed"):
                await provider.get_result("gen_fail")


class TestAuthHeaders:
    """Tests for cookie-based authentication."""

    def test_headers_include_cookies(self):
        """Auth headers contain jwt and cf_clearance cookies."""
        provider = _make_provider()
        headers = provider._get_headers()
        assert "jwt=test_jwt_token" in headers["Cookie"]
        assert "cf_clearance=test_cf_clearance" in headers["Cookie"]

    def test_headers_include_browser_fingerprint(self):
        """Headers include User-Agent, Origin, and Referer."""
        provider = _make_provider()
        headers = provider._get_headers()
        assert "Mozilla" in headers["User-Agent"]
        assert headers["Origin"] == "https://tungsten.run"
        assert headers["Referer"] == "https://tungsten.run/"
