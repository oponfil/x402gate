"""Unit tests for the SocialDownload provider."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
from x402gate.providers.base import ProviderError, outcome_unknown
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
        price = await provider.get_price("download", {"url": "https://youtu.be/x"})
        assert price == Decimal("0.005")

    @pytest.mark.asyncio
    async def test_accepts_schemeless_url(self):
        """A link without a scheme is still priced; RapidAPI resolves it."""
        provider = _make_provider(fixed_price_usd=0.005)
        assert await provider.get_price("download", {"url": "youtu.be/x"}) == Decimal("0.005")

    @pytest.mark.asyncio
    async def test_raises_if_price_not_configured(self):
        """Raises ProviderError if fixed_price_usd is 0."""
        provider = _make_provider(fixed_price_usd=0.0)
        with pytest.raises(Exception, match="fixed_price_usd is not configured"):
            await provider.get_price("download", {"url": "https://youtu.be/x"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body", [{}, {"url": ""}, {"url": "   "}, {"url": 123}, {"url": ["x"]}, {"url": {"a": 1}}]
    )
    async def test_rejects_missing_url_before_payment(self, body):
        """A body without a url is refused by get_price(), before payment."""
        provider = _make_provider()
        with pytest.raises(ProviderError, match="'url' field") as exc_info:
            await provider.get_price("download", body)
        assert exc_info.value.status_code == 400


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

        with patch("x402gate.providers.socialdownload.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

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

        with patch("x402gate.providers.socialdownload.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            with pytest.raises(Exception, match="RapidAPI error"):
                await provider.submit("download", {"url": "https://www.youtube.com/watch?v=test"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            ({"error": "URL not supported"}, "RapidAPI returned error: URL not supported"),
            (
                {"error": True, "message": "Unable to extract media"},
                "RapidAPI returned error: Unable to extract media",
            ),
            ({"error": True, "msg": "Private video"}, "RapidAPI returned error: Private video"),
            ({"error": True}, 'RapidAPI returned error: {"error": true}'),
        ],
    )
    async def test_submit_error_item_keeps_the_message(self, item, expected):
        """A 200 response with an error item reports RapidAPI's own message."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [item]

        with patch("x402gate.providers.socialdownload.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            with pytest.raises(ProviderError) as exc_info:
                await provider.submit("download", {"url": "https://www.youtube.com/watch?v=test"})

        assert exc_info.value.detail == expected
        # RapidAPI answered 200, so it ran and billed the call: keep the payment.
        assert outcome_unknown(exc_info.value) is True

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

        with patch("x402gate.providers.socialdownload.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            result = await provider.submit(
                "download", {"url": "https://www.tiktok.com/@user/video/123"}
            )

        assert "best_media" not in result


@contextmanager
def _rapidapi_answers(response: MagicMock):
    """Patch RapidAPI to answer with ``response``."""
    with patch("x402gate.providers.socialdownload.shared_client") as mock_shared:
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        mock_shared.return_value = mock_client
        yield


class TestBilling:
    """Which failures refund the client and which keep the payment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_is_502_and_refunded(self, status):
        """An upstream 402 must not reach x402 clients as a payment request."""
        response = MagicMock(status_code=status, text="Invalid API key")
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @pytest.mark.asyncio
    async def test_error_status_is_refunded(self):
        response = MagicMock(status_code=429, text="Too Many Requests")
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert caught.value.status_code == 429
        assert outcome_unknown(caught.value) is False

    @pytest.mark.asyncio
    async def test_redirect_is_refunded(self):
        """httpx does not follow redirects, so a 3xx never ran the call."""
        response = MagicMock(status_code=307, text="")
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False
        response.json.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            ValueError("Expecting value"),
            "just a string",
            [None],
            [{"medias": {"not": "a list"}}],
        ],
    )
    async def test_unreadable_success_body_keeps_the_payment(self, body):
        """A 200 we cannot parse still ran, and was billed, at RapidAPI."""
        response = MagicMock(status_code=200, text="garbled")
        if isinstance(body, Exception):
            response.json.side_effect = body
        else:
            response.json.return_value = body
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert "unreadable response" in caught.value.detail
        assert outcome_unknown(caught.value) is True

    @pytest.mark.asyncio
    async def test_null_title_does_not_fail(self):
        response = MagicMock(status_code=200)
        response.json.return_value = [{"title": None, "source": "x", "medias": []}]
        with _rapidapi_answers(response):
            result = await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert result["title"] is None


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


# ---------------------------------------------------------------------------
# pooled connection
# ---------------------------------------------------------------------------


_API_URL = "https://social-download-all-in-one.p.rapidapi.com/v1/social/autolink"


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
        route = respx.post(_API_URL).mock(
            return_value=httpx.Response(200, json=[{"title": "T", "medias": []}])
        )
        with patch("x402gate.core.http.httpx.AsyncClient", wraps=httpx.AsyncClient) as ctor:
            await provider.submit("download", {"url": "https://youtu.be/x"})
            await provider.submit("download", {"url": "https://youtu.be/x"})
        assert ctor.call_count == 1
        assert route.call_count == 2

    @respx.mock
    async def test_keeps_the_read_timeout_with_a_short_connect_limit(self, pooled):
        """The old per-call timeout still bounds the read; connecting fails fast."""
        provider = _make_provider()
        route = respx.post(_API_URL).mock(
            return_value=httpx.Response(200, json=[{"title": "T", "medias": []}])
        )
        await provider.submit("download", {"url": "https://youtu.be/x"})
        limits = route.calls.last.request.extensions["timeout"]
        assert limits["read"] == 30
        assert limits["connect"] == http.CONNECT_TIMEOUT_S

    @respx.mock
    async def test_read_timeout_keeps_the_payment(self, pooled):
        """The request was sent and may have run: outcome unknown, as before."""
        provider = _make_provider()
        respx.post(_API_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("download", {"url": "https://youtu.be/x"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    async def test_connection_never_opened_is_refunded(self, pooled, error):
        """Nothing was sent, so the client is refunded, as before."""
        provider = _make_provider()
        respx.post(_API_URL).mock(side_effect=error("down"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("download", {"url": "https://youtu.be/x"})
        assert outcome_unknown(caught.value) is False


class TestGatewayTimeout:
    """A gateway timeout came after the call was forwarded: outcome unknown."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_gateway_timeout_keeps_the_payment(self, status):
        response = MagicMock(status_code=status, text="Gateway Timeout")
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True
        response.json.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503, 521])
    async def test_other_server_errors_are_refunded(self, status):
        response = MagicMock(status_code=status, text="down")
        with _rapidapi_answers(response), pytest.raises(ProviderError) as caught:
            await _make_provider().submit("download", {"url": "https://x.com/a/status/1"})
        assert caught.value.status_code == status
        assert outcome_unknown(caught.value) is False
