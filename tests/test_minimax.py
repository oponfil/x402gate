"""Unit tests for the MiniMax TTS provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.providers.base import ProviderError, outcome_unknown
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

    def test_default_is_hd(self):
        """Unknown model defaults to HD (expensive) pricing."""
        assert _rate_for_model("speech-02") == Decimal("0.10")


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
    async def test_config_pricing_overrides_defaults(self):
        """Rates from config.yaml pricing are used instead of the defaults."""
        provider = _make_provider(pricing={"per_1k_chars_turbo": 0.5, "per_1k_chars_hd": 2})
        body = {"text": "a" * 1000}
        assert await provider.get_price("speech-2.8-turbo", body) == Decimal("0.5")
        assert await provider.get_price("speech-2.8-hd", body) == Decimal("2")

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

        with patch("x402gate.providers.minimax.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

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

        with patch("x402gate.providers.minimax.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

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

        with patch("x402gate.providers.minimax.shared_client") as mock_shared:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_shared.return_value = mock_client

            with pytest.raises(Exception, match="MiniMax error"):
                await provider.submit(
                    "speech-02-turbo",
                    {"text": "Hello", "model": "speech-02-turbo"},
                )


# ---------------------------------------------------------------------------
# calculate_actual_cost
# ---------------------------------------------------------------------------


def _mock_client(mock_shared, payload: dict) -> None:
    """Make the patched shared client return a 200 with ``payload``."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = payload
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_shared.return_value = mock_client


class TestActualCost:
    """Actual cost follows MiniMax's own usage_characters."""

    @pytest.mark.asyncio
    async def test_uses_usage_characters(self):
        """The billed count from extra_info prices the call, at the path model's rate."""
        provider = _make_provider()
        body = {"text": "Hello world"}
        with patch("x402gate.providers.minimax.shared_client") as mock_shared:
            _mock_client(
                mock_shared,
                {
                    "base_resp": {"status_code": 0},
                    "data": {"audio": b"\xff\xfb".hex()},
                    "extra_info": {"usage_characters": 500},
                },
            )
            result = await provider.submit("speech-02-turbo", body)

        assert result["usage_characters"] == 500
        cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal("500") / Decimal("1000") * Decimal("0.06")

    @pytest.mark.asyncio
    async def test_missing_usage_falls_back_to_estimate(self):
        """Without usage_characters the estimate is used (None)."""
        provider = _make_provider()
        body = {"text": "Hello", "model": "speech-02-hd"}
        with patch("x402gate.providers.minimax.shared_client") as mock_shared:
            _mock_client(
                mock_shared,
                {
                    "base_resp": {"status_code": 0},
                    "data": {"audio": b"\xff\xfb".hex()},
                    "extra_info": {"usage_characters": "lots"},
                },
            )
            result = await provider.submit("speech-02-hd", body)

        assert "usage_characters" not in result
        assert await provider.calculate_actual_cost(body, result) is None


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


# ---------------------------------------------------------------------------
# pooled connection
# ---------------------------------------------------------------------------


_TTS_URL = "https://api.minimaxi.com/v1/t2a_v2"


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
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": b"\xff\xfb".hex()}}
            )
        )
        with patch("x402gate.core.http.httpx.AsyncClient", wraps=httpx.AsyncClient) as ctor:
            await provider.submit("speech-02-hd", {"text": "Hello"})
            await provider.submit("speech-02-hd", {"text": "Hello"})
        assert ctor.call_count == 1
        assert route.call_count == 2

    @respx.mock
    async def test_keeps_the_read_timeout_with_a_short_connect_limit(self, pooled):
        """The old per-call timeout still bounds the read; connecting fails fast."""
        provider = _make_provider()
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": b"\xff\xfb".hex()}}
            )
        )
        await provider.submit("speech-02-hd", {"text": "Hello"})
        limits = route.calls.last.request.extensions["timeout"]
        assert limits["read"] == 600
        assert limits["connect"] == http.CONNECT_TIMEOUT_S

    @respx.mock
    async def test_read_timeout_keeps_the_payment(self, pooled):
        """The request was sent and may have run: outcome unknown, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    async def test_connection_never_opened_is_refunded(self, pooled, error):
        """Nothing was sent, so the client is refunded, as before."""
        provider = _make_provider()
        respx.post(_TTS_URL).mock(side_effect=error("down"))
        with pytest.raises(ProviderError) as caught:
            await provider.submit("speech-02-hd", {"text": "Hello"})
        assert outcome_unknown(caught.value) is False
