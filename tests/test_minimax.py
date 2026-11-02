"""Unit tests for the MiniMax TTS provider."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
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


class TestUpstreamStatus:
    """How MiniMax's error statuses are billed and shown to the client."""

    @respx.mock
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_hides_the_body(self, pooled, status):
        """Our account state stays private, and a 402 is not an x402 request."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(status, text="insufficient balance: 0.40")
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @respx.mock
    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_gateway_timeout_keeps_the_payment(self, pooled, status):
        """A proxy gave up after forwarding: the audio may be made and billed."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="timeout"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == 504
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("status", [400, 500, 503])
    async def test_reported_error_is_refunded(self, pooled, status):
        respx.post(_TTS_URL).mock(return_value=httpx.Response(status, text="bad"))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == status
        assert outcome_unknown(caught.value) is False

    @respx.mock
    @pytest.mark.parametrize("content", [b"not json", b"[1, 2]", b"null"])
    async def test_unreadable_200_keeps_the_payment(self, pooled, content):
        respx.post(_TTS_URL).mock(return_value=httpx.Response(200, content=content))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert "unreadable response" in caught.value.detail
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize(
        ("code", "msg"),
        [
            (1004, "login fail: Please carry the API secret key"),
            (1008, "insufficient balance"),
            (2049, "invalid api key"),
        ],
    )
    async def test_rejected_key_in_base_resp_hides_the_message(self, pooled, code, msg):
        """MiniMax reports a bad key or low balance as HTTP 200 with base_resp."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": code, "status_msg": msg}}
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert msg not in caught.value.detail
        assert outcome_unknown(caught.value) is False

    @respx.mock
    async def test_content_error_in_base_resp_keeps_the_message(self, pooled):
        """An error the client caused still tells the client what was wrong."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 1026, "status_msg": "sensitive input"}}
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.detail == "MiniMax API error: sensitive input"
        assert outcome_unknown(caught.value) is False

    @respx.mock
    async def test_null_parts_do_not_crash(self, pooled):
        """A null base_resp or data is read as missing, not an AttributeError."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(200, json={"base_resp": None, "data": None})
        )
        with pytest.raises(ProviderError, match="no audio data") as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        # No error was reported on a 2xx, so the audio may be billed.
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize(
        "body",
        [
            {"base_resp": {"status_code": 0}, "data": None, "extra_info": {"usage_characters": 5}},
            {"base_resp": {"status_code": 0}, "data": {"audio": ""}},
            {"base_resp": {"status_code": 0}, "data": {}},
            {"data": {}},
            {},
        ],
    )
    async def test_success_without_audio_keeps_the_payment(self, pooled, body):
        """A 2xx that reports success but sends no audio may still be billed."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.detail == "MiniMax returned no audio data"
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("audio", [5, {"x": 1}, ["ff"], True, None])
    async def test_non_string_audio_keeps_the_payment(self, pooled, audio):
        """A non-string audio field is a ProviderError, not a TypeError crash."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": audio}}
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.detail == "MiniMax returned no audio data"
        assert outcome_unknown(caught.value) is True

    @respx.mock
    @pytest.mark.parametrize("audio", ["zz!!", "not base64 at all", "QUJD@@"])
    async def test_undecodable_audio_keeps_the_payment(self, pooled, audio):
        """Junk is neither hex nor strict base64, so it is not served as audio."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": audio}}
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.detail == "MiniMax returned audio that could not be decoded"
        assert outcome_unknown(caught.value) is True

    @respx.mock
    async def test_base64_audio_is_still_accepted(self, pooled):
        """Audio sent as valid base64 instead of hex is still decoded."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": "SUQzBA=="}}
            )
        )
        result = await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert result["audio_base64"] == "SUQzBA=="

    @respx.mock
    async def test_reported_error_without_audio_is_refunded(self, pooled):
        """A nonzero base_resp code is a failure MiniMax reported, even with data null."""
        respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200,
                json={"base_resp": {"status_code": 2013, "status_msg": "bad voice"}, "data": None},
            )
        )
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.detail == "MiniMax API error: bad voice"
        assert outcome_unknown(caught.value) is False

    @respx.mock
    async def test_redirect_is_refunded(self, pooled):
        """httpx does not follow redirects, so a 3xx never reached the API."""
        respx.post(_TTS_URL).mock(return_value=httpx.Response(307, text=""))
        with pytest.raises(ProviderError) as caught:
            await _make_provider().submit("speech-02-hd", {"text": "Hello"})
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False


class TestInputTypes:
    """Wrong types are a 400 before payment, not a 500 or a crash after billing."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [["hi"], 5, "   "])
    async def test_bad_text_is_rejected(self, text):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("speech-02-hd", {"text": text})
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [5, ["speech-02-turbo"]])
    async def test_non_string_model_is_rejected(self, model):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("speech-02-hd", {"text": "hi", "model": model})
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_null_model_uses_the_path(self):
        provider = _make_provider()
        price = await provider.get_price("speech-02-turbo", {"text": "a" * 1000, "model": None})
        assert price == Decimal("0.06")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "audio_setting", ["mp3", 5, ["mp3"], {"format": ["mp3"]}, {"format": 123}, {"format": {}}]
    )
    async def test_bad_audio_setting_is_rejected(self, audio_setting):
        with pytest.raises(ProviderError) as caught:
            await _make_provider().get_price(
                "speech-02-hd", {"text": "hi", "audio_setting": audio_setting}
            )
        assert caught.value.status_code == 400
        assert "audio_setting" in caught.value.detail

    @respx.mock
    async def test_null_audio_setting_is_dropped(self, pooled):
        """Priced, sent without audio_setting, and answered as mp3."""
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": "ff00"}}
            )
        )
        body = {"text": "hi", "audio_setting": None}
        await _make_provider().get_price("speech-02-hd", body)
        result = await _make_provider().submit("speech-02-hd", body)
        assert result["content_type"] == "audio/mpeg"
        assert b"audio_setting" not in route.calls.last.request.content

    @respx.mock
    async def test_audio_setting_format_sets_the_content_type(self, pooled):
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": "ff00"}}
            )
        )
        body = {"text": "hi", "audio_setting": {"format": "wav"}}
        result = await _make_provider().submit("speech-02-hd", body)
        assert result["content_type"] == "audio/wav"
        assert b'"format":"wav"' in route.calls.last.request.content.replace(b" ", b"")

    @pytest.mark.asyncio
    async def test_null_format_is_priced_normally(self):
        """A null format counts as missing, like a null audio_setting."""
        price = await _make_provider().get_price(
            "speech-02-hd", {"text": "a" * 1000, "audio_setting": {"format": None}}
        )
        assert price == Decimal("0.10")

    @respx.mock
    async def test_null_format_is_left_out_and_answered_as_mp3(self, pooled):
        """MiniMax applies its own mp3 default, so audio/mpeg is right."""
        route = respx.post(_TTS_URL).mock(
            return_value=httpx.Response(
                200, json={"base_resp": {"status_code": 0}, "data": {"audio": "ff00"}}
            )
        )
        body = {"text": "hi", "audio_setting": {"format": None, "sample_rate": 32000}}
        result = await _make_provider().submit("speech-02-hd", body)
        assert result["content_type"] == "audio/mpeg"
        sent = json.loads(route.calls.last.request.content)
        assert sent["audio_setting"] == {"sample_rate": 32000}
        assert body["audio_setting"] == {"format": None, "sample_rate": 32000}
