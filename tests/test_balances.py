"""Unit tests for upstream provider balance checks."""

import time
from decimal import Decimal

import httpx
import pytest
import respx

from x402gate.core import balances
from x402gate.core.balances import (
    clear_balance_cache,
    fetch_balances,
    parse_cloudconvert,
    parse_elevenlabs,
    parse_fishaudio,
    parse_openrouter,
    parse_openrouter_key,
    parse_spicyapi,
    parse_wavespeed,
)
from x402gate.core.config import ProviderConfig


def _config(base_url: str, api_key: str = "test-key") -> ProviderConfig:
    return ProviderConfig(base_url=base_url, api_key=api_key)


class TestParsers:
    """Balance payloads from each provider."""

    def test_openrouter_remaining_credits(self):
        parsed = parse_openrouter({"data": {"total_credits": 20, "total_usage": 12.5}})
        assert parsed["label"] == "$7.50"
        assert parsed["low"] is True

    def test_openrouter_key_limit(self):
        parsed = parse_openrouter_key({"data": {"limit": 100, "limit_remaining": 74.5}})
        assert parsed is not None
        assert parsed["label"] == "$74.50"
        assert parsed["low"] is False

    def test_openrouter_key_uses_the_tighter_budget(self):
        parsed = parse_openrouter_key(
            {
                "data": {
                    "limit_remaining": 20,
                    "effective_budget": {"remaining_usd": 3.5},
                }
            }
        )
        assert parsed is not None
        assert parsed["label"] == "$3.50"
        assert parsed["low"] is True

    def test_openrouter_unlimited_key_has_no_balance(self):
        assert parse_openrouter_key({"data": {"limit": None, "limit_remaining": None}}) is None

    def test_openrouter_under_five_dollars_is_low(self):
        parsed = parse_openrouter({"data": {"total_credits": 4, "total_usage": 0}})
        assert parsed["label"] == "$4.00"
        assert parsed["low"] is True

    def test_wavespeed_balance(self):
        parsed = parse_wavespeed({"code": 200, "data": {"balance": 50.25}})
        assert parsed["label"] == "$50.25"
        assert parsed["unit"] == "usd"

    def test_spicyapi_uses_spendable_balance(self):
        parsed = parse_spicyapi(
            {"code": 200, "data": {"available": "1.20", "held": "3.00", "total": "4.20"}}
        )
        assert parsed["label"] == "$1.20"
        assert parsed["low"] is True

    def test_spicyapi_negative_balance(self):
        parsed = parse_spicyapi({"code": 200, "data": {"available": "-0.50"}})
        assert parsed["label"] == "-$0.50"
        assert parsed["low"] is True

    def test_cloudconvert_credits(self):
        assert parse_cloudconvert({"data": {"credits": 500}})["low"] is False
        low = parse_cloudconvert({"data": {"credits": 299}})
        assert low["label"] == "299 credits"
        assert low["low"] is True

    def test_fishaudio_credit_is_dollars(self):
        parsed = parse_fishaudio({"credit": "12.5"})
        assert parsed["label"] == "$12.50"
        assert parsed["low"] is True

    def test_elevenlabs_characters_remaining(self):
        parsed = parse_elevenlabs({"character_limit": 100000, "character_count": 1000})
        assert parsed["label"] == "99,000 chars"
        assert parsed["unit"] == "characters"
        assert parsed["low"] is False

    def test_elevenlabs_under_ten_thousand_characters_is_low(self):
        parsed = parse_elevenlabs({"character_limit": 100000, "character_count": 90001})
        assert parsed["label"] == "9,999 chars"
        assert parsed["low"] is True

    def test_elevenlabs_ten_thousand_characters_is_enough(self):
        parsed = parse_elevenlabs({"character_limit": 100000, "character_count": 90000})
        assert parsed["label"] == "10,000 chars"
        assert parsed["low"] is False

    def test_elevenlabs_used_up_is_low(self):
        parsed = parse_elevenlabs({"character_limit": 10000, "character_count": 12000})
        assert parsed["label"] == "-2,000 chars"
        assert parsed["low"] is True


class TestFetch:
    """HTTP fetch, cache, and providers without a balance API."""

    def setup_method(self) -> None:
        """Drop balances cached by the previous test."""
        clear_balance_cache()

    @respx.mock
    async def test_fetches_openrouter_and_caches(self):
        route = respx.get("https://openrouter.ai/api/v1/key").mock(
            return_value=httpx.Response(200, json={"data": {"limit": 10, "limit_remaining": 7}})
        )
        configs = {"openrouter": _config("https://openrouter.ai/api/v1")}
        first = await fetch_balances(configs)
        second = await fetch_balances(configs)
        assert first["openrouter"]["label"] == "$7.00"
        assert second["openrouter"]["label"] == "$7.00"
        assert route.call_count == 1

    @respx.mock
    async def test_failed_refresh_keeps_the_previous_balance(self):
        route = respx.get("https://api.wavespeed.ai/api/v3/balance").mock(
            side_effect=[
                httpx.Response(200, json={"code": 200, "data": {"balance": 9}}),
                httpx.Response(500, json={"message": "down"}),
            ]
        )
        configs = {"wavespeed": _config("https://api.wavespeed.ai/api/v3")}
        first = await fetch_balances(configs)
        balances._cache["wavespeed"].fetched_at = time.monotonic() - balances._CACHE_TTL_S - 1
        second = await fetch_balances(configs)
        third = await fetch_balances(configs)
        assert first["wavespeed"]["label"] == "$9.00"
        assert second["wavespeed"]["label"] == "$9.00"
        assert third["wavespeed"]["label"] == "$9.00"
        assert route.call_count == 2

    @respx.mock
    async def test_openrouter_management_key_reads_the_account_wallet(self):
        credits = respx.get("https://openrouter.ai/api/v1/credits").mock(
            return_value=httpx.Response(200, json={"data": {"total_credits": 30, "total_usage": 4}})
        )
        key = respx.get("https://openrouter.ai/api/v1/key").mock(
            return_value=httpx.Response(401, json={"error": {"message": "User not found."}})
        )
        configs = {
            "openrouter": ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key="inference-key",
                balance_api_key="management-key",
            )
        }
        result = await fetch_balances(configs)
        assert result["openrouter"]["label"] == "$26.00"
        assert credits.call_count == 1
        assert key.call_count == 0
        assert credits.calls[0].request.headers["authorization"] == "Bearer management-key"

    @respx.mock
    async def test_openrouter_account_credits_when_the_key_has_no_limit(self):
        respx.get("https://openrouter.ai/api/v1/key").mock(
            return_value=httpx.Response(
                200, json={"data": {"limit": None, "limit_remaining": None}}
            )
        )
        respx.get("https://openrouter.ai/api/v1/credits").mock(
            return_value=httpx.Response(200, json={"data": {"total_credits": 8, "total_usage": 1}})
        )
        configs = {"openrouter": _config("https://openrouter.ai/api/v1")}
        result = await fetch_balances(configs)
        assert result["openrouter"]["label"] == "$7.00"

    @respx.mock
    async def test_elevenlabs_falls_back_to_the_user_endpoint(self):
        respx.get("https://api.elevenlabs.io/v1/user/subscription").mock(
            return_value=httpx.Response(400, json={"detail": {"status": "invalid_api_key"}})
        )
        respx.get("https://api.elevenlabs.io/v1/user").mock(
            return_value=httpx.Response(
                200,
                json={"subscription": {"character_limit": 10000, "character_count": 100}},
            )
        )
        configs = {"elevenlabs": _config("https://api.elevenlabs.io")}
        result = await fetch_balances(configs)
        assert result["elevenlabs"]["label"] == "9,900 chars"

    @respx.mock
    async def test_elevenlabs_uses_its_own_header(self):
        route = respx.get("https://api.elevenlabs.io/v1/user/subscription").mock(
            return_value=httpx.Response(200, json={"character_limit": 1000, "character_count": 100})
        )
        configs = {"elevenlabs": _config("https://api.elevenlabs.io")}
        result = await fetch_balances(configs)
        assert result["elevenlabs"]["label"] == "900 chars"
        assert route.calls[0].request.headers["xi-api-key"] == "test-key"
        assert "authorization" not in route.calls[0].request.headers

    @respx.mock
    async def test_skips_providers_without_a_balance_api(self):
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(200, json={}))
        configs = {
            "blockrun": _config("https://blockrun.ai/api", api_key=""),
            "minimax": _config("https://api.minimax.io"),
            "socialdownload": _config("https://example.test"),
            "openrouter": _config("https://openrouter.ai/api/v1", api_key=""),
        }
        assert await fetch_balances(configs) == {}

    @respx.mock
    async def test_http_error_omits_the_balance(self):
        respx.get("https://api.cloudconvert.com/v2/users/me").mock(
            return_value=httpx.Response(401, json={"message": "unauthenticated"})
        )
        configs = {"cloudconvert": _config("https://api.cloudconvert.com/v2")}
        assert await fetch_balances(configs) == {}

    def test_spicy_rejects_a_failed_envelope(self):
        with pytest.raises(ValueError):
            parse_spicyapi({"code": 401, "data": {"available": "1"}})

    def test_low_threshold_constant(self):
        """$20.00 is still enough; one cent under is low."""
        assert (
            parse_openrouter({"data": {"total_credits": Decimal("20"), "total_usage": 0}})["low"]
            is False
        )
        assert (
            parse_openrouter({"data": {"total_credits": Decimal("19.99"), "total_usage": 0}})["low"]
            is True
        )
