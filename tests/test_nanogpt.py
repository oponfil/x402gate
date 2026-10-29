"""Unit tests for the NanoGPT chat provider."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import ProviderError
from x402gate.providers.nanogpt import NanoGptProvider

PROMPT_PER_MILLION = Decimal("1")
COMPLETION_PER_MILLION = Decimal("2")
CACHE_READ_PER_1K = Decimal("0.0001")
CACHE_WRITE_PER_1K = Decimal("0.0025")


@pytest.fixture
def provider():
    """NanoGPT provider with a 2048-token output ceiling."""
    config = ProviderConfig(
        enabled=True,
        base_url="https://api.nano-gpt.com/api/v1",
        api_key="test-key",
        poll_timeout=10,
    )
    return NanoGptProvider(config=config, default_max_tokens=2048)


def _catalog(*models: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json={"data": list(models)})


def _priced(model_id: str, **extra_pricing: object) -> dict:
    pricing = {
        "prompt": str(PROMPT_PER_MILLION),
        "completion": str(COMPLETION_PER_MILLION),
        "unit": "per_million_tokens",
        "cacheReadInputPer1kTokens": str(CACHE_READ_PER_1K),
        "cacheWriteInputPer1kTokens": str(CACHE_WRITE_PER_1K),
    }
    pricing.update(extra_pricing)
    return {"id": model_id, "pricing": pricing}


def _expected(input_tokens: int, max_tokens: int = 2048, completions: int = 1) -> Decimal:
    prompt = Decimal(input_tokens) * PROMPT_PER_MILLION / Decimal("1000000")
    completion = (
        Decimal(max_tokens) * COMPLETION_PER_MILLION / Decimal("1000000") * Decimal(completions)
    )
    return prompt + completion


class TestGetPrice:
    """Catalog prices are USD per million tokens."""

    @pytest.mark.asyncio
    async def test_prices_per_million_not_per_token(self, provider):
        """A $1 / $2 per-million model must not be billed as $1 per token."""
        body = {
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        catalog = _catalog(_priced("deepseek/deepseek-v4.1-flash-1017"))

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("chat/completions", body)

        assert price == _expected(1)
        assert price < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_raises_short_max_tokens_to_the_ceiling(self, provider):
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 32,
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)

        assert price == _expected(1, max_tokens=2048)

    @pytest.mark.asyncio
    async def test_multiplies_completion_by_n(self, provider):
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 3,
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)

        assert price == _expected(1, completions=3)

    @pytest.mark.asyncio
    async def test_strips_a_routing_suffix_to_find_the_catalog_row(self, provider):
        body = {
            "model": "demo/model:abliteration",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)

        assert price == _expected(1)

    @pytest.mark.asyncio
    async def test_cache_write_uses_the_higher_input_rate(self, provider):
        """An enabled prompt cache can bill the whole input at the write rate."""
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "prompt_caching": {"enabled": True},
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)

        write_rate = CACHE_WRITE_PER_1K / Decimal("1000")
        prompt_rate = PROMPT_PER_MILLION / Decimal("1000000")
        completion = Decimal(2048) * COMPLETION_PER_MILLION / Decimal("1000000")
        assert price == Decimal(1) * max(prompt_rate, write_rate) + completion

    @pytest.mark.asyncio
    async def test_rejects_web_search_before_calling_the_catalog(self, provider):
        body = {
            "model": "demo/model:online",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_price("chat/completions", body)

        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_streaming(self, provider):
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_model(self, provider):
        body = {
            "model": "missing/model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as caught,
        ):
            mock_get.return_value = _catalog(_priced("demo/model"))
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 404

    @pytest.mark.asyncio
    async def test_rejects_a_non_chat_path(self, provider):
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("images/generations", {"model": "demo/model"})
        assert caught.value.status_code == 400


class TestSubmit:
    """Forwarding and the output ceiling."""

    @pytest.mark.asyncio
    async def test_posts_chat_and_strips_gateway_fields(self, provider):
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
            "_caller": "wallet",
        }
        response = httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            result = await provider.submit("chat/completions", body)

        assert result["choices"][0]["message"]["content"] == "Hello"
        sent = mock_post.call_args.kwargs["json"]
        assert sent["max_tokens"] == 2048
        assert "_caller" not in sent
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert body["_caller"] == "wallet"

    @pytest.mark.asyncio
    async def test_maps_upstream_client_errors(self, provider):
        body = {
            "model": "demo/model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(400, text="bad model")
            await provider.submit("chat/completions", body)
        assert caught.value.status_code == 400


class TestActualCost:
    """Token counts from the completion, with cache reads at the cache rate."""

    @pytest.mark.asyncio
    async def test_cache_read_is_not_billed_as_fresh_input(self, provider):
        provider._models_cache = {"demo/model": _priced("demo/model")}
        provider._cache_updated_at = 10**12
        result = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "cache_read_input_tokens": 400,
            }
        }
        cost = await provider.calculate_actual_cost({"model": "demo/model"}, result)

        fresh = Decimal(600) * PROMPT_PER_MILLION / Decimal("1000000")
        cached = Decimal(400) * CACHE_READ_PER_1K / Decimal("1000")
        completion = Decimal(100) * COMPLETION_PER_MILLION / Decimal("1000000")
        assert cost == fresh + cached + completion
