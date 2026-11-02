"""Unit tests for the NanoGPT chat provider."""

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests.helpers import make_wav_b64
from x402gate.core.config import ProviderConfig
from x402gate.providers import nanogpt, openrouter
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.nanogpt import NanoGptProvider
from x402gate.providers.openrouter import _CATALOG_RETRY_S, estimate_chat_input_tokens

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


def _chat_body(**extra: object) -> dict:
    return {"model": "demo/model", "messages": [{"role": "user", "content": "Hi"}], **extra}


def _tokens(chars: int) -> int:
    return max(int(Decimal(chars) / Decimal("2.7")), 1)


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
        # Settings objects are counted as input too: "Hi" + {"enabled":true}.
        input_tokens = int(Decimal(len("Hi") + len('{"enabled":true}')) / Decimal("2.7"))
        assert price == Decimal(input_tokens) * max(prompt_rate, write_rate) + completion

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


class TestUnpricedFeatures:
    """Flags that add a charge are caught by truthiness, not only by ``is True``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [True, "true", 1])
    async def test_rejects_any_truthy_stream(self, provider, stream):
        with pytest.raises(ProviderError, match="Streaming") as caught:
            await provider.get_price("chat/completions", _chat_body(stream=stream))
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"web_search": "true"},
            {"web_search": 1},
            {"webSearch": {"enabled": "yes"}},
            {"linkup": {"depth": "deep"}},
            {"web_search_options": ["high"]},
            {"model": "demo/model:Online"},
            {"model": "demo/model:online/linkup-deep"},
        ],
    )
    async def test_rejects_web_search_in_any_form(self, provider, extra):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="Web search") as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [{"model": "demo/model:memory"}, {"model": "demo/model:memory-30"}, {"memory": "1"}],
    )
    async def test_rejects_context_memory(self, provider, extra):
        """:memory is billed on top of tokens; the suffix fallback must not price it as free."""
        with pytest.raises(ProviderError, match="Context Memory") as caught:
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_disabled_flags_are_allowed(self, provider):
        body = _chat_body(web_search=False, webSearch={"enabled": False}, memory=0)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        chars = len("Hi") + len('{"enabled":false}')
        assert price == _expected(_tokens(chars))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"caching": "true"},
            {"prompt_caching": True},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                ]
            },
        ],
    )
    async def test_caching_in_any_form_uses_the_write_rate(self, provider, extra):
        body = _chat_body(**extra)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        write_rate = CACHE_WRITE_PER_1K / Decimal("1000")
        completion = Decimal(2048) * COMPLETION_PER_MILLION / Decimal("1000000")
        input_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == Decimal(input_tokens) * write_rate + completion


class TestInputEstimate:
    """Everything the model reads is counted."""

    @pytest.mark.asyncio
    async def test_tools_are_counted(self, provider):
        tools = [{"type": "function", "function": {"name": "f", "description": "d" * 5000}}]
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", _chat_body(tools=tools))
        chars = len("Hi") + len(json.dumps(tools, separators=(",", ":")))
        assert price == _expected(_tokens(chars))

    @pytest.mark.asyncio
    async def test_image_parts_add_a_fixed_allowance(self, provider):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 9000}}
        body = _chat_body(messages=[{"role": "user", "content": [image]}])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        assert price == _expected(1 + 4800)

    @pytest.mark.asyncio
    async def test_rejects_unpriced_content_parts(self, provider):
        part = {"type": "file", "file": {"file_data": "https://example.com/a.pdf"}}
        body = _chat_body(messages=[{"role": "user", "content": [part]}])
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_max_completion_tokens_is_priced(self, provider):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price(
                "chat/completions", _chat_body(max_completion_tokens=40_000)
            )
        assert price == _expected(1, max_tokens=40_000)


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

    @pytest.mark.asyncio
    async def test_read_timeout_is_not_resent(self, provider):
        """A request that timed out after it was sent may be running; resending bills twice."""
        body = {"model": "demo/model", "messages": [{"role": "user", "content": "Hi"}]}
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.nanogpt.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.side_effect = httpx.ReadTimeout("slow")
            await provider.submit("chat/completions", body)
        assert mock_post.call_count == 1
        assert caught.value.status_code == 504

    @pytest.mark.asyncio
    async def test_connect_timeout_is_retried(self, provider):
        """A request that never reached NanoGPT is safe to send again."""
        body = {"model": "demo/model", "messages": [{"role": "user", "content": "Hi"}]}
        ok = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.nanogpt.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_post.side_effect = [httpx.ConnectTimeout("no route"), ok]
            result = await provider.submit("chat/completions", body)
        assert mock_post.call_count == 2
        assert result["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_connect_error_is_retried(self, provider):
        """The connection never opened, so nothing reached NanoGPT."""
        ok = httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1}})
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.nanogpt.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_post.side_effect = [httpx.ConnectError("refused"), ok]
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_server_error_is_not_resent_and_is_a_known_failure(self, provider):
        """The call got an answer and may have run, so it is never sent twice."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.nanogpt.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(503, text="model overloaded")
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 1
        assert caught.value.status_code == 503
        assert not outcome_unknown(caught.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "text"), [(504, ""), (524, ""), (520, ""), (502, "upstream timed out")]
    )
    async def test_gateway_timeout_is_an_unknown_outcome(self, provider, status, text):
        """The request was forwarded and may have billed: kept, not resent."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.nanogpt.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(status, text=text)
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 1
        assert outcome_unknown(caught.value)

    @pytest.mark.asyncio
    async def test_refuses_stream_without_calling_nanogpt(self, provider):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("chat/completions", _chat_body(stream=1))
        assert caught.value.status_code == 400
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreadable_success_body_is_an_unknown_outcome(self, provider):
        """NanoGPT billed the call, so the payment must be kept, not refunded."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(200, text="data: {}\n\n")
            await provider.submit("chat/completions", _chat_body())
        assert outcome_unknown(caught.value)
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_drops_unpriced_max_completion_tokens(self, provider):
        ok = httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1}})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ok
            await provider.submit("chat/completions", _chat_body(max_completion_tokens=-5))
        assert "max_completion_tokens" not in mock_post.call_args.kwargs["json"]


class TestActualCost:
    """Token counts from the completion, with cache reads at the cache rate."""

    @pytest.mark.asyncio
    async def test_cache_read_is_not_billed_as_fresh_input(self, provider):
        provider._catalog.put("", {"demo/model": _priced("demo/model")})
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


class TestBilledAddons:
    """Add-ons NanoGPT bills on top of tokens cannot be switched on for free."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            ({"youtube_transcripts": True}, "YouTube"),
            ({"youtube_transcripts": "true"}, "YouTube"),
            ({"scraping": True}, "scraping"),
            ({"scraping": {"enabled": True, "byok": False}}, "scraping"),
            ({"advisor": {"model": "big/model", "mode": "required"}}, "advisor"),
            ({"service_tier": "fast"}, "service_tier"),
            ({"service_tier": "priority"}, "service_tier"),
            ({"service_tier": {"tier": "fast"}}, "service_tier"),
            ({"modalities": ["image", "text"]}, "output"),
            ({"audio": {"voice": "alloy"}}, "output"),
        ],
    )
    async def test_rejected_before_payment(self, provider, extra, match):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match=match) as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"youtube_transcripts": False},
            {"scraping": {"enabled": False}},
            {"service_tier": "flex"},
            {"service_tier": "auto"},
        ],
    )
    async def test_off_or_cheaper_settings_are_allowed(self, provider, extra):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", _chat_body(**extra))
        assert price == _expected(estimate_chat_input_tokens(_chat_body(**extra), "nanogpt"))

    @pytest.mark.asyncio
    async def test_submit_pins_youtube_transcripts_off(self, provider):
        ok = httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 1}})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ok
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_args.kwargs["json"]["youtube_transcripts"] is False


class TestReasoningBudget:
    """Thinking counts inside a cap above the budget, on top of a cap at or below it."""

    @pytest.mark.asyncio
    async def test_reasoning_max_tokens_is_added(self, provider):
        body = _chat_body(reasoning={"max_tokens": 16_000})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        input_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == _expected(input_tokens, max_tokens=2048 + 16_000)

    @pytest.mark.asyncio
    async def test_thinking_suffix_budget_is_added(self, provider):
        """The suffix fallback prices the base model; the budget must still be counted."""
        body = _chat_body(model="demo/model:thinking:8192")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        assert price == _expected(1, max_tokens=2048 + 8192)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("extra", "ceiling"),
        [
            ({"model": "demo/model:thinking:8192", "max_tokens": 16_000}, 16_000),
            ({"model": "demo/model:thinking:8192", "max_tokens": 8192}, 8192 + 8192),
            ({"reasoning": {"max_tokens": 8000}, "max_tokens": 16_000}, 16_000),
            # Suffix and body budgets: the larger one counts, once.
            (
                {"model": "demo/model:thinking:4000", "reasoning": {"max_tokens": 8000}},
                2048 + 8000,
            ),
        ],
    )
    async def test_suffix_and_body_budgets_follow_the_same_rule(self, provider, extra, ceiling):
        body = _chat_body(**extra)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        input_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == _expected(input_tokens, max_tokens=ceiling)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("suffix", ["\u00b2", "\u0663", "1" * 10, "9" * 5000])
    async def test_unpriceable_suffix_budget_is_rejected(self, provider, suffix):
        """Non-ASCII digits and over-long runs are a 400, not a crash or a misread."""
        body = _chat_body(model=f"demo/model:thinking:{suffix}")
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="thinking") as caught,
        ):
            mock_get.return_value = _catalog(_priced("demo/model"))
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_malformed_budget_is_rejected(self, provider):
        body = _chat_body(thinking={"type": "enabled", "budget_tokens": "lots"})
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="budget_tokens") as caught,
        ):
            mock_get.return_value = _catalog(_priced("demo/model"))
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400


def _expire(provider: NanoGptProvider) -> None:
    """Age the loaded catalog past its TTL."""
    catalogs = provider._catalog
    catalogs._loaded_at[""] -= 2 * catalogs._ttl
    catalogs._refresh_at[""] -= 2 * catalogs._ttl


class TestCatalogLoading:
    """The catalog loads at startup and refreshes without making requests wait."""

    async def test_warm_up_loads_the_catalog(self, provider):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            await provider.warm_up()
            price = await provider.get_price("chat/completions", _chat_body())

        assert price == _expected(_tokens(len("Hi")))
        assert mock_get.call_count == 1
        assert mock_get.call_args.kwargs["timeout"].connect == 10.0

    async def test_expired_catalog_is_served_while_it_refreshes(self, provider):
        provider._catalog.put("", {"demo/model": _priced("demo/model")})
        _expire(provider)
        release = asyncio.Event()

        async def slow_catalog(url, **kwargs):
            await release.wait()
            return _catalog(_priced("demo/model", prompt="2", completion="4"))

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = slow_catalog
            old = await provider.get_price("chat/completions", _chat_body())
            refresh = provider._catalog._tasks[""]
            await provider.get_price("chat/completions", _chat_body())
            assert provider._catalog._tasks[""] is refresh
            release.set()
            await refresh
            new = await provider.get_price("chat/completions", _chat_body())

        assert old == _expected(_tokens(len("Hi")))
        assert new == old * 2
        assert mock_get.call_count == 1

    async def test_failed_refresh_keeps_the_old_catalog(self, provider):
        provider._catalog.put("", {"demo/model": _priced("demo/model")})
        _expire(provider)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = httpx.Response(400, text="down")
            price = await provider.get_price("chat/completions", _chat_body())
            await asyncio.gather(provider._catalog._tasks[""], return_exceptions=True)
            again = await provider.get_price("chat/completions", _chat_body())

        assert price == again == _expected(_tokens(len("Hi")))
        assert mock_get.call_count == 1

    async def test_model_missing_from_an_expired_catalog_waits_for_the_refresh(self, provider):
        provider._catalog.put("", {"old/model": _priced("old/model")})
        _expire(provider)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", _chat_body())

        assert price == _expected(_tokens(len("Hi")))
        assert mock_get.call_count == 1

    async def test_model_missing_from_a_fresh_catalog_is_not_refetched(self, provider):
        provider._catalog.put("", {"old/model": _priced("old/model")})
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as exc_info,
        ):
            await provider.get_price("chat/completions", _chat_body())

        assert exc_info.value.status_code == 404
        mock_get.assert_not_called()

    async def test_unknown_model_after_a_failed_refresh_does_not_fetch_again(self, provider):
        """In the retry delay an unknown id is a 404 at once, not a fetch per request."""
        provider._catalog.put("", {"old/model": _priced("old/model")})
        _expire(provider)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = httpx.Response(400, text="down")
            statuses = []
            for _ in range(3):
                with pytest.raises(ProviderError) as exc_info:
                    await provider.get_price("chat/completions", _chat_body())
                statuses.append(exc_info.value.status_code)
            # The first one waited for the refresh and got its error.
            assert statuses == [400, 404, 404]
            assert mock_get.call_count == 1

            # After the retry delay a missing model waits for a fetch again.
            provider._catalog._refresh_at[""] -= _CATALOG_RETRY_S
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", _chat_body())

        assert price == _expected(_tokens(len("Hi")))
        assert mock_get.call_count == 2

    async def test_suffixed_model_is_found_in_a_stale_catalog_without_waiting(self, provider):
        provider._catalog.put("", {"demo/model": _priced("demo/model")})
        _expire(provider)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = asyncio.Event().wait  # never answers
            price = await provider.get_price(
                "chat/completions", _chat_body(model="demo/model:thinking")
            )
            provider._catalog._tasks[""].cancel()

        assert price == _expected(_tokens(len("Hi")))


class TestPooledSubmit:
    """Completions go through the shared pooled client."""

    async def test_submits_share_one_client(self, provider):
        response = httpx.Response(200, json={"choices": [], "usage": {}})
        with patch.object(httpx.AsyncClient, "post", autospec=True) as mock_post:
            mock_post.return_value = response
            await provider.submit("chat/completions", _chat_body())
            await provider.submit("chat/completions", _chat_body())

        first, second = mock_post.call_args_list
        assert first.args[0] is second.args[0]
        assert first.kwargs["timeout"].connect == 10.0
        assert first.kwargs["timeout"].read == 10.0

    async def test_get_price_weighs_once_and_submit_not_at_all(self, provider):
        provider._catalog.put("", {"demo/model": _priced("demo/model")})
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch.object(
                nanogpt, "estimate_chat_input_tokens", wraps=nanogpt.estimate_chat_input_tokens
            ) as weigh,
        ):
            mock_post.return_value = httpx.Response(200, json={"choices": [], "usage": {}})
            await provider.get_price("chat/completions", _chat_body())
            assert weigh.call_count == 1
            await provider.submit("chat/completions", _chat_body())

        assert weigh.call_count == 1


class TestThinkingVariantSuffix:
    """A '<model>-thinking:<budget>' id is priced with its budget."""

    @pytest.mark.asyncio
    async def test_dash_thinking_budget_is_added(self, provider):
        body = _chat_body(model="claude-3-7-sonnet-thinking:128000")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("claude-3-7-sonnet-thinking"))
            price = await provider.get_price("chat/completions", body)
        input_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == _expected(input_tokens, max_tokens=2048 + 128_000)

    @pytest.mark.asyncio
    async def test_any_numeric_suffix_counts_as_a_budget(self, provider):
        body = _chat_body(model="demo/model:16000:cheap")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        assert price == _expected(1, max_tokens=2048 + 16_000)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("suffix", ["²", "1" * 10])
    async def test_unpriceable_dash_thinking_budget_is_rejected(self, provider, suffix):
        body = _chat_body(model=f"claude-3-7-sonnet-thinking:{suffix}")
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="thinking") as caught,
        ):
            mock_get.return_value = _catalog(_priced("claude-3-7-sonnet-thinking"))
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400


# A 1.25x write rate (USD per 1k tokens), below the 2x a 1-hour write bills.
SHORT_WRITE_PER_1K = Decimal("0.00125")


class TestLongCacheTtl:
    """A 1-hour cache write is priced at 2x input, above the listed write rate."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("extra", "rate_per_million"),
        [
            ({"prompt_caching": {"enabled": True}}, Decimal("1.25")),
            ({"prompt_caching": {"enabled": True, "ttl": "5m"}}, Decimal("1.25")),
            ({"prompt_caching": {"enabled": True, "ttl": "1h"}}, Decimal(2)),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Hi",
                                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                                }
                            ],
                        }
                    ]
                },
                Decimal(2),
            ),
        ],
    )
    async def test_quote_uses_the_ttl_write_rate(self, provider, extra, rate_per_million):
        body = _chat_body(**extra)
        catalog = _catalog(
            _priced("demo/model", cacheWriteInputPer1kTokens=str(SHORT_WRITE_PER_1K))
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("chat/completions", body)
        input_tokens = estimate_chat_input_tokens(body, "nanogpt")
        completion = Decimal(2048) * COMPLETION_PER_MILLION / Decimal("1000000")
        assert price == Decimal(input_tokens) * rate_per_million / Decimal("1000000") + completion

    @pytest.mark.asyncio
    async def test_actual_cost_bills_long_writes_at_2x(self, provider):
        provider._catalog.put(
            "",
            {
                "demo/model": _priced(
                    "demo/model", cacheWriteInputPer1kTokens=str(SHORT_WRITE_PER_1K)
                )
            },
        )
        body = _chat_body(prompt_caching={"enabled": True, "ttl": "1h"})
        result = {"usage": {"prompt_tokens": 1000, "cache_creation_input_tokens": 1000}}
        cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal(1000) * Decimal(2) / Decimal("1000000")


def _long_body(model: str, chars: int = 540_027) -> dict:
    """A body of 200_010 estimated tokens by default."""
    return _chat_body(model=model, messages=[{"role": "user", "content": "x" * chars}])


def _per_m(tokens: int, per_million: Decimal) -> Decimal:
    return Decimal(tokens) * per_million / Decimal("1000000")


class TestLongContextTier:
    """Long prompts bill the tier NanoGPT lists; the hold covers upstream tiers it does not."""

    GROK = {
        **_priced("x-ai/grok-4.7"),
        "context_length": 500_000,
        "effective_pricing": {
            "list_price": {
                "inputPer1kTokens": 0.001,
                "outputPer1kTokens": 0.002,
                "inputPer1kTokensAbove200k": 0.004,
                "outputPer1kTokensAbove200k": 0.012,
                "highContextThresholdTokens": 200_000,
                "cacheReadInputPer1kTokensAbove200k": 0.001,
            },
        },
    }

    @pytest.mark.asyncio
    async def test_listed_tier_is_quoted(self, provider):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.GROK)
            price = await provider.get_price("chat/completions", _long_body("x-ai/grok-4.7"))
        assert price == _per_m(200_010, Decimal(4)) + _per_m(2048, Decimal(12))

    @pytest.mark.asyncio
    async def test_prompt_at_the_threshold_keeps_the_base_rate(self, provider):
        body = _long_body("x-ai/grok-4.7", chars=540_000)  # exactly 200k tokens
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.GROK)
            price = await provider.get_price("chat/completions", body)
        assert price == _expected(200_000)

    @pytest.mark.asyncio
    async def test_listed_tier_is_billed(self, provider):
        provider._catalog.put("", {"x-ai/grok-4.7": self.GROK})
        usage = {
            "prompt_tokens": 250_000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 10_000},
        }
        cost = await provider.calculate_actual_cost({"model": "x-ai/grok-4.7"}, {"usage": usage})
        assert cost == (
            _per_m(240_000, Decimal(4)) + _per_m(10_000, Decimal(1)) + _per_m(100, Decimal(12))
        )

    @pytest.mark.asyncio
    async def test_cache_write_rate_the_tier_leaves_out_scales(self, provider):
        provider._catalog.put("", {"x-ai/grok-4.7": self.GROK})
        result = {"usage": {"prompt_tokens": 250_000, "cache_creation_input_tokens": 250_000}}
        cost = await provider.calculate_actual_cost({"model": "x-ai/grok-4.7"}, result)
        # The $2.5/M write rate times the tier's 4x prompt rate.
        assert cost == _per_m(250_000, Decimal(10))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4.6",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-opus-5.5",
            "anthropic/claude-fable-5",
            "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted",
            "gemini-2.0-pro-exp-02-05",
        ],
    )
    async def test_model_without_a_tier_is_flat(self, provider, model):
        entry = {**_priced(model), "context_length": 1_000_000}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(entry)
            price = await provider.get_price("chat/completions", _long_body(model))
        assert price == _expected(200_010)
        result = {"usage": {"prompt_tokens": 300_000, "completion_tokens": 1000}}
        cost = await provider.calculate_actual_cost({"model": model}, result)
        assert cost == _per_m(300_000, PROMPT_PER_MILLION) + _per_m(1000, COMPLETION_PER_MILLION)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        ["gemini-2.5-pro", "gemini-2.5-pro-preview-06-05", "google/gemini-3.1-pro-preview"],
    )
    async def test_gemini_pro_hold_covers_the_upstream_tier(self, provider, model):
        entry = {**_priced(model), "context_length": 1_048_576}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(entry)
            price = await provider.get_price("chat/completions", _long_body(model))
        prompt = _per_m(200_010, PROMPT_PER_MILLION * 2)
        assert price == prompt + _per_m(2048, COMPLETION_PER_MILLION * Decimal("1.5"))

    @pytest.mark.asyncio
    async def test_gemini_pro_actual_cost_bills_the_listed_rates(self, provider):
        provider._catalog.put("", {"gemini-2.5-pro": _priced("gemini-2.5-pro")})
        result = {"usage": {"prompt_tokens": 250_000, "completion_tokens": 100}}
        cost = await provider.calculate_actual_cost({"model": "gemini-2.5-pro"}, result)
        assert cost == _per_m(250_000, PROMPT_PER_MILLION) + _per_m(100, COMPLETION_PER_MILLION)

    @pytest.mark.asyncio
    async def test_hold_tier_needs_a_window_above_the_threshold(self, provider):
        model = "gemini-3-pro-image-preview"
        entry = {**_priced(model), "context_length": 200_000}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(entry)
            price = await provider.get_price("chat/completions", _long_body(model))
        assert price == _expected(200_010)


class TestMalformedBodies:
    """Bodies json.loads accepts but that cannot be priced or sent get a 400, not a 500."""

    @pytest.mark.asyncio
    async def test_deep_message_nesting_is_priced_without_recursion(self, provider):
        nested: list = []
        for _ in range(5000):
            nested = [nested]
        body = _chat_body(messages=[{"role": "user", "content": "Hi", "extra": nested}])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            assert await provider.get_price("chat/completions", body) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("part_type", [["text"], {}])
    async def test_non_string_part_type_is_a_400(self, provider, part_type):
        body = _chat_body(messages=[{"role": "user", "content": [{"type": part_type}]}])
        for call in (provider.get_price, provider.submit):
            with pytest.raises(ProviderError) as exc_info:
                await call("chat/completions", body)
            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"temperature": float("nan")},
            {"frequency_penalty": float("inf")},
            {"messages": [{"role": "user", "content": "a\ud800"}]},
        ],
    )
    async def test_unsendable_json_is_refused_before_payment(self, provider, extra):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as exc_info,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert exc_info.value.status_code == 400
        mock_get.assert_not_called()


class TestErrorInSuccessBody:
    """A 2xx body with only an error object is a failure, not a billed success."""

    async def _submit(self, provider, payload: dict):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(status_code=200, json=payload)
            return await provider.submit("chat/completions", _chat_body())

    @pytest.mark.asyncio
    async def test_reported_error_is_a_known_failure(self, provider):
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": {"code": 503, "message": "overloaded"}})
        assert exc_info.value.status_code == 503
        assert not outcome_unknown(exc_info.value)

    @pytest.mark.asyncio
    async def test_reported_timeout_is_an_unknown_outcome(self, provider):
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": {"code": 524, "message": "timeout"}})
        assert outcome_unknown(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_with_usage_is_returned_and_billed(self, provider):
        payload = {"error": {"code": 502}, "usage": {"prompt_tokens": 7, "completion_tokens": 0}}
        assert await self._submit(provider, payload) == payload

    @pytest.mark.asyncio
    async def test_error_with_zeroed_usage_is_a_known_failure(self, provider):
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": {"code": 502}, "usage": usage})
        assert exc_info.value.status_code == 502
        assert not outcome_unknown(exc_info.value)


class TestSearchToolsAndAudio:
    """Web search tools and xAI Live Search are refused before payment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"tools": [{"type": "web_search"}]},
            {"tools": [{"type": "web_search_preview"}]},
            {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            {"tools": "web_search"},
            {"tools": [{"google_search": {}}]},
            {"search_parameters": {"mode": "auto"}},
            {"search_parameters": {}},
        ],
    )
    async def test_web_search_tools_are_refused(self, provider, extra):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_function_tools_and_search_off_are_priced(self, provider):
        tools = [{"type": "function", "function": {"name": "f"}}]
        body = _chat_body(tools=tools, search_parameters={"mode": "off"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_priced("demo/model"))
            price = await provider.get_price("chat/completions", body)
        assert price == _expected(estimate_chat_input_tokens(body, "nanogpt"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("extra", [{"tools": [{"type": "web_search"}]}])
    async def test_submit_refuses_them_too(self, provider, extra):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_post.assert_not_called()


_MILLION = Decimal("1000000")


def _audio_body(*parts: dict, model: str = "gemini-2.5-flash") -> dict:
    content = [{"type": "text", "text": "Transcribe this"}, *parts]
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def _audio_part(data: str, fmt: str = "wav") -> dict:
    return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def _gemini_flash() -> dict:
    """gemini-2.5-flash as NanoGPT's catalog lists it: audio at $1/M, text at $0.3/M."""
    return {
        "id": "gemini-2.5-flash",
        "capabilities": {"audio_input": True},
        "pricing": {
            "audio_input": 1,
            "prompt": 0.3,
            "completion": 2.5,
            "cacheReadInputPer1kTokens": 0.00003,
            "unit": "per_million_tokens",
        },
    }


def _audio_capable_unpriced() -> dict:
    """An audio-capable model whose entry lists no audio_input rate."""
    model = _priced("openai/gpt-4o-audio-preview")
    model["capabilities"] = {"audio_input": True}
    return model


class TestAudioInput:
    """Audio parts are priced by duration at the model's catalog audio_input rate."""

    async def _price(self, provider, body, *models):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(*models)
            return await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    async def test_one_second_wav_is_priced_at_the_audio_rate(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        price = await self._price(provider, body, _gemini_flash())
        text_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == (
            Decimal(text_tokens) * Decimal("0.3") / _MILLION
            + Decimal(50) * Decimal(1) / _MILLION
            + Decimal(2048) * Decimal("2.5") / _MILLION
        )

    @pytest.mark.asyncio
    async def test_audio_is_timed_not_weighed_as_text(self, provider):
        """A minute of 16 kHz WAV (~2.6 MB of base64) is 3000 audio tokens."""
        body = _audio_body(_audio_part(make_wav_b64(60.0, rate=16000)))
        price = await self._price(provider, body, _gemini_flash())
        text_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert text_tokens < 100
        assert price == (
            Decimal(text_tokens) * Decimal("0.3") / _MILLION
            + Decimal(3000) / _MILLION
            + Decimal(2048) * Decimal("2.5") / _MILLION
        )

    @pytest.mark.asyncio
    async def test_audio_seconds_of_every_part_add_up(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(1.2)), _audio_part(make_wav_b64(0.5)))
        price = await self._price(provider, body, _gemini_flash())
        text_tokens = estimate_chat_input_tokens(body, "nanogpt")
        # 2 s + 1 s, rounded up per part.
        assert price == (
            Decimal(text_tokens) * Decimal("0.3") / _MILLION
            + Decimal(150) / _MILLION
            + Decimal(2048) * Decimal("2.5") / _MILLION
        )

    @pytest.mark.asyncio
    async def test_audio_rate_below_the_input_rate_bills_the_input_rate(self, provider):
        model = _priced("demo/model", audio_input="0.5")
        body = _audio_body(_audio_part(make_wav_b64(1.0)), model="demo/model")
        price = await self._price(provider, body, model)
        text_tokens = estimate_chat_input_tokens(body, "nanogpt")
        assert price == _expected(text_tokens + 50)

    @pytest.mark.asyncio
    async def test_audio_tokens_count_toward_the_long_context_tier(self, provider):
        """3000 audio tokens lift a short prompt over a 1000-token tier."""
        model = _gemini_flash()
        model["effective_pricing"] = {
            "list_price": {
                "highContextThresholdTokens": 1000,
                "inputPer1kTokensAbove1k": "0.0006",
                "outputPer1kTokensAbove1k": "0.0025",
            }
        }
        body = _audio_body(_audio_part(make_wav_b64(60.0, rate=16000)))
        price = await self._price(provider, body, model)
        text_tokens = estimate_chat_input_tokens(body, "nanogpt")
        # The tier doubles the prompt rate and lists no audio rate, so audio
        # doubles with it.
        assert price == (
            Decimal(text_tokens) * Decimal("0.6") / _MILLION
            + Decimal(3000) * Decimal(2) / _MILLION
            + Decimal(2048) * Decimal("2.5") / _MILLION
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        [
            _audio_capable_unpriced(),
            _priced("demo/model", audio_input=0),
            _priced("demo/model", audio_input="free"),
        ],
        ids=["no-audio-rate", "zero-rate", "malformed-rate"],
    )
    async def test_model_without_an_audio_rate_is_refused(self, provider, model):
        body = _audio_body(_audio_part(make_wav_b64(1.0)), model=model["id"])
        with pytest.raises(ProviderError, match="audio_input rate") as caught:
            await self._price(provider, body, model)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_refuses_audio_for_a_model_without_an_audio_rate(self, provider):
        model = _audio_capable_unpriced()
        provider._catalog.put("", {model["id"]: model})
        body = _audio_body(_audio_part(make_wav_b64(1.0)), model=model["id"])
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError, match="audio_input rate") as caught,
        ):
            await provider.submit("chat/completions", body)
        assert caught.value.status_code == 400
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_forwards_audio_for_a_model_with_an_audio_rate(self, provider):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        ok = httpx.Response(200, json={"choices": [], "usage": {"prompt_tokens": 60}})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ok
            await provider.submit("chat/completions", body)
        assert mock_post.call_args.kwargs["json"]["messages"] == body["messages"]

    @pytest.mark.asyncio
    async def test_oversized_audio_part_is_refused(self):
        config = ProviderConfig(
            enabled=True,
            base_url="https://api.nano-gpt.com/api/v1",
            api_key="k",
            max_stt_audio_mb=1,
        )
        small = NanoGptProvider(config=config, default_max_tokens=2048)
        body = _audio_body(_audio_part(make_wav_b64(70.0)))  # ~1.1 MB
        with pytest.raises(ProviderError) as caught:
            await self._price(small, body, _gemini_flash())
        assert caught.value.status_code == 413

    @pytest.mark.asyncio
    async def test_audio_longer_than_the_limit_is_refused(self):
        config = ProviderConfig(
            enabled=True,
            base_url="https://api.nano-gpt.com/api/v1",
            api_key="k",
            max_stt_audio_seconds=5,
        )
        short = NanoGptProvider(config=config, default_max_tokens=2048)
        assert await self._price(
            short, _audio_body(_audio_part(make_wav_b64(5.0))), _gemini_flash()
        )
        body = _audio_body(_audio_part(make_wav_b64(6.0)))
        with pytest.raises(ProviderError, match="Audio too long") as caught:
            await self._price(short, body, _gemini_flash())
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "audio",
        [
            {"data": "QUJD0123" * 100, "format": "wav"},  # not audio
            {"data": "not base64!", "format": "wav"},
            {"data": make_wav_b64(1.0)},  # no format
        ],
    )
    async def test_unreadable_audio_is_refused(self, provider, audio):
        body = _audio_body({"type": "input_audio", "input_audio": audio})
        with pytest.raises(ProviderError) as caught:
            await self._price(provider, body, _gemini_flash())
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_busy_pricing_slots_refuse_with_429(self, provider):
        slots = asyncio.Semaphore(0)
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        with (
            patch.object(openrouter, "_STT_PRICING_WAIT_S", 0.01),
            patch.object(openrouter, "_stt_pricing_slots", lambda: slots),
            pytest.raises(ProviderError) as caught,
        ):
            await self._price(provider, body, _gemini_flash())
        assert caught.value.status_code == 429

    @pytest.mark.asyncio
    async def test_actual_cost_bills_reported_audio_tokens_at_the_audio_rate(self, provider):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "prompt_tokens_details": {"audio_tokens": 32},
        }
        cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == (
            Decimal(8) * Decimal("0.3") / _MILLION
            + Decimal(32) / _MILLION
            + Decimal(10) * Decimal("2.5") / _MILLION
        )

    @pytest.mark.asyncio
    async def test_actual_cost_without_audio_tokens_bills_a_short_prompt_as_audio(self, provider):
        """A short prompt fits under the 50 audio tokens held for one second."""
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        usage = {"prompt_tokens": 40, "completion_tokens": 10}
        cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == Decimal(40) / _MILLION + Decimal(10) * Decimal("2.5") / _MILLION

    @pytest.mark.asyncio
    async def test_actual_cost_without_audio_tokens_bills_text_at_the_prompt_rate(self, provider):
        """Only the audio tokens held for the timed audio bill at the audio rate."""
        body = _audio_body(_audio_part(make_wav_b64(10.0, rate=16000)))
        body["messages"][0]["content"][0]["text"] = "word " * 5400
        usage = {"prompt_tokens": 10320, "completion_tokens": 200}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(_gemini_flash())
            hold = await provider.get_price("chat/completions", body)
            cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == (
            Decimal(500) / _MILLION
            + Decimal(10320 - 500) * Decimal("0.3") / _MILLION
            + Decimal(200) * Decimal("2.5") / _MILLION
        )
        assert cost < hold

    @pytest.mark.asyncio
    async def test_actual_cost_bills_the_prompt_as_audio_when_the_audio_cannot_be_timed(
        self, provider
    ):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(10.0, rate=16000)))
        usage = {"prompt_tokens": 10320, "completion_tokens": 200}
        failing = AsyncMock(
            side_effect=ProviderError(provider="nanogpt", detail="busy", status_code=429)
        )
        with patch.object(provider, "_measure_chat_audio", failing):
            cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == Decimal(10320) / _MILLION + Decimal(200) * Decimal("2.5") / _MILLION

    @pytest.mark.asyncio
    async def test_actual_cost_does_not_time_audio_the_usage_reports(self, provider):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        usage = {"prompt_tokens": 40, "prompt_tokens_details": {"audio_tokens": 32}}
        measure = AsyncMock(return_value=1)
        with patch.object(provider, "_measure_chat_audio", measure):
            await provider.calculate_actual_cost(body, {"usage": usage})
        measure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_actual_cost_keeps_cache_reads_at_the_cache_rate(self, provider):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = _audio_body(_audio_part(make_wav_b64(1.0)))
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "cache_read_input_tokens": 400,
            "prompt_tokens_details": {"audio_tokens": 100},
        }
        cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == (
            Decimal(500) * Decimal("0.3") / _MILLION
            + Decimal(100) / _MILLION
            + Decimal(400) * Decimal("0.00003") / Decimal(1000)
        )

    @pytest.mark.asyncio
    async def test_text_only_request_ignores_the_audio_rate(self, provider):
        provider._catalog.put("", {"gemini-2.5-flash": _gemini_flash()})
        body = {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "Hi"}]}
        usage = {"prompt_tokens": 40, "completion_tokens": 10}
        cost = await provider.calculate_actual_cost(body, {"usage": usage})
        assert cost == Decimal(40) * Decimal("0.3") / _MILLION + Decimal(25) / _MILLION
