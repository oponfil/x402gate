"""Unit tests for the NanoGPT chat provider."""

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.nanogpt import NanoGptProvider
from x402gate.providers.openrouter import estimate_chat_input_tokens

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
