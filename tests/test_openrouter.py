"""Unit tests for OpenRouter provider."""

import asyncio
import json
import threading
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests.helpers import make_wav_b64
from x402gate.core.config import ProviderConfig
from x402gate.providers import openrouter
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.openrouter import OpenRouterProvider, estimate_chat_input_tokens

MINI_PROMPT = Decimal("0.00000015")
MINI_COMPLETION = Decimal("0.0000006")


def _chat_tokens(chars: int) -> int:
    return max(int(Decimal(chars) / Decimal("2.7")), 1)


def _ascii_tokens(text: str) -> int:
    """Chat estimate for ASCII text: 2.7 characters per token, a digit one token."""
    digits = sum(ch.isdigit() for ch in text)
    return max((len(text) - digits) * 10 // 27 + digits, 1)


def _mini_price(input_tokens: int, max_tokens: int = 1024) -> Decimal:
    return Decimal(input_tokens) * MINI_PROMPT + Decimal(max_tokens) * MINI_COMPLETION


def _catalog(*models: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json={"data": list(models)})


def _chat_body(**extra) -> dict:
    return {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], **extra}


@pytest.fixture
def provider():
    """Create an OpenRouter provider with test config."""
    config = ProviderConfig(
        enabled=True,
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key-123",
        docs_url="https://openrouter.ai/docs",
    )
    return OpenRouterProvider(config=config)


@pytest.fixture
def model_info_response():
    """Mock response from GET /api/v1/models (full catalog)."""
    return httpx.Response(
        status_code=200,
        json={
            "data": [
                {
                    "id": "openai/gpt-4o-mini",
                    "name": "OpenAI: GPT-4o Mini",
                    "pricing": {
                        "prompt": "0.00000015",  # $0.15 per 1M tokens
                        "completion": "0.0000006",  # $0.60 per 1M tokens
                    },
                },
                {
                    "id": "openai/gpt-4o",
                    "name": "OpenAI: GPT-4o",
                    "pricing": {
                        "prompt": "0.0000025",
                        "completion": "0.00001",
                    },
                },
            ],
        },
    )


@pytest.fixture
def embedding_model_info_response():
    """Mock response from GET /api/v1/models?output_modalities=embeddings."""
    return httpx.Response(
        status_code=200,
        json={
            "data": [
                {
                    "id": "openai/text-embedding-3-small",
                    "name": "OpenAI: text-embedding-3-small",
                    "pricing": {
                        "prompt": "0.00000002",
                        "completion": "0",
                    },
                }
            ],
        },
    )


@pytest.fixture
def transcription_model_info_response():
    """Mock response from GET /api/v1/models?output_modalities=transcription."""
    return httpx.Response(
        status_code=200,
        json={
            "data": [
                {
                    "id": "openai/whisper-1",
                    "name": "OpenAI: Whisper 1",
                    "pricing": {
                        "prompt": "0.0001",
                        "completion": "0",
                    },
                },
                {
                    "id": "openai/gpt-4o-mini-transcribe",
                    "name": "OpenAI: GPT-4o Mini Transcribe",
                    "pricing": {
                        "prompt": "0.00000125",
                        "completion": "0.000005",
                    },
                },
            ],
        },
    )


@pytest.fixture
def chat_completion_response():
    """Mock response from POST /api/v1/chat/completions."""
    return httpx.Response(
        status_code=200,
        json={
            "id": "gen-abc123",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        },
    )


class TestGetPrice:
    """Tests for OpenRouterProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_calculates_price_from_model_pricing(self, provider, model_info_response):
        """Should estimate cost based on input text length and max_tokens."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is quantum computing?"}],
            "max_tokens": 100,  # below default 1024 → will be raised to 1024
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        assert isinstance(price, Decimal)
        assert price > Decimal("0")

        # Verify: 25 chars / 2.7 ≈ 9 input tokens × $0.00000015
        #        + 1024 output tokens × $0.0000006 (max_tokens raised to default)
        expected_input_tokens = max(
            int(Decimal(len("What is quantum computing?")) / Decimal("2.7")),
            1,
        )
        expected = Decimal(expected_input_tokens) * Decimal("0.00000015") + Decimal(
            "1024"
        ) * Decimal("0.0000006")
        assert price == expected

    @pytest.mark.asyncio
    async def test_uses_default_max_tokens(self, provider, model_info_response):
        """Should use default_max_tokens when max_tokens is not specified."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        # Default is 1024 max_tokens
        # "Hi" is 2 chars / 2.7, which floors to 0 and is raised to 1 token
        # 1 * 0.00000015 + 1024 * 0.0000006 = 0.00061455
        expected = Decimal("1") * Decimal("0.00000015") + Decimal("1024") * Decimal("0.0000006")
        assert price == expected

    @pytest.mark.asyncio
    async def test_multiplies_completion_price_by_n(self, provider, model_info_response):
        """n completions multiply the completion portion of the estimate."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 3,
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        prompt = Decimal("1") * Decimal("0.00000015")
        completion = Decimal("1024") * Decimal("0.0000006")
        assert price == prompt + completion * 3

    @pytest.mark.asyncio
    async def test_rejects_non_positive_n(self, provider, model_info_response):
        """n must be a positive integer."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 0,
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            with pytest.raises(ProviderError, match="n must be a positive integer"):
                await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    async def test_caches_model_info(self, provider, model_info_response):
        """Should cache model info and not re-fetch on second call."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price1 = await provider.get_price("chat/completions", body)
            price2 = await provider.get_price("chat/completions", body)

        assert mock_get.call_count == 1
        assert price1 == price2

    @pytest.mark.asyncio
    async def test_fetches_embedding_catalog_with_output_modalities(
        self, provider, embedding_model_info_response
    ):
        """Embedding lookup should query the dedicated embeddings catalog."""
        body = {
            "model": "openai/text-embedding-3-small",
            "input": "hello world",
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = embedding_model_info_response
            price = await provider.get_price("embeddings", body)

        assert price == Decimal("0.000001")
        assert mock_get.call_count == 1
        assert mock_get.call_args_list[0].args[0].endswith("/models?output_modalities=embeddings")

    @pytest.mark.asyncio
    async def test_stt_whisper_price_duration_based(
        self, provider, transcription_model_info_response
    ):
        """Whisper STT should price by audio duration (USD per second)."""
        body = {
            "model": "openai/whisper-1",
            "input_audio": {
                "data": make_wav_b64(1.0),
                "format": "wav",
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = transcription_model_info_response
            price = await provider.get_price("audio/transcriptions", body)

        # 1 billing second * $0.0001 per second
        assert price == Decimal("0.0001")
        assert (
            mock_get.call_args_list[0].args[0].endswith("/models?output_modalities=transcription")
        )

    @pytest.mark.asyncio
    async def test_stt_small_per_second_price_is_accepted(self, provider):
        """A sub-cent per-second rate is duration pricing, not a token model."""
        body = {
            "model": "openai/whisper-large-v3",
            "input_audio": {
                "data": make_wav_b64(1.0),
                "format": "wav",
            },
        }
        catalog = httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "id": "openai/whisper-large-v3",
                        "pricing": {"prompt": "0.0000075", "completion": "0"},
                    }
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("audio/transcriptions", body)

        assert price == Decimal("0.0000075")

    @pytest.mark.asyncio
    async def test_stt_hourly_catalog_price(self, provider):
        """Microsoft publishes the hourly rate in pricing.prompt."""
        body = {
            "model": "microsoft/mai-transcribe-1.5",
            "input_audio": {
                "data": make_wav_b64(1.0),
                "format": "wav",
            },
        }
        catalog = httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "id": "microsoft/mai-transcribe-1.5",
                        "pricing": {"prompt": "0.36", "completion": "0"},
                    }
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("audio/transcriptions", body)

        assert price == Decimal("0.36") / Decimal("3600")

    @pytest.mark.asyncio
    async def test_stt_token_model_rejected(self, provider, transcription_model_info_response):
        """Token-based STT models cannot be safely priced before payment."""
        body = {
            "model": "openai/gpt-4o-mini-transcribe",
            "input_audio": {
                "data": make_wav_b64(1.0),
                "format": "wav",
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = transcription_model_info_response
            with pytest.raises(ProviderError, match="Token-based STT model"):
                await provider.get_price("audio/transcriptions", body)

    @pytest.mark.asyncio
    async def test_stt_missing_audio_raises(self, provider, transcription_model_info_response):
        """STT requests without input_audio should fail fast."""
        body = {"model": "openai/whisper-1"}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = transcription_model_info_response
            with pytest.raises(ProviderError, match="Missing input_audio"):
                await provider.get_price("audio/transcriptions", body)

    @pytest.mark.asyncio
    async def test_stt_oversized_audio_raises(self, transcription_model_info_response):
        """STT audio exceeding max_stt_audio_mb should return 413."""
        config = ProviderConfig(
            enabled=True,
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key-123",
            max_stt_audio_mb=0,
        )
        small_provider = OpenRouterProvider(config=config)
        body = {
            "model": "openai/whisper-1",
            "input_audio": {
                "data": make_wav_b64(0.1),
                "format": "wav",
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = transcription_model_info_response
            with pytest.raises(ProviderError, match="too large") as exc_info:
                await small_provider.get_price("audio/transcriptions", body)

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_model_not_found(self, provider):
        """Should raise ProviderError for unknown model."""
        body = {
            "model": "nonexistent/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }

        # Return a valid models list that doesn't contain the requested model
        empty_catalog = httpx.Response(
            status_code=200,
            json={
                "data": [
                    {
                        "id": "some/other-model",
                        "pricing": {"prompt": "0.001", "completion": "0.002"},
                    }
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = empty_catalog

            with pytest.raises(ProviderError, match="not found"):
                await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    async def test_null_max_tokens_uses_default(self, provider, model_info_response):
        """max_tokens: null (JSON null) should not crash, uses default."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": None,  # JSON null
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        # Should use default_max_tokens (1024), same as omitting the key
        expected = Decimal("1") * Decimal("0.00000015") + Decimal("1024") * Decimal("0.0000006")
        assert price == expected


class TestRequestChecks:
    """Path and body checks that run in get_price, before payment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["completions", "responses", "images/generations", "x"])
    async def test_rejects_unpriced_paths(self, provider, path):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_price(path, _chat_body())
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_path_cannot_be_priced_as_an_embedding(self, provider):
        """A body with input and no messages on chat/completions was billed as ~1 token."""
        body = {"model": "openai/gpt-4o-mini", "input": "x" * 100_000, "prompt": "y" * 100_000}
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_with_input_key_is_priced_as_chat(self, provider, model_info_response):
        body = _chat_body(input="z" * 270)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(_chat_tokens(272))
        assert mock_get.call_args.args[0].endswith("/models")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            {"model": "openai/text-embedding-3-small"},
            {"model": "openai/text-embedding-3-small", "input": ""},
            {"model": "openai/text-embedding-3-small", "input": "a", "messages": []},
        ],
    )
    async def test_embeddings_need_input_only(self, provider, body):
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("embeddings", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_model_is_rejected(self, provider):
        """OpenRouter would fall back to the account default model, which is unpriced."""
        with pytest.raises(ProviderError, match="Missing model id"):
            await provider.get_price(
                "chat/completions", {"messages": [{"role": "user", "content": "Hi"}]}
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [True, "true", 1, {"x": 1}])
    async def test_rejects_any_truthy_stream(self, provider, stream):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="Streaming is not supported") as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(stream=stream))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_false_is_fine(self, provider, model_info_response):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", _chat_body(stream=False))
        assert price == _mini_price(1)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"plugins": [{"id": "file-parser"}]},
            {"plugins": ["web"]},
            {"plugins": {"id": "web"}},
            {"plugins": [{"id": "web", "engine": "native"}]},
            {"web_search_options": {"search_context_size": "high"}},
        ],
    )
    async def test_rejects_unpriced_plugins_and_native_search(self, provider, extra):
        with pytest.raises(ProviderError) as caught:
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("part_type", ["file", "video_url", None])
    async def test_rejects_unpriced_content_parts(self, provider, part_type):
        part = {"type": part_type, "file": {"file_data": "https://example.com/a.pdf"}}
        body = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": [part]}]}
        with pytest.raises(ProviderError, match="not supported") as caught:
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_routers_without_a_fixed_price(self, provider):
        catalog = _catalog(
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}
        )
        body = _chat_body(model="openrouter/auto")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            with pytest.raises(ProviderError, match="no fixed price") as caught:
                await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400


class TestChatInputEstimate:
    """Everything the model reads is counted, not only messages[].content."""

    @pytest.mark.asyncio
    async def test_tools_are_counted(self, provider, model_info_response):
        tools = [{"type": "function", "function": {"name": "f", "description": "d" * 5000}}]
        body = _chat_body(tools=tools)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        tools_chars = len(json.dumps(tools, separators=(",", ":")))
        assert price == _mini_price(_chat_tokens(len("Hi") + tools_chars))

    @pytest.mark.asyncio
    async def test_prompt_tool_calls_and_response_format_are_counted(
        self, provider, model_info_response
    ):
        tool_calls = [{"id": "c1", "function": {"name": "f", "arguments": "a" * 300}}]
        schema = {"type": "json_schema", "json_schema": {"schema": {"d": "s" * 200}}}
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "assistant", "content": None, "tool_calls": tool_calls}],
            "prompt": "p" * 400,
            "response_format": schema,
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        chars = (
            len(json.dumps(tool_calls, separators=(",", ":")))
            + 400
            + len(json.dumps(schema, separators=(",", ":")))
        )
        assert price == _mini_price(_chat_tokens(chars))

    @pytest.mark.asyncio
    async def test_text_parts_count_only_their_text(self, provider, model_info_response):
        content = [{"type": "text", "text": "What is quantum computing?"}]
        body = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(_chat_tokens(len("What is quantum computing?")))

    @pytest.mark.asyncio
    async def test_image_parts_add_a_fixed_allowance(self, provider, model_info_response):
        """An image costs a per-image allowance; its base64 bytes are not priced as text."""
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 500_000},
        }
        content = [{"type": "text", "text": "Describe"}, image, image]
        body = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": content}]}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(_chat_tokens(len("Describe")) + 2 * 4800)

    @pytest.mark.asyncio
    async def test_max_completion_tokens_is_priced(self, provider, model_info_response):
        body = _chat_body(max_completion_tokens=50_000)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(1, max_tokens=50_000)

    @pytest.mark.asyncio
    async def test_cache_control_uses_the_cache_write_rate(self, provider):
        catalog = _catalog(
            {
                "id": "anthropic/claude",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "input_cache_write": "0.00000375",
                },
            }
        )
        content = [{"type": "text", "text": "x" * 2700, "cache_control": {"type": "ephemeral"}}]
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": content}]}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("chat/completions", body)
        assert price == Decimal(1000) * Decimal("0.00000375") + Decimal(1024) * Decimal("0.000015")

    @pytest.mark.asyncio
    async def test_per_request_fee_is_added(self, provider):
        catalog = _catalog(
            {
                "id": "perplexity/sonar",
                "pricing": {"prompt": "0.000001", "completion": "0.000001", "request": "0.005"},
            }
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price(
                "chat/completions", _chat_body(model="perplexity/sonar")
            )
        assert price == Decimal("0.000001") + Decimal(1024) * Decimal("0.000001") + Decimal("0.005")


class TestWebPluginPricing:
    """max_results is clamped and the plugin is pinned to the priced engine."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("max_results", "priced"),
        [(None, 3), (-100, 3), (0, 3), ("50", 3), (True, 3), (1, 1), (7, 7), (10_000, 10)],
    )
    async def test_max_results_is_clamped(self, provider, model_info_response, max_results, priced):
        plugin = {"id": "web"} if max_results is None else {"id": "web", "max_results": max_results}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", _chat_body(plugins=[plugin]))
        plugins_json = json.dumps([plugin], separators=(",", ":"))
        tokens = _ascii_tokens("Hi" + plugins_json) + priced * 2048
        assert price == _mini_price(tokens) + Decimal(priced) * Decimal("0.004")


class TestModelsFallback:
    """A cheap primary model cannot route to an expensive fallback for free."""

    @pytest.mark.asyncio
    async def test_prices_the_most_expensive_candidate(self, provider, model_info_response):
        body = _chat_body(models=["openai/gpt-4o"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)

        tokens = _chat_tokens(len("Hi") + len('["openai/gpt-4o"]'))
        expensive = Decimal(tokens) * Decimal("0.0000025") + Decimal(1024) * Decimal("0.00001")
        assert price == expensive

    @pytest.mark.asyncio
    async def test_models_without_model_is_priced(self, provider, model_info_response):
        body = {
            "models": ["openai/gpt-4o-mini", "openai/gpt-4o"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price > _mini_price(1) * 10

    @pytest.mark.asyncio
    async def test_unknown_fallback_is_rejected(self, provider, model_info_response):
        body = _chat_body(models=["missing/model"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            with pytest.raises(ProviderError, match="not found"):
                await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("models", ["openai/gpt-4o", [1], [""]])
    async def test_malformed_models_is_rejected(self, provider, models):
        with pytest.raises(ProviderError, match="models must be") as caught:
            await provider.get_price("chat/completions", _chat_body(models=models))
        assert caught.value.status_code == 400


class TestSubmit:
    """Tests for OpenRouterProvider.submit()."""

    @pytest.fixture(autouse=True)
    def warm_catalog(self, provider, model_info_response):
        """submit reads the catalog get_price already loaded to cap the endpoint price."""
        models = {model["id"]: model for model in model_info_response.json()["data"]}
        provider._catalogs.put("", models)

    @pytest.mark.asyncio
    async def test_forwards_request(self, provider, chat_completion_response):
        """Should forward request body to OpenRouter and return response."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 20,
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            result = await provider.submit("chat/completions", body)

        assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert result["usage"]["total_tokens"] == 18

        # Verify correct URL and headers
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chat/completions" in call_args.args[0]
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key-123"

    @pytest.mark.asyncio
    async def test_api_error(self, provider):
        """Should raise ProviderError on API failure."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(status_code=429, text="Rate limit exceeded")

            with pytest.raises(ProviderError):
                await provider.submit("chat/completions", body)

    @pytest.mark.asyncio
    async def test_read_timeout_is_not_resent(self, provider):
        """A request that timed out after it was sent may be running; resending bills twice."""
        body = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.side_effect = httpx.ReadTimeout("slow")
            await provider.submit("chat/completions", body)

        assert mock_post.call_count == 1
        assert caught.value.status_code == 504

    @pytest.mark.asyncio
    async def test_connect_timeout_is_retried(self, provider, chat_completion_response):
        """A request that never reached OpenRouter is safe to send again."""
        body = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_post.side_effect = [httpx.ConnectTimeout("no route"), chat_completion_response]
            result = await provider.submit("chat/completions", body)

        assert mock_post.call_count == 2
        assert result["usage"]["total_tokens"] == 18

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure", [httpx.ConnectError("refused"), httpx.PoolTimeout("pool full")]
    )
    async def test_unopened_connection_is_retried_then_refunded(self, provider, failure):
        """Nothing reached OpenRouter, so it is retried and a final failure is known."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.side_effect = failure
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 3
        assert not outcome_unknown(caught.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_server_error_is_not_resent_and_is_a_known_failure(self, provider, status):
        """The call got an answer and may have run, so it is never sent twice."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(status, json={"error": {"message": "down"}})
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 1
        assert caught.value.status_code == status
        assert not outcome_unknown(caught.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "text"),
        [
            (504, "Gateway Timeout"),
            (520, ""),
            (524, "A timeout occurred"),
            (502, '{"error":{"message":"Upstream request timed out"}}'),
            (500, "<html>504 Gateway Time-out</html>"),
        ],
    )
    async def test_gateway_timeout_is_an_unknown_outcome(self, provider, status, text):
        """The request was forwarded and may have billed: kept, not resent."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch("x402gate.providers.openrouter.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(status, text=text)
            await provider.submit("chat/completions", _chat_body())
        assert mock_post.call_count == 1
        assert outcome_unknown(caught.value)

    @pytest.mark.asyncio
    async def test_redirect_is_a_known_failure(self, provider):
        """httpx does not follow redirects, so the model never ran."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(301, headers={"location": "/elsewhere"})
            await provider.submit("chat/completions", _chat_body())
        assert caught.value.status_code == 502
        assert not outcome_unknown(caught.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_is_not_passed_to_the_client(self, provider, status):
        """A 402 from OpenRouter is about our credit, not an x402 payment request."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = httpx.Response(status, text="Insufficient credits")
            await provider.submit("chat/completions", _chat_body())
        assert caught.value.status_code == 502
        assert "Insufficient credits" in caught.value.detail
        assert not outcome_unknown(caught.value)

    @pytest.mark.asyncio
    async def test_read_timeout_is_an_unknown_outcome(self, provider):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.side_effect = httpx.ReadTimeout("slow")
            await provider.submit("chat/completions", _chat_body())
        assert outcome_unknown(caught.value)

    @pytest.mark.asyncio
    async def test_no_task_id_in_response(self, provider, chat_completion_response):
        """Response should not contain 'id' field that triggers polling."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            result = await provider.submit("chat/completions", body)

        # OpenRouter responses have 'id' at top level (gen-xxx),
        # but managed handler checks result.get("data", result).get("id")
        # Since there's no "data" key, it checks result.get("id") = "gen-abc123"
        # This would trigger polling — let's verify the managed handler won't
        # by checking the response structure
        assert "data" not in result  # No "data" wrapper
        assert "choices" in result  # Direct OpenAI format

    @pytest.mark.asyncio
    async def test_null_max_tokens_enforced(self, provider, chat_completion_response):
        """submit() with max_tokens=None should enforce default minimum."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": None,  # JSON null
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_prepaid_submit_still_caps_max_tokens(self, provider, chat_completion_response):
        """Prepaid chat is capped the same way as an x402 payment."""
        body = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body, prepaid=True)

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_refuses_unpriced_path(self, provider):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("responses", _chat_body())
        assert caught.value.status_code == 400
        assert not outcome_unknown(caught.value)
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_refuses_stream(self, provider):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("chat/completions", _chat_body(stream="yes"))
        assert caught.value.status_code == 400
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text='data: {"choices": []}\n\ndata: [DONE]\n\n'),
            httpx.Response(200, json=["not", "an", "object"]),
        ],
    )
    async def test_unreadable_success_body_is_an_unknown_outcome(self, provider, response):
        """OpenRouter billed the call, so the payment must be kept, not refunded."""
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            mock_post.return_value = response
            await provider.submit("chat/completions", _chat_body())
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value)
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_web_plugin_is_pinned_to_the_priced_values(
        self, provider, chat_completion_response
    ):
        body = _chat_body(plugins=[{"id": "web", "max_results": 500}])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["plugins"] == [{"id": "web", "max_results": 10, "engine": "exa"}]

    @pytest.mark.asyncio
    async def test_drops_unpriced_max_completion_tokens(self, provider, chat_completion_response):
        body = _chat_body(max_completion_tokens="100000")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)
        sent = mock_post.call_args.kwargs["json"]
        assert "max_completion_tokens" not in sent
        assert sent["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_chat_with_input_audio_key_is_still_capped(
        self, provider, chat_completion_response
    ):
        """A top-level input_audio key used to skip the max_tokens cap on chat."""
        body = _chat_body(input_audio={"data": "AAAA", "format": "wav"})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)
        assert mock_post.call_args.kwargs["json"]["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_stt_submit_strips_internal_keys(self, provider):
        """STT submit should forward clean JSON without gateway-internal keys."""
        body = {
            "model": "openai/whisper-1",
            "input_audio": {"data": make_wav_b64(0.1), "format": "wav"},
            "_caller": "test-wallet",
        }
        stt_response = httpx.Response(
            status_code=200,
            json={
                "text": "hello",
                "usage": {"seconds": 1.0, "cost": 0.0001},
            },
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = stt_response
            result = await provider.submit("audio/transcriptions", body)

        assert result["text"] == "hello"
        sent_body = mock_post.call_args.kwargs["json"]
        assert "_caller" not in sent_body
        assert "audio/transcriptions" in mock_post.call_args.args[0]
        assert "max_tokens" not in sent_body


class TestGetResult:
    """Tests for OpenRouterProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_raises_not_implemented(self, provider):
        """Should raise NotImplementedError as OpenRouter is synchronous."""
        with pytest.raises(NotImplementedError):
            await provider.get_result("some-task-id")


class TestCalculateActualCost:
    """Tests for OpenRouterProvider.calculate_actual_cost()."""

    @pytest.mark.asyncio
    async def test_calculates_from_usage(self, provider, model_info_response):
        """Should compute actual cost from prompt + completion tokens."""
        body = {"model": "openai/gpt-4o-mini"}
        result = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)

        assert isinstance(cost, Decimal)
        expected = Decimal(10) * Decimal("0.00000015") + Decimal(50) * Decimal("0.0000006")
        assert cost == expected

    @pytest.mark.asyncio
    async def test_returns_none_without_usage(self, provider):
        """Should return None when response has no usage data."""
        body = {"model": "openai/gpt-4o-mini"}
        result = {"choices": [{"message": {"content": "Hello"}}]}

        cost = await provider.calculate_actual_cost(body, result)
        assert cost is None

    @pytest.mark.asyncio
    async def test_returns_none_with_partial_usage(self, provider):
        """Should return None when usage is missing token counts."""
        body = {"model": "openai/gpt-4o-mini"}
        result = {"usage": {"total_tokens": 60}}  # missing prompt/completion

        cost = await provider.calculate_actual_cost(body, result)
        assert cost is None

    @pytest.mark.asyncio
    async def test_stt_uses_reported_cost(self, provider):
        """STT responses should prefer usage.cost from OpenRouter."""
        body = {"model": "openai/whisper-1"}
        result = {"text": "hi", "usage": {"seconds": 2.0, "cost": 0.0002}}

        cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal("0.0002")

    @pytest.mark.asyncio
    async def test_chat_prefers_usage_cost(self, provider):
        """usage.cost is what OpenRouter charged: fallback model, provider and cache included."""
        body = {"model": "openai/gpt-4o-mini", "models": ["openai/gpt-4o"]}
        result = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
                "cost": 0.0123,
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            cost = await provider.calculate_actual_cost(body, result)

        assert cost == Decimal("0.0123")
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "usage_extra",
        [{"cost": -1}, {"cost": True}, {"cost": "abc"}, {"cost": 0.5, "is_byok": True}],
    )
    async def test_unusable_usage_cost_falls_back_to_catalog(
        self, provider, model_info_response, usage_extra
    ):
        """A negative, non-numeric or BYOK usage.cost is not the real charge."""
        body = {"model": "openai/gpt-4o-mini"}
        result = {"usage": {"prompt_tokens": 10, "completion_tokens": 50, **usage_extra}}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)

        expected = Decimal(10) * Decimal("0.00000015") + Decimal(50) * Decimal("0.0000006")
        assert cost == expected

    @pytest.mark.asyncio
    async def test_negative_web_results_do_not_lower_the_actual_cost(
        self, provider, model_info_response
    ):
        body = {"model": "openai/gpt-4o-mini", "plugins": [{"id": "web", "max_results": -50}]}
        result = {"usage": {"prompt_tokens": 10, "completion_tokens": 50}}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)

        tokens = Decimal(10) * Decimal("0.00000015") + Decimal(50) * Decimal("0.0000006")
        assert cost == tokens + Decimal(3) * Decimal("0.004")

    @pytest.mark.asyncio
    async def test_stt_duration_fallback_without_cost(
        self, provider, transcription_model_info_response
    ):
        """STT duration models should fall back to seconds-based pricing."""
        body = {"model": "openai/whisper-1"}
        result = {"text": "hi", "usage": {"seconds": 60.0}}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = transcription_model_info_response
            cost = await provider.calculate_actual_cost(body, result)

        assert cost == Decimal("0.006")  # 60 seconds * $0.0001 per second


class TestNonTextOutput:
    """Image and audio output bill above the text rate, so they are refused before payment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"modalities": ["image", "text"]},
            {"modalities": ["text", "audio"]},
            {"modalities": "image"},
            {"audio": {"voice": "alloy", "format": "wav"}},
        ],
    )
    async def test_rejects_image_and_audio_output(self, provider, extra):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="output") as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_only_modalities_is_fine(self, provider, model_info_response):
        body = _chat_body(modalities=["text"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(_chat_tokens(len("Hi") + len('["text"]')))


class TestReasoningBudget:
    """Thinking counts inside a cap above the budget, on top of a cap at or below it."""

    @staticmethod
    def _input_tokens(body: dict) -> int:
        text = "Hi" + "".join(
            json.dumps(body[key], separators=(",", ":"))
            for key in ("reasoning", "thinking")
            if key in body
        )
        return _ascii_tokens(text)

    async def _price(self, provider, model_info_response, body):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            return await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [{"reasoning": {"max_tokens": 32_000}}, {"thinking": {"budget_tokens": 32_000}}],
    )
    async def test_budget_is_added_without_an_explicit_cap(
        self, provider, model_info_response, extra
    ):
        """The injected default max_tokens is raised upstream to fit the budget."""
        body = _chat_body(**extra)
        price = await self._price(provider, model_info_response, body)
        assert price == _mini_price(self._input_tokens(body), max_tokens=1024 + 32_000)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"max_tokens": 16_000, "reasoning": {"max_tokens": 8000}},
            {"max_tokens": 16_000, "thinking": {"type": "enabled", "budget_tokens": 8000}},
            {"max_completion_tokens": 16_000, "reasoning": {"max_tokens": 8000}},
        ],
    )
    async def test_cap_above_the_budget_bounds_the_output(
        self, provider, model_info_response, extra
    ):
        """Thinking counts inside max_tokens, so 16000 is not priced as 24000."""
        body = _chat_body(**extra)
        price = await self._price(provider, model_info_response, body)
        assert price == _mini_price(self._input_tokens(body), max_tokens=16_000)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cap", [4000, 8000])
    async def test_cap_at_or_below_the_budget_adds_it(self, provider, model_info_response, cap):
        """Anthropic needs max_tokens above the budget, so the upstream raises the cap."""
        body = _chat_body(max_tokens=cap, reasoning={"max_tokens": 8000})
        price = await self._price(provider, model_info_response, body)
        assert price == _mini_price(self._input_tokens(body), max_tokens=cap + 8000)

    @pytest.mark.asyncio
    async def test_small_cap_is_raised_to_the_default_first(self, provider, model_info_response):
        """submit raises max_tokens to the 1024 default, which is above a 500 budget."""
        body = _chat_body(max_tokens=600, reasoning={"max_tokens": 500})
        price = await self._price(provider, model_info_response, body)
        assert price == _mini_price(self._input_tokens(body), max_tokens=1024)

    @pytest.mark.asyncio
    async def test_each_forwarded_cap_is_checked(self, provider, model_info_response):
        """A max_completion_tokens at the budget may be the cap the upstream honours."""
        body = _chat_body(
            max_tokens=9000, max_completion_tokens=8000, reasoning={"max_tokens": 8000}
        )
        price = await self._price(provider, model_info_response, body)
        assert price == _mini_price(self._input_tokens(body), max_tokens=16_000)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("budget", ["32000", -1, True, 1.5])
    async def test_malformed_budget_is_rejected(self, provider, budget):
        body = _chat_body(reasoning={"max_tokens": budget})
        with pytest.raises(ProviderError, match="reasoning.max_tokens") as caught:
            await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400


class TestImagePricing:
    """Each image is reserved at the larger of the token allowance and the catalog image rate."""

    @pytest.mark.asyncio
    async def test_catalog_image_rate_wins_when_higher(self, provider):
        catalog = _catalog(
            {
                "id": "vision/model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002", "image": "0.02"},
            }
        )
        image = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        body = {"model": "vision/model", "messages": [{"role": "user", "content": [image]}]}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("chat/completions", body)
        # 4800 tokens x $1/M = $0.0048 < $0.02 catalog rate.
        assert price == Decimal("0.000001") + Decimal("0.02") + Decimal(1024) * Decimal("0.000002")


class TestEndpointPriceCap:
    """Client routing that picks endpoints gets max_price; default routing keeps its fallback."""

    @pytest.fixture(autouse=True)
    def warm_catalog(self, provider, model_info_response, embedding_model_info_response):
        for key, response in (
            ("", model_info_response),
            ("embeddings", embedding_model_info_response),
        ):
            provider._catalogs.put(key, {model["id"]: model for model in response.json()["data"]})

    async def _sent(self, provider, chat_completion_response, path, body):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit(path, body)
        return mock_post.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_default_routing_gets_no_cap(self, provider, chat_completion_response):
        """A cap would stop OpenRouter falling back to a pricier endpoint."""
        sent = await self._sent(
            provider, chat_completion_response, "chat/completions", _chat_body()
        )
        assert "provider" not in sent

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "routing",
        [
            {},
            {"allow_fallbacks": True},
            {"data_collection": "allow"},
            {"zdr": False},
            {"require_parameters": False, "allow_fallbacks": True},
            {"order": [], "sort": None},
            {"max_price": {"prompt": 0.1}},
        ],
    )
    async def test_neutral_routing_is_forwarded_unchanged(
        self, provider, chat_completion_response, routing
    ):
        body = _chat_body(provider=routing)
        sent = await self._sent(provider, chat_completion_response, "chat/completions", body)
        assert sent["provider"] == routing

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "routing",
        [
            {"order": ["pricey-host"]},
            {"only": ["pricey-host"]},
            {"ignore": ["cheap-host"]},
            {"sort": "throughput"},
            {"quantizations": ["fp16"]},
            {"allow_fallbacks": False},
            {"require_parameters": True},
            # Privacy filters drop endpoints on every request, like ignore.
            {"data_collection": "deny"},
            {"zdr": True},
            {"zdr": True, "preferred_min_throughput": 100},
        ],
    )
    async def test_steering_routing_gets_the_catalog_cap(
        self, provider, chat_completion_response, routing
    ):
        body = _chat_body(provider=routing)
        sent = await self._sent(provider, chat_completion_response, "chat/completions", body)
        assert sent["provider"] == {
            **routing,
            "max_price": {"prompt": 0.15, "completion": 0.6, "request": 0.0},
        }

    @pytest.mark.asyncio
    async def test_cap_covers_the_priciest_fallback(self, provider, chat_completion_response):
        body = _chat_body(models=["openai/gpt-4o"], provider={"sort": "latency"})
        sent = await self._sent(provider, chat_completion_response, "chat/completions", body)
        assert sent["provider"]["max_price"] == {"prompt": 2.5, "completion": 10.0, "request": 0.0}

    @pytest.mark.asyncio
    async def test_merges_into_client_routing(self, provider, chat_completion_response):
        """Client routing is kept, a lower client cap wins, a higher one is lowered."""
        routing = {
            "order": ["pricey-host"],
            "sort": "throughput",
            "max_price": {"prompt": 0.1, "completion": 99, "image": 1},
        }
        body = _chat_body(provider=routing)
        sent = await self._sent(provider, chat_completion_response, "chat/completions", body)
        assert sent["provider"] == {
            "order": ["pricey-host"],
            "sort": "throughput",
            "max_price": {"prompt": 0.1, "completion": 0.6, "image": 1, "request": 0.0},
        }
        assert body["provider"]["max_price"]["completion"] == 99  # caller's dict untouched

    @pytest.mark.asyncio
    async def test_embeddings_follow_the_same_rule(self, provider, chat_completion_response):
        body = {"model": "openai/text-embedding-3-small", "input": "hello"}
        sent = await self._sent(provider, chat_completion_response, "embeddings", body)
        assert "provider" not in sent
        body = {**body, "provider": {"order": ["pricey-host"]}}
        sent = await self._sent(provider, chat_completion_response, "embeddings", body)
        assert sent["provider"]["max_price"]["prompt"] == 0.02

    @pytest.mark.asyncio
    @pytest.mark.parametrize("routing", ["openai", ["openai"], {"max_price": 5}])
    async def test_malformed_routing_is_rejected(self, provider, routing):
        with pytest.raises(ProviderError, match="provider must be") as caught:
            await provider.get_price("chat/completions", _chat_body(provider=routing))
        assert caught.value.status_code == 400


class TestCatalogFallbackCost:
    """Without usage.cost the fallback prices what OpenRouter may have billed."""

    @pytest.mark.asyncio
    async def test_prices_the_served_fallback_model(self, provider, model_info_response):
        body = _chat_body(models=["openai/gpt-4o"])
        result = {"model": "openai/gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 50}}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal(10) * Decimal("0.0000025") + Decimal(50) * Decimal("0.00001")

    @pytest.mark.asyncio
    async def test_unnamed_served_model_uses_the_priciest_candidate(
        self, provider, model_info_response
    ):
        body = _chat_body(models=["openai/gpt-4o"])
        result = {"model": "openai/gpt-4o-2024-08-06", "usage": {"prompt_tokens": 10}}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal(10) * Decimal("0.0000025")

    @pytest.mark.asyncio
    async def test_served_primary_model_keeps_its_own_rate(self, provider, model_info_response):
        body = _chat_body(models=["openai/gpt-4o"])
        result = {"model": "openai/gpt-4o-mini", "usage": {"prompt_tokens": 10}}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal(10) * MINI_PROMPT

    @pytest.mark.asyncio
    async def test_cache_breakpoint_uses_the_write_rate(self, provider):
        catalog = _catalog(
            {
                "id": "anthropic/claude",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "input_cache_write": "0.00000375",
                },
            }
        )
        content = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": content}]}
        result = {"usage": {"prompt_tokens": 1000, "completion_tokens": 10, "is_byok": True}}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            cost = await provider.calculate_actual_cost(body, result)
        assert cost == Decimal(1000) * Decimal("0.00000375") + Decimal(10) * Decimal("0.000015")


class TestInputTokenEstimate:
    """Dense text is weighed by character class so it cannot be priced below its bill."""

    @staticmethod
    def _tokens(text: str) -> int:
        body = {"model": "m", "messages": [{"role": "user", "content": text}]}
        return estimate_chat_input_tokens(body, "openrouter")

    def test_ascii_prose_is_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog. " * 40
        assert self._tokens(text) == _chat_tokens(len(text))

    def test_digits_count_one_token_each(self):
        """Many tokenizers split numbers into single digits."""
        assert self._tokens("1234567890" * 100) == 1000

    @pytest.mark.parametrize("char", ["\U00020000", "\U0002a6d6", "\U0001f600"])
    def test_astral_characters_are_at_least_three_tokens(self, char):
        """CJK Extension B and emoji bill 2-3 tokens each."""
        assert self._tokens(char * 1000) >= 3000

    def test_russian_is_priced_within_a_sane_factor(self):
        """Cyrillic bills about 0.3-0.6 tokens per character; it is not priced far above."""
        text = "Привет! Сегодня отличная погода, и мы пойдём гулять в парк у реки. " * 20
        tokens = self._tokens(text)
        assert 0.5 * len(text) <= tokens <= 0.8 * len(text)

    def test_common_cjk_is_one_and_a_half_tokens(self):
        assert self._tokens("你好世界" * 100) == 600

    @pytest.mark.parametrize(
        ("text", "tokens"),
        [
            ("\x01" * 100, 100),  # control codes: one byte each
            ("e" + "́" * 99, 198),  # stacked combining marks: two bytes each
            ("" * 100, 300),  # private use: three bytes each
        ],
    )
    def test_characters_no_vocabulary_holds_bill_per_byte(self, text, tokens):
        assert self._tokens(text) == tokens

    def test_cjk_extension_a_is_at_least_three_tokens(self):
        """The BMP twin of the Extension B attack."""
        assert self._tokens("\u3400\u4dbf\u3aaa" * 1000) >= 9000

    @pytest.mark.parametrize("mark", ["\u05b0", "\u05b8", "\u064e", "\u0670", "\u0483"])
    def test_stacked_two_byte_marks_are_at_least_two_tokens(self, mark):
        """Hebrew points, Arabic harakat and Cyrillic marks stack without limit."""
        assert self._tokens("a" + mark * 1000) >= 2000

    @pytest.mark.parametrize("char", ["\U000e0041", "\U000e0101", "\U000f0000", "\U0010fffd"])
    def test_rare_astral_characters_are_four_tokens(self, char):
        assert self._tokens(char * 1000) == 4000

    @pytest.mark.parametrize(
        "text",
        [
            "Привет! Сегодня отличная погода, и мы пойдём гулять в парк у реки.",
            "Καλημέρα, πώς είσαι σήμερα;",
            "你好，今天天气很好。",
            "こんにちは、元気ですか。",
            "안녕하세요, 오늘 날씨가 좋네요.",
        ],
    )
    def test_normal_text_has_no_rare_characters(self, text):
        """The per-byte classes never touch ordinary prose."""
        assert openrouter._COMBINING_MARKS.search(text) is None
        assert openrouter._RARE_THREE_BYTE.search(text) is None
        assert openrouter._RARE_FOUR_BYTE.search(text) is None

    def test_tool_definitions_use_the_same_weights(self):
        tools = [{"type": "function", "function": {"name": "f", "description": "\U00020000" * 500}}]
        body = {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "tools": tools}
        assert estimate_chat_input_tokens(body, "openrouter") >= 1500

    def test_inline_audio_keeps_the_plain_size_rule(self):
        """Base64 audio is not tokenized text; its digits are not weighed as digits."""
        part = {"type": "input_audio", "input_audio": {"data": "QUJD0123" * 100, "format": "wav"}}
        body = {"model": "m", "messages": [{"role": "user", "content": [part]}]}
        serialized = json.dumps(part, separators=(",", ":"))
        assert estimate_chat_input_tokens(body, "openrouter") == _chat_tokens(len(serialized))

    def test_lone_surrogate_does_not_raise(self):
        assert self._tokens("a\ud800b") >= 1


def _expire(catalogs: openrouter.CatalogCache, key: str) -> None:
    """Age a loaded catalog past its TTL."""
    catalogs._loaded_at[key] -= 2 * catalogs._ttl
    catalogs._refresh_at[key] -= 2 * catalogs._ttl


def _gpt4o_mini(prompt: str) -> httpx.Response:
    return _catalog({"id": "openai/gpt-4o-mini", "pricing": {"prompt": prompt, "completion": "0"}})


class TestPooledClient:
    """Billed calls and catalog fetches reuse one pooled client."""

    async def test_submits_share_one_client_with_a_short_connect_timeout(
        self, provider, chat_completion_response
    ):
        provider._catalogs.put("", {})
        with patch.object(httpx.AsyncClient, "post", autospec=True) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", _chat_body())
            await provider.submit("chat/completions", _chat_body())

        first, second = mock_post.call_args_list
        assert first.args[0] is second.args[0]
        assert not first.args[0].is_closed
        sent_timeout = first.kwargs["timeout"]
        assert sent_timeout.connect == 10.0
        assert sent_timeout.read == float(provider._config.poll_timeout)
        # A full shared pool fails fast (never sent: retried, then refunded).
        assert sent_timeout.pool == 10.0

    async def test_catalog_fetch_uses_the_shared_client(self, provider, model_info_response):
        with (
            patch.object(httpx.AsyncClient, "get", autospec=True) as mock_get,
            patch.object(httpx.AsyncClient, "post", autospec=True) as mock_post,
        ):
            mock_get.return_value = model_info_response
            mock_post.return_value = httpx.Response(200, json={"usage": {}})
            await provider.get_price("chat/completions", _chat_body())
            await provider.submit("chat/completions", _chat_body())

        assert mock_get.call_args.args[0] is mock_post.call_args.args[0]
        assert mock_get.call_args.kwargs["timeout"].read == 15.0


class TestCatalogWarmUp:
    """Catalogs load at startup, so the first request does not wait for them."""

    async def test_warm_up_loads_every_catalog(
        self,
        provider,
        model_info_response,
        embedding_model_info_response,
        transcription_model_info_response,
    ):
        def by_url(url, **kwargs):
            if url.endswith("embeddings"):
                return embedding_model_info_response
            if url.endswith("transcription"):
                return transcription_model_info_response
            return model_info_response

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = by_url
            await provider.warm_up()
            assert mock_get.call_count == 3
            await provider.get_price("chat/completions", _chat_body())
            await provider.get_price(
                "embeddings", {"model": "openai/text-embedding-3-small", "input": "hi"}
            )
            await provider.get_price(
                "audio/transcriptions",
                {
                    "model": "openai/whisper-1",
                    "input_audio": {"data": make_wav_b64(1.0), "format": "wav"},
                },
            )

        assert mock_get.call_count == 3

    async def test_warm_up_failure_is_raised_for_the_log(self, provider, model_info_response):
        """One failed catalog is reported; the ones that loaded are kept."""

        def by_url(url, **kwargs):
            if url.endswith("transcription"):
                return httpx.Response(400, text="bad request")
            return model_info_response

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = by_url
            with pytest.raises(ProviderError):
                await provider.warm_up()
            await provider.get_price("chat/completions", _chat_body())

        assert mock_get.call_count == 3

    async def test_concurrent_first_requests_share_one_fetch(self, provider, model_info_response):
        release = asyncio.Event()

        async def slow_catalog(url, **kwargs):
            await release.wait()
            return model_info_response

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = slow_catalog
            requests = [
                asyncio.create_task(provider.get_price("chat/completions", _chat_body()))
                for _ in range(5)
            ]
            await asyncio.sleep(0)
            release.set()
            prices = await asyncio.gather(*requests)

        assert mock_get.call_count == 1
        assert len(set(prices)) == 1

    async def test_cancelled_request_does_not_cancel_the_shared_fetch(
        self, provider, model_info_response
    ):
        release = asyncio.Event()

        async def slow_catalog(url, **kwargs):
            await release.wait()
            return model_info_response

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = slow_catalog
            first = asyncio.create_task(provider.get_price("chat/completions", _chat_body()))
            second = asyncio.create_task(provider.get_price("chat/completions", _chat_body()))
            await asyncio.sleep(0)
            first.cancel()
            release.set()
            price = await second

        assert price == _mini_price(_chat_tokens(len("Hi")))
        assert mock_get.call_count == 1


class TestCatalogStaleWhileRevalidate:
    """An expired catalog is served while one background task refreshes it."""

    async def test_expired_catalog_is_served_while_one_refresh_runs(self, provider):
        provider._catalogs.put("", {"openai/gpt-4o-mini": _gpt4o_mini("0.001").json()["data"][0]})
        _expire(provider._catalogs, "")
        release = asyncio.Event()

        async def slow_catalog(url, **kwargs):
            await release.wait()
            return _gpt4o_mini("0.002")

        body = _chat_body(max_tokens=1024)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = slow_catalog
            # Both answer at once from the old prices; only one fetch starts.
            old = await provider.get_price("chat/completions", body)
            refresh = provider._catalogs._tasks[""]
            again = await provider.get_price("chat/completions", body)
            assert provider._catalogs._tasks[""] is refresh
            await asyncio.sleep(0)
            assert mock_get.call_count == 1
            release.set()
            await refresh
            new = await provider.get_price("chat/completions", body)

        tokens = _chat_tokens(len("Hi"))
        assert old == again == Decimal(tokens) * Decimal("0.001")
        assert new == Decimal(tokens) * Decimal("0.002")
        assert mock_get.call_count == 1

    async def test_failed_refresh_keeps_the_old_catalog_and_retries_later(self, provider):
        provider._catalogs.put("", {"openai/gpt-4o-mini": _gpt4o_mini("0.001").json()["data"][0]})
        _expire(provider._catalogs, "")
        body = _chat_body(max_tokens=1024)
        expected = Decimal(_chat_tokens(len("Hi"))) * Decimal("0.001")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = httpx.Response(400, text="down")
            assert await provider.get_price("chat/completions", body) == expected
            await asyncio.gather(provider._catalogs._tasks[""], return_exceptions=True)
            # Within the retry delay no new fetch starts.
            assert await provider.get_price("chat/completions", body) == expected
            assert mock_get.call_count == 1

            provider._catalogs._refresh_at[""] -= openrouter._CATALOG_RETRY_S
            mock_get.return_value = _gpt4o_mini("0.002")
            assert await provider.get_price("chat/completions", body) == expected
            await provider._catalogs._tasks[""]

        assert mock_get.call_count == 2
        assert await provider.get_price("chat/completions", body) == expected * 2

    async def test_empty_refresh_keeps_the_old_catalog(self, provider):
        provider._catalogs.put("", {"openai/gpt-4o-mini": _gpt4o_mini("0.001").json()["data"][0]})
        _expire(provider._catalogs, "")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog()
            await provider.get_price("chat/completions", _chat_body())
            await asyncio.gather(provider._catalogs._tasks[""], return_exceptions=True)

        assert "openai/gpt-4o-mini" in provider._catalogs._catalogs[""]

    async def test_empty_first_load_is_an_error(self, provider):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog()
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_price("chat/completions", _chat_body())

        assert exc_info.value.status_code == 503

    async def test_model_missing_from_an_expired_catalog_waits_for_the_refresh(
        self, provider, model_info_response
    ):
        provider._catalogs.put("", {"old/model": {"id": "old/model", "pricing": {}}})
        _expire(provider._catalogs, "")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", _chat_body())

        assert price == _mini_price(_chat_tokens(len("Hi")))
        assert mock_get.call_count == 1

    async def test_model_missing_from_a_fresh_catalog_is_not_refetched(
        self, provider, model_info_response
    ):
        provider._catalogs.put("", {"old/model": {"id": "old/model", "pricing": {}}})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_price("chat/completions", _chat_body())

        assert exc_info.value.status_code == 404
        mock_get.assert_not_called()

    async def test_unknown_model_after_a_failed_refresh_does_not_fetch_again(self, provider):
        """In the retry delay an unknown id is a 404 at once, not a fetch per request."""
        provider._catalogs.put("", {"old/model": {"id": "old/model", "pricing": {}}})
        _expire(provider._catalogs, "")
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
            provider._catalogs._refresh_at[""] -= openrouter._CATALOG_RETRY_S
            mock_get.return_value = _gpt4o_mini("0.001")
            price = await provider.get_price("chat/completions", _chat_body(max_tokens=1024))

        assert price == Decimal(_chat_tokens(len("Hi"))) * Decimal("0.001")
        assert mock_get.call_count == 2


class TestCapAfterRefresh:
    """A refresh between get_price and submit never lifts the cap above the priced rates."""

    _ROUTING = {"order": ["pricey-host"]}

    async def _refreshed_cap(self, provider, chat_completion_response, old, new, model=None):
        provider._catalogs.put("", old)
        provider._catalogs.put("", new)
        body = _chat_body(provider=self._ROUTING, **({"model": model} if model else {}))
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)
        return mock_post.call_args.kwargs["json"]["provider"]["max_price"]

    @staticmethod
    def _mini(prompt: str, completion: str = "0.000001") -> dict:
        return {
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "pricing": {"prompt": prompt, "completion": completion},
            }
        }

    async def test_raised_rate_keeps_the_replaced_cap(self, provider, chat_completion_response):
        cap = await self._refreshed_cap(
            provider,
            chat_completion_response,
            self._mini("0.000001"),
            self._mini("0.000003", completion="0.0000005"),
        )
        # Prompt at the old (priced) rate; completion fell, so the new one.
        assert cap == {"prompt": 1.0, "completion": 0.5, "request": 0.0}

    async def test_model_new_in_the_refresh_uses_the_fresh_rates(
        self, provider, chat_completion_response
    ):
        new = {
            **self._mini("0.000001"),
            "new/model": {"id": "new/model", "pricing": {"prompt": "0.000002", "completion": "0"}},
        }
        cap = await self._refreshed_cap(
            provider, chat_completion_response, self._mini("0.000001"), new, model="new/model"
        )
        assert cap["prompt"] == 2.0

    async def test_cap_follows_the_fresh_rates_after_the_grace_window(
        self, provider, chat_completion_response
    ):
        provider._catalogs.put("", self._mini("0.000001"))
        provider._catalogs.put("", self._mini("0.000003"))
        old, replaced_at = provider._catalogs._previous[""]
        provider._catalogs._previous[""] = (old, replaced_at - openrouter._CAP_GRACE_S - 1)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", _chat_body(provider=self._ROUTING))
        assert mock_post.call_args.kwargs["json"]["provider"]["max_price"]["prompt"] == 3.0


class TestRequestWeighing:
    """Large bodies are weighed once per request, and audio off the event loop."""

    async def test_get_price_weighs_chat_text_once_and_submit_not_at_all(
        self, provider, model_info_response, chat_completion_response
    ):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            patch.object(openrouter, "chat_input_size", wraps=openrouter.chat_input_size) as weigh,
        ):
            mock_get.return_value = model_info_response
            mock_post.return_value = chat_completion_response
            await provider.get_price("chat/completions", _chat_body())
            assert weigh.call_count == 1
            await provider.submit("chat/completions", _chat_body())

        assert weigh.call_count == 1

    async def test_submit_still_refuses_unpriced_content_parts(self, provider):
        body = _chat_body(
            messages=[{"role": "user", "content": [{"type": "file", "file": {"data": "x"}}]}]
        )
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as exc_info,
        ):
            await provider.submit("chat/completions", body)

        assert exc_info.value.status_code == 400
        assert "file" in exc_info.value.detail
        mock_post.assert_not_called()

    async def test_stt_audio_is_measured_once_in_a_worker_thread(
        self, provider, transcription_model_info_response
    ):
        threads = []
        real_duration = openrouter.get_audio_duration_seconds

        def recording_duration(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_duration(*args, **kwargs)

        body = {
            "model": "openai/whisper-1",
            "models": ["openai/whisper-1", "openai/whisper-1-copy"],
            "input_audio": {"data": make_wav_b64(2.0), "format": "wav"},
        }
        catalog = transcription_model_info_response.json()["data"]
        catalog.append(
            {"id": "openai/whisper-1-copy", "pricing": {"prompt": "0.0003", "completion": "0"}}
        )
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            patch.object(openrouter, "get_audio_duration_seconds", recording_duration),
        ):
            mock_get.return_value = _catalog(*catalog)
            price = await provider.get_price("audio/transcriptions", body)

        assert price == Decimal(2) * Decimal("0.0003")
        assert len(threads) == 1
        assert threads[0] is not threading.main_thread()
