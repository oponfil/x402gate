"""Unit tests for OpenRouter provider."""

import asyncio
import base64
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

    def test_inline_audio_data_is_not_weighed_as_text(self):
        """Base64 audio is priced by its duration; the rest of the part is text."""
        part = {"type": "input_audio", "input_audio": {"data": "QUJD0123" * 100, "format": "wav"}}
        body = {"model": "m", "messages": [{"role": "user", "content": [part]}]}
        text = '{"type":"input_audio"}' + '{"format":"wav"}'
        assert estimate_chat_input_tokens(body, "openrouter") == _ascii_tokens(text)

    def test_text_beside_inline_audio_is_weighed(self):
        """Extra keys in an audio part cannot carry an unpriced prompt."""
        audio = {"data": "QUJD", "format": "wav", "note": "y" * 2700}
        part = {"type": "input_audio", "input_audio": audio, "extra": "z" * 2700}
        body = {"model": "m", "messages": [{"role": "user", "content": [part]}]}
        assert estimate_chat_input_tokens(body, "openrouter") >= 2000

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


_STT_BODY = {"model": "openai/whisper-1", "input_audio": {"data": "", "format": "wav"}}


class TestSttPricingSlots:
    """Unpaid STT pricing decodes and parses audio in at most two threads at once."""

    @pytest.fixture
    def stt_body(self):
        return {**_STT_BODY, "input_audio": {"data": make_wav_b64(1.0), "format": "wav"}}

    async def _price(self, provider, body, catalog):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            return await provider.get_price("audio/transcriptions", body)

    async def test_base64_decode_runs_in_a_worker_thread_holding_a_slot(
        self, provider, stt_body, transcription_model_info_response
    ):
        seen = []
        real_extract = openrouter._extract_stt_audio

        def recording_extract(*args, **kwargs):
            free = openrouter._stt_slots._value
            seen.append((threading.current_thread(), free))
            return real_extract(*args, **kwargs)

        with patch.object(openrouter, "_extract_stt_audio", recording_extract):
            price = await self._price(provider, stt_body, transcription_model_info_response)

        assert price == Decimal("0.0001")
        [(thread, free)] = seen
        assert thread is not threading.main_thread()
        assert free == openrouter._STT_PRICING_SLOTS - 1
        assert openrouter._stt_slots._value == openrouter._STT_PRICING_SLOTS

    async def test_busy_slots_refuse_with_429(
        self, provider, stt_body, transcription_model_info_response
    ):
        slots = openrouter._stt_pricing_slots()
        for _ in range(openrouter._STT_PRICING_SLOTS):
            await slots.acquire()
        try:
            with (
                patch.object(openrouter, "_STT_PRICING_WAIT_S", 0.01),
                patch.object(provider, "_stt_billing_seconds") as measure,
                pytest.raises(ProviderError, match="retry shortly") as exc,
            ):
                await self._price(provider, stt_body, transcription_model_info_response)
        finally:
            for _ in range(openrouter._STT_PRICING_SLOTS):
                slots.release()
        # Unpaid clients can fill the slots: not a provider failure (5xx).
        assert exc.value.status_code == 429
        measure.assert_not_called()

    async def test_waiter_gets_a_slot_when_one_frees(
        self, provider, stt_body, transcription_model_info_response
    ):
        slots = openrouter._stt_pricing_slots()
        for _ in range(openrouter._STT_PRICING_SLOTS):
            await slots.acquire()
        pricing = asyncio.create_task(
            self._price(provider, stt_body, transcription_model_info_response)
        )
        await asyncio.sleep(0.05)
        assert not pricing.done()
        slots.release()
        assert await pricing == Decimal("0.0001")
        slots.release()
        assert slots._value == openrouter._STT_PRICING_SLOTS

    async def test_slot_held_until_the_thread_ends_after_cancel(
        self, provider, stt_body, transcription_model_info_response
    ):
        entered, release = threading.Event(), threading.Event()

        def slow_measure(inputs):
            entered.set()
            release.wait(5)
            return 1

        with patch.object(provider, "_stt_billing_seconds", slow_measure):
            pricing = asyncio.create_task(
                self._price(provider, stt_body, transcription_model_info_response)
            )
            await asyncio.to_thread(entered.wait, 5)
            pricing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pricing
            slots = openrouter._stt_pricing_slots()
            # The thread still runs, so its slot is still taken.
            assert slots._value == openrouter._STT_PRICING_SLOTS - 1
            release.set()
            for _ in range(100):
                if slots._value == openrouter._STT_PRICING_SLOTS:
                    break
                await asyncio.sleep(0.01)
        assert slots._value == openrouter._STT_PRICING_SLOTS

    async def test_failed_measure_releases_its_slot(
        self, provider, transcription_model_info_response
    ):
        body = {**_STT_BODY, "input_audio": {"data": "not base64!", "format": "wav"}}
        for _ in range(openrouter._STT_PRICING_SLOTS + 1):
            with pytest.raises(ProviderError, match="Invalid base64") as exc:
                await self._price(provider, body, transcription_model_info_response)
            assert exc.value.status_code == 400
        assert openrouter._stt_pricing_slots()._value == openrouter._STT_PRICING_SLOTS

    async def test_one_semaphore_per_event_loop(self):
        slots = openrouter._stt_pricing_slots()
        assert openrouter._stt_pricing_slots() is slots
        # A semaphore from another loop is never reused: it binds to the
        # loop that first waited on it.
        with patch.multiple(openrouter, _stt_slots=slots, _stt_slots_loop=object()):
            assert openrouter._stt_pricing_slots() is not slots


def _claude_catalog(**pricing: str) -> httpx.Response:
    base = {"prompt": "0.000003", "completion": "0.000015", "input_cache_write": "0.00000375"}
    return _catalog({"id": "anthropic/claude", "pricing": {**base, **pricing}})


def _cached_body(ttl: str | None) -> dict:
    control = {"type": "ephemeral"} if ttl is None else {"type": "ephemeral", "ttl": ttl}
    content = [{"type": "text", "text": "x" * 2700, "cache_control": control}]
    return {"model": "anthropic/claude", "messages": [{"role": "user", "content": content}]}


class TestNativeSearchAndReasoning:
    """Per-search and reasoning rates of the catalog are part of the quote."""

    SEARCH_MODEL = {
        "id": "perplexity/sonar-pro",
        "pricing": {
            "prompt": "0.000002",
            "completion": "0.000008",
            "web_search": "0.005",
            "internal_reasoning": "0.00001",
        },
    }

    @pytest.mark.asyncio
    async def test_searches_and_reasoning_rate_are_priced(self, provider):
        body = _chat_body(model="perplexity/sonar-pro")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.SEARCH_MODEL)
            price = await provider.get_price("chat/completions", body)
        searches = openrouter._NATIVE_SEARCHES * Decimal("0.005")
        assert price == Decimal("0.000002") + Decimal(1024) * Decimal("0.00001") + searches

    @pytest.mark.asyncio
    async def test_each_completion_gets_its_searches(self, provider):
        body = _chat_body(model="perplexity/sonar-pro", n=2)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.SEARCH_MODEL)
            price = await provider.get_price("chat/completions", body)
        searches = 2 * openrouter._NATIVE_SEARCHES * Decimal("0.005")
        assert price == Decimal("0.000002") + 2 * 1024 * Decimal("0.00001") + searches

    @pytest.mark.asyncio
    async def test_lower_reasoning_rate_keeps_the_completion_rate(self, provider):
        model = {"id": "x/thinker", "pricing": {**self.SEARCH_MODEL["pricing"]}}
        model["pricing"].update(web_search="0", internal_reasoning="0.000001")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(model)
            price = await provider.get_price("chat/completions", _chat_body(model="x/thinker"))
        assert price == Decimal("0.000002") + Decimal(1024) * Decimal("0.000008")

    @pytest.mark.asyncio
    async def test_deep_research_is_refused(self, provider):
        model = {"id": "perplexity/sonar-deep-research", "pricing": self.SEARCH_MODEL["pricing"]}
        body = _chat_body(model="perplexity/sonar-deep-research")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(model)
            with pytest.raises(ProviderError, match="deep research") as caught:
                await provider.get_price("chat/completions", body)
        assert caught.value.status_code == 400

    @pytest.mark.asyncio
    async def test_fallback_cost_bills_reasoning_tokens_at_their_rate(self, provider):
        body = _chat_body(model="perplexity/sonar-pro")
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 60},
            "is_byok": True,
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.SEARCH_MODEL)
            cost = await provider.calculate_actual_cost(body, {"usage": usage})
        # The native searches get_price reserved are billed with the tokens.
        assert cost == (
            Decimal(10) * Decimal("0.000002")
            + Decimal(40) * Decimal("0.000008")
            + Decimal(60) * Decimal("0.00001")
            + openrouter._NATIVE_SEARCHES * Decimal("0.005")
        )


class TestLongCacheTtl:
    """A 1-hour cache write bills 2x input, above the 5-minute write rate."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("ttl", "rate"),
        [(None, "0.00000375"), ("5m", "0.00000375"), ("1h", "0.000006"), ("2h", "0.000006")],
    )
    async def test_quote_uses_the_ttl_write_rate(self, provider, ttl, rate):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _claude_catalog()
            price = await provider.get_price("chat/completions", _cached_body(ttl))
        assert price == Decimal(1000) * Decimal(rate) + Decimal(1024) * Decimal("0.000015")

    @pytest.mark.asyncio
    async def test_fallback_cost_uses_the_ttl_write_rate(self, provider):
        result = {"usage": {"prompt_tokens": 1000, "completion_tokens": 10, "is_byok": True}}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _claude_catalog()
            cost = await provider.calculate_actual_cost(_cached_body("1h"), result)
        assert cost == Decimal(1000) * Decimal("0.000006") + Decimal(10) * Decimal("0.000015")


class TestLongContextTier:
    """The catalog's pricing.overrides set the rates of a long prompt."""

    LONG_TEXT = "x" * 810_000  # 300_000 tokens at 2.7 characters per token
    TIERED = {
        "id": "openai/gpt-5.5",
        "pricing": {
            "prompt": "0.000005",
            "completion": "0.00003",
            "web_search": "0.01",
            "overrides": [
                {"min_prompt_tokens": 272_000, "prompt": "0.00001", "completion": "0.000045"}
            ],
        },
    }

    @staticmethod
    def _body(model: str, text: str, **extra) -> dict:
        return {"model": model, "messages": [{"role": "user", "content": text}], **extra}

    async def _price(self, provider, body: dict, *models: dict) -> Decimal:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(*models)
            return await provider.get_price("chat/completions", body)

    async def _fallback_cost(self, provider, body: dict, usage: dict, *models: dict):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(*models)
            return await provider.calculate_actual_cost(body, {"usage": {**usage, "is_byok": True}})

    @pytest.mark.asyncio
    async def test_long_prompt_uses_the_override_rates(self, provider):
        body = self._body("openai/gpt-5.5", self.LONG_TEXT)
        price = await self._price(provider, body, self.TIERED)
        assert price == Decimal(300_000) * Decimal("0.00001") + Decimal(1024) * Decimal("0.000045")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text", "tiered"),
        [
            ("x" * 734_400, True),  # exactly 272k tokens: min_prompt_tokens is inclusive
            ("x" * 734_397, False),  # 271_998 tokens
        ],
    )
    async def test_threshold_is_the_minimum_prompt(self, provider, text, tiered):
        tokens = len(text) * 10 // 27
        price = await self._price(provider, self._body("openai/gpt-5.5", text), self.TIERED)
        prompt, completion = ("0.00001", "0.000045") if tiered else ("0.000005", "0.00003")
        assert price == Decimal(tokens) * Decimal(prompt) + Decimal(1024) * Decimal(completion)

    @pytest.mark.asyncio
    async def test_tier_below_200k_and_above_2x(self, provider):
        haiku = {
            "id": "anthropic/claude-haiku-5.5",
            "pricing": {
                "prompt": "0.0000001",
                "completion": "0.0000005",
                "overrides": [
                    {
                        "min_prompt_tokens": 100_000,
                        "prompt": "0.0000005",
                        "completion": "0.0000025",
                    }
                ],
            },
        }
        body = self._body("anthropic/claude-haiku-5.5", "x" * 405_000)  # 150k tokens
        price = await self._price(provider, body, haiku)
        assert price == Decimal(150_000) * Decimal("0.0000005") + Decimal(1024) * Decimal(
            "0.0000025"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text", "prompt", "completion"),
        [
            ("x" * 81_000, "0.00000003", "0.00000013"),  # 30k tokens: base
            ("x" * 120_000, "0.0000001", "0.0000004"),  # 44_444 tokens: 32k tier
            ("x" * 810_000, "0.0000002", "0.0000008"),  # 300k tokens: 256k tier
        ],
    )
    async def test_highest_tier_reached_applies(self, provider, text, prompt, completion):
        qwen = {
            "id": "qwen/qwen3.7-flash",
            "pricing": {
                "prompt": "0.00000003",
                "completion": "0.00000013",
                # Listed out of order: the highest threshold reached wins.
                "overrides": [
                    {
                        "min_prompt_tokens": 256_000,
                        "prompt": "0.0000002",
                        "completion": "0.0000008",
                    },
                    {"min_prompt_tokens": 32_000, "prompt": "0.0000001", "completion": "0.0000004"},
                ],
            },
        }
        tokens = len(text) * 10 // 27
        price = await self._price(provider, self._body("qwen/qwen3.7-flash", text), qwen)
        assert price == Decimal(tokens) * Decimal(prompt) + Decimal(1024) * Decimal(completion)

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
            "google/gemini-3-flash",
        ],
    )
    async def test_model_without_overrides_is_flat(self, provider, model):
        entry = {
            "id": model,
            "context_length": 1_000_000,
            "pricing": {"prompt": "0.000005", "completion": "0.000025", "web_search": "0.01"},
        }
        price = await self._price(provider, self._body(model, self.LONG_TEXT), entry)
        assert price == Decimal(300_000) * Decimal("0.000005") + Decimal(1024) * Decimal("0.000025")

    @pytest.mark.asyncio
    async def test_missing_cache_write_and_reasoning_rates_scale(self, provider):
        gemini = {
            "id": "google/gemini-2.5-pro",
            "pricing": {
                "prompt": "0.000002",
                "completion": "0.00001",
                "input_cache_write": "0.000003",
                "internal_reasoning": "0.00002",
                "overrides": [
                    {"min_prompt_tokens": 200_000, "prompt": "0.000004", "completion": "0.000015"}
                ],
            },
        }
        content = [{"type": "text", "text": self.LONG_TEXT, "cache_control": {"type": "ephemeral"}}]
        body = {
            "model": "google/gemini-2.5-pro",
            "messages": [{"role": "user", "content": content}],
        }
        price = await self._price(provider, body, gemini)
        tokens = estimate_chat_input_tokens(body, "openrouter")
        # Cache write 3e-6 x (4e-6 / 2e-6); reasoning 2e-5 x (1.5e-5 / 1e-5).
        assert price == Decimal(tokens) * Decimal("0.000006") + Decimal(1024) * Decimal("0.00003")

    @pytest.mark.asyncio
    async def test_listed_cache_write_of_the_tier_is_used(self, provider):
        sonnet = {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_write": "0.00000375",
                "overrides": [
                    {
                        "min_prompt_tokens": 200_000,
                        "prompt": "0.000006",
                        "completion": "0.0000225",
                        "input_cache_write": "0.0000075",
                    }
                ],
            },
        }
        content = [{"type": "text", "text": self.LONG_TEXT, "cache_control": {"type": "ephemeral"}}]
        body = {
            "model": "anthropic/claude-sonnet-4.5",
            "messages": [{"role": "user", "content": content}],
        }
        price = await self._price(provider, body, sonnet)
        tokens = estimate_chat_input_tokens(body, "openrouter")
        assert price == Decimal(tokens) * Decimal("0.0000075") + Decimal(1024) * Decimal(
            "0.0000225"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "condition",
        [
            {"utc_days": ["monday"], "utc_start": 100, "utc_end": 400},
            {"utc_start": 0, "utc_end": 1600},
            {"region": "eu"},  # a condition this code does not know
            {"min_prompt_tokens": "1000"},  # a threshold it cannot read
        ],
    )
    async def test_other_overrides_may_apply_so_the_higher_rate_counts(self, provider, condition):
        entry = {
            "id": "deepseek/deepseek-v4.1-flash",
            "pricing": {
                "prompt": "0.0000003",
                "completion": "0.0000012",
                "overrides": [
                    {**condition, "prompt": "0.0000006", "completion": "0.0000024"},
                    {"utc_days": ["sunday"], "prompt": "0.0000001", "completion": "0.0000004"},
                ],
            },
        }
        body = self._body("deepseek/deepseek-v4.1-flash", "x" * 2700)
        price = await self._price(provider, body, entry)
        assert price == Decimal(1000) * Decimal("0.0000006") + Decimal(1024) * Decimal("0.0000024")

    @pytest.mark.asyncio
    async def test_lower_time_window_keeps_the_base_rate(self, provider):
        entry = {
            "id": "tencent/hy3",
            "pricing": {
                "prompt": "0.000000132",
                "completion": "0.000000528",
                "overrides": [
                    {"utc_start": 1600, "utc_end": 0, "prompt": "0.00000008", "completion": "3e-7"}
                ],
            },
        }
        price = await self._price(provider, self._body("tencent/hy3", "x" * 2700), entry)
        assert price == Decimal(1000) * Decimal("0.000000132") + Decimal(1024) * Decimal(
            "0.000000528"
        )

    @pytest.mark.asyncio
    async def test_tier_is_applied_per_candidate_before_the_max(self, provider):
        flat = {"id": "openai/gpt-4.1", "pricing": {"prompt": "0.000002", "completion": "0.000008"}}
        gemini = {
            "id": "google/gemini-2.5-pro",
            "pricing": {
                "prompt": "0.0000015",
                "completion": "1e-6",
                "overrides": [
                    {"min_prompt_tokens": 200_000, "prompt": "0.000003", "completion": "1.5e-6"}
                ],
            },
        }
        body = self._body("openai/gpt-4.1", self.LONG_TEXT, models=["google/gemini-2.5-pro"])
        price = await self._price(provider, body, flat, gemini)
        tokens = estimate_chat_input_tokens(body, "openrouter")
        # Gemini's tier input (3e-6) tops GPT-4.1's flat 2e-6; GPT-4.1's flat
        # output 8e-6 tops Gemini's tier 1.5e-6.
        assert price == Decimal(tokens) * Decimal("0.000003") + Decimal(1024) * Decimal("0.000008")

    @pytest.mark.asyncio
    async def test_cap_keeps_the_listed_rates_of_a_tiered_prompt(self, provider):
        body = self._body("openai/gpt-5.5", self.LONG_TEXT)
        await self._price(provider, body, self.TIERED)
        assert dict(body["_priced_rates"]) == {
            "prompt": Decimal("0.000005"),
            "completion": Decimal("0.00003"),
            "request": Decimal(0),
        }

    @pytest.mark.asyncio
    async def test_fallback_cost_uses_the_tier_of_the_real_prompt(self, provider):
        usage = {"prompt_tokens": 280_000, "completion_tokens": 100}
        cost = await self._fallback_cost(
            provider, self._body("openai/gpt-5.5", "x"), usage, self.TIERED
        )
        assert cost == Decimal(280_000) * Decimal("0.00001") + Decimal(100) * Decimal("0.000045")

    @pytest.mark.asyncio
    async def test_fallback_cost_tiers_only_the_model_that_reaches_the_tier(self, provider):
        flat = {"id": "openai/gpt-4.1", "pricing": {"prompt": "0.000002", "completion": "0.000008"}}
        gemini = {
            "id": "google/gemini-2.5-pro",
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000001",
                "overrides": [
                    {"min_prompt_tokens": 200_000, "prompt": "0.000002", "completion": "0.0000015"}
                ],
            },
        }
        body = self._body("openai/gpt-4.1", "x", models=["google/gemini-2.5-pro"])
        usage = {"prompt_tokens": 250_000, "completion_tokens": 100}
        cost = await self._fallback_cost(provider, body, usage, flat, gemini)
        assert cost == Decimal(250_000) * Decimal("0.000002") + Decimal(100) * Decimal("0.000008")

    @pytest.mark.asyncio
    async def test_embeddings_use_the_override_rates(self, provider):
        entry = {
            "id": "qwen/qwen3-embedding",
            "pricing": {
                "prompt": "0.00000001",
                "completion": "0",
                "overrides": [{"utc_start": 0, "utc_end": 800, "prompt": "0.00000002"}],
            },
        }
        body = {"model": "qwen/qwen3-embedding", "input": "x" * 27_000}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(entry)
            price = await provider.get_price("embeddings", body)
        assert price == Decimal(10_000) * Decimal("0.00000002")


class TestNativeSearchOnlyForSearchModels:
    """Only models that search on every request are priced with native searches."""

    MAINSTREAM = {
        "id": "openai/gpt-5.4-nano",
        "pricing": {"prompt": "0.0000002", "completion": "0.00000125", "web_search": "0.01"},
    }
    SONAR = {
        "id": "perplexity/sonar",
        "pricing": {"prompt": "0.000001", "completion": "0.000001", "web_search": "0.005"},
    }

    async def _price(self, provider, body: dict, *models: dict) -> Decimal:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(*models)
            return await provider.get_price("chat/completions", body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        ["openai/gpt-5.4-nano", "anthropic/claude-haiku-5.5", "~anthropic/claude-opus-latest"],
    )
    async def test_per_search_rate_of_a_chat_model_is_not_charged(self, provider, model):
        entry = {**self.MAINSTREAM, "id": model}
        price = await self._price(provider, _chat_body(model=model), entry)
        assert price == Decimal(1) * Decimal("0.0000002") + Decimal(1024) * Decimal("0.00000125")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model", ["perplexity/sonar", "~perplexity/sonar-latest", "openai/gpt-4o-search-preview"]
    )
    async def test_models_that_always_search_are_charged(self, provider, model):
        entry = {**self.SONAR, "id": model}
        price = await self._price(provider, _chat_body(model=model), entry)
        searches = openrouter._NATIVE_SEARCHES * Decimal("0.005")
        assert price == Decimal("0.000001") + Decimal(1024) * Decimal("0.000001") + searches

    @pytest.mark.asyncio
    async def test_fallback_to_a_search_model_keeps_its_searches(self, provider):
        body = _chat_body(model="openai/gpt-5.4-nano", models=["perplexity/sonar"])
        price = await self._price(provider, body, self.MAINSTREAM, self.SONAR)
        tokens = estimate_chat_input_tokens(body, "openrouter")
        searches = openrouter._NATIVE_SEARCHES * Decimal("0.005")
        # Sonar's input and searches, GPT's output.
        assert price == (
            Decimal(tokens) * Decimal("0.000001") + Decimal(1024) * Decimal("0.00000125") + searches
        )

    @pytest.mark.asyncio
    async def test_fallback_cost_leaves_out_searches_of_a_chat_model(self, provider):
        usage = {"prompt_tokens": 10, "completion_tokens": 100, "is_byok": True}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.MAINSTREAM)
            cost = await provider.calculate_actual_cost(
                _chat_body(model="openai/gpt-5.4-nano"), {"usage": usage}
            )
        assert cost == Decimal(10) * Decimal("0.0000002") + Decimal(100) * Decimal("0.00000125")

    @pytest.mark.asyncio
    async def test_fallback_cost_counts_the_searches_of_each_completion(self, provider):
        usage = {"prompt_tokens": 10, "completion_tokens": 100, "is_byok": True}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(self.SONAR)
            cost = await provider.calculate_actual_cost(
                _chat_body(model="perplexity/sonar", n=2), {"usage": usage}
            )
        searches = 2 * openrouter._NATIVE_SEARCHES * Decimal("0.005")
        assert cost == Decimal(110) * Decimal("0.000001") + searches


class TestEmbeddingEstimate:
    """Embeddings input is weighed like chat text, so dense text is not under-quoted."""

    RATE = Decimal("0.00000002")

    async def _price(self, provider, embedding_model_info_response, text_input) -> Decimal:
        body = {"model": "openai/text-embedding-3-small", "input": text_input}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = embedding_model_info_response
            return await provider.get_price("embeddings", body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text_input", "tokens"),
        [
            ("汉" * 4000, 6000),  # common CJK: 1.5 tokens each
            ("\U00020000" * 4000, 12_000),  # CJK Extension B: 3 tokens each
            (["\U0001f600" * 1000, "汉" * 1000], 4500),  # emoji 3, CJK 1.5
            ("7" * 5000, 5000),  # one token per digit
            ("x" * 27_000, 10_000),  # ASCII at 2.7 characters per token
            ([[1, 2, 3], [4, 5]], 5),  # token arrays
            ([101, 102, 103, 104], 4),  # one token array as ints
        ],
    )
    async def test_dense_input_is_not_under_quoted(
        self, provider, embedding_model_info_response, text_input, tokens
    ):
        price = await self._price(provider, embedding_model_info_response, text_input)
        assert price == max(Decimal(tokens) * self.RATE, Decimal("0.000001"))


def _nested(depth: int, leaf=None) -> list:
    value = [] if leaf is None else leaf
    for _ in range(depth):
        value = [value]
    return value


class TestMalformedBodies:
    """Bodies json.loads accepts but that cannot be priced or sent get a 400, not a 500."""

    async def test_deep_nesting_is_priced_without_recursion(self, provider, model_info_response):
        # Deeper than the default recursion limit allows a recursive walk.
        body = _chat_body(foo=_nested(5000))
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price > 0

    def test_nested_key_walks_are_iterative(self):
        deep = _nested(10_000, {"cache_control": {"type": "ephemeral", "ttl": "1h"}})
        assert openrouter.has_nested_key(deep, "cache_control")
        assert openrouter.long_cache_ttl(deep)
        assert not openrouter.has_nested_key(_nested(10_000), "cache_control")

    async def test_nesting_too_deep_to_serialize_is_a_400(self, provider, model_info_response):
        body = _chat_body(foo=_nested(100_000))
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_price("chat/completions", body)
        assert exc_info.value.status_code == 400
        assert "nested too deeply" in exc_info.value.detail

    async def test_deep_embedding_input_is_a_400(self, provider, embedding_model_info_response):
        body = {"model": "openai/text-embedding-3-small", "input": [_nested(100_000, {})]}
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = embedding_model_info_response
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_price("embeddings", body)
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize("part_type", [["text"], {}, {"a": 1}, 7])
    async def test_non_string_part_type_is_a_400(self, provider, part_type):
        body = _chat_body(
            messages=[{"role": "user", "content": [{"type": part_type, "text": "hi"}]}]
        )
        for call in (provider.get_price, provider.submit):
            with pytest.raises(ProviderError) as exc_info:
                await call("chat/completions", body)
            assert exc_info.value.status_code == 400
            assert "must be a string" in exc_info.value.detail

    @pytest.mark.parametrize(
        "extra",
        [
            {"temperature": float("nan")},
            {"top_p": float("inf")},
            {"tools": [{"type": "function", "x": float("-inf")}]},
            {"messages": [{"role": "user", "content": "a\ud800b"}]},
            {"messages": [{"role": "user", "content": "hi", "name\udfff": "x"}]},
        ],
    )
    async def test_unsendable_json_is_refused_before_payment(self, provider, extra):
        """httpx would refuse to encode these on every send, after the client paid."""
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as exc_info,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert exc_info.value.status_code == 400
        mock_get.assert_not_called()

    async def test_unsendable_embedding_input_is_refused(self, provider):
        body = {"model": "openai/text-embedding-3-small", "input": "x\udc00"}
        with pytest.raises(ProviderError) as exc_info:
            await provider.get_price("embeddings", body)
        assert exc_info.value.status_code == 400

    async def test_surrogate_pairs_and_internal_keys_are_fine(self, provider, model_info_response):
        body = _chat_body(
            messages=[{"role": "user", "content": "smile \U0001f600 привет"}],
            _file_bytes=b"\xff",
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            assert await provider.get_price("chat/completions", body) > 0


class TestErrorInSuccessBody:
    """A 2xx body with only an error object is a failure, not a billed success."""

    @pytest.fixture(autouse=True)
    def warm_catalog(self, provider, model_info_response):
        provider._catalogs.put(
            "", {model["id"]: model for model in model_info_response.json()["data"]}
        )

    async def _submit(self, provider, payload: dict):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(status_code=200, json=payload)
            return await provider.submit("chat/completions", _chat_body())

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            ({"code": 502, "message": "upstream failed"}, 502),
            ({"code": 429, "message": "rate limited"}, 429),
            ({"code": 400, "message": "context too long"}, 400),
            ({"code": 402, "message": "insufficient credits"}, 502),
            ({"code": "server_error", "message": "boom"}, 502),
            ({"message": "boom"}, 502),
            ("model failed", 502),
        ],
    )
    async def test_reported_error_is_a_known_failure(self, provider, error, status):
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": error})
        assert exc_info.value.status_code == status
        assert not outcome_unknown(exc_info.value)

    @pytest.mark.parametrize(
        "error",
        [
            {"code": 524, "message": "provider timeout"},
            {"code": 504, "message": "gateway"},
            {"code": 502, "message": "Upstream timed out"},
            {"message": "request timed out"},
        ],
    )
    async def test_reported_timeout_is_an_unknown_outcome(self, provider, error):
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": error})
        assert outcome_unknown(exc_info.value)

    @pytest.mark.parametrize(
        "usage",
        [
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0},
            {"cost": 0.0},
            {"cost": "0", "total_tokens": "0"},
            {"cost": "abc", "prompt_tokens": True, "completion_tokens": [5], "total_tokens": None},
            {"cost": "NaN", "prompt_tokens": "Infinity", "completion_tokens": -3},
        ],
    )
    async def test_error_with_zeroed_usage_is_a_known_failure(self, provider, usage):
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": {"code": 502, "message": "x"}, "usage": usage})
        assert exc_info.value.status_code == 502
        assert not outcome_unknown(exc_info.value)

    async def test_timeout_with_zeroed_usage_is_an_unknown_outcome(self, provider):
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}
        with pytest.raises(ProviderError) as exc_info:
            await self._submit(provider, {"error": {"code": 524}, "usage": usage})
        assert outcome_unknown(exc_info.value)

    @pytest.mark.parametrize(
        "payload",
        [
            {"error": {"code": 502}, "usage": {"cost": 0.0001}},
            {"error": {"code": 502}, "usage": {"cost": "0.0001"}},
            {"error": {"code": 502}, "usage": {"prompt_tokens": 10}},
            {"error": {"code": 502}, "usage": {"cost": 0, "prompt_tokens": 10, "is_byok": True}},
            {"error": {"code": 502}, "usage": {"total_tokens": 12}},
            {"error": {"code": 502}, "choices": [{"message": {"content": "partial"}}]},
            {"error": None, "choices": []},
            {"error": {}, "id": "gen-1"},
        ],
    )
    async def test_billed_or_answered_bodies_are_returned(self, provider, payload):
        assert await self._submit(provider, payload) == payload


class TestCapAtPricedRates:
    """submit caps steered endpoints at the rates get_price priced the request at."""

    _ROUTING = {"order": ["pricey-host"]}

    @staticmethod
    def _mini(prompt: str, completion: str) -> dict:
        return {
            "openai/gpt-4o-mini": {
                "id": "openai/gpt-4o-mini",
                "pricing": {"prompt": prompt, "completion": completion},
            }
        }

    async def _cap(self, provider, chat_completion_response, body) -> dict:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = chat_completion_response
            await provider.submit("chat/completions", body)
        sent = mock_post.call_args.kwargs["json"]
        assert openrouter._PRICED_RATES not in sent
        return sent["provider"]["max_price"]

    async def test_request_priced_after_a_raise_is_capped_at_the_new_rates(
        self, provider, chat_completion_response
    ):
        provider._catalogs.put("", self._mini("0.000001", "0.000002"))
        provider._catalogs.put("", self._mini("0.000002", "0.000004"))
        body = _chat_body(provider=self._ROUTING)
        await provider.get_price("chat/completions", body)
        cap = await self._cap(provider, chat_completion_response, body)
        assert cap == {"prompt": 2.0, "completion": 4.0, "request": 0.0}

    async def test_request_priced_before_a_raise_keeps_its_rates(
        self, provider, chat_completion_response
    ):
        provider._catalogs.put("", self._mini("0.000001", "0.000002"))
        body = _chat_body(provider=self._ROUTING)
        await provider.get_price("chat/completions", body)
        provider._catalogs.put("", self._mini("0.000002", "0.000004"))
        # Even once the grace window for unpriced bodies is over.
        old, replaced_at = provider._catalogs._previous[""]
        provider._catalogs._previous[""] = (old, replaced_at - openrouter._CAP_GRACE_S - 1)
        cap = await self._cap(provider, chat_completion_response, body)
        assert cap == {"prompt": 1.0, "completion": 2.0, "request": 0.0}

    async def test_client_cannot_forge_the_priced_rates(self, provider, chat_completion_response):
        provider._catalogs.put("", self._mini("0.000001", "0.000002"))
        forged = {"prompt": Decimal(1), "completion": Decimal(1), "request": Decimal(1)}
        body = _chat_body(provider=self._ROUTING, **{openrouter._PRICED_RATES: forged})
        cap = await self._cap(provider, chat_completion_response, body)
        assert cap == {"prompt": 1.0, "completion": 2.0, "request": 0.0}


def _audio_part(data: str, fmt: str = "wav") -> dict:
    return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def _audio_body(*parts: dict, model: str = "google/gemini-audio", **extra) -> dict:
    content = [{"type": "text", "text": "Hi"}, *parts]
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


# Text weight of the "Hi" part plus one audio part's type and format.
_AUDIO_TEXT_UNITS = len("Hi") * 100
_AUDIO_PART_UNITS = (len('{"type":"input_audio"}') + len('{"format":"wav"}')) * 100


def _audio_text_tokens(parts: int = 1) -> int:
    return (_AUDIO_TEXT_UNITS + parts * _AUDIO_PART_UNITS) // 270


class TestChatAudioInput:
    """Audio parts are timed and billed at the catalog audio rate."""

    PROMPT = Decimal("0.0000003")
    COMPLETION = Decimal("0.0000025")
    AUDIO = Decimal("0.000001")

    def _model(self, **pricing: str) -> dict:
        rates = {"prompt": str(self.PROMPT), "completion": str(self.COMPLETION)}
        rates.update(pricing)
        return {"id": "google/gemini-audio", "pricing": rates}

    async def _price(self, provider, body: dict, model: dict) -> Decimal:
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(model)
            return await provider.get_price("chat/completions", body)

    def _expected(self, audio_tokens: int, audio_rate: Decimal, parts: int = 1) -> Decimal:
        return (
            Decimal(_audio_text_tokens(parts)) * self.PROMPT
            + Decimal(audio_tokens) * audio_rate
            + Decimal(1024) * self.COMPLETION
        )

    @pytest.mark.asyncio
    async def test_audio_is_priced_by_duration_at_the_audio_rate(self, provider):
        threads = []
        real_duration = openrouter.get_audio_duration_seconds

        def recording_duration(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_duration(*args, **kwargs)

        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        with patch.object(openrouter, "get_audio_duration_seconds", recording_duration):
            price = await self._price(provider, body, self._model(audio=str(self.AUDIO)))

        assert price == self._expected(2 * 50, self.AUDIO)
        assert len(threads) == 1
        assert threads[0] is not threading.main_thread()

    @pytest.mark.asyncio
    async def test_audio_rate_below_the_input_rate_uses_the_input_rate(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        price = await self._price(provider, body, self._model(audio="0.0000001"))
        assert price == self._expected(100, self.PROMPT)

    @pytest.mark.asyncio
    async def test_no_audio_rate_bills_audio_as_input_tokens(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        price = await self._price(provider, body, self._model())
        assert price == self._expected(100, self.PROMPT)

    @pytest.mark.asyncio
    async def test_partial_seconds_round_up_per_part_and_parts_add_up(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(1.2)), _audio_part(make_wav_b64(0.5)))
        price = await self._price(provider, body, self._model(audio=str(self.AUDIO)))
        assert price == self._expected((2 + 1) * 50, self.AUDIO, parts=2)

    @pytest.mark.asyncio
    async def test_a_large_wav_is_not_priced_as_text(self, provider):
        """A minute of WAV is ~1.3M base64 characters, ~470k text tokens; it bills 3k."""
        body = _audio_body(_audio_part(make_wav_b64(60.0, rate=16000)))
        price = await self._price(provider, body, self._model(audio=str(self.AUDIO)))
        assert price == self._expected(60 * 50, self.AUDIO)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("fmt", "per_second"), [("pcm16", 16_000), ("PCM24", 24_000)])
    async def test_raw_pcm_is_timed_from_its_length_at_8khz(self, provider, fmt, per_second):
        data = base64.b64encode(b"\x00" * (3 * per_second)).decode()
        body = _audio_body(
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
        )
        price = await self._price(provider, body, self._model(audio=str(self.AUDIO)))
        # The data is not weighed, so the text estimate stays tiny.
        text_tokens = estimate_chat_input_tokens(body, "openrouter")
        assert text_tokens < 20
        assert price == (
            Decimal(text_tokens) * self.PROMPT
            + Decimal(3 * 50) * self.AUDIO
            + Decimal(1024) * self.COMPLETION
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("audio", "status"),
        [
            ({"data": "QUJD0123" * 100, "format": "wav"}, 400),  # not audio
            ({"data": "not base64!", "format": "wav"}, 400),
            ({"data": make_wav_b64(1.0)}, 400),  # no format
            ({"format": "wav"}, 400),  # no data
            ("AAAA", 400),  # not an object
        ],
    )
    async def test_audio_that_cannot_be_timed_is_refused(self, provider, audio, status):
        body = _audio_body({"type": "input_audio", "input_audio": audio})
        with pytest.raises(ProviderError) as caught:
            await self._price(provider, body, self._model(audio=str(self.AUDIO)))
        assert caught.value.status_code == status

    @pytest.mark.asyncio
    async def test_oversized_audio_part_is_refused(self):
        config = ProviderConfig(
            enabled=True,
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            max_stt_audio_mb=1,
        )
        small = OpenRouterProvider(config=config)
        body = _audio_body(_audio_part(make_wav_b64(70.0)))  # ~1.1 MB
        with pytest.raises(ProviderError) as caught:
            await self._price(small, body, self._model(audio=str(self.AUDIO)))
        assert caught.value.status_code == 413

    @pytest.mark.asyncio
    async def test_audio_tokens_count_toward_the_long_context_tier(self, provider):
        """100 audio tokens lift a 14-token prompt over a 50-token tier."""
        model = self._model(
            audio=str(self.AUDIO),
            overrides=[{"min_prompt_tokens": 50, "prompt": "0.0000006"}],
        )
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        price = await self._price(provider, body, model)
        # The tier doubles the prompt rate and lists no audio rate, so audio
        # doubles with it.
        assert price == (
            Decimal(_audio_text_tokens()) * Decimal("0.0000006")
            + Decimal(100) * self.AUDIO * 2
            + Decimal(1024) * self.COMPLETION
        )

    @pytest.mark.asyncio
    async def test_tier_audio_rate_is_used_when_listed(self, provider):
        model = self._model(
            audio=str(self.AUDIO),
            overrides=[{"min_prompt_tokens": 50, "prompt": "0.0000006", "audio": "0.000005"}],
        )
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        price = await self._price(provider, body, model)
        assert price == (
            Decimal(_audio_text_tokens()) * Decimal("0.0000006")
            + Decimal(100) * Decimal("0.000005")
            + Decimal(1024) * self.COMPLETION
        )

    @pytest.mark.asyncio
    async def test_fallback_candidate_audio_rate_counts(self, provider):
        cheap = self._model()
        dear = {
            "id": "openai/gpt-audio",
            "pricing": {"prompt": "0.0000003", "completion": "0.0000025", "audio": "0.00004"},
        }
        body = _audio_body(_audio_part(make_wav_b64(2.0)), models=["openai/gpt-audio"])
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(cheap, dear)
            price = await provider.get_price("chat/completions", body)
        models_units = len('["openai/gpt-audio"]') * 100
        text_tokens = (_AUDIO_TEXT_UNITS + _AUDIO_PART_UNITS + models_units) // 270
        assert price == (
            Decimal(text_tokens) * self.PROMPT
            + Decimal(100) * Decimal("0.00004")
            + Decimal(1024) * self.COMPLETION
        )

    async def _fallback_cost(self, provider, body: dict, usage: dict, model: dict):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _catalog(model)
            return await provider.calculate_actual_cost(body, {"usage": {**usage, "is_byok": True}})

    @pytest.mark.asyncio
    async def test_fallback_cost_bills_reported_audio_tokens_at_the_audio_rate(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 10,
            "prompt_tokens_details": {"audio_tokens": 64},
        }
        cost = await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        assert cost == (
            Decimal(56) * self.PROMPT + Decimal(64) * self.AUDIO + Decimal(10) * self.COMPLETION
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("details", [None, {}, {"audio_tokens": 0}, {"audio_tokens": "64"}])
    async def test_fallback_cost_without_an_audio_count_bills_the_held_audio_tokens_as_audio(
        self, provider, details
    ):
        """Two timed seconds held 100 audio tokens; the rest bills at the prompt rate."""
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {"prompt_tokens": 120, "completion_tokens": 10}
        if details is not None:
            usage["prompt_tokens_details"] = details
        cost = await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        assert cost == (
            Decimal(100) * self.AUDIO + Decimal(20) * self.PROMPT + Decimal(10) * self.COMPLETION
        )

    @pytest.mark.asyncio
    async def test_fallback_cost_without_an_audio_count_bills_a_short_prompt_as_audio(
        self, provider
    ):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {"prompt_tokens": 80, "completion_tokens": 10}
        cost = await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        assert cost == Decimal(80) * self.AUDIO + Decimal(10) * self.COMPLETION

    @pytest.mark.asyncio
    async def test_fallback_cost_bills_the_prompt_as_audio_when_the_audio_cannot_be_timed(
        self, provider
    ):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {"prompt_tokens": 120, "completion_tokens": 10}
        failing = AsyncMock(
            side_effect=ProviderError(provider="openrouter", detail="busy", status_code=429)
        )
        with patch.object(provider, "_measure_chat_audio", failing):
            cost = await self._fallback_cost(
                provider, body, usage, self._model(audio=str(self.AUDIO))
            )
        assert cost == Decimal(120) * self.AUDIO + Decimal(10) * self.COMPLETION

    @pytest.mark.asyncio
    async def test_fallback_cost_does_not_time_audio_the_usage_reports(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {"prompt_tokens": 120, "prompt_tokens_details": {"audio_tokens": 64}}
        measure = AsyncMock(return_value=2)
        with patch.object(provider, "_measure_chat_audio", measure):
            await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        measure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_cost_caps_audio_tokens_at_the_prompt(self, provider):
        body = _audio_body(_audio_part(make_wav_b64(2.0)))
        usage = {"prompt_tokens": 50, "prompt_tokens_details": {"audio_tokens": 500}}
        cost = await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        assert cost == Decimal(50) * self.AUDIO

    @pytest.mark.asyncio
    async def test_fallback_cost_of_text_ignores_the_audio_rate(self, provider):
        body = _chat_body(model="google/gemini-audio")
        usage = {"prompt_tokens": 120, "prompt_tokens_details": {"audio_tokens": 0}}
        cost = await self._fallback_cost(provider, body, usage, self._model(audio=str(self.AUDIO)))
        assert cost == Decimal(120) * self.PROMPT


class TestWebSearchRoutes:
    """Every way to ask for a billed web search is priced or refused."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tools",
        [
            [{"type": "web_search"}],
            [{"type": "web_search_preview", "search_context_size": "high"}],
            [{"type": "openrouter:web_search", "parameters": {"max_results": 5}}],
            [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            [{"type": "function", "function": {"name": "f"}}, {"type": "code_interpreter"}],
            [{"type": ["function"]}],
            [{"type": None}],
            [{"google_search": {}}],
            [{"googleSearchRetrieval": {}}],
            ["web_search"],
            {"type": "web_search"},
        ],
    )
    async def test_server_tools_are_refused_before_payment(self, provider, tools):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(tools=tools))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tools",
        [
            [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            [{"function": {"name": "f"}}],  # no type: a function tool
            [{"type": "custom", "custom": {"name": "grammar"}}],
            [],
            None,
        ],
    )
    async def test_function_tools_are_priced_as_text(self, provider, model_info_response, tools):
        body = _chat_body(tools=tools)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(estimate_chat_input_tokens(body, "openrouter"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "search",
        [
            {},
            {"mode": "auto"},
            {"mode": "on", "max_search_results": 30},
            {"mode": "OFF "},
            "off",
            [],
        ],
    )
    async def test_xai_live_search_is_refused(self, provider, search):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match="search_parameters") as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(search_parameters=search))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_xai_live_search_off_is_allowed(self, provider, model_info_response):
        body = _chat_body(search_parameters={"mode": "off"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            price = await provider.get_price("chat/completions", body)
        assert price == _mini_price(estimate_chat_input_tokens(body, "openrouter"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"model": "openai/gpt-4o-mini:online"},
            {"model": "openai/gpt-4o-mini:Online"},
            {"model": "openai/gpt-4o-mini:online:nitro"},
            {"models": ["openai/gpt-4o:online"]},
        ],
    )
    async def test_online_variant_is_refused_before_payment(self, provider, extra):
        with (
            patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
            pytest.raises(ProviderError, match=":online") as caught,
        ):
            await provider.get_price("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_online_variant_in_the_catalog_is_still_refused(self, provider):
        """Even a listed :online row would price no searches."""
        catalog = _catalog(
            {"id": "openai/gpt-4o-mini:online", "pricing": {"prompt": "0", "completion": "0"}}
        )
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            with pytest.raises(ProviderError, match=":online"):
                await provider.get_price(
                    "chat/completions", _chat_body(model="openai/gpt-4o-mini:online")
                )

    @pytest.mark.asyncio
    async def test_online_in_a_model_name_is_not_a_variant(self, provider):
        catalog = _catalog(
            {"id": "acme/online-chat", "pricing": {"prompt": "0.000001", "completion": "0"}}
        )
        body = _chat_body(model="acme/online-chat")
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = catalog
            price = await provider.get_price("chat/completions", body)
        assert price == Decimal(_ascii_tokens("Hi")) * Decimal("0.000001")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "extra",
        [
            {"tools": [{"type": "web_search_preview"}]},
            {"search_parameters": {"mode": "auto"}},
            {"model": "openai/gpt-4o-mini:online"},
        ],
    )
    async def test_submit_refuses_them_too(self, provider, extra):
        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("chat/completions", _chat_body(**extra))
        assert caught.value.status_code == 400
        mock_post.assert_not_called()
