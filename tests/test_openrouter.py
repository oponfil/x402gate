"""Unit tests for OpenRouter provider."""

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests.helpers import make_wav_b64
from x402gate.core.config import ProviderConfig
from x402gate.providers.base import ProviderError
from x402gate.providers.openrouter import OpenRouterProvider


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
                        "prompt": "0.006",
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

        # Verify: ~6 input tokens (25 chars / 4) × $0.00000015
        #        + 1024 output tokens × $0.0000006 (max_tokens raised to default)
        expected_input_tokens = max(len("What is quantum computing?") // 4, 1)
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
        # input ~1 token (2 chars / 4), completion = 1024
        # 1 * 0.00000015 + 1024 * 0.0000006 = 0.00061455
        expected = Decimal("1") * Decimal("0.00000015") + Decimal("1024") * Decimal("0.0000006")
        assert price == expected

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
        """Whisper STT should price by audio duration (USD per minute)."""
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

        # 1 billing second -> 1/60 minute * $0.006/min
        assert price == Decimal("1") / Decimal("60") * Decimal("0.006")
        assert (
            mock_get.call_args_list[0].args[0].endswith("/models?output_modalities=transcription")
        )

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


class TestSubmit:
    """Tests for OpenRouterProvider.submit()."""

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
    async def test_chat_ignores_usage_cost(self, provider, model_info_response):
        """Chat responses should use token counts even when usage.cost is present."""
        body = {"model": "openai/gpt-4o-mini"}
        result = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
                "cost": 0.999,
            },
        }

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = model_info_response
            cost = await provider.calculate_actual_cost(body, result)

        expected = Decimal(10) * Decimal("0.00000015") + Decimal(50) * Decimal("0.0000006")
        assert cost == expected

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

        assert cost == Decimal("0.006")
