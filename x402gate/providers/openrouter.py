"""OpenRouter provider for x402gate.

Implements the BaseProvider interface for OpenRouter's API:
- Pricing via GET /api/v1/models (chat, embedding, and transcription models)
- Chat completion via POST /api/v1/chat/completions (synchronous)
- Embeddings via POST /api/v1/embeddings (synchronous)
- Speech-to-text via POST /api/v1/audio/transcriptions (synchronous)
- No polling needed (responses are synchronous)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.audio_duration import (
    billing_seconds,
    get_audio_duration_seconds,
)
from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4

# OpenRouter STT: Whisper models use pricing.prompt as USD per minute of audio.
_DURATION_STT_PROMPT_THRESHOLD = Decimal("0.0001")


def _is_stt_request(path: str, inputs: dict[str, Any]) -> bool:
    return "transcription" in path.lower() or "input_audio" in inputs


def _is_duration_based_stt(pricing: dict[str, Any]) -> bool:
    prompt = Decimal(str(pricing.get("prompt", "0")))
    completion = Decimal(str(pricing.get("completion", "0")))
    return prompt >= _DURATION_STT_PROMPT_THRESHOLD and completion == 0


def _extract_stt_audio(
    inputs: dict[str, Any],
    provider_name: str,
    *,
    max_audio_bytes: int,
) -> tuple[bytes, str]:
    input_audio = inputs.get("input_audio")
    if not isinstance(input_audio, dict):
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio object",
            status_code=400,
        )

    data_b64 = input_audio.get("data", "")
    fmt = input_audio.get("format", "")
    if not data_b64:
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio.data (base64-encoded audio)",
            status_code=400,
        )
    if not fmt:
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio.format",
            status_code=400,
        )

    try:
        audio_bytes = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ProviderError(
            provider=provider_name,
            detail="Invalid base64 in input_audio.data",
            status_code=400,
        ) from exc

    if len(audio_bytes) > max_audio_bytes:
        max_mb = max_audio_bytes // 1024 // 1024
        raise ProviderError(
            provider=provider_name,
            detail=(
                f"Audio file too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum for OpenRouter STT: {max_mb} MB"
            ),
            status_code=413,
        )

    return audio_bytes, str(fmt)


def _upstream_body(body: dict[str, Any]) -> dict[str, Any]:
    """Strip gateway-internal keys before forwarding to OpenRouter."""
    return {k: v for k, v in body.items() if not k.startswith("_")}


def _apply_cost_floor(cost: Decimal) -> Decimal:
    if cost < Decimal("0.000001"):
        return Decimal("0.000001")
    return cost


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation.

    OpenRouter is an LLM aggregator with 300+ chat, embedding, and STT models
    and an OpenAI-compatible API.  Supports chat completions, embeddings, and
    speech-to-text transcriptions.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        default_max_tokens: int = 1024,
        web_search_tokens_per_result: int = 2048,
        default_web_search_max_results: int = 3,
        web_search_cost_per_result: float = 0.004,
    ) -> None:
        super().__init__(name="openrouter", config=config)
        self._models_cache: dict[str, dict[str, Any]] = {}
        self._fetched_catalogs: set[str] = set()
        self._cache_updated_at: float = 0.0
        self._cache_ttl: float = float(config.models_cache_ttl)
        self._default_max_tokens = default_max_tokens
        self._web_search_tokens_per_result = web_search_tokens_per_result
        self._default_web_search_max_results = default_web_search_max_results
        self._web_search_cost_per_result = Decimal(str(web_search_cost_per_result))
        self._max_stt_audio_bytes = config.max_stt_audio_mb * 1024 * 1024

    async def _fetch_model_info(
        self,
        model_id: str,
        *,
        output_modalities: str = "",
    ) -> dict[str, Any]:
        """Fetch model info (including pricing) from OpenRouter Models API.

        Fetches only the catalog needed by the request type, then adds it to
        the unified cache.

        Args:
            model_id: Model identifier (e.g. 'openai/gpt-4o-mini').

        Returns:
            Model info dict with 'pricing' sub-dict.

        Raises:
            ProviderError: If the model is not found or the API call fails.
        """
        current_time = time.time()

        if self._models_cache and (current_time - self._cache_updated_at > self._cache_ttl):
            logger.info("OpenRouter models cache expired (1 day TTL), clearing...")
            self._models_cache.clear()
            self._fetched_catalogs.clear()

        if output_modalities not in self._fetched_catalogs:
            await self._fetch_models_list(output_modalities=output_modalities)
            self._fetched_catalogs.add(output_modalities)
            self._cache_updated_at = time.time()

        if model_id in self._models_cache:
            return self._models_cache[model_id]

        raise ProviderError(
            provider=self.name,
            detail=f"Model '{model_id}' not found on OpenRouter",
            status_code=404,
        )

    async def _fetch_models_list(self, output_modalities: str = "") -> None:
        """Fetch models from OpenRouter and add to cache."""
        url = f"{self._config.base_url.rstrip('/')}/models"
        if output_modalities:
            url += f"?output_modalities={output_modalities}"

        max_retries = 2
        retry_delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url)

                if resp.status_code >= 500 and attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API returned %d (attempt %d/%d), retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    raise ProviderError(
                        provider=self.name,
                        detail=f"OpenRouter Models API error: {resp.text}",
                        status_code=resp.status_code,
                    )

                models_list = resp.json().get("data", [])
                count = 0
                for model in models_list:
                    mid = model.get("id", "")
                    if mid:
                        self._models_cache[mid] = model
                        count += 1
                label = f" ({output_modalities})" if output_modalities else ""
                logger.info(
                    "Cached %d models from OpenRouter%s (total: %d)",
                    count,
                    label,
                    len(self._models_cache),
                )
                return

            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API error: %s (attempt %d/%d), retrying in %.1fs",
                        e,
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

        raise ProviderError(
            provider=self.name,
            detail=f"Failed to fetch models list after retries: {last_error}",
            status_code=503,
        ) from last_error

    async def _estimate_stt_price(
        self,
        model_id: str,
        inputs: dict[str, Any],
        model_info: dict[str, Any],
        prompt_price_dec: Decimal,
    ) -> Decimal:
        audio_bytes, fmt = _extract_stt_audio(
            inputs,
            self.name,
            max_audio_bytes=self._max_stt_audio_bytes,
        )
        duration = get_audio_duration_seconds(audio_bytes, fmt)
        seconds = billing_seconds(duration)
        pricing = model_info.get("pricing", {})

        if _is_duration_based_stt(pricing):
            # pricing.prompt is USD per minute of audio (Whisper family).
            estimated_cost = Decimal(seconds) / Decimal("60") * prompt_price_dec
            logger.info(
                "OpenRouter price estimate for %s: $%s (%d billing seconds, duration-based STT)",
                model_id,
                estimated_cost,
                seconds,
            )
            return _apply_cost_floor(estimated_cost)

        raise ProviderError(
            provider=self.name,
            detail=(
                f"Token-based STT model '{model_id}' is not supported because "
                "x402gate cannot guarantee an upfront maximum cost. Use a "
                "duration-priced Whisper model such as 'openai/whisper-1'."
            ),
            status_code=400,
        )

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate maximum request cost from model pricing and request inputs.

        Supports chat completions, embeddings, and speech-to-text transcriptions.

        Args:
            model_path: API sub-path or model id fallback.
            inputs: Request body.

        Returns:
            Estimated maximum cost in USD as a Decimal.

        Raises:
            ProviderError: If pricing info is unavailable.
        """
        model_id = inputs.get("model", model_path)
        if _is_stt_request(model_path, inputs):
            output_modalities = "transcription"
        elif "input" in inputs and "messages" not in inputs:
            output_modalities = "embeddings"
        else:
            output_modalities = ""
        model_info = await self._fetch_model_info(
            model_id,
            output_modalities=output_modalities,
        )

        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            raise ProviderError(
                provider=self.name,
                detail=f"No pricing info for model '{model_id}'",
                status_code=503,
            )

        prompt_price_dec = Decimal(str(prompt_price))
        completion_price_dec = Decimal(str(completion_price))

        if _is_stt_request(model_path, inputs):
            return await self._estimate_stt_price(
                model_id,
                inputs,
                model_info,
                prompt_price_dec,
            )

        if "input" in inputs and "messages" not in inputs:
            text_input = inputs.get("input", "")
            if isinstance(text_input, list):
                total_chars = sum(len(str(item)) for item in text_input)
            else:
                total_chars = len(str(text_input))
            estimated_input_tokens = max(total_chars // _CHARS_PER_TOKEN, 1)

            estimated_cost = Decimal(estimated_input_tokens) * prompt_price_dec
            logger.info(
                "OpenRouter price estimate for %s: $%s (~%d input tokens, embedding)",
                model_id,
                estimated_cost,
                estimated_input_tokens,
            )
            return _apply_cost_floor(estimated_cost)

        messages = inputs.get("messages", [])
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_input_tokens = max(total_chars // _CHARS_PER_TOKEN, 1)

        web_search_cost = Decimal(0)
        plugins = inputs.get("plugins", [])
        for plugin in plugins:
            if isinstance(plugin, dict) and plugin.get("id") == "web":
                max_results = plugin.get("max_results", self._default_web_search_max_results)
                estimated_input_tokens += max_results * self._web_search_tokens_per_result
                web_search_cost = Decimal(max_results) * self._web_search_cost_per_result

        raw_max = inputs.get("max_tokens")
        max_tokens = max(
            raw_max if isinstance(raw_max, int) else self._default_max_tokens,
            self._default_max_tokens,
        )

        estimated_cost = (
            Decimal(estimated_input_tokens) * prompt_price_dec
            + Decimal(max_tokens) * completion_price_dec
            + web_search_cost
        )

        logger.info(
            "OpenRouter price estimate for %s: $%s (~%d input tokens, %d max output tokens)",
            model_id,
            estimated_cost,
            estimated_input_tokens,
            max_tokens,
        )
        return _apply_cost_floor(estimated_cost)

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a request to OpenRouter (chat, embeddings, or STT).

        Routes to the appropriate endpoint based on path:
        - 'chat/completions' → POST /api/v1/chat/completions
        - 'embeddings' → POST /api/v1/embeddings
        - 'audio/transcriptions' → POST /api/v1/audio/transcriptions

        OpenRouter returns results synchronously (no polling needed).

        Args:
            path: API sub-path.
            body: Request body, forwarded to OpenRouter.
            prepaid: If True, skip max_tokens injection for chat requests.

        Returns:
            OpenRouter response dict.

        Raises:
            ProviderError: If the request fails.
        """
        is_embedding = "embedding" in path.lower()
        is_stt = "transcription" in path.lower() or "input_audio" in body

        if not prepaid and not is_embedding and not is_stt:
            raw_max = body.get("max_tokens")
            user_max = raw_max if isinstance(raw_max, int) else 0
            if user_max < self._default_max_tokens:
                body = {**body, "max_tokens": self._default_max_tokens}
                logger.info(
                    "max_tokens %s -> %d (enforced minimum)",
                    raw_max,
                    self._default_max_tokens,
                )

        if not is_embedding and not is_stt:
            plugins = body.get("plugins", [])
            patched_plugins = []
            plugins_changed = False
            for plugin in plugins:
                if (
                    isinstance(plugin, dict)
                    and plugin.get("id") == "web"
                    and "max_results" not in plugin
                ):
                    plugin = {**plugin, "max_results": self._default_web_search_max_results}
                    plugins_changed = True
                    logger.info(
                        "Injected plugins.web.max_results=%d (default)",
                        self._default_web_search_max_results,
                    )
                patched_plugins.append(plugin)
            if plugins_changed:
                body = {**body, "plugins": patched_plugins}

        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"
        payload = _upstream_body(body)
        max_retries = 2
        retry_delay = 2.0
        last_error: ProviderError | None = None

        for attempt in range(1 + max_retries):
            try:
                async with httpx.AsyncClient(timeout=float(self._config.poll_timeout)) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self._config.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
            except httpx.TimeoutException:
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"OpenRouter request timed out after {self._config.poll_timeout}s",
                    status_code=504,
                )
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_error from None
            except Exception as e:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Failed to submit request: {e}",
                ) from e

            if resp.status_code >= 500 and attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "OpenRouter returned %d (attempt %d/%d), retrying in %.1fs: %s",
                    resp.status_code,
                    attempt + 1,
                    1 + max_retries,
                    delay,
                    resp.text[:200],
                )
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"OpenRouter API error: {resp.text}",
                    status_code=resp.status_code,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"OpenRouter API error: {resp.text}",
                    status_code=resp.status_code,
                )

            result = resp.json()
            usage = result.get("usage", {})
            if "text" in result:
                logger.info(
                    "OpenRouter STT completed: model=%s, seconds=%s, cost=$%s",
                    body.get("model", "unknown"),
                    usage.get("seconds", "?"),
                    usage.get("cost", "?"),
                )
            else:
                logger.info(
                    "OpenRouter request completed: model=%s, tokens=%s",
                    body.get("model", "unknown"),
                    usage.get("total_tokens", "?"),
                )
            return result

        raise last_error or ProviderError(
            provider=self.name,
            detail="All retries exhausted",
            status_code=502,
        )

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — OpenRouter returns results synchronously."""
        raise NotImplementedError(
            "OpenRouter returns results synchronously; polling is not needed."
        )

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Calculate actual cost from usage data in the response.

        For STT, prefers ``usage.cost`` from OpenRouter.
        For chat/embeddings, uses prompt and completion token counts.

        Args:
            body: Original request body (used for model ID).
            result: Provider response containing 'usage' dict.

        Returns:
            Actual cost in USD, or None if usage data is unavailable.
        """
        usage = result.get("usage")
        if not usage:
            return None

        is_stt = _is_stt_request("", body) or "text" in result or usage.get("seconds") is not None

        reported_cost = usage.get("cost")
        if reported_cost is not None and is_stt:
            actual_cost = _apply_cost_floor(Decimal(str(reported_cost)))
            logger.info(
                "OpenRouter actual cost for %s: $%s (from usage.cost)",
                body.get("model", ""),
                actual_cost,
            )
            return actual_cost

        model_id = body.get("model", "")
        try:
            is_embedding = "input" in body and "messages" not in body
            if is_stt:
                output_modalities = "transcription"
            elif is_embedding:
                output_modalities = "embeddings"
            else:
                output_modalities = ""
            model_info = await self._fetch_model_info(
                model_id,
                output_modalities=output_modalities,
            )
        except ProviderError:
            return None

        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is None or completion_price is None:
            return None

        prompt_price_dec = Decimal(str(prompt_price))
        completion_price_dec = Decimal(str(completion_price))

        audio_seconds = usage.get("seconds")
        if audio_seconds is not None and _is_duration_based_stt(pricing):
            seconds = billing_seconds(float(audio_seconds))
            actual_cost = Decimal(seconds) / Decimal("60") * prompt_price_dec
            actual_cost = _apply_cost_floor(actual_cost)
            logger.info(
                "OpenRouter actual cost for %s: $%s (%d billing seconds, duration STT)",
                model_id,
                actual_cost,
                seconds,
            )
            return actual_cost

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            return None

        completion_tokens = usage.get("completion_tokens", 0)

        web_search_cost = Decimal(0)
        plugins = body.get("plugins", [])
        for plugin in plugins:
            if isinstance(plugin, dict) and plugin.get("id") == "web":
                max_results = plugin.get("max_results", self._default_web_search_max_results)
                web_search_cost = Decimal(max_results) * self._web_search_cost_per_result

        actual_cost = (
            Decimal(prompt_tokens) * prompt_price_dec
            + Decimal(completion_tokens) * completion_price_dec
            + web_search_cost
        )
        actual_cost = _apply_cost_floor(actual_cost)

        logger.info(
            "OpenRouter actual cost for %s: $%s (prompt=%d, completion=%d tokens)",
            model_id,
            actual_cost,
            prompt_tokens,
            completion_tokens,
        )
        return actual_cost
