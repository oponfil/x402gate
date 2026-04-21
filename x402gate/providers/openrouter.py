"""OpenRouter provider for x402gate.

Implements the BaseProvider interface for OpenRouter's API:
- Pricing via GET /api/v1/models (per-token pricing, chat + embedding models)
- Chat completion via POST /api/v1/chat/completions (synchronous)
- Embeddings via POST /api/v1/embeddings (synchronous)
- No polling needed (responses are synchronous)
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation.

    OpenRouter is an LLM aggregator with 300+ chat and embedding models and
    an OpenAI-compatible API.  Supports both chat completions and embeddings.
    Pricing is per-token and varies by model.  We estimate the cost upfront
    using the model's published pricing and the request's max_tokens
    (chat) or input length (embedding).
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
        self._cache_updated_at: float = 0.0
        self._cache_ttl: float = float(config.models_cache_ttl)
        self._default_max_tokens = default_max_tokens
        self._web_search_tokens_per_result = web_search_tokens_per_result
        self._default_web_search_max_results = default_web_search_max_results
        self._web_search_cost_per_result = Decimal(str(web_search_cost_per_result))

    async def _fetch_model_info(self, model_id: str) -> dict[str, Any]:
        """Fetch model info (including pricing) from OpenRouter Models API.

        On first access, fetches both the chat model catalog and the embedding
        model catalog (?output_modalities=embeddings) into a unified cache.

        Args:
            model_id: Model identifier (e.g. 'openai/gpt-4o-mini').

        Returns:
            Model info dict with 'pricing' sub-dict.

        Raises:
            ProviderError: If the model is not found or the API call fails.
        """
        current_time = time.time()

        # Invalidate cache if TTL expired
        if self._models_cache and (current_time - self._cache_updated_at > self._cache_ttl):
            logger.info("OpenRouter models cache expired (1 day TTL), clearing...")
            self._models_cache.clear()

        if model_id in self._models_cache:
            return self._models_cache[model_id]

        # Fetch both chat and embedding models on first access.
        # This is intentional: partial catalog state is considered invalid,
        # so if either catalog cannot be loaded we fail fast instead of
        # serving OpenRouter with incomplete pricing/model metadata.
        if not self._models_cache:
            await self._fetch_models_list()
            await self._fetch_models_list(output_modalities="embeddings")
            self._cache_updated_at = time.time()

        if model_id not in self._models_cache:
            raise ProviderError(
                provider=self.name,
                detail=f"Model '{model_id}' not found on OpenRouter",
                status_code=404,
            )

        return self._models_cache[model_id]

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

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate maximum request cost from per-token pricing.

        For chat completions:
            estimated_cost = input_tokens × prompt_price
                           + max_tokens × completion_price
        For embeddings:
            estimated_cost = input_tokens × prompt_price

        Input tokens are estimated from total text length (messages or input).

        Args:
            model_path: Ignored (model is in inputs['model']).
            inputs: Request body containing 'model' and either 'messages'
                    (chat) or 'input' (embedding).

        Returns:
            Estimated maximum cost in USD as a Decimal.

        Raises:
            ProviderError: If pricing info is unavailable.
        """

        model_id = inputs.get("model", model_path)
        model_info = await self._fetch_model_info(model_id)

        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            raise ProviderError(
                provider=self.name,
                detail=f"No pricing info for model '{model_id}'",
                status_code=503,
            )

        # Convert from string price-per-token to Decimal
        prompt_price_dec = Decimal(str(prompt_price))
        completion_price_dec = Decimal(str(completion_price))

        # Embedding requests: price based on input only (no completion tokens)
        if "input" in inputs and "messages" not in inputs:
            text_input = inputs.get("input", "")
            if isinstance(text_input, list):
                total_chars = sum(len(str(item)) for item in text_input)
            else:
                total_chars = len(str(text_input))
            estimated_input_tokens = max(total_chars // _CHARS_PER_TOKEN, 1)

            estimated_cost = Decimal(estimated_input_tokens) * prompt_price_dec
            if estimated_cost < Decimal("0.000001"):
                estimated_cost = Decimal("0.000001")

            logger.info(
                "OpenRouter price estimate for %s: $%s (~%d input tokens, embedding)",
                model_id,
                estimated_cost,
                estimated_input_tokens,
            )
            return estimated_cost

        # Chat completion requests
        # Estimate input tokens from message text
        messages = inputs.get("messages", [])
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_input_tokens = max(total_chars // _CHARS_PER_TOKEN, 1)

        # Web search plugins inject search results into the prompt,
        # adding significant extra input tokens not visible in messages.
        # They also have a fixed cost per result (OpenRouter Exa: $4/1000).
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

        # Reasoning tokens are a subset of completion tokens and
        # capped by max_tokens, so the estimate is simply:
        estimated_cost = (
            Decimal(estimated_input_tokens) * prompt_price_dec
            + Decimal(max_tokens) * completion_price_dec
            + web_search_cost
        )

        # Ensure a minimum cost floor (some models are very cheap)
        if estimated_cost < Decimal("0.000001"):
            estimated_cost = Decimal("0.000001")

        logger.info(
            "OpenRouter price estimate for %s: $%s (~%d input tokens, %d max output tokens)",
            model_id,
            estimated_cost,
            estimated_input_tokens,
            max_tokens,
        )
        return estimated_cost

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a request to OpenRouter (chat completions or embeddings).

        Routes to the appropriate endpoint based on path:
        - 'chat/completions' → POST /api/v1/chat/completions
        - 'embeddings' → POST /api/v1/embeddings

        OpenRouter returns results synchronously (no polling needed).

        Args:
            path: API sub-path ('chat/completions' or 'embeddings').
            body: Request body, forwarded to OpenRouter.
            prepaid: If True, skip max_tokens injection (actual usage
                will be charged post-request from prepaid balance).

        Returns:
            OpenRouter response dict (OpenAI-compatible format).

        Raises:
            ProviderError: If the request fails.
        """
        # max_tokens and plugin injection only apply to chat completions
        is_embedding = "embedding" in path.lower()

        # In x402 mode, ensure max_tokens is set so OpenRouter doesn't
        # fall back to a large provider default (which would cost more
        # than our estimate).  In prepaid mode, skip this — we charge
        # actual usage, so let the model respond fully.
        if not prepaid and not is_embedding:
            raw_max = body.get("max_tokens")
            user_max = raw_max if isinstance(raw_max, int) else 0
            if user_max < self._default_max_tokens:
                body = {**body, "max_tokens": self._default_max_tokens}
                logger.info(
                    "max_tokens %s -> %d (enforced minimum)",
                    raw_max,
                    self._default_max_tokens,
                )

        # Inject default max_results into web search plugins so
        # actual usage matches our cost estimate (chat only).
        if not is_embedding:
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
        max_retries = 2
        retry_delay = 2.0
        last_error: ProviderError | None = None

        for attempt in range(1 + max_retries):
            try:
                async with httpx.AsyncClient(timeout=float(self._config.poll_timeout)) as client:
                    resp = await client.post(
                        url,
                        json=body,
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
            logger.info(
                "OpenRouter request completed: model=%s, tokens=%s",
                body.get("model", "unknown"),
                result.get("usage", {}).get("total_tokens", "?"),
            )
            return result

        # Safety net — all retries exhausted
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

        For chat completions, uses real prompt_tokens and completion_tokens.
        For embeddings, uses prompt_tokens only (completion_tokens=0).

        Args:
            body: Original request body (used for model ID).
            result: Provider response containing 'usage' dict.

        Returns:
            Actual cost in USD, or None if usage data is unavailable.
        """
        usage = result.get("usage")
        if not usage:
            return None

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            return None

        # Embeddings have no completion_tokens — default to 0
        completion_tokens = usage.get("completion_tokens", 0)

        model_id = body.get("model", "")
        try:
            model_info = await self._fetch_model_info(model_id)
        except ProviderError:
            return None

        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is None or completion_price is None:
            return None

        # Fixed web search cost per result (OpenRouter Exa: $4/1000)
        web_search_cost = Decimal(0)
        plugins = body.get("plugins", [])
        for plugin in plugins:
            if isinstance(plugin, dict) and plugin.get("id") == "web":
                max_results = plugin.get("max_results", self._default_web_search_max_results)
                web_search_cost = Decimal(max_results) * self._web_search_cost_per_result

        actual_cost = (
            Decimal(prompt_tokens) * Decimal(str(prompt_price))
            + Decimal(completion_tokens) * Decimal(str(completion_price))
            + web_search_cost
        )

        if actual_cost < Decimal("0.000001"):
            actual_cost = Decimal("0.000001")

        logger.info(
            "OpenRouter actual cost for %s: $%s (prompt=%d, completion=%d tokens)",
            model_id,
            actual_cost,
            prompt_tokens,
            completion_tokens,
        )
        return actual_cost
