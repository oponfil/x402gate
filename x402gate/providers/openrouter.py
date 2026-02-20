"""OpenRouter provider for x402gate.

Implements the BaseProvider interface for OpenRouter's API:
- Pricing via GET /api/v1/models/{model} (per-token pricing)
- Task submission via POST /api/v1/chat/completions (synchronous)
- No polling needed (responses are synchronous)
"""

from __future__ import annotations

import logging
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

    OpenRouter is an LLM aggregator with 300+ models and an OpenAI-compatible
    API.  Pricing is per-token and varies by model.  We estimate the cost
    upfront using the model's published pricing and the request's max_tokens.
    """

    def __init__(self, config: ProviderConfig, *, default_max_tokens: int = 1024) -> None:
        super().__init__(name="openrouter", config=config)
        self._models_cache: dict[str, dict[str, Any]] = {}
        self._default_max_tokens = default_max_tokens

    async def _fetch_model_info(self, model_id: str) -> dict[str, Any]:
        """Fetch model info (including pricing) from OpenRouter Models API.

        OpenRouter only provides a list endpoint (GET /api/v1/models), not a
        single-model lookup.  We fetch the full catalog once and cache it.

        Args:
            model_id: Model identifier (e.g. 'openai/gpt-4o-mini').

        Returns:
            Model info dict with 'pricing' sub-dict.

        Raises:
            ProviderError: If the model is not found or the API call fails.
        """
        if model_id in self._models_cache:
            return self._models_cache[model_id]

        # Fetch full model catalog if cache is empty
        if not self._models_cache:
            url = f"{self._config.base_url.rstrip('/')}/models"
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url)

                if resp.status_code >= 400:
                    raise ProviderError(
                        provider=self.name,
                        detail=f"OpenRouter Models API error: {resp.text}",
                        status_code=resp.status_code,
                    )

                models_list = resp.json().get("data", [])
                for model in models_list:
                    mid = model.get("id", "")
                    if mid:
                        self._models_cache[mid] = model
                logger.info("Cached %d models from OpenRouter", len(self._models_cache))

            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Failed to fetch models list: {e}",
                    status_code=503,
                ) from e

        if model_id not in self._models_cache:
            raise ProviderError(
                provider=self.name,
                detail=f"Model '{model_id}' not found on OpenRouter",
                status_code=404,
            )

        return self._models_cache[model_id]

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate maximum request cost from per-token pricing.

        Fetches the model's pricing from OpenRouter and calculates:
            estimated_cost = input_tokens × prompt_price
                           + max_tokens × completion_price

        Input tokens are estimated from total message text length.
        If max_tokens is not provided, uses the configured default.

        Args:
            model_path: Ignored (model is in inputs['model']).
            inputs: Request body containing 'model', 'messages', and
                    optionally 'max_tokens'.

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

        # Estimate input tokens from message text
        messages = inputs.get("messages", [])
        total_chars = sum(
            len(str(m.get("content", ""))) for m in messages
        )
        estimated_input_tokens = max(total_chars // _CHARS_PER_TOKEN, 1)

        max_tokens = inputs.get("max_tokens", self._default_max_tokens)

        estimated_cost = (
            Decimal(estimated_input_tokens) * prompt_price_dec
            + Decimal(max_tokens) * completion_price_dec
        )

        # Ensure a minimum cost floor (some models are very cheap)
        if estimated_cost < Decimal("0.000001"):
            estimated_cost = Decimal("0.000001")

        logger.info(
            "OpenRouter price estimate for %s: $%s "
            "(~%d input tokens, %d max output tokens)",
            model_id,
            estimated_cost,
            estimated_input_tokens,
            max_tokens,
        )
        return estimated_cost

    async def submit(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Submit a chat completion request to OpenRouter.

        Sends the request to POST /api/v1/chat/completions.  OpenRouter
        returns the result synchronously (no polling needed).

        Args:
            path: API sub-path (typically 'chat/completions').
            body: Request body, forwarded without modification.

        Returns:
            OpenRouter response dict (OpenAI-compatible format).

        Raises:
            ProviderError: If the request fails.
        """
        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                )

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

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to submit request: {e}",
            ) from e

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — OpenRouter returns results synchronously."""
        raise NotImplementedError(
            "OpenRouter returns results synchronously; polling is not needed."
        )

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Calculate actual cost from usage data in the response.

        Uses real prompt_tokens and completion_tokens from the provider
        response instead of the estimated max_tokens ceiling.

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
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None

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

        actual_cost = (
            Decimal(prompt_tokens) * Decimal(str(prompt_price))
            + Decimal(completion_tokens) * Decimal(str(completion_price))
        )

        if actual_cost < Decimal("0.000001"):
            actual_cost = Decimal("0.000001")

        logger.info(
            "OpenRouter actual cost for %s: $%s "
            "(prompt=%d, completion=%d tokens)",
            model_id,
            actual_cost,
            prompt_tokens,
            completion_tokens,
        )
        return actual_cost
