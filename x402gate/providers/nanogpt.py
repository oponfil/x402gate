"""NanoGPT provider for x402gate.

Chat completions against NanoGPT's text catalog. Detailed model prices are
USD per million tokens. Cache read and cache write rates in that catalog are
USD per thousand tokens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

_CHAT_CHARS_PER_TOKEN = Decimal("2.7")
_MILLION = Decimal("1000000")
_THOUSAND = Decimal("1000")
_CHAT_PATH = "chat/completions"


def _apply_cost_floor(cost: Decimal) -> Decimal:
    if cost < Decimal("0.000001"):
        return Decimal("0.000001")
    return cost


def _content_chars(content: Any) -> int:
    """Count characters in a string or a list of chat content parts."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(part) for part in content)
    if isinstance(content, dict):
        if "text" in content:
            return _content_chars(content["text"])
        return len(str(content))
    return len(str(content))


def _estimate_input_tokens(messages: list[Any]) -> int:
    total = 0
    for message in messages:
        if isinstance(message, dict):
            total += _content_chars(message.get("content", ""))
        else:
            total += len(str(message))
    return max(int(Decimal(total) / _CHAT_CHARS_PER_TOKEN), 1)


def _upstream_body(body: dict[str, Any]) -> dict[str, Any]:
    """Strip gateway-internal keys before forwarding to NanoGPT."""
    return {key: value for key, value in body.items() if not key.startswith("_")}


def _money(value: Any, label: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProviderError(
            provider="nanogpt",
            detail=f"Invalid NanoGPT {label}: {value}",
            status_code=503,
        ) from exc


def _per_million(value: Any) -> Decimal:
    return _money(value, "per-million price") / _MILLION


def _per_thousand(value: Any) -> Decimal:
    return _money(value, "per-thousand price") / _THOUSAND


def _web_search_enabled(body: dict[str, Any]) -> bool:
    """True when the request asks NanoGPT to run a billed web search."""
    model_id = str(body.get("model", ""))
    suffixes = model_id.split(":")[1:]
    if any(part == "online" or part.startswith("online/") for part in suffixes):
        return True
    for key in ("web_search", "webSearch", "linkup"):
        value = body.get(key)
        if value is True:
            return True
        if isinstance(value, dict) and value.get("enabled") is True:
            return True
    options = body.get("web_search_options")
    return isinstance(options, dict) and bool(options)


def _caching_requested(body: dict[str, Any]) -> bool:
    if body.get("caching") is True:
        return True
    prompt_caching = body.get("prompt_caching")
    return isinstance(prompt_caching, dict) and prompt_caching.get("enabled") is True


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderError(
            provider="nanogpt",
            detail=f"{field} must be a positive integer",
            status_code=400,
        )
    return value


class NanoGptProvider(BaseProvider):
    """NanoGPT chat provider.

    NanoGPT is an OpenAI-compatible aggregator. This gateway path serves the
    text catalog at ``POST /v1/nanogpt/chat/completions``.
    """

    def __init__(self, config: ProviderConfig, *, default_max_tokens: int = 1024) -> None:
        super().__init__(name="nanogpt", config=config)
        self._models_cache: dict[str, dict[str, Any]] = {}
        self._cache_updated_at: float = 0.0
        self._cache_ttl: float = float(config.models_cache_ttl)
        self._default_max_tokens = default_max_tokens

    def _require_chat(self, path: str) -> None:
        if path.strip("/") != _CHAT_PATH:
            raise ProviderError(
                provider=self.name,
                detail=(
                    "NanoGPT on x402gate supports POST /v1/nanogpt/chat/completions. "
                    f"Path '{path}' is not priced."
                ),
                status_code=400,
            )

    def _reject_unpriced_options(self, body: dict[str, Any]) -> None:
        if body.get("stream") is True:
            raise ProviderError(
                provider=self.name,
                detail="Streaming is not supported. Send the request with stream set to false.",
                status_code=400,
            )
        if _web_search_enabled(body):
            raise ProviderError(
                provider=self.name,
                detail=(
                    "Web search is not priced by x402gate because NanoGPT bills each "
                    "search on top of tokens and the count is not known up front. "
                    "Send the request without :online, web_search, or web_search_options."
                ),
                status_code=400,
            )

    async def _fetch_models_list(self) -> None:
        """Load the detailed text catalog. Prices are USD per million tokens."""
        url = f"{self._config.base_url.rstrip('/')}/models?detailed=true"
        max_retries = 2
        retry_delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url)
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                break

            if resp.status_code >= 500 and attempt < max_retries:
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"NanoGPT Models API error: {resp.text}",
                    status_code=resp.status_code,
                )
                await asyncio.sleep(retry_delay * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"NanoGPT Models API error: {resp.text}",
                    status_code=resp.status_code,
                )

            models = resp.json().get("data", [])
            cached: dict[str, dict[str, Any]] = {}
            for model in models:
                model_id = model.get("id", "")
                if model_id:
                    cached[model_id] = model
            self._models_cache = cached
            self._cache_updated_at = time.time()
            logger.info("Cached %d models from NanoGPT", len(cached))
            return

        raise ProviderError(
            provider=self.name,
            detail=f"Failed to fetch NanoGPT models after retries: {last_error}",
            status_code=503,
        ) from last_error

    async def _model_info(self, model_id: str) -> dict[str, Any]:
        now = time.time()
        if not self._models_cache or now - self._cache_updated_at > self._cache_ttl:
            await self._fetch_models_list()

        if model_id in self._models_cache:
            return self._models_cache[model_id]

        # Routing suffixes (:thinking, :abliteration, :cheap) are not always
        # their own catalog rows. The price is the underlying model.
        candidate = model_id
        while ":" in candidate:
            candidate = candidate.rsplit(":", 1)[0]
            if candidate in self._models_cache:
                return self._models_cache[candidate]

        raise ProviderError(
            provider=self.name,
            detail=f"Model '{model_id}' not found on NanoGPT",
            status_code=404,
        )

    def _rates(self, model_id: str, model_info: dict[str, Any]) -> tuple[Decimal, Decimal, dict]:
        pricing = model_info.get("pricing")
        if not isinstance(pricing, dict):
            raise ProviderError(
                provider=self.name,
                detail=f"No pricing info for model '{model_id}'",
                status_code=503,
            )
        unit = pricing.get("unit") or "per_million_tokens"
        if unit != "per_million_tokens":
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"Model '{model_id}' uses pricing unit '{unit}'. "
                    "x402gate prices NanoGPT models in USD per million tokens."
                ),
                status_code=503,
            )
        if pricing.get("prompt") is None or pricing.get("completion") is None:
            raise ProviderError(
                provider=self.name,
                detail=f"No pricing info for model '{model_id}'",
                status_code=503,
            )
        return _per_million(pricing["prompt"]), _per_million(pricing["completion"]), pricing

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate the maximum chat cost from the catalog and max_tokens."""
        self._require_chat(model_path)
        self._reject_unpriced_options(inputs)

        model_id = inputs.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise ProviderError(
                provider=self.name,
                detail="Missing model id",
                status_code=400,
            )
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            raise ProviderError(
                provider=self.name,
                detail="messages must be a list",
                status_code=400,
            )

        raw_n = inputs.get("n", 1)
        if raw_n is None:
            raw_n = 1
        completions = _positive_int(raw_n, "n")
        raw_max = inputs.get("max_tokens")
        max_tokens = max(
            raw_max if isinstance(raw_max, int) and raw_max > 0 else 0,
            self._default_max_tokens,
        )

        model_info = await self._model_info(model_id)
        prompt_rate, completion_rate, pricing = self._rates(model_id, model_info)
        input_tokens = _estimate_input_tokens(messages)
        input_rate = prompt_rate
        if _caching_requested(inputs) and pricing.get("cacheWriteInputPer1kTokens") is not None:
            input_rate = max(input_rate, _per_thousand(pricing["cacheWriteInputPer1kTokens"]))

        estimated = Decimal(input_tokens) * input_rate + Decimal(
            max_tokens
        ) * completion_rate * Decimal(completions)
        logger.info(
            "NanoGPT price estimate for %s: $%s (~%d input tokens, %d max output tokens)",
            model_id,
            estimated,
            input_tokens,
            max_tokens,
        )
        return _apply_cost_floor(estimated)

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Forward one chat completion. NanoGPT answers synchronously."""
        del prepaid
        self._require_chat(path)
        self._reject_unpriced_options(body)

        raw_max = body.get("max_tokens")
        user_max = raw_max if isinstance(raw_max, int) else 0
        if user_max < self._default_max_tokens:
            body = {**body, "max_tokens": self._default_max_tokens}
            logger.info(
                "max_tokens %s -> %d (enforced minimum)",
                raw_max,
                self._default_max_tokens,
            )

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
                    detail=f"NanoGPT request timed out after {self._config.poll_timeout}s",
                    status_code=504,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                raise last_error from None
            except Exception as exc:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Failed to submit request: {exc}",
                ) from exc

            if resp.status_code >= 500 and attempt < max_retries:
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"NanoGPT API error: {resp.text}",
                    status_code=resp.status_code,
                )
                await asyncio.sleep(retry_delay * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"NanoGPT API error: {resp.text}",
                    status_code=resp.status_code,
                )

            result = resp.json()
            usage = result.get("usage", {})
            logger.info(
                "NanoGPT request completed: model=%s, tokens=%s",
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
        """Not used. NanoGPT returns the completion in the submit response."""
        raise NotImplementedError("NanoGPT returns results synchronously; polling is not needed.")

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Price the real token counts. Cache reads are cheaper than fresh input."""
        usage = result.get("usage")
        if not isinstance(usage, dict) or usage.get("prompt_tokens") is None:
            return None

        model_id = body.get("model", "")
        if not isinstance(model_id, str) or not model_id:
            return None
        try:
            model_info = await self._model_info(model_id)
            prompt_rate, completion_rate, pricing = self._rates(model_id, model_info)
            read_rate = prompt_rate
            write_rate = prompt_rate
            if pricing.get("cacheReadInputPer1kTokens") is not None:
                read_rate = _per_thousand(pricing["cacheReadInputPer1kTokens"])
            if pricing.get("cacheWriteInputPer1kTokens") is not None:
                write_rate = _per_thousand(pricing["cacheWriteInputPer1kTokens"])
        except ProviderError:
            return None

        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        if not cache_read:
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cache_read = int(details.get("cached_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)

        fresh_tokens = prompt_tokens
        if cache_read + cache_write <= prompt_tokens:
            fresh_tokens = prompt_tokens - cache_read - cache_write

        actual = (
            Decimal(fresh_tokens) * prompt_rate
            + Decimal(cache_read) * read_rate
            + Decimal(cache_write) * write_rate
            + Decimal(completion_tokens) * completion_rate
        )
        actual = _apply_cost_floor(actual)
        logger.info(
            "NanoGPT actual cost for %s: $%s (prompt=%d, cache_read=%d, completion=%d)",
            model_id,
            actual,
            prompt_tokens,
            cache_read,
            completion_tokens,
        )
        return actual
