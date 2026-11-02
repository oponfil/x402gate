"""NanoGPT provider for x402gate.

Chat completions against NanoGPT's text catalog. Detailed model prices are
USD per million tokens. Cache read and cache write rates in that catalog are
USD per thousand tokens. Audio input is accepted only on models whose entry
lists ``pricing.audio_input`` (per million tokens), timed like OpenRouter's.
"""

from __future__ import annotations

import asyncio
import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError
from x402gate.providers.openrouter import (
    _AUDIO_TOKENS_PER_SECOND,
    _LONG_CACHE_WRITE,
    _SHORT_CACHE_TTLS,
    Catalog,
    CatalogCache,
    _billed_audio_tokens,
    _chat_audio_seconds,
    chat_audio_inputs,
    check_content_parts,
    check_json_encodable,
    estimate_chat_input_tokens,
    has_nested_key,
    held_audio_tokens,
    in_audio_pricing_slot,
    long_cache_ttl,
    output_ceiling,
    parse_billed_json,
    post_billed_request,
    raise_for_error_body,
    reject_non_text_output,
    reject_server_tools,
)

logger = logging.getLogger(__name__)

_MILLION = Decimal("1000000")
_THOUSAND = Decimal("1000")
_CHAT_PATH = "chat/completions"

# A thinking budget suffix (``:thinking:<n>``, ``-thinking:<n>``): ASCII
# digits, nine at most, far above any real budget and well inside Python's
# int-string limit.
_BUDGET_DIGITS = re.compile(r"[0-9]{1,9}")

# Tiers billed at or below the standard rate. "fast" and "priority" bill a
# higher rate the catalog does not list.
_PRICED_SERVICE_TIERS = frozenset({"auto", "default", "flex"})

# Per-request add-ons NanoGPT bills on top of tokens, by the flag that
# switches them on: YouTube transcripts ($0.01 each), URL scraping ($0.0015
# per URL) and the advisor model (its own billed phase).
_UNPRICED_ADDONS = {
    "youtube_transcripts": "YouTube transcripts",
    "scraping": "URL scraping",
    "advisor": "The advisor model",
}

# Long-context tier for the hold only, on models whose upstream bills one
# that NanoGPT's catalog does not list: Gemini 2.5 and 3.x Pro bill 2x input
# and 1.5x output above 200k prompt tokens. Anchored on the id, and only for
# a model whose context window is above the threshold. The actual cost uses
# only the tier NanoGPT publishes (see _published_tier).
_HOLD_TIERS = (
    (
        re.compile(r"(google/)?gemini-(2\.5|3(\.[0-9]+)?)-pro\b"),
        200_000,
        Decimal(2),
        Decimal("1.5"),
    ),
)


def _apply_cost_floor(cost: Decimal) -> Decimal:
    if cost < Decimal("0.000001"):
        return Decimal("0.000001")
    return cost


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


def _audio_input_rate(pricing: Any) -> Decimal | None:
    """Per-token audio input rate a catalog ``pricing`` entry lists, if usable.

    ``pricing.audio_input`` is USD per million tokens, like ``prompt``. None
    when it is missing, malformed, or not a positive finite number.
    """
    listed = pricing.get("audio_input") if isinstance(pricing, dict) else None
    if listed is None or isinstance(listed, bool):
        return None
    try:
        rate = Decimal(str(listed))
    except (InvalidOperation, ValueError):
        return None
    if not rate.is_finite() or rate <= 0:
        return None
    return rate / _MILLION


def _model_suffixes(body: dict[str, Any]) -> list[str]:
    return str(body.get("model", "")).lower().split(":")[1:]


def _flag_on(value: Any) -> bool:
    """True when a feature flag may switch the feature on.

    Any truthy value counts ("true", 1), not just True, since NanoGPT may
    accept loose values. An object counts unless its ``enabled`` is falsy.
    """
    if isinstance(value, dict):
        return bool(value) and bool(value.get("enabled", True))
    return bool(value)


def _web_search_enabled(body: dict[str, Any]) -> bool:
    """True when the request asks NanoGPT to run a billed web search."""
    if any(part == "online" or part.startswith("online/") for part in _model_suffixes(body)):
        return True
    if any(_flag_on(body.get(key)) for key in ("web_search", "webSearch", "linkup")):
        return True
    options = body.get("web_search_options")
    return bool(options)


def _memory_enabled(body: dict[str, Any]) -> bool:
    """True when the request asks for NanoGPT Context Memory, billed separately."""
    if any(part == "memory" or part.startswith("memory-") for part in _model_suffixes(body)):
        return True
    return _flag_on(body.get("memory"))


def _suffix_thinking_budget(body: dict[str, Any]) -> int:
    """Thinking budget from a numeric model suffix.

    NanoGPT reads it after ``:thinking`` (``model:thinking:8192``) and after
    a thinking variant (``claude-3-7-sonnet-thinking:128000``). Any suffix of
    digits counts, wherever it sits, so no spelling of the budget is priced
    at the default output. The suffix fallback in ``_model_info`` prices the
    base model, so the budget is passed to ``output_ceiling``, which counts
    it like a body budget: on top of a max_tokens at or below it, inside a
    larger one.
    """
    budget = 0
    for part in _model_suffixes(body):
        if _BUDGET_DIGITS.fullmatch(part):
            budget = max(budget, int(part))
        elif part.isdigit():
            # "²", "٣" or a huge run: int() would fail or NanoGPT may read it
            # differently, so the budget cannot be priced.
            raise ProviderError(
                provider="nanogpt",
                detail="A thinking budget model suffix must be 1 to 9 ASCII digits",
                status_code=400,
            )
    return budget


def _caching_requested(body: dict[str, Any]) -> bool:
    if has_nested_key(body.get("messages"), "cache_control"):
        return True
    return _flag_on(body.get("caching")) or _flag_on(body.get("prompt_caching"))


def _long_cache_requested(body: dict[str, Any]) -> bool:
    """True when the request asks for a cache TTL above 5 minutes.

    In a ``cache_control`` anywhere in the body, or in the ``caching`` or
    ``prompt_caching`` options. Anthropic bills such a write at 2x the input
    rate, above the 5-minute rate the catalog lists as
    cacheWriteInputPer1kTokens.
    """
    if long_cache_ttl(body):
        return True
    for key in ("caching", "prompt_caching"):
        options = body.get(key)
        if isinstance(options, dict) and options.get("ttl") not in _SHORT_CACHE_TTLS:
            return True
    return False


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderError(
            provider="nanogpt",
            detail=f"{field} must be a positive integer",
            status_code=400,
        )
    return value


def _context_length(model_info: dict[str, Any]) -> int | None:
    """Context window a catalog entry lists, or None when it lists none."""
    length = model_info.get("context_length")
    if isinstance(length, bool) or not isinstance(length, (int, float)) or length <= 0:
        return None
    return int(length)


def _hold_multipliers(
    model_id: str, model_info: dict[str, Any], input_tokens: int
) -> tuple[Decimal, Decimal]:
    """(input, output) multipliers of the hold for a prompt of ``input_tokens``.

    A tier counts only when the model's context window is above its
    threshold; an entry without a window is assumed to reach it.
    """
    context = _context_length(model_info)
    for pattern, threshold, tier_input, tier_output in _HOLD_TIERS:
        if (
            input_tokens > threshold
            and (context is None or context > threshold)
            and pattern.match(model_id.lower())
        ):
            return tier_input, tier_output
    return Decimal(1), Decimal(1)


def _published_tier(model_info: dict[str, Any], input_tokens: int) -> dict[str, Decimal]:
    """Long-context rates NanoGPT lists for a prompt of ``input_tokens``, per token.

    ``effective_pricing.list_price`` gives them per 1k tokens beside
    ``highContextThresholdTokens``, named for the threshold
    (``inputPer1kTokensAbove200k``). Empty below the threshold or when the
    model lists none.
    """
    effective = model_info.get("effective_pricing")
    listed = effective.get("list_price") if isinstance(effective, dict) else None
    if not isinstance(listed, dict):
        return {}
    threshold = listed.get("highContextThresholdTokens")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1000:
        return {}
    if input_tokens <= threshold:
        return {}
    suffix = f"Above{threshold // 1000}k"
    tier = {}
    for key, field in (
        ("prompt", "input"),
        ("completion", "output"),
        ("cache_read", "cacheReadInput"),
        ("cache_write", "cacheWriteInput"),
    ):
        value = listed.get(f"{field}Per1kTokens{suffix}")
        if value is not None:
            tier[key] = _per_thousand(value)
    return tier


def _lookup(models: Catalog, model_id: str) -> dict[str, Any] | None:
    """Catalog entry of a model id, or None."""
    if model_id in models:
        return models[model_id]
    # Routing suffixes (:thinking, :abliteration, :cheap) are not always
    # their own catalog rows. The price is the underlying model. Suffixes
    # that add a separate charge (:online, :memory) are refused before this.
    candidate = model_id
    while ":" in candidate:
        candidate = candidate.rsplit(":", 1)[0]
        if candidate in models:
            return models[candidate]
    return None


class NanoGptProvider(BaseProvider):
    """NanoGPT chat provider.

    NanoGPT is an OpenAI-compatible aggregator. This gateway path serves the
    text catalog at ``POST /v1/nanogpt/chat/completions``.
    """

    def __init__(self, config: ProviderConfig, *, default_max_tokens: int = 1024) -> None:
        super().__init__(name="nanogpt", config=config)
        # One catalog, stored under the key "".
        self._catalog = CatalogCache(
            float(config.models_cache_ttl), self._fetch_models_list, self.name
        )
        self._default_max_tokens = default_max_tokens
        # Cap on each input_audio part, decoded and timed before payment.
        self._max_audio_bytes = config.max_stt_audio_mb * 1024 * 1024

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
        if body.get("stream"):
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
        for flag, feature in _UNPRICED_ADDONS.items():
            if _flag_on(body.get(flag)):
                raise ProviderError(
                    provider=self.name,
                    detail=(
                        f"{feature} is not priced by x402gate because NanoGPT bills it "
                        f"on top of tokens. Send the request without {flag}."
                    ),
                    status_code=400,
                )
        tier = body.get("service_tier")
        if tier is not None and (not isinstance(tier, str) or tier not in _PRICED_SERVICE_TIERS):
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"service_tier '{tier}' is billed above the catalog rate and is not "
                    "priced by x402gate. Use auto, default or flex."
                ),
                status_code=400,
            )
        reject_non_text_output(body, self.name)
        if _memory_enabled(body):
            raise ProviderError(
                provider=self.name,
                detail=(
                    "Context Memory is not priced by x402gate because NanoGPT bills it "
                    "on top of tokens. Send the request without :memory or memory."
                ),
                status_code=400,
            )
        # A web search tool or xAI Live Search bills per search, like the
        # options refused above.
        reject_server_tools(body, self.name)
        # Rejects content parts (files, video) the estimate cannot price,
        # without weighing the text, which get_price does once.
        check_content_parts(body, self.name)
        # Audio parts are checked per model, against the catalog
        # (_require_audio_rate).

    def _require_audio_rate(self, model_id: str, model_info: dict[str, Any]) -> None:
        """Refuse audio input for a model whose catalog entry lists no audio rate.

        Models bill audio tokens at their own rate, up to 16x the text rate
        (gpt-4o audio). Only a few entries (Gemini Flash) list it, as
        ``pricing.audio_input`` per million tokens; ``capabilities.audio_input``
        alone says nothing of the price.
        """
        if _audio_input_rate(model_info.get("pricing")) is None:
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"Audio input is not priced by x402gate for NanoGPT model '{model_id}' "
                    "(its catalog entry lists no audio_input rate). Use a model that lists "
                    "one, or send the request without input_audio parts."
                ),
                status_code=400,
            )

    async def _measure_chat_audio(self, audio_inputs: list[Any]) -> int:
        """Billing seconds of a chat request's audio parts, measured in a worker thread."""
        return await in_audio_pricing_slot(
            lambda: _chat_audio_seconds(
                audio_inputs, self.name, max_audio_bytes=self._max_audio_bytes
            ),
            self.name,
        )

    async def warm_up(self) -> None:
        """Load the catalog at startup, so the first request does not wait."""
        await self._catalog.load("")

    async def _fetch_models_list(self, _key: str = "") -> Catalog:
        """Load the detailed text catalog. Prices are USD per million tokens."""
        url = f"{self._config.base_url.rstrip('/')}/models?detailed=true"
        max_retries = 2
        retry_delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                resp = await http.shared_client().get(url, timeout=http.timeout(15.0))
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
            cached: Catalog = {}
            for model in models:
                model_id = model.get("id", "")
                if model_id:
                    cached[model_id] = model
            logger.info("Cached %d models from NanoGPT", len(cached))
            return cached

        raise ProviderError(
            provider=self.name,
            detail=f"Failed to fetch NanoGPT models after retries: {last_error}",
            status_code=503,
        ) from last_error

    async def _model_info(self, model_id: str) -> dict[str, Any]:
        # An expired catalog is served while it refreshes in the background,
        # except for a model it lacks (perhaps added since), which waits for
        # the fresh catalog; in the retry delay after a failed refresh it is
        # not found without a fetch, so unknown ids cannot start one each.
        info = _lookup(await self._catalog.get(""), model_id)
        if info is None and self._catalog.may_fetch(""):
            info = _lookup(await self._catalog.load(""), model_id)
        if info is not None:
            return info
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

    def _token_rates(
        self, model_id: str, model_info: dict[str, Any], input_tokens: int, *, hold: bool
    ) -> dict[str, Decimal]:
        """Per-token rates of one model for a prompt of ``input_tokens``.

        ``prompt``, ``completion``, ``cache_read`` and ``cache_write`` (the
        cache rates fall back to the prompt rate), and ``audio`` when the
        model lists an audio input rate. Above the long-context
        threshold NanoGPT lists, its tier rates apply; a cache rate it leaves
        out scales with the prompt rate. With ``hold``, a model in
        ``_HOLD_TIERS`` that lists no tier is raised to that tier.
        """
        prompt, completion, pricing = self._rates(model_id, model_info)
        rates = {"prompt": prompt, "completion": completion}
        for key, field in (
            ("cache_read", "cacheReadInputPer1kTokens"),
            ("cache_write", "cacheWriteInputPer1kTokens"),
        ):
            listed = pricing.get(field)
            rates[key] = prompt if listed is None else _per_thousand(listed)
        audio = _audio_input_rate(pricing)
        if audio is not None:
            rates["audio"] = audio

        tier = _published_tier(model_info, input_tokens)
        if tier:
            scale = tier.get("prompt", prompt) / prompt if prompt > 0 else Decimal(1)
            return {key: tier.get(key, rate * scale) for key, rate in rates.items()}
        if not hold:
            return rates
        input_mult, output_mult = _hold_multipliers(model_id, model_info, input_tokens)
        return {
            key: rate * (output_mult if key == "completion" else input_mult)
            for key, rate in rates.items()
        }

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate the maximum chat cost from the catalog and max_tokens."""
        self._require_chat(model_path)
        self._reject_unpriced_options(inputs)
        check_json_encodable(inputs, self.name)

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
        max_tokens = output_ceiling(
            inputs,
            self._default_max_tokens,
            self.name,
            extra_budget=_suffix_thinking_budget(inputs),
        )

        model_info = await self._model_info(model_id)
        input_tokens = estimate_chat_input_tokens(inputs, self.name)
        # Audio parts are timed, not weighed as text (chat_input_size skips
        # their base64), and billed at the model's audio rate.
        audio_inputs = chat_audio_inputs(inputs)
        audio_tokens = 0
        if audio_inputs:
            self._require_audio_rate(model_id, model_info)
            audio_tokens = await self._measure_chat_audio(audio_inputs) * _AUDIO_TOKENS_PER_SECOND
        # The estimate is weighed to stay above the real prompt, so a prompt
        # billed at the long-context tier is priced at it.
        rates = self._token_rates(model_id, model_info, input_tokens + audio_tokens, hold=True)
        input_rate = rates["prompt"]
        if _caching_requested(inputs):
            input_rate = max(input_rate, rates["cache_write"])
        if _long_cache_requested(inputs):
            input_rate = max(input_rate, _LONG_CACHE_WRITE * rates["prompt"])
        audio_rate = max(rates.get("audio", input_rate), input_rate)

        estimated = (
            Decimal(input_tokens) * input_rate
            + Decimal(audio_tokens) * audio_rate
            + Decimal(max_tokens) * rates["completion"] * Decimal(completions)
        )
        logger.info(
            "NanoGPT price estimate for %s: $%s (~%d input tokens, %d audio tokens, "
            "%d max output tokens)",
            model_id,
            estimated,
            input_tokens,
            audio_tokens,
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
        if chat_audio_inputs(body):
            # get_price checks this too; a direct call must not forward audio
            # a model's catalog entry does not price.
            model_id = body.get("model")
            if not isinstance(model_id, str) or not model_id:
                raise ProviderError(provider=self.name, detail="Missing model id", status_code=400)
            self._require_audio_rate(model_id, await self._model_info(model_id))

        raw_max = body.get("max_tokens")
        user_max = raw_max if isinstance(raw_max, int) else 0
        if user_max < self._default_max_tokens:
            body = {**body, "max_tokens": self._default_max_tokens}
            logger.info(
                "max_tokens %s -> %d (enforced minimum)",
                raw_max,
                self._default_max_tokens,
            )
        if "max_completion_tokens" in body and (
            _positive_int_or_none(body["max_completion_tokens"]) is None
        ):
            # get_price only counts a positive integer; drop anything else.
            body = {k: v for k, v in body.items() if k != "max_completion_tokens"}
        # Off by default upstream; pinned off so transcripts are never billed.
        body = {**body, "youtube_transcripts": False}

        resp = await post_billed_request(
            f"{self._config.base_url.rstrip('/')}/{path.strip('/')}",
            _upstream_body(body),
            api_key=self._config.api_key,
            request_timeout=float(self._config.poll_timeout),
            provider_name=self.name,
            label="NanoGPT",
        )
        result = parse_billed_json(resp, self.name)
        raise_for_error_body(result, self.name, "NanoGPT")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        logger.info(
            "NanoGPT request completed: model=%s, tokens=%s",
            body.get("model", "unknown"),
            usage.get("total_tokens", "?"),
        )
        return result

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
        prompt_tokens = int(usage["prompt_tokens"])
        try:
            model_info = await self._model_info(model_id)
            # Only the long-context tier NanoGPT lists is billed, not the hold's.
            rates = self._token_rates(model_id, model_info, prompt_tokens, hold=False)
        except ProviderError:
            return None
        write_rate = rates["cache_write"]
        if _long_cache_requested(body):
            write_rate = max(write_rate, _LONG_CACHE_WRITE * rates["prompt"])

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
        # Audio tokens bill at the audio rate. When the usage reports none
        # for a body with audio, the audio tokens get_price held for the
        # timed parts are priced as audio and the rest of the fresh prompt
        # at the prompt rate; if the audio cannot be timed again, the whole
        # fresh prompt is priced as audio (the app never charges more than
        # the hold).
        audio_ceiling = await held_audio_tokens(usage, body, self._measure_chat_audio, self.name)
        audio_tokens = min(
            int(_billed_audio_tokens(usage, Decimal(prompt_tokens), body, audio_ceiling)),
            fresh_tokens,
        )
        audio_rate = max(rates.get("audio", rates["prompt"]), rates["prompt"])

        actual = (
            Decimal(fresh_tokens - audio_tokens) * rates["prompt"]
            + Decimal(audio_tokens) * audio_rate
            + Decimal(cache_read) * rates["cache_read"]
            + Decimal(cache_write) * write_rate
            + Decimal(completion_tokens) * rates["completion"]
        )
        actual = _apply_cost_floor(actual)
        logger.info(
            "NanoGPT actual cost for %s: $%s (prompt=%d, audio=%d, cache_read=%d, completion=%d)",
            model_id,
            actual,
            prompt_tokens,
            audio_tokens,
            cache_read,
            completion_tokens,
        )
        return actual
