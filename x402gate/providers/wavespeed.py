"""WaveSpeed AI provider for x402gate.

Implements the BaseProvider interface for WaveSpeed AI's API:
- Pricing via POST /api/v3/model/pricing
- Task submission via POST /api/v3/{model-path}
- Result polling via GET /api/v3/predictions/{task-id}/result
- Model catalog via GET /api/v3/models (model types, for the price cache)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import (
    UPSTREAM_CREDENTIALS_DETAIL,
    ProxyError,
    TaskTimeoutError,
    UpstreamMayHaveRun,
    poll_result,
    proxy_request,
    rejected_credentials,
)
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# The app's price cache keys on the whole body, so every new prompt or seed
# cost a pricing API call before the job could start. This cache keys on the
# fields a price can depend on. WaveSpeed prices on parameters such as
# duration, size, aspect ratio and sound, and on the measured length or size of
# input media, so everything but the fields below stays in the key.
#
# A random seed never changes the price.
_NEVER_PRICED_FIELDS = frozenset({"seed"})
# Free text. Image and video models are priced per output, but text-to-speech
# and LLM models are priced per character or token of their text, so these
# leave the key only for an image or video model (by its catalog type) whose
# pricing API gave the same price for a much longer text (see get_price).
_FREE_TEXT_FIELDS = ("prompt", "negative_prompt")
_TEXT_FREE_TYPE_SUFFIXES = ("-image", "-video")
# Appended to each free text field for that check: over 1000 characters, past
# the usual per-1000-characters billing step.
_PROBE_FILLER = " a quiet street at dusk" * 50

# Same as the gateway's default price_cache_ttl. Bounded: unpaid requests
# reach get_price too.
_PRICE_CACHE_TTL_S = 60
_PRICE_CACHE_MAX_ENTRIES = 2048
# How long a "price ignores the text" check holds. A probe that failed gave
# no answer, so the check runs again soon (but not on every request, in case
# the long text is what WaveSpeed rejects).
_TEXT_CHECK_TTL_S = 3600
_FAILED_CHECK_TTL_S = 60
# One check per combination of the other fields, so bounded like the prices.
_TEXT_CHECK_MAX_ENTRIES = 2048
# A failed catalog load is retried on demand at most this often.
_CATALOG_RETRY_S = 60

# Results that came back finished in the submit answer, kept until the app
# asks get_result for them. Bounded in case it never does.
_FINISHED_MAX_ENTRIES = 256

# Inputs with more text than this are hashed off the event loop, as app.py
# does for its own price key: serializing 40 MB of base64 takes tens of ms.
_HASH_IN_THREAD_CHARS = 1024 * 1024
# Inputs with more values than this are hashed off the event loop too,
# without walking the rest of them.
_ROUGH_SIZE_MAX_VALUES = 10_000


def _price_keys(
    model_path: str, inputs: dict[str, Any], text_fields: tuple[str, ...]
) -> tuple[str, str]:
    """Return the cache keys with and without ``text_fields``.

    The other fields are serialized once for both. The key without the text
    names the fields it leaves out, so a check run on a prompt alone is not
    reused for a prompt and a negative prompt, and it never equals a full key.
    """
    rest = {
        key: value
        for key, value in inputs.items()
        if key not in _NEVER_PRICED_FIELDS and key not in text_fields
    }
    rest_hash = hashlib.sha256(json.dumps(rest, sort_keys=True, default=str).encode())
    text_key = f"{model_path}:{rest_hash.hexdigest()[:32]}:without:{','.join(text_fields)}"
    text = {field: inputs[field] for field in text_fields}
    rest_hash.update(json.dumps(text, sort_keys=True).encode())
    full_key = f"{model_path}:{rest_hash.hexdigest()[:32]}"
    return full_key, text_key


def _rough_size(inputs: dict[str, Any], limit: int = _HASH_IN_THREAD_CHARS) -> int:
    """Return about how many characters ``inputs`` serializes to.

    Walks nested dicts and lists, so text hidden in a list of dicts counts
    too. Strings and keys count their length, an int about its digits (a
    4000-digit int takes ms to turn into text), any other value one. Stops
    once the size passes ``limit``, and returns more than ``limit`` after
    ``_ROUGH_SIZE_MAX_VALUES`` values: walking 200 000 small dicts would
    itself stall the event loop for tens of ms.
    """
    size = 0
    queued = 1
    stack: list[Any] = [inputs]
    while stack and size <= limit:
        value = stack.pop()
        if isinstance(value, str):
            size += len(value)
            continue
        if isinstance(value, int):
            # bit_length() // 3 is a little over the digit count, and costs nothing
            size += 1 + value.bit_length() // 3
            continue
        size += 1
        if isinstance(value, dict | list):
            # Checked before copying, so a list of 10 million items is not.
            queued += len(value)
            if queued > _ROUGH_SIZE_MAX_VALUES:
                return limit + 1
            if isinstance(value, dict):
                size += sum(len(key) for key in value if isinstance(key, str))
                stack.extend(value.values())
            else:
                stack.extend(value)
    return size


def _text_fields(inputs: dict[str, Any]) -> tuple[str, ...]:
    """Return the free text fields the request sets as strings."""
    return tuple(field for field in _FREE_TEXT_FIELDS if isinstance(inputs.get(field), str))


def _parse_catalog(response: httpx.Response) -> dict[str, str]:
    """Return model id -> type from a GET /api/v3/models response."""
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    types = {
        model["model_id"]: model["type"]
        for model in models or []
        if isinstance(model, dict)
        and isinstance(model.get("model_id"), str)
        and isinstance(model.get("type"), str)
    }
    if not types:
        raise ValueError(f"WaveSpeed model catalog lists no models: {str(payload)[:200]}")
    return types


class WaveSpeedProvider(BaseProvider):
    """WaveSpeed AI provider implementation.

    Acts as a transparent proxy to WaveSpeed's API. Requests are forwarded
    as-is, and pricing is fetched dynamically from their pricing endpoint.
    """

    def __init__(self, config: ProviderConfig, price_cache_ttl: int = _PRICE_CACHE_TTL_S) -> None:
        super().__init__(name="wavespeed", config=config)
        # The gateway's price_cache_ttl; 0 turns this cache off too.
        self._price_cache_ttl = price_cache_ttl
        self._prices: OrderedDict[str, tuple[Decimal, float]] = OrderedDict()
        # Price key without the text -> (price ignores the text, valid until)
        self._text_checks: OrderedDict[str, tuple[bool, float]] = OrderedDict()
        # Price key without the text -> its probe still running
        self._probes: dict[str, asyncio.Task[Decimal]] = {}
        self._model_types: dict[str, str] = {}
        self._catalog_loaded_at: float | None = None
        self._catalog_tried_at: float | None = None
        self._catalog_task: asyncio.Task[None] | None = None
        self._finished: OrderedDict[str, dict[str, Any]] = OrderedDict()

    async def warm_up(self) -> None:
        """Load the model catalog, which also opens a pooled connection."""
        await self._load_catalog()

    async def close(self) -> None:
        """Stop a catalog refresh and probes still running.

        They are awaited, so none is left using the shared client the app
        closes next.
        """
        tasks = [*self._probes.values()]
        if self._catalog_task is not None:
            tasks.append(self._catalog_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Return the price of a request, from cache or WaveSpeed's Pricing API.

        A cached quote for the same model and price-relevant fields is reused
        for the gateway's ``price_cache_ttl``. The prompt counts as
        price-relevant until the model is known to ignore it with these other
        fields: on the first miss for an image or video model, a quote for a
        much longer prompt is fetched next to the real one, and an equal
        answer lets later requests with the same other fields and any prompt
        reuse the quote. That probe runs in the background, so the request
        never waits for it.

        Args:
            model_path: Model path (e.g. 'wavespeed-ai/flux-dev').
            inputs: Request parameters to price.

        Returns:
            Base price in USD as a Decimal.

        Raises:
            ProviderError: If the pricing API call fails, or with 400 for a
                multipart upload.
        """
        self._reject_upload(inputs)
        self._refresh_catalog_soon()
        text_fields = _text_fields(inputs)
        # text_key also names the check: a prompt may count only while some
        # other option is on, so a check holds only for the fields it was run with.
        if _rough_size(inputs) > _HASH_IN_THREAD_CHARS:
            full_key, text_key = await asyncio.to_thread(
                _price_keys, model_path, inputs, text_fields
            )
        else:
            full_key, text_key = _price_keys(model_path, inputs, text_fields)
        text_free = self._text_check(text_key) if text_fields else None

        cached = self._cached_price(text_key if text_free else full_key)
        if cached is not None:
            logger.debug("WaveSpeed price for %s from cache: $%s", model_path, cached)
            return cached

        probe = None
        if (
            text_free is None
            and text_fields
            and self._price_cache_ttl > 0
            and text_key not in self._probes
            and self._priced_per_output(model_path)
        ):
            probe = self._start_probe(model_path, inputs, text_fields, text_key)
        try:
            price = await self._fetch_price(model_path, inputs)
        except BaseException:
            if probe is not None:
                probe.cancel()
            raise
        quoted_at = time.monotonic()

        self._cache_price(full_key, price, quoted_at)
        if text_free:
            self._cache_price(text_key, price, quoted_at)
        elif probe is not None:
            probe.add_done_callback(
                functools.partial(
                    self._record_text_check, model_path, text_fields, text_key, price, quoted_at
                )
            )
        return price

    def _start_probe(
        self,
        model_path: str,
        inputs: dict[str, Any],
        text_fields: tuple[str, ...],
        text_key: str,
    ) -> asyncio.Task[Decimal]:
        """Start a background quote for the same request with longer text."""
        probe_inputs = {**inputs}
        for field in text_fields:
            probe_inputs[field] = inputs[field] + _PROBE_FILLER
        # One attempt only: it only decides whether later quotes can be reused
        probe = asyncio.create_task(self._fetch_price(model_path, probe_inputs, max_retries=0))
        self._probes[text_key] = probe

        def forget(task: asyncio.Task[Decimal]) -> None:
            if self._probes.get(text_key) is task:
                del self._probes[text_key]
            # Read the probe's error even when nobody awaits it
            if not task.cancelled():
                task.exception()

        probe.add_done_callback(forget)
        return probe

    def _record_text_check(
        self,
        model_path: str,
        text_fields: tuple[str, ...],
        text_key: str,
        price: Decimal,
        quoted_at: float,
        probe: asyncio.Task[Decimal],
    ) -> None:
        """Compare a finished probe with the real quote and keep the answer."""
        if probe.cancelled():
            return  # checked again on the next miss
        now = time.monotonic()
        if probe.exception() is not None:
            # A failed probe is no answer, so the text stays in the key for now
            self._store_text_check(text_key, False, now + _FAILED_CHECK_TTL_S)
            return
        text_free = probe.result() == price
        self._store_text_check(text_key, text_free, now + _TEXT_CHECK_TTL_S)
        if text_free:
            # Ages from the real quote, like the full-key entry
            self._cache_price(text_key, price, quoted_at)
        logger.info(
            "WaveSpeed %s price %s on %s",
            model_path,
            "ignores" if text_free else "may depend",
            "/".join(text_fields),
        )

    async def _fetch_price(
        self, model_path: str, inputs: dict[str, Any], max_retries: int = 2
    ) -> Decimal:
        """Fetch a quote from POST /api/v3/model/pricing."""
        try:
            pricing_body = {
                "model_id": model_path,
                "inputs": inputs,
            }
            result = await proxy_request(
                base_url=self._config.base_url,
                path="model/pricing",
                body=pricing_body,
                api_key=self._config.api_key,
                max_retries=max_retries,
                # A price quote starts no job, so a 5xx can be retried.
                idempotent=True,
            )

            data = result.get("data", result)
            unit_price = data.get("unit_price")

            if unit_price is None:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Pricing API returned no unit_price: {result}",
                )

            price = Decimal(str(unit_price))
            logger.info("WaveSpeed price for %s: $%s", model_path, price)
            return price

        except ProviderError:
            raise
        except Exception as e:
            self._raise_if_key_rejected(e)
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to fetch pricing: {e}",
                status_code=503,
            ) from e

    def _cached_price(self, key: str) -> Decimal | None:
        """Return a cached quote that has not expired."""
        entry = self._prices.get(key)
        if entry is None:
            return None
        price, stored_at = entry
        if time.monotonic() - stored_at > self._price_cache_ttl:
            del self._prices[key]
            return None
        return price

    def _cache_price(self, key: str, price: Decimal, quoted_at: float) -> None:
        """Store a quote, dropping the oldest once the cache is full."""
        if self._price_cache_ttl <= 0:
            return
        self._prices[key] = (price, quoted_at)
        self._prices.move_to_end(key)
        while len(self._prices) > _PRICE_CACHE_MAX_ENTRIES:
            self._prices.popitem(last=False)

    def _text_check(self, text_key: str) -> bool | None:
        """Return whether the price ignores the text with these other fields, if known."""
        entry = self._text_checks.get(text_key)
        if entry is None or time.monotonic() > entry[1]:
            return None
        return entry[0]

    def _store_text_check(self, text_key: str, text_free: bool, valid_until: float) -> None:
        """Keep a check's answer, dropping the oldest once there are too many."""
        self._text_checks[text_key] = (text_free, valid_until)
        self._text_checks.move_to_end(text_key)
        while len(self._text_checks) > _TEXT_CHECK_MAX_ENTRIES:
            self._text_checks.popitem(last=False)

    def _priced_per_output(self, model_path: str) -> bool:
        """Return True for an image or video model, which is not priced on its text.

        A model the catalog does not list (or a catalog not loaded yet) is not.
        """
        model_type = self._model_types.get(model_path, "")
        return model_type.endswith(_TEXT_FREE_TYPE_SUFFIXES)

    async def _load_catalog(self) -> None:
        """Load model types from GET /api/v3/models."""
        self._catalog_tried_at = time.monotonic()
        response = await http.shared_client().get(
            f"{self._config.base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            timeout=http.timeout(30.0),
        )
        if response.status_code != 200:
            raise ProviderError(
                provider=self.name,
                detail=f"Model catalog returned HTTP {response.status_code}: {response.text[:200]}",
            )
        # The catalog runs to several MB; parsing it on the loop would stall
        # every request in flight.
        types = await asyncio.to_thread(_parse_catalog, response)
        self._model_types = types
        self._catalog_loaded_at = time.monotonic()
        logger.info("WaveSpeed catalog loaded: %d models", len(types))

    def _refresh_catalog_soon(self) -> None:
        """Reload a missing or stale catalog in the background."""
        now = time.monotonic()
        if (
            self._catalog_loaded_at is not None
            and now - self._catalog_loaded_at < self._config.models_cache_ttl
        ):
            return
        if self._catalog_task is not None and not self._catalog_task.done():
            return
        if self._catalog_tried_at is not None and now - self._catalog_tried_at < _CATALOG_RETRY_S:
            return
        self._catalog_task = asyncio.create_task(self._refresh_catalog())

    async def _refresh_catalog(self) -> None:
        try:
            await self._load_catalog()
        except Exception:
            logger.warning("WaveSpeed catalog load failed; prompts stay in the price key")

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a task to WaveSpeed's API.

        Forwards the request body to POST /api/v3/{path}, without
        ``enable_sync_mode``.

        Args:
            path: Model path (e.g. 'wavespeed-ai/flux-dev').
            body: Request body, forwarded as-is but for ``enable_sync_mode``.

        Returns:
            WaveSpeed response dict with task ID and status.

        Raises:
            ProviderError: If the submission fails, or WaveSpeed accepted it
                without a task id to poll, or with 400 for a multipart upload.
        """
        self._reject_upload(body)
        # Sync mode holds the submit open until the output is ready (up to
        # 110 s), past the 30 s read timeout, and WaveSpeed may answer 504
        # first. Either one ends it with no task id while the job still runs
        # and bills, so the payment is kept for a result nobody can poll. The
        # gateway polls anyway, from 0.3 s after submit.
        body = {key: value for key, value in body.items() if key != "enable_sync_mode"}
        try:
            result = await proxy_request(
                base_url=self._config.base_url,
                path=path,
                body=body,
                api_key=self._config.api_key,
            )
            # Read the body the way app.py does to tell async from sync.
            data = result.get("data", result)
            is_task = isinstance(data, dict) and "id" in data and "status" in data
            is_sync = isinstance(data, dict) and ("outputs" in data or "output" in data)
            if not (is_task or is_sync):
                # With neither a task id nor an output the gateway has nothing
                # to poll or deliver, so this is no success. WaveSpeed answered
                # 2xx, so a task may still exist: the payment is kept.
                raise ProviderError(
                    provider=self.name,
                    detail=f"WaveSpeed accepted the request but returned no task id: "
                    f"{str(result)[:200]}",
                ) from UpstreamMayHaveRun("No task id in the submit response")
            logger.info("WaveSpeed task submitted for %s: %s", path, data.get("id", "sync"))
            if is_task:
                self._remember_if_finished(data)
            return result

        except ProviderError:
            raise
        except Exception as e:
            self._raise_if_key_rejected(e)
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to submit task: {e}",
            ) from e

    def _reject_upload(self, body: dict[str, Any]) -> None:
        """Refuse a multipart upload: WaveSpeed takes JSON, with media as URLs.

        The file bytes cannot be sent as JSON, and hashing them into the price
        key would stall the event loop for seconds on a large upload.
        """
        if "_file_bytes" in body:
            raise ProviderError(
                provider=self.name,
                detail="WaveSpeed takes a JSON body. Multipart uploads are not accepted.",
                status_code=400,
            )

    def _raise_if_key_rejected(self, exc: Exception) -> None:
        """Turn an upstream 401/402/403 into a 502 that hides WaveSpeed's text.

        A 402 passed on would read to x402 clients as the gateway asking for
        payment, and 401/403 would tell them our key is bad.
        """
        if isinstance(exc, ProxyError) and rejected_credentials(exc.status_code):
            logger.error("WaveSpeed rejected our key (%d): %s", exc.status_code, exc.detail)
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL) from exc

    def _remember_if_finished(self, data: dict[str, Any]) -> None:
        """Keep a task the submit answer already shows finished.

        The app polls every answer with an id and a status; get_result then
        returns this instead of asking WaveSpeed again. A sync wait that ran
        out answers "processing" with no outputs and is polled as usual.
        """
        task_id = data.get("id")
        outputs = data.get("outputs")
        if (
            isinstance(task_id, str)
            and data.get("status") == "completed"
            and isinstance(outputs, list)
            and outputs
        ):
            self._finished[task_id] = data
            while len(self._finished) > _FINISHED_MAX_ENTRIES:
                self._finished.popitem(last=False)

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll WaveSpeed for task completion.

        Polls GET /api/v3/predictions/{task_id}/result until the task
        completes, fails, or times out. A task the submit answer already
        returned finished is returned without polling.

        Args:
            task_id: Task ID from submit() response.

        Returns:
            Completed task data including outputs.

        Raises:
            ProviderError: If the task fails.
            TaskTimeoutError: If polling exceeds poll_timeout.
        """
        finished = self._finished.pop(task_id, None)
        if finished is not None:
            logger.info("WaveSpeed task %s finished at submit; not polled", task_id)
            return finished
        try:
            return await poll_result(
                base_url=self._config.base_url,
                task_id=task_id,
                api_key=self._config.api_key,
                poll_interval=self._config.poll_interval,
                poll_timeout=self._config.poll_timeout,
            )
        except TaskTimeoutError:
            raise
        except ProxyError as e:
            logger.error("Task %s failed: %s", task_id, e.detail)
            raise ProviderError(
                provider=self.name,
                detail=e.detail,
                status_code=e.status_code,
            ) from e
        except Exception as e:
            logger.error("Task %s error: %s", task_id, e)
            raise ProviderError(
                provider=self.name,
                detail=str(e),
            ) from e
