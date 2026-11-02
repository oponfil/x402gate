"""SpicyAPI provider for x402gate.

Image and video jobs use SpicyAPI's task API:

- price via POST /api/v1/jobs/quote (no hold)
- submit via POST /api/v1/jobs/createTask with that quote
- poll via GET /api/v1/jobs/recordInfo

The client calls ``POST /v1/spicyapi/jobs`` with ``{model, input}``.
The gateway adds ``quoteId`` and ``expectedCost`` and, for images, holds
the create call briefly so a fast job can return without a poll loop.

Text models use the same quote, then ``POST /api/v1/jobs/stream``. The
client calls ``POST /v1/spicyapi/chat/completions`` with an OpenAI chat
body. The gateway returns one JSON completion after the stream finishes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, NoReturn

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.http import shared_client
from x402gate.core.proxy import (
    _GATEWAY_TIMEOUT_STATUSES,
    _NOT_FOUND_GRACE_POLLS,
    _NOT_FOUND_GRACE_S,
    UPSTREAM_CREDENTIALS_DETAIL,
    TaskTimeoutError,
    UpstreamMayHaveRun,
    rejected_credentials,
)
from x402gate.providers.base import BaseProvider, ProviderError, outcome_unknown

logger = logging.getLogger(__name__)

_JOBS_PATH = "jobs"
_CHAT_PATH = "chat/completions"
# Only these mean SpicyAPI gave up on the task. A missing or unlisted state
# may be a job still running (and billing), so it is polled like a pending one.
_FAILED_STATES = frozenset({"failed", "canceled", "cancelled", "expired"})
_QUOTE_SKEW_SECONDS = 20
_IMAGE_WAIT_SECONDS = 45
_IMAGE_REQUEST_TIMEOUT = 75.0
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 2
_RETRY_DELAY = 2.0
# Pauses between reads of a finished chat's settled cost. SpicyAPI settles a
# moment after the stream ends, so the first read goes out at once and the
# next ones follow quickly. They add up to about the two poll intervals
# (2 s each) the lookup used to wait, so a settlement that lands as late as
# before is still found and billed exactly instead of at the quote.
_COST_LOOKUP_DELAYS = (0.25, 0.5, 1.0, 2.0)
# Unpaid quote requests fill the quote and price caches, so both are bounded.
# A price outlives the gateway's price cache (60s by default) and has the same
# entry cap, so a price served from that cache still finds its record here.
_PRICED_TTL_SECONDS = 3600
_CACHE_MAX_ENTRIES = 10_000
# Same as the gateway's default price_cache_ttl. The gateway caches prices by
# raw body, so a price it showed may still be charged that long after a newer
# quote for an equivalent body. The grace covers the time from that cache
# read to submit: payment checks and, on Solana, settlement, whose confirm
# wait alone may run svm_signer._CONFIRM_RUN_TIMEOUT_S (150 s). Only used when
# submit is not told the price the client was charged.
_PRICE_CACHE_TTL_S = 60
_PRICE_GRACE_SECONDS = 180
_POLL_BACKOFF_MIN = 1.0
_POLL_BACKOFF_MAX = 30.0

_INTERNAL_FIELDS = frozenset({"_caller", "_file_bytes", "_file_name"})
_REQUEST_FIELDS = frozenset({"model", "input", "callBackUrl"})
_CHAT_INPUT_FIELDS = frozenset(
    {
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "response_format",
        "reasoning_effort",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
    }
)
_CHAT_IGNORED_FIELDS = frozenset({"stream_options", "user", "metadata", "store"})


@dataclass
class _Quote:
    """A SpicyAPI price quote bound to one exact request."""

    quote_id: str
    cost: Decimal
    cost_text: str
    unit: str
    expires_at: float


class SpicyApiProvider(BaseProvider):
    """SpicyAPI image and video provider.

    Quotes every request before the client pays, then submits that same
    quote so SpicyAPI cannot accept a higher price than the one shown.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        default_max_tokens: int = 2048,
        price_cache_ttl: int = _PRICE_CACHE_TTL_S,
    ) -> None:
        super().__init__(name="spicyapi", config=config)
        self._default_max_tokens = default_max_tokens
        # How long a price shown for a request key may still be charged.
        self._price_window = max(price_cache_ttl, 0) + _PRICE_GRACE_SECONDS
        self._quotes: dict[str, _Quote] = {}
        # Prices shown per request key as (cost, monotonic time), oldest and
        # cheapest first; the last one is the latest. See _remember_price.
        self._priced: dict[str, list[tuple[Decimal, float]]] = {}
        self._actual_costs: dict[str, Decimal] = {}

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Quote the exact job or chat. The quote is reused when the client pays."""
        payload = self._payload(model_path, inputs)
        quote = await self._quote(payload)
        self._remember_price(self._cache_key(payload), quote.cost)
        logger.info(
            "SpicyAPI quote for %s: $%s (%s)",
            payload["model"],
            quote.cost_text,
            quote.unit or "unit",
        )
        return quote.cost

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
        paid_price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Run a quoted image, video, or chat request.

        Image jobs pass ``wait`` so a finished image comes back on this
        call. Video jobs return immediately and are polled by ``get_result``.
        Chat is streamed at the quoted price and returned as one completion.
        ``paid_price`` is the base price the request was charged at; without
        it the quote is checked against the prices recently shown.
        """
        payload = self._payload(path, body)
        if path.strip("/") == _CHAT_PATH:
            return await self._submit_chat(payload, paid_price)
        quote = await self._quote_for_submit(payload, paid_price)
        wait = _IMAGE_WAIT_SECONDS if quote.unit == "per_image" else None
        try:
            record = await self._create_task(payload, quote, wait=wait)
        except ProviderError as exc:
            # A new quote means a new Idempotency-Key, so a create that may
            # already have made the task is never sent again.
            if outcome_unknown(exc):
                raise
            if "40901" not in exc.detail and "no longer" not in exc.detail.lower():
                raise
            logger.info("SpicyAPI quote rejected, quoting %s again", payload["model"])
            self._quotes.pop(self._cache_key(payload), None)
            quote = await self._quote_for_submit(payload, paid_price)
            record = await self._create_task(payload, quote, wait=wait)
        return self._outcome(record)

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll recordInfo until the task succeeds or SpicyAPI marks it failed.

        The task was accepted at createTask, so only a task SpicyAPI marks
        failed is a known failure. A task that stays missing, a rejected key
        or a redirect is an unknown outcome: the job may still run, and bill.
        """
        interval = self._config.poll_interval
        deadline = time.monotonic() + self._config.poll_timeout
        not_found = 0  # consecutive not-found answers
        not_found_since = 0.0  # monotonic time of the first of them
        transient = 0

        while time.monotonic() < deadline:
            try:
                response = await self._send(
                    "GET",
                    "/api/v1/jobs/recordInfo",
                    params={"taskId": task_id},
                    request_timeout=_DEFAULT_TIMEOUT,
                    retry=False,
                    check=False,
                )
            except ProviderError:
                # A timeout or dropped connection says nothing about the job.
                transient += 1
                await self._sleep(_poll_backoff(interval, transient, deadline))
                continue

            status = _poll_status(response)
            if status != 404:
                not_found = 0
            if status == 429 or status >= 500:
                # A rate limit or an upstream hiccup, as an HTTP status, an
                # envelope code or a proxy's non-JSON page, says nothing about
                # the job, which keeps running and billing. Back off and poll
                # again; running out of time is an unknown outcome.
                transient += 1
                await self._sleep(_poll_backoff(interval, transient, deadline))
                continue
            if status == 404:
                # A new task may not be readable yet, so a missing task counts
                # as lost only after a grace period, as in proxy.poll_result.
                now = time.monotonic()
                if not not_found:
                    not_found_since = now
                not_found += 1
                if not_found > _NOT_FOUND_GRACE_POLLS and now - not_found_since >= (
                    _NOT_FOUND_GRACE_S
                ):
                    logger.error("SpicyAPI no longer finds task %s", task_id)
                    raise ProviderError(
                        provider=self.name,
                        detail=f"SpicyAPI no longer finds task {task_id}",
                        status_code=502,
                    ) from UpstreamMayHaveRun(f"Task {task_id} not found")
                await self._sleep(interval)
                continue
            if rejected_credentials(status):
                logger.error("SpicyAPI rejected our key while polling task %s", task_id)
                raise ProviderError(
                    provider=self.name,
                    detail=UPSTREAM_CREDENTIALS_DETAIL,
                    status_code=502,
                ) from UpstreamMayHaveRun(f"Poll of task {task_id} rejected our key")
            if 300 <= status < 400:
                # httpx does not follow redirects, so the task cannot be read.
                raise ProviderError(
                    provider=self.name,
                    detail=f"SpicyAPI redirected the poll of task {task_id}",
                    status_code=502,
                ) from UpstreamMayHaveRun(f"Poll redirect {status}")
            if status >= 400:
                # Any other request error will not change on retry. It mirrors
                # proxy.poll_result's rule for a 4xx while polling.
                raise self._error_from_response(response)

            try:
                data = _envelope_data(response)
            except ProviderError:
                # A garbled 2xx says nothing about the task; ask again.
                transient += 1
                await self._sleep(_poll_backoff(interval, transient, deadline))
                continue
            transient = 0

            state = str(data.get("state") or "")
            if state == "succeeded":
                self._remember_cost(task_id, data.get("cost"))
                logger.info("SpicyAPI task %s succeeded", task_id)
                return data
            if state in _FAILED_STATES:
                self._log_terminal_failure(task_id, data)
                raise ProviderError(
                    provider=self.name,
                    detail=_failure_detail(data),
                    status_code=502,
                )

            await self._sleep(interval)

        raise TaskTimeoutError(task_id=task_id, timeout=self._config.poll_timeout)

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Return the settled SpicyAPI cost when the task recorded one."""
        if isinstance(result, dict) and result.get("object") == "chat.completion":
            task_id = str(result.get("task_id") or "")
            if task_id and task_id in self._actual_costs:
                return self._actual_costs.pop(task_id)
            return None
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            return None
        task_id = str(data.get("id") or data.get("taskId") or "")
        if task_id and task_id in self._actual_costs:
            return self._actual_costs.pop(task_id)
        if data.get("state") == "succeeded" and data.get("cost") is not None:
            return _parse_cost(data.get("cost"))
        return None

    async def close(self) -> None:
        """Nothing to close: calls use the gateway's shared pool, closed on shutdown."""

    def _payload(self, model_path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Validate the gateway body and return the quoted SpicyAPI request."""
        path = model_path.strip("/")
        if path == _CHAT_PATH:
            return self._chat_request(body)
        if path == _JOBS_PATH:
            return self._task_request(model_path, body)
        raise ProviderError(
            provider=self.name,
            detail=(
                "SpicyAPI image and video jobs use POST /v1/spicyapi/jobs. "
                "Text models use POST /v1/spicyapi/chat/completions."
            ),
            status_code=400,
        )

    def _chat_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Turn an OpenAI chat body into a quoted SpicyAPI text request."""
        if "_file_bytes" in body:
            raise ProviderError(
                provider=self.name,
                detail="Chat requests are JSON. Multipart uploads are not accepted.",
                status_code=400,
            )
        if body.get("stream") not in (None, False):
            raise ProviderError(
                provider=self.name,
                detail="Streaming is not supported. Omit stream or set it to false.",
                status_code=400,
            )
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain a non-empty 'model' string",
                status_code=400,
            )
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain a non-empty 'messages' list",
                status_code=400,
            )
        if any(not isinstance(message, dict) for message in messages):
            raise ProviderError(
                provider=self.name,
                detail="Each message must be an object",
                status_code=400,
            )
        raw_n = body.get("n", 1)
        if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n != 1:
            raise ProviderError(
                provider=self.name,
                detail="n must be 1",
                status_code=400,
            )
        max_tokens = _chat_max_tokens(body, self._default_max_tokens)
        allowed = _CHAT_INPUT_FIELDS | _CHAT_IGNORED_FIELDS | _INTERNAL_FIELDS
        allowed |= {"model", "stream", "max_completion_tokens", "n"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise ProviderError(
                provider=self.name,
                detail=f"Unknown fields: {', '.join(unknown)}.",
                status_code=400,
            )
        task_input: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        for field in _CHAT_INPUT_FIELDS - {"messages", "max_tokens"}:
            if field in body:
                task_input[field] = body[field]
        return {"model": model, "input": task_input}

    async def _submit_chat(
        self, payload: dict[str, Any], paid_price: Decimal | None
    ) -> dict[str, Any]:
        """Stream a quoted chat and return one OpenAI completion."""
        quote = await self._quote_for_submit(payload, paid_price)
        try:
            completion, task_id = await self._stream_chat(payload, quote)
        except ProviderError as exc:
            # A stream cut off after output may have run, so it is never re-sent.
            if outcome_unknown(exc):
                raise
            if "40901" not in exc.detail and "no longer" not in exc.detail.lower():
                raise
            logger.info("SpicyAPI chat quote rejected, quoting %s again", payload["model"])
            self._quotes.pop(self._cache_key(payload), None)
            quote = await self._quote_for_submit(payload, paid_price)
            completion, task_id = await self._stream_chat(payload, quote)
        cost = await self._lookup_settled_cost(task_id)
        if cost is None:
            cost = quote.cost
        if task_id:
            self._remember_cost(task_id, cost)
            completion["task_id"] = task_id
        return completion

    async def _stream_chat(
        self,
        payload: dict[str, Any],
        quote: _Quote,
    ) -> tuple[dict[str, Any], str]:
        """POST jobs/stream and fold the SSE chunks into one completion."""
        request_body = {
            **payload,
            "quoteId": quote.quote_id,
            "expectedCost": quote.cost_text,
        }
        url = f"{self._config.base_url.rstrip('/')}/api/v1/jobs/stream"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4()),
        }
        timeout = httpx.Timeout(
            30.0,
            connect=http.CONNECT_TIMEOUT_S,
            read=float(self._config.poll_timeout),
        )
        client = self._get_client()
        try:
            # The read timeout is per read, and keepalive comments reset it,
            # so the whole stream gets the same limit as a polled job.
            async with asyncio.timeout(float(self._config.poll_timeout)):
                async with client.stream(
                    "POST",
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        error = self._error_from_response(response)
                        if response.status_code in _GATEWAY_TIMEOUT_STATUSES:
                            # A proxy gave up waiting; SpicyAPI may still run the chat.
                            raise error from UpstreamMayHaveRun(
                                f"jobs/stream gateway timeout {response.status_code}"
                            )
                        raise error
                    task_id = response.headers.get("x-spicy-task-id") or ""
                    chunks = await _read_sse_chunks(response)
        except ProviderError:
            raise
        except TimeoutError as exc:
            # The request was sent, so the chat may run and bill, as with a
            # polled job that times out.
            raise ProviderError(
                provider=self.name,
                detail=f"SpicyAPI chat did not finish within {self._config.poll_timeout}s",
                status_code=504,
            ) from UpstreamMayHaveRun(f"jobs/stream exceeded poll_timeout: {exc!r}")
        except httpx.TimeoutException as exc:
            raise ProviderError(
                provider=self.name,
                detail=f"SpicyAPI chat timed out after {self._config.poll_timeout}s",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to connect to SpicyAPI: {exc}",
                status_code=503,
            ) from exc
        completion = _completion_from_chunks(chunks, payload["model"])
        logger.info(
            "SpicyAPI chat %s for %s finished",
            task_id or completion.get("id") or "unknown",
            payload["model"],
        )
        return completion, task_id

    async def _lookup_settled_cost(self, task_id: str) -> Decimal | None:
        """Read the settled chat cost once the stream has finished."""
        if not task_id:
            return None
        delays = (0.0, *_COST_LOOKUP_DELAYS)
        for attempt, delay in enumerate(delays):
            await self._sleep(delay)
            try:
                response = await self._send(
                    "GET",
                    "/api/v1/jobs/recordInfo",
                    params={"taskId": task_id},
                    request_timeout=_DEFAULT_TIMEOUT,
                    retry=False,
                    check=False,
                )
            except ProviderError as exc:
                # A read that timed out already waited its full timeout, so
                # the quote is used rather than making the client wait again.
                # A connection that failed or dropped is read again.
                if isinstance(exc.__cause__, httpx.TimeoutException):
                    return None
                logger.debug("SpicyAPI cost read %s for %s failed: %s", attempt + 1, task_id, exc)
                continue
            status = response.status_code
            if not 200 <= status < 300:
                if status in (404, 429) or status >= 500:
                    # Not readable yet, rate limited or a hiccup: read again.
                    logger.debug(
                        "SpicyAPI cost read %s for %s got %s", attempt + 1, task_id, status
                    )
                    continue
                # A rejected key or a bad request will not change on another read.
                return None
            try:
                data = _envelope_data(response)
            except ProviderError:
                # The chat already ran and billed; an unreadable record (a task
                # not readable yet, a 5xx code, a non-JSON page) only means the
                # quoted cost may be used instead, never a failed request.
                data = {}
            cost = _parse_cost(data.get("cost"))
            if data.get("settled") is True and cost is not None:
                return cost
            if data.get("state") == "succeeded" and cost is not None:
                return cost
            logger.debug("SpicyAPI cost of %s not settled (read %s)", task_id, attempt + 1)
        return None

    def _task_request(self, model_path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Validate the gateway body and return the SpicyAPI task request."""
        if model_path.strip("/") != _JOBS_PATH:
            raise ProviderError(
                provider=self.name,
                detail=(
                    "SpicyAPI image and video jobs use POST /v1/spicyapi/jobs. "
                    "Text models use POST /v1/spicyapi/chat/completions."
                ),
                status_code=400,
            )
        if "_file_bytes" in body:
            raise ProviderError(
                provider=self.name,
                detail=(
                    "Multipart uploads are not accepted. Put an https or spicy:// "
                    "URI in the model input."
                ),
                status_code=400,
            )
        unknown = sorted(set(body) - _INTERNAL_FIELDS - _REQUEST_FIELDS)
        if unknown:
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"Unknown fields: {', '.join(unknown)}. "
                    "Send model, input, and an optional callBackUrl."
                ),
                status_code=400,
            )
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain a non-empty 'model' string",
                status_code=400,
            )
        task_input = body.get("input")
        if not isinstance(task_input, dict):
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain an 'input' object",
                status_code=400,
            )
        payload: dict[str, Any] = {"model": model, "input": task_input}
        callback = body.get("callBackUrl")
        if callback is not None:
            if not isinstance(callback, str) or not callback.startswith("https://"):
                raise ProviderError(
                    provider=self.name,
                    detail="callBackUrl must be a public https URL",
                    status_code=400,
                )
            payload["callBackUrl"] = callback
        return payload

    async def _quote_for_submit(
        self, payload: dict[str, Any], paid_price: Decimal | None = None
    ) -> _Quote:
        """Reuse the price shown to the client, or quote again if it is stale.

        A fresh quote that costs more than the price the client paid is
        rejected, so a paid retry cannot be accepted above that amount. When
        that amount is not passed in, the lowest price recently shown stands
        in for it. A request with neither is refused before anything runs:
        the amount the client paid is unknown, so no quote can be checked.
        """
        key = self._cache_key(payload)
        priced = paid_price if paid_price is not None else self._price_for(key)
        if priced is None:
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"SpicyAPI price for {payload['model']} is no longer on record. "
                    "Retry the request."
                ),
                status_code=409,
            )
        quote = self._quotes.get(key)
        if quote is None or quote.expires_at <= time.time() + _QUOTE_SKEW_SECONDS:
            quote = await self._quote(payload)
        if quote.cost > priced:
            raise ProviderError(
                provider=self.name,
                detail=(
                    f"SpicyAPI price for {payload['model']} increased from "
                    f"${priced} to ${quote.cost_text}. Retry the request to "
                    "pay the new price."
                ),
                status_code=409,
            )
        return quote

    async def _quote(self, payload: dict[str, Any]) -> _Quote:
        """Ask SpicyAPI for a five-minute quote and remember it."""
        try:
            response = await self._send(
                "POST",
                "/api/v1/jobs/quote",
                json_body=payload,
                request_timeout=_DEFAULT_TIMEOUT,
            )
        except ProviderError as exc:
            failure = exc
        else:
            failure = None
        if failure is not None:
            # A quote starts no job, so even one that timed out after it was
            # sent is a known failure. Raised outside the except block, the
            # transport error is in neither __cause__ nor __context__.
            raise ProviderError(
                provider=self.name,
                detail=failure.detail,
                status_code=failure.status_code,
            )
        data = _envelope_data(response)
        cost_text = data.get("estimatedCost")
        cost = _parse_cost(cost_text)
        quote_id = data.get("quoteId")
        if cost is None or not isinstance(quote_id, str) or not quote_id:
            raise ProviderError(
                provider=self.name,
                detail=f"SpicyAPI quote response missing price: {data}",
                status_code=502,
            )
        quote = _Quote(
            quote_id=quote_id,
            cost=cost,
            cost_text=str(cost_text),
            unit=str(data.get("unit") or ""),
            expires_at=_expires_at(data.get("expiresAt")),
        )
        key = self._cache_key(payload)
        self._drop_expired(self._quotes)
        self._quotes.pop(key, None)
        while len(self._quotes) >= _CACHE_MAX_ENTRIES:
            del self._quotes[next(iter(self._quotes))]
        self._quotes[key] = quote
        return quote

    def _remember_price(self, key: str, cost: Decimal) -> None:
        """Record the price shown for a request. Old and excess entries go first.

        Equivalent bodies share a key here but not in the gateway's price
        cache, so a lower price shown a moment ago may still be what a client
        pays. A newer, higher quote must not raise the guard for it: lower
        prices inside the window are kept. A price that is not lower than a
        newer one can never be the minimum, so it is dropped.
        """
        now = time.monotonic()
        # Re-inserting moves the key to the end, so dict order stays oldest first.
        shown = self._priced.pop(key, [])
        shown = [entry for entry in shown if now - entry[1] <= self._price_window]
        while shown and shown[-1][0] >= cost:
            shown.pop()
        shown.append((cost, now))
        while self._priced:
            oldest = next(iter(self._priced))
            stored_at = self._priced[oldest][-1][1]
            if now - stored_at <= _PRICED_TTL_SECONDS and len(self._priced) < _CACHE_MAX_ENTRIES:
                break
            del self._priced[oldest]
        self._priced[key] = shown

    def _price_for(self, key: str) -> Decimal | None:
        """Return the lowest price a client may have paid for a request.

        That is the lowest price shown inside the window, else the latest
        one. None if it is gone or stale.
        """
        shown = self._priced.get(key)
        now = time.monotonic()
        if not shown or now - shown[-1][1] > _PRICED_TTL_SECONDS:
            return None
        for cost, shown_at in shown:
            if now - shown_at <= self._price_window:
                return cost
        return shown[-1][0]

    async def _create_task(
        self,
        payload: dict[str, Any],
        quote: _Quote,
        *,
        wait: int | None,
    ) -> dict[str, Any]:
        """POST createTask. Retries reuse one Idempotency-Key."""
        request_body = {
            **payload,
            "quoteId": quote.quote_id,
            "expectedCost": quote.cost_text,
        }
        params = {"wait": wait} if wait else None
        timeout = _IMAGE_REQUEST_TIMEOUT if wait else _DEFAULT_TIMEOUT
        idempotency_key = str(uuid.uuid4())
        response, may_have_run = await self._send_tracked(
            "POST",
            "/api/v1/jobs/createTask",
            json_body=request_body,
            params=params,
            request_timeout=timeout,
            extra_headers={"Idempotency-Key": idempotency_key},
            billable=True,
        )
        data = self._created_record(response, may_have_run)
        logger.info(
            "SpicyAPI task %s for %s (%s)",
            data["taskId"],
            payload["model"],
            data.get("state") or "unknown",
        )
        return data

    def _created_record(self, response: httpx.Response, may_have_run: str) -> dict[str, Any]:
        """Return the task record from a 2xx createTask answer.

        SpicyAPI accepted the call, so an answer without a readable task is an
        unknown outcome: the task may exist and bill. The exception is a
        4xx-class envelope code (40001 invalid input, 40201/40202 balance,
        40901 stale quote): SpicyAPI uses those to refuse a job before it
        creates one, as it does with the same codes on an HTTP 4xx, and 40901
        must stay a known failure so submit can quote again.
        """
        try:
            data = _envelope_data(response)
        except ProviderError as exc:
            status = _envelope_status(response)
            if status is not None and 400 <= status < 500 and status != 429:
                error = self._error_from_response(response)
                if may_have_run:
                    raise error from UpstreamMayHaveRun(may_have_run)
                raise error from None
            raise ProviderError(
                provider=self.name,
                detail=f"SpicyAPI accepted the job but its answer was unreadable: {exc.detail}",
                status_code=502,
            ) from UpstreamMayHaveRun(may_have_run or "Unreadable createTask answer")
        task_id = data.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError(
                provider=self.name,
                detail="SpicyAPI response missing taskId",
                status_code=502,
            ) from UpstreamMayHaveRun(may_have_run or "No task id in the createTask answer")
        return data

    def _outcome(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map a createTask record onto the gateway's sync or async shape."""
        task_id = str(data["taskId"])
        state = str(data.get("state") or "")
        if state == "succeeded":
            self._remember_cost(task_id, data.get("cost"))
            return {"data": data}
        if state in _FAILED_STATES:
            self._log_terminal_failure(task_id, data)
            raise ProviderError(
                provider=self.name,
                detail=_failure_detail(data),
                status_code=502,
            )
        # Pending, missing or unlisted: get_result polls until a terminal state.
        return {"data": {"id": task_id, "status": "processing", "state": state or "queued"}}

    def _remember_cost(self, task_id: str, cost: Any) -> None:
        """Keep a task's settled cost until the gateway asks for it."""
        parsed = _parse_cost(cost)
        if parsed is None:
            return
        if len(self._actual_costs) > 200:
            self._actual_costs.clear()
        self._actual_costs[task_id] = parsed

    def _log_terminal_failure(self, task_id: str, data: dict[str, Any]) -> None:
        logger.warning(
            "SpicyAPI task %s %s (cost=%s settled=%s)",
            task_id,
            data.get("state"),
            data.get("cost"),
            data.get("settled"),
        )

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        request_timeout: float = _DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
        retry: bool = True,
        check: bool = True,
    ) -> httpx.Response:
        """Call SpicyAPI. 5xx and timeouts retry with the same headers.

        With ``check=False`` an HTTP error answer is returned, not raised.
        """
        response, _ = await self._send_tracked(
            method,
            path,
            json_body=json_body,
            params=params,
            request_timeout=request_timeout,
            extra_headers=extra_headers,
            retry=retry,
            check=check,
        )
        return response

    async def _send_tracked(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        request_timeout: float = _DEFAULT_TIMEOUT,
        extra_headers: dict[str, str] | None = None,
        retry: bool = True,
        check: bool = True,
        billable: bool = False,
    ) -> tuple[httpx.Response, str]:
        """Call SpicyAPI and say why a billable call may already have run.

        A billable call (createTask) retries with one Idempotency-Key. When
        any attempt timed out or dropped after it was sent, or hit a gateway
        timeout, the task may exist whatever the last attempt says, so a final
        failure is chained to ``UpstreamMayHaveRun`` and the reason is
        returned with a 2xx answer.
        """
        url = f"{self._config.base_url.rstrip('/')}{path}"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        if extra_headers:
            headers.update(extra_headers)

        attempts = 1 + _MAX_RETRIES if retry else 1
        may_have_run = ""
        client = self._get_client()

        for attempt in range(attempts):
            cause: BaseException | None = None
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                    timeout=http.timeout(request_timeout),
                )
            except httpx.TimeoutException as exc:
                cause = exc
                error = ProviderError(
                    provider=self.name,
                    detail=f"SpicyAPI request timed out after {request_timeout}s",
                    status_code=504,
                )
            except httpx.HTTPError as exc:
                cause = exc
                error = ProviderError(
                    provider=self.name,
                    detail=f"Failed to connect to SpicyAPI: {exc}",
                    status_code=503,
                )
            else:
                status = response.status_code
                if 200 <= status < 300 or not check:
                    return response, may_have_run
                if billable and status in _GATEWAY_TIMEOUT_STATUSES and not may_have_run:
                    may_have_run = f"{path} attempt {attempt + 1}: gateway timeout {status}"
                if status < 500:
                    # A 3xx or 4xx will not change on retry.
                    error = self._error_from_response(response)
                    if may_have_run:
                        raise error from UpstreamMayHaveRun(may_have_run)
                    raise error
                logger.warning(
                    "SpicyAPI %s %s returned %s (attempt %s/%s)",
                    method,
                    path,
                    status,
                    attempt + 1,
                    attempts,
                )
                error = self._error_from_response(response)

            if billable and cause is not None and outcome_unknown(cause) and not may_have_run:
                may_have_run = f"{path} attempt {attempt + 1}: {type(cause).__name__}"
            if attempt + 1 < attempts:
                await self._sleep(_RETRY_DELAY * (2**attempt))
                continue
            if may_have_run:
                raise error from UpstreamMayHaveRun(may_have_run)
            if cause is not None:
                raise error from cause
            raise error

        raise ProviderError(
            provider=self.name,
            detail="SpicyAPI request failed",
            status_code=502,
        )

    def _error_from_response(self, response: httpx.Response) -> ProviderError:
        """Map a SpicyAPI error envelope onto ProviderError."""
        status = response.status_code
        detail = response.text[:500]
        code: Any = None
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            code = body.get("code")
            message = body.get("msg") or body.get("message")
            if message:
                detail = str(message)
        if not 400 <= status < 600:
            # An error envelope in a 2xx or 3xx answer: use its code's status.
            envelope = _status_class(code)
            status = envelope if envelope is not None and 400 <= envelope < 600 else 502
        if code in (40201, 40202, "40201", "40202") or status == 402:
            return ProviderError(
                provider=self.name,
                detail=f"SpicyAPI account cannot accept this charge: {detail}",
                status_code=503,
            )
        if rejected_credentials(status):
            # A client reads 401/403 as its own credentials being bad, so a
            # rejected gateway key is a 502, as in proxy.py.
            return ProviderError(
                provider=self.name,
                detail=UPSTREAM_CREDENTIALS_DETAIL,
                status_code=502,
            )
        if code == 40901 or code == "40901":
            detail = f"40901 {detail}"
        return ProviderError(
            provider=self.name,
            detail=detail,
            status_code=status,
        )

    @staticmethod
    def _get_client() -> httpx.AsyncClient:
        # The shared pool keeps the connection opened by the quote for the
        # create or stream call that follows, and for the next request.
        return shared_client()

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        # Hashed, so a large chat body is not kept in memory as a cache key.
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _drop_expired(quotes: dict[str, _Quote]) -> None:
        now = time.time()
        stale = [key for key, quote in quotes.items() if quote.expires_at <= now]
        for key in stale:
            del quotes[key]

    @staticmethod
    async def _sleep(seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


def _envelope_data(response: httpx.Response) -> dict[str, Any]:
    """Return the data object from a successful SpicyAPI envelope."""
    try:
        body = response.json()
    except Exception as exc:
        raise ProviderError(
            provider="spicyapi",
            detail="SpicyAPI returned a non-JSON response",
            status_code=502,
        ) from exc
    if not isinstance(body, dict):
        raise ProviderError(
            provider="spicyapi",
            detail="SpicyAPI returned an unexpected response",
            status_code=502,
        )
    code = body.get("code")
    if code not in (200, "200", None):
        message = body.get("msg") or body.get("message") or f"code {code}"
        raise ProviderError(
            provider="spicyapi",
            detail=str(message),
            status_code=502,
        )
    data = body.get("data")
    if not isinstance(data, dict):
        raise ProviderError(
            provider="spicyapi",
            detail="SpicyAPI response missing data",
            status_code=502,
        )
    return data


def _status_class(code: Any) -> int | None:
    """Return the HTTP-style status of a SpicyAPI envelope code, or None.

    Codes are HTTP statuses (429, 500) or five digits that start with one
    (40201 is a 402, 40901 a 409, 50001 a 500).
    """
    try:
        value = int(code)
    except (TypeError, ValueError):
        return None
    return value // 100 if 10_000 <= value < 100_000 else value


def _envelope_status(response: httpx.Response) -> int | None:
    """Return the HTTP-style status of an error envelope, or None.

    None for success, no code, or a body that is not a JSON object.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict) or body.get("code") in (None, 200, "200"):
        return None
    return _status_class(body.get("code"))


def _envelope_transient(response: httpx.Response) -> bool:
    """True when a 2xx body is unreadable or its envelope code is 429 or 5xx class."""
    try:
        body = response.json()
    except Exception:
        return True
    if not isinstance(body, dict):
        return True
    status = _status_class(body.get("code") or 200)
    return status is not None and (status == 429 or 500 <= status < 600)


def _poll_status(response: httpx.Response) -> int:
    """Return the HTTP-style status of a poll answer.

    A 2xx can carry an error in its envelope code. An unreadable 2xx counts
    as a 503, a transient hiccup that says nothing about the task.
    """
    status = response.status_code
    if not 200 <= status < 300:
        return status
    if _envelope_transient(response):
        return 503
    return _envelope_status(response) or status


def _parse_cost(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _expires_at(value: Any) -> float:
    """Parse an RFC 3339 expiry, or fall back to four minutes from now."""
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp()
        except ValueError:
            pass
    return time.time() + 240


def _poll_backoff(interval: float, failures: int, deadline: float) -> float:
    """Delay before the next poll after transient failures, within the deadline."""
    delay = min(max(interval, _POLL_BACKOFF_MIN) * 2 ** (failures - 1), _POLL_BACKOFF_MAX)
    return min(delay, max(0.0, deadline - time.monotonic()))


def _failure_detail(data: dict[str, Any]) -> str:
    state = data.get("state") or "failed"
    reason = data.get("errorMessage") or data.get("errorCode") or state
    return f"Generation {state}: {reason}"


def _chat_max_tokens(body: dict[str, Any], default_max_tokens: int) -> int:
    """Resolve the output cap sent to SpicyAPI."""
    raw_max = body.get("max_tokens")
    raw_completion = body.get("max_completion_tokens")
    if raw_max is not None and raw_completion is not None and raw_max != raw_completion:
        raise ProviderError(
            provider="spicyapi",
            detail="Send max_tokens or max_completion_tokens, not two different values",
            status_code=400,
        )
    chosen = raw_max if raw_max is not None else raw_completion
    if chosen is None:
        return default_max_tokens
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < 1:
        raise ProviderError(
            provider="spicyapi",
            detail="max_tokens must be a positive integer",
            status_code=400,
        )
    return chosen


async def _read_sse_chunks(response: httpx.Response) -> list[dict[str, Any]]:
    """Read a jobs/stream body into JSON chunks, ignoring keepalive comments.

    A failure before any output is a known failure. Once output has streamed,
    SpicyAPI has billed it, so a failure from then on is an unknown outcome.
    """
    chunks: list[dict[str, Any]] = []
    event_name = ""
    produced = False
    async for line in response.aiter_lines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            if produced:
                _raise_cut_off("SpicyAPI chat stream returned a non-JSON event after output")
            raise ProviderError(
                provider="spicyapi",
                detail="SpicyAPI chat stream returned a non-JSON event",
                status_code=502,
            ) from exc
        if not isinstance(payload, dict):
            if produced:
                _raise_cut_off("SpicyAPI chat stream returned an unexpected event after output")
            raise ProviderError(
                provider="spicyapi",
                detail="SpicyAPI chat stream returned an unexpected event",
                status_code=502,
            )
        if event_name == "error" or "error" in payload:
            if produced:
                message = _stream_error_message(payload)
                _raise_cut_off(f"SpicyAPI chat failed after output: {message}")
            _raise_stream_error(payload)
        chunks.append(payload)
        produced = produced or _has_output(payload)
        event_name = ""
    return chunks


def _has_output(chunk: dict[str, Any]) -> bool:
    """Return True when a stream chunk carries generated, billable output."""
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
            continue
        delta = choice["delta"]
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True
    return False


def _raise_cut_off(detail: str) -> NoReturn:
    """Raise for a stream that failed after output, as an unknown outcome.

    The output was generated and billed, so the payment is kept. Chaining to
    a transport error is how ``outcome_unknown()`` recognises that case.
    """
    raise ProviderError(provider="spicyapi", detail=detail, status_code=502) from (
        httpx.RemoteProtocolError("SpicyAPI chat stream ended after output")
    )


def _stream_error_message(payload: dict[str, Any]) -> str:
    """Return the message of a stream error event."""
    error = payload.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or "")
        return str(error.get("message") or code or "chat failed")
    code = str(payload.get("code") or "")
    return str(payload.get("message") or payload.get("msg") or code or "chat failed")


def _raise_stream_error(payload: dict[str, Any]) -> None:
    """Turn a stream error event before any output into ProviderError."""
    error = payload.get("error")
    code = str((error.get("code") if isinstance(error, dict) else payload.get("code")) or "")
    message = _stream_error_message(payload)
    client_error = code in {"content_rejected", "invalid_request", "unsupported_parameter"}
    status = 400 if client_error else 502
    raise ProviderError(provider="spicyapi", detail=message, status_code=status)


def _completion_from_chunks(chunks: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Fold OpenAI chat chunks into one completion object."""
    choices: dict[int, dict[str, Any]] = {}
    completion_id = ""
    seen_model = model
    usage: dict[str, Any] | None = None
    for chunk in chunks:
        if isinstance(chunk.get("id"), str):
            completion_id = chunk["id"]
        if isinstance(chunk.get("model"), str) and chunk["model"]:
            seen_model = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if isinstance(index, bool) or not isinstance(index, int):
                index = 0
            slot = choices.setdefault(
                index,
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                    "tool_calls": [],
                    "finish_reason": None,
                },
            )
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str):
                slot["content"] += delta["content"]
            if isinstance(delta.get("reasoning_content"), str):
                slot["reasoning_content"] += delta["reasoning_content"]
            if choice.get("finish_reason"):
                slot["finish_reason"] = choice["finish_reason"]
            for tool_call in delta.get("tool_calls") or []:
                if isinstance(tool_call, dict):
                    _merge_tool_call(slot, tool_call)
    if not choices or any(slot["finish_reason"] is None for slot in choices.values()):
        if any(_has_output(chunk) for chunk in chunks):
            _raise_cut_off("SpicyAPI chat stream ended before the answer finished")
        raise ProviderError(
            provider="spicyapi",
            detail="SpicyAPI chat stream ended before the answer finished",
            status_code=502,
        )
    messages = []
    for index in sorted(choices):
        slot = choices[index]
        message: dict[str, Any] = {"role": slot["role"], "content": slot["content"]}
        if slot["reasoning_content"]:
            message["reasoning_content"] = slot["reasoning_content"]
        if slot["tool_calls"]:
            message["tool_calls"] = slot["tool_calls"]
        messages.append(
            {
                "index": index,
                "message": message,
                "finish_reason": slot["finish_reason"],
            }
        )
    completion: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "model": seen_model,
        "choices": messages,
    }
    if usage is not None:
        completion["usage"] = usage
    return completion


def _merge_tool_call(slot: dict[str, Any], delta_call: dict[str, Any]) -> None:
    """Append one streamed tool-call fragment onto the choice being built."""
    index = delta_call.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        index = 0
    while len(slot["tool_calls"]) <= index:
        slot["tool_calls"].append(
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
    current = slot["tool_calls"][index]
    if isinstance(delta_call.get("id"), str):
        current["id"] = delta_call["id"]
    if isinstance(delta_call.get("type"), str):
        current["type"] = delta_call["type"]
    function = delta_call.get("function") if isinstance(delta_call.get("function"), dict) else {}
    if isinstance(function.get("name"), str):
        current["function"]["name"] += function["name"]
    if isinstance(function.get("arguments"), str):
        current["function"]["arguments"] += function["arguments"]
