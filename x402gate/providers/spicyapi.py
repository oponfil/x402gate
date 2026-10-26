"""SpicyAPI provider for x402gate.

Image and video jobs use SpicyAPI's task API:

- price via POST /api/v1/jobs/quote (no hold)
- submit via POST /api/v1/jobs/createTask with that quote
- poll via GET /api/v1/jobs/recordInfo

The client calls ``POST /v1/spicyapi/jobs`` with ``{model, input}``.
The gateway adds ``quoteId`` and ``expectedCost`` and, for images, holds
the create call briefly so a fast job can return without a poll loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

_JOBS_PATH = "jobs"
_PENDING_STATES = frozenset({"queued", "running"})
_QUOTE_SKEW_SECONDS = 20
_IMAGE_WAIT_SECONDS = 45
_IMAGE_REQUEST_TIMEOUT = 75.0
_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 2
_RETRY_DELAY = 2.0

_INTERNAL_FIELDS = frozenset({"_caller", "_file_bytes", "_file_name"})
_REQUEST_FIELDS = frozenset({"model", "input", "callBackUrl"})


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

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="spicyapi", config=config)
        self._client: httpx.AsyncClient | None = None
        self._quotes: dict[str, _Quote] = {}
        self._priced: dict[str, Decimal] = {}
        self._actual_costs: dict[str, Decimal] = {}

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Quote the exact job. The quote is reused when the client pays."""
        payload = self._task_request(model_path, inputs)
        quote = await self._quote(payload)
        self._priced[self._cache_key(payload)] = quote.cost
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
    ) -> dict[str, Any]:
        """Create a task at the quoted price.

        Image jobs pass ``wait`` so a finished image comes back on this
        call. Video jobs return immediately and are polled by ``get_result``.
        """
        payload = self._task_request(path, body)
        quote = await self._quote_for_submit(payload)
        wait = _IMAGE_WAIT_SECONDS if quote.unit == "per_image" else None
        try:
            record = await self._create_task(payload, quote, wait=wait)
        except ProviderError as exc:
            if "40901" not in exc.detail and "no longer" not in exc.detail.lower():
                raise
            logger.info("SpicyAPI quote rejected, quoting %s again", payload["model"])
            self._quotes.pop(self._cache_key(payload), None)
            quote = await self._quote_for_submit(payload)
            record = await self._create_task(payload, quote, wait=wait)
        return self._outcome(record)

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll recordInfo until the task leaves queued/running."""
        interval = self._config.poll_interval
        deadline = time.monotonic() + self._config.poll_timeout
        not_found = 0

        while time.monotonic() < deadline:
            try:
                response = await self._send(
                    "GET",
                    "/api/v1/jobs/recordInfo",
                    params={"taskId": task_id},
                    request_timeout=_DEFAULT_TIMEOUT,
                    retry=False,
                )
            except ProviderError as exc:
                if exc.status_code == 404 and not_found < 5:
                    not_found += 1
                    await self._sleep(interval)
                    continue
                if exc.status_code >= 500:
                    await self._sleep(interval)
                    continue
                raise

            data = _envelope_data(response)
            state = str(data.get("state") or "")
            if state == "succeeded":
                self._remember_cost(task_id, data.get("cost"))
                logger.info("SpicyAPI task %s succeeded", task_id)
                return data
            if state not in _PENDING_STATES:
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
        """Close the shared HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _task_request(self, model_path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Validate the gateway body and return the SpicyAPI task request."""
        if model_path.strip("/") != _JOBS_PATH:
            raise ProviderError(
                provider=self.name,
                detail=(
                    "SpicyAPI requests use POST /v1/spicyapi/jobs "
                    "with a JSON body of {model, input}"
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

    async def _quote_for_submit(self, payload: dict[str, Any]) -> _Quote:
        """Reuse the price shown to the client, or quote again if it is stale.

        A fresh quote that costs more than the price already offered is
        rejected, so a paid retry cannot be accepted above that amount.
        """
        key = self._cache_key(payload)
        quote = self._quotes.get(key)
        if quote is None or quote.expires_at <= time.time() + _QUOTE_SKEW_SECONDS:
            quote = await self._quote(payload)
        priced = self._priced.get(key)
        if priced is not None and quote.cost > priced:
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
        response = await self._send(
            "POST",
            "/api/v1/jobs/quote",
            json_body=payload,
            request_timeout=_DEFAULT_TIMEOUT,
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
        self._quotes[key] = quote
        return quote

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
        response = await self._send(
            "POST",
            "/api/v1/jobs/createTask",
            json_body=request_body,
            params=params,
            request_timeout=timeout,
            extra_headers={"Idempotency-Key": idempotency_key},
        )
        data = _envelope_data(response)
        task_id = data.get("taskId")
        logger.info(
            "SpicyAPI task %s for %s (%s)",
            task_id or "unknown",
            payload["model"],
            data.get("state") or "unknown",
        )
        return data

    def _outcome(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map a createTask record onto the gateway's sync or async shape."""
        task_id = data.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            raise ProviderError(
                provider=self.name,
                detail="SpicyAPI response missing taskId",
                status_code=502,
            )
        state = str(data.get("state") or "")
        if state == "succeeded":
            self._remember_cost(task_id, data.get("cost"))
            return {"data": data}
        if state in _PENDING_STATES or not state:
            return {"data": {"id": task_id, "status": "processing", "state": state or "queued"}}
        self._log_terminal_failure(task_id, data)
        raise ProviderError(
            provider=self.name,
            detail=_failure_detail(data),
            status_code=502,
        )

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
    ) -> httpx.Response:
        """Call SpicyAPI. 5xx and timeouts retry with the same headers."""
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
        last_error: ProviderError | None = None
        client = await self._get_client()

        for attempt in range(attempts):
            try:
                response = await client.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"SpicyAPI request timed out after {request_timeout}s",
                    status_code=504,
                )
                if attempt + 1 < attempts:
                    await self._sleep(_RETRY_DELAY * (2**attempt))
                    continue
                raise last_error from exc
            except httpx.HTTPError as exc:
                last_error = ProviderError(
                    provider=self.name,
                    detail=f"Failed to connect to SpicyAPI: {exc}",
                    status_code=503,
                )
                if attempt + 1 < attempts:
                    await self._sleep(_RETRY_DELAY * (2**attempt))
                    continue
                raise last_error from exc

            if response.status_code < 400:
                return response
            if response.status_code >= 500 and attempt + 1 < attempts:
                logger.warning(
                    "SpicyAPI %s %s returned %s (attempt %s/%s)",
                    method,
                    path,
                    response.status_code,
                    attempt + 1,
                    attempts,
                )
                await self._sleep(_RETRY_DELAY * (2**attempt))
                continue
            raise self._error_from_response(response)

        raise last_error or ProviderError(
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
        if code in (40201, 40202, "40201", "40202") or status == 402:
            return ProviderError(
                provider=self.name,
                detail=f"SpicyAPI account cannot accept this charge: {detail}",
                status_code=503,
            )
        if status == 401:
            return ProviderError(
                provider=self.name,
                detail="SpicyAPI authentication failed",
                status_code=502,
            )
        if code == 40901 or code == "40901":
            detail = f"40901 {detail}"
        return ProviderError(
            provider=self.name,
            detail=detail,
            status_code=status if 400 <= status < 600 else 502,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_IMAGE_REQUEST_TIMEOUT)
        return self._client

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

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


def _failure_detail(data: dict[str, Any]) -> str:
    state = data.get("state") or "failed"
    reason = data.get("errorMessage") or data.get("errorCode") or state
    return f"Generation {state}: {reason}"
