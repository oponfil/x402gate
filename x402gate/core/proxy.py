"""Transparent proxy and async task polling for x402gate.

Forwards requests to provider APIs and polls for async task results.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Our own API key was refused or ran out of credit. Passed on as is, a 402
# reads to x402 clients as the gateway asking for payment, and 401/403 tell
# them our key is bad, so the client gets a 502 with this text instead.
_CREDENTIAL_STATUSES = frozenset({401, 402, 403})
UPSTREAM_CREDENTIALS_DETAIL = (
    "Upstream provider rejected the gateway's credentials or is out of credit"
)

# Gateway errors from a proxy in front of the provider that come after the
# request was forwarded: a gateway timeout (504, Cloudflare's 524) or an empty
# or reset answer from the backend (Cloudflare's 520). Like our own read
# timeout or dropped connection, the task may exist. 521-523 are left out:
# there Cloudflare never reached the backend.
_GATEWAY_TIMEOUT_STATUSES = frozenset({504, 520, 524})

# A missing task (404) counts as lost only after this many 404 polls in a row
# and this many seconds since the first one. Right after creation the task
# may not be readable yet, and a poll count alone is only a few seconds at
# a 1s poll interval.
_NOT_FOUND_GRACE_POLLS = 3
_NOT_FOUND_GRACE_S = 20

# Longest wait between polls while the provider keeps failing transiently.
_MAX_POLL_BACKOFF = 30


class ProxyError(Exception):
    """Raised when the upstream provider returns an error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Proxy error {status_code}: {detail}")


def rejected_credentials(status_code: int) -> bool:
    """Return True when an upstream status is about our API key, not the request."""
    return status_code in _CREDENTIAL_STATUSES


class UpstreamMayHaveRun(httpx.ReadError):
    """Cause to chain when a failed call may have run, and billed us, upstream.

    ``outcome_unknown()`` in providers/base.py keeps the client's payment when
    the exception chain holds a transport error that came after the request
    was sent. This subclass makes a failure seen in a response (a gateway
    timeout, a task the provider lost, a call that ran but failed) count the
    same way: ``raise ProviderError(...) from UpstreamMayHaveRun(reason)``.
    """


def safe_to_resend(exc: httpx.TimeoutException) -> bool:
    """Return True when a timed-out request never reached the provider.

    A connect or pool timeout fails before anything is sent. A read or write
    timeout comes after: the provider may already be running, and billing,
    the job, so sending it again could pay for it twice.
    """
    return isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout))


class TaskTimeoutError(Exception):
    """Raised when task polling exceeds the timeout."""

    def __init__(self, task_id: str, timeout: int) -> None:
        self.task_id = task_id
        self.timeout = timeout
        super().__init__(f"Task {task_id} did not complete within {timeout}s")


async def proxy_request(
    base_url: str,
    path: str,
    body: dict[str, Any],
    api_key: str,
    request_timeout: float = 30.0,
    max_retries: int = 2,
    retry_delay: float = 2.0,
    idempotent: bool = False,
) -> dict[str, Any]:
    """Forward a request body to the provider API as-is.

    Retries connections that failed before the request was sent (up to
    max_retries times) with exponential backoff. 5xx server errors are retried
    only for idempotent calls such as pricing: a submit that got a 5xx may
    still have created a task, and sending it again would start a second paid
    job. 3xx and 4xx answers and read timeouts fail immediately.

    Args:
        base_url: Provider API base URL (e.g. https://api.wavespeed.ai/api/v3).
        path: Request path to append (e.g. wavespeed-ai/flux-dev).
        body: Request body dict, forwarded without modification.
        api_key: Provider API key for Authorization header.
        request_timeout: HTTP request timeout in seconds.
        max_retries: Max retry attempts (default: 2).
        retry_delay: Base delay between retries in seconds (doubles each attempt).
        idempotent: True when sending the request twice costs nothing extra,
            which allows retrying it after a 5xx.

    Returns:
        Provider response as a dict.

    Raises:
        ProxyError: If the provider returns a non-2xx response after all retries.
            Chained to ``UpstreamMayHaveRun`` when a submit may have landed
            (a gateway timeout or reset, an unreadable 2xx body).
    """
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    last_error: ProxyError | None = None

    for attempt in range(1 + max_retries):
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
        except httpx.TimeoutException as exc:
            last_error = ProxyError(
                status_code=504,
                detail=f"Request to {url} timed out after {request_timeout}s",
            )
            if attempt < max_retries and safe_to_resend(exc):
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Proxy timeout (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    1 + max_retries,
                    delay,
                    url,
                )
                await asyncio.sleep(delay)
                continue
            raise last_error from None
        except httpx.ConnectError:
            # The connection never opened, so nothing reached the provider.
            if attempt < max_retries:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Proxy connect error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    1 + max_retries,
                    delay,
                    url,
                )
                await asyncio.sleep(delay)
                continue
            raise

        # Only 2xx is success. httpx does not follow redirects, and a 3xx
        # (a moved base_url, say) means the provider never ran the call.
        if 200 <= response.status_code < 300:
            try:
                data = response.json()
            except ValueError:
                data = None
            if not isinstance(data, dict):
                # The provider accepted the call; a submit may have created a task.
                raise ProxyError(
                    status_code=502,
                    detail=f"Unreadable response from {url}: {response.text[:200]}",
                ) from UpstreamMayHaveRun(f"Unreadable {response.status_code} body")
            return data

        if response.status_code >= 500 and idempotent and attempt < max_retries:
            # 5xx server error — transient, retry with backoff
            delay = retry_delay * (2**attempt)
            logger.warning(
                "Provider returned %d (attempt %d/%d), retrying in %.1fs: %s",
                response.status_code,
                attempt + 1,
                1 + max_retries,
                delay,
                response.text[:200],
            )
            last_error = ProxyError(
                status_code=response.status_code,
                detail=response.text,
            )
            await asyncio.sleep(delay)
            continue

        # 3xx, 4xx, a non-idempotent 5xx or the final attempt — fail immediately
        error = ProxyError(
            status_code=response.status_code,
            detail=response.text,
        )
        if not idempotent and response.status_code in _GATEWAY_TIMEOUT_STATUSES:
            raise error from UpstreamMayHaveRun(f"Gateway timeout {response.status_code}")
        raise error

    # Should not reach here, but safety net
    raise last_error or ProxyError(status_code=502, detail="All retries exhausted")


async def poll_result(
    base_url: str,
    task_id: str,
    api_key: str,
    poll_interval: int = 2,
    poll_timeout: int = 300,
) -> dict[str, Any]:
    """Poll a provider API for async task completion.

    429, 5xx and network errors are transient: polling goes on, backing off,
    until poll_timeout. A 404 is tolerated for a grace period, since a new
    task may not be readable yet. The task was accepted at submission, so a
    task that stays missing, a redirect or a rejected key while polling is an
    unknown outcome rather than a known failure: the job may still run, and
    bill, upstream.

    Args:
        base_url: Provider API base URL.
        task_id: Task ID to poll.
        api_key: Provider API key.
        poll_interval: Seconds between poll requests.
        poll_timeout: Maximum seconds to wait before timing out.

    Returns:
        Task result data dict when status is "completed".

    Raises:
        TaskTimeoutError: If the task doesn't complete within poll_timeout.
        ProxyError: If the task fails or the API returns an error. Chained to
            ``UpstreamMayHaveRun`` when the task stays missing (404), the poll
            is redirected (3xx) or our key is rejected (401/402/403).
    """
    url = f"{base_url.rstrip('/')}/predictions/{task_id}/result"
    elapsed = 0
    poll_count = 0
    not_found = 0  # consecutive 404 answers
    not_found_since = 0  # elapsed seconds at the first of them
    transient = 0  # consecutive transient failures, drives the backoff

    # Brief delay before first poll to let the task register
    await asyncio.sleep(1)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < poll_timeout:
            poll_count += 1
            failed_transiently = False
            try:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if response.status_code != 404:
                    not_found = 0

                if response.status_code >= 500 or response.status_code == 429:
                    # Server error or rate limit — transient, the task keeps running
                    failed_transiently = True
                    logger.warning(
                        "Task %s poll #%d: HTTP %d after %ds — %s",
                        task_id,
                        poll_count,
                        response.status_code,
                        elapsed,
                        response.text[:200],
                    )
                    # Fall through to sleep and retry

                elif response.status_code == 404:
                    if not not_found:
                        not_found_since = elapsed
                    not_found += 1
                    logger.warning(
                        "Task %s poll #%d: HTTP 404 (%d in a row) after %ds",
                        task_id,
                        poll_count,
                        not_found,
                        elapsed,
                    )
                    if (
                        not_found > _NOT_FOUND_GRACE_POLLS
                        and elapsed - not_found_since >= _NOT_FOUND_GRACE_S
                    ):
                        raise ProxyError(
                            status_code=502,
                            detail=f"Provider no longer finds task {task_id}",
                        ) from UpstreamMayHaveRun(f"Task {task_id} not found")

                elif rejected_credentials(response.status_code):
                    logger.error(
                        "Task %s poll #%d: HTTP %d — %s",
                        task_id,
                        poll_count,
                        response.status_code,
                        response.text[:200],
                    )
                    # The task was accepted with this key and may still run
                    # upstream, so the payment is kept, as for a lost task.
                    raise ProxyError(
                        status_code=502, detail=UPSTREAM_CREDENTIALS_DETAIL
                    ) from UpstreamMayHaveRun(f"Poll of task {task_id} rejected our key")

                elif 300 <= response.status_code < 400:
                    # httpx does not follow redirects, so the task cannot be
                    # read here, but it was accepted and may still run.
                    logger.error(
                        "Task %s poll #%d: HTTP %d redirect to %s",
                        task_id,
                        poll_count,
                        response.status_code,
                        response.headers.get("location", "?"),
                    )
                    raise ProxyError(
                        status_code=502,
                        detail=f"Provider redirected the poll of task {task_id}",
                    ) from UpstreamMayHaveRun(f"Poll redirect {response.status_code}")

                elif response.status_code >= 400:
                    # Client error (4xx) — won't change on retry, fail immediately
                    logger.warning(
                        "Task %s poll #%d: HTTP %d after %ds — %s",
                        task_id,
                        poll_count,
                        response.status_code,
                        elapsed,
                        response.text[:200],
                    )
                    raise ProxyError(
                        status_code=response.status_code,
                        detail=response.text,
                    )

                else:
                    try:
                        data = response.json()
                    except ValueError:
                        data = None
                    task_data = data.get("data", data) if isinstance(data, dict) else None
                    if not isinstance(task_data, dict):
                        # A garbled answer says nothing about the task; ask again
                        failed_transiently = True
                        task_data = {}
                        logger.warning(
                            "Task %s poll #%d: unreadable body — %s",
                            task_id,
                            poll_count,
                            response.text[:200],
                        )
                    status = task_data.get("status", "")

                    if status == "completed":
                        logger.info(
                            "Task %s completed after %ds (%d polls)",
                            task_id,
                            elapsed,
                            poll_count,
                        )
                        return task_data

                    if status == "failed":
                        error_msg = task_data.get("error", "Task failed without details")
                        logger.error(
                            "Task %s failed after %ds (%d polls): %s",
                            task_id,
                            elapsed,
                            poll_count,
                            str(error_msg)[:200],
                        )
                        # Try to extract clean validation errors from provider response
                        if isinstance(error_msg, dict):
                            # Some providers return structured errors
                            error_msg = error_msg.get("message", str(error_msg))
                        elif isinstance(error_msg, str):
                            # Try to extract "Validation errors: [...]" patterns
                            match = re.search(r"Validation errors: \[(.+?)\]", error_msg)
                            if match:
                                error_msg = match.group(1).strip("'\"")
                        raise ProxyError(
                            status_code=502,
                            detail=error_msg,
                        )

                    # Log every poll at INFO for first 5, then every 10th
                    if poll_count <= 5 or poll_count % 10 == 0:
                        logger.info(
                            "Task %s poll #%d: status=%s, elapsed=%ds/%ds",
                            task_id,
                            poll_count,
                            status,
                            elapsed,
                            poll_timeout,
                        )

            except httpx.RequestError as e:
                logger.warning(
                    "Task %s poll #%d failed after %ds: %s: %s",
                    task_id,
                    poll_count,
                    elapsed,
                    type(e).__name__,
                    e,
                )
                failed_transiently = True
                # Fall through to sleep and retry

            transient = transient + 1 if failed_transiently else 0
            delay = poll_interval
            if transient:
                delay = min(poll_interval * 2**transient, max(poll_interval, _MAX_POLL_BACKOFF))
            # Do not sleep past the deadline.
            delay = min(delay, poll_timeout - elapsed)
            await asyncio.sleep(delay)
            # NOTE: elapsed counts sleep time only, not HTTP request duration.
            # Real wall-clock time may be higher if requests are slow (up to
            # 30s httpx timeout each).  Actual user-facing latency is tracked
            # via time.monotonic() in app.py, not here.
            elapsed += delay

    logger.error(
        "Task %s timed out after %ds (%d polls, url=%s)",
        task_id,
        poll_timeout,
        poll_count,
        url,
    )
    raise TaskTimeoutError(task_id=task_id, timeout=poll_timeout)
