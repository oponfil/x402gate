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


class ProxyError(Exception):
    """Raised when the upstream provider returns an error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Proxy error {status_code}: {detail}")


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
) -> dict[str, Any]:
    """Forward a request body to the provider API as-is.

    Args:
        base_url: Provider API base URL (e.g. https://api.wavespeed.ai/api/v3).
        path: Request path to append (e.g. wavespeed-ai/flux-dev).
        body: Request body dict, forwarded without modification.
        api_key: Provider API key for Authorization header.
        request_timeout: HTTP request timeout in seconds.

    Returns:
        Provider response as a dict.

    Raises:
        ProxyError: If the provider returns a non-2xx response.
    """
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    async with httpx.AsyncClient(timeout=request_timeout) as client:
        response = await client.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code >= 400:
        raise ProxyError(
            status_code=response.status_code,
            detail=response.text,
        )

    return response.json()


async def poll_result(
    base_url: str,
    task_id: str,
    api_key: str,
    poll_interval: int = 2,
    poll_timeout: int = 300,
) -> dict[str, Any]:
    """Poll a provider API for async task completion.

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
        ProxyError: If the task fails or the API returns an error.
    """
    url = f"{base_url.rstrip('/')}/predictions/{task_id}/result"
    elapsed = 0
    poll_count = 0

    # Brief delay before first poll to let the task register
    await asyncio.sleep(1)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < poll_timeout:
            poll_count += 1
            try:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if response.status_code >= 400:
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

                data = response.json()
                task_data = data.get("data", data)
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
                # Fall through to sleep and retry

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    logger.error(
        "Task %s timed out after %ds (%d polls, url=%s)",
        task_id,
        poll_timeout,
        poll_count,
        url,
    )
    raise TaskTimeoutError(task_id=task_id, timeout=poll_timeout)
