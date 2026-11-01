"""Base provider interface for x402gate.

All AI service providers must implement this interface.
The gateway uses it to fetch prices, submit tasks, and poll results
without knowing provider-specific details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import TaskTimeoutError

# Transport failures that happen after the request was sent. The provider may
# have accepted it and started (and billed) the job.
_MAY_HAVE_LANDED = (
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


class BaseProvider(ABC):
    """Abstract base class for AI service providers.

    Providers act as transparent proxies — they forward requests as-is
    and expose a standard interface for pricing, submission, and polling.
    """

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self._name = name
        self._config = config

    @property
    def name(self) -> str:
        """Provider name (e.g. 'wavespeed')."""
        return self._name

    @property
    def config(self) -> ProviderConfig:
        """Provider configuration."""
        return self._config

    @abstractmethod
    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Query the provider's pricing API for a given model and inputs.

        Args:
            model_path: Model path from the URL (e.g. 'wavespeed-ai/flux-dev').
            inputs: Request body / parameters to price.

        Returns:
            Base price in USD as a Decimal (before commission).

        Raises:
            ProviderError: If the pricing API call fails.
        """

    @abstractmethod
    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a task to the provider API.

        The request body is forwarded as-is without modification.

        Args:
            path: API path (e.g. 'wavespeed-ai/flux-dev').
            body: Request body dict.
            prepaid: True for prepaid-balance requests. Providers must still
                apply cost limits such as default max_tokens, because the
                gateway reserves the get_price() estimate before the call
                and never charges more than that reservation.

        Returns:
            Provider response including task ID and status.

        Raises:
            ProviderError: If the submission fails.
        """

    @abstractmethod
    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll for a task result by ID.

        Args:
            task_id: Task identifier from submit() response.

        Returns:
            Task result data when completed.

        Raises:
            ProviderError: If the task failed.
            TaskTimeoutError: If polling times out.
        """

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Calculate actual cost from provider response.

        Override for token-based providers to compute real cost from
        usage data in the response.  Returns None by default, meaning
        the estimated price (from get_price) will be used for settlement.

        Args:
            body: Original request body.
            result: Provider response (may contain 'usage' etc.).

        Returns:
            Actual cost in USD, or None to use the estimate.
        """
        return None

    async def warm_up(self) -> None:  # noqa: B027
        """Load what the first request would otherwise wait for (catalogs).

        Called in the background at startup; a failure is logged and the
        provider loads lazily as before. Override if needed.
        """

    async def close(self) -> None:  # noqa: B027
        """Clean up provider resources. Override if needed."""


def require_text(inputs: dict[str, Any], provider_name: str) -> str:
    """Extract and validate non-empty ``text`` from request inputs.

    Shared by all TTS providers to avoid duplicating the same check.

    Args:
        inputs: Request body / parameters.
        provider_name: Provider name for error messages.

    Returns:
        The non-empty text string.

    Raises:
        ProviderError: If ``text`` is missing or empty.
    """
    text = inputs.get("text", "")
    if not text:
        raise ProviderError(
            provider=provider_name,
            detail="Request body must contain non-empty 'text' field",
            status_code=400,
        )
    return text


def outcome_unknown(exc: BaseException) -> bool:
    """Return True when a failed call may still have run, and billed us.

    A poll timeout leaves the job running. A request that timed out or
    dropped after it was sent may have been accepted. Failures the provider
    reported itself (an HTTP error status, a failed task) and connections
    that never opened are known failures.

    Providers wrap transport errors in ``ProviderError``, often with
    ``from None``, so the whole ``__cause__`` / ``__context__`` chain is
    checked.
    """
    if isinstance(exc, TaskTimeoutError):
        return True
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, _MAY_HAVE_LANDED):
            return True
        stack.extend((current.__cause__, current.__context__))
    return False


class ProviderError(Exception):
    """Raised when a provider API call fails."""

    def __init__(self, provider: str, detail: str, status_code: int = 502) -> None:
        self.provider = provider
        self.detail = detail
        self.status_code = status_code
        super().__init__(f"[{provider}] {detail}")
