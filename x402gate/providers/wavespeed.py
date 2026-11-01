"""WaveSpeed AI provider for x402gate.

Implements the BaseProvider interface for WaveSpeed AI's API:
- Pricing via POST /api/v3/model/pricing
- Task submission via POST /api/v3/{model-path}
- Result polling via GET /api/v3/predictions/{task-id}
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

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


class WaveSpeedProvider(BaseProvider):
    """WaveSpeed AI provider implementation.

    Acts as a transparent proxy to WaveSpeed's API. Requests are forwarded
    as-is, and pricing is fetched dynamically from their pricing endpoint.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="wavespeed", config=config)

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Fetch dynamic pricing from WaveSpeed's Pricing API.

        Calls POST /api/v3/model/pricing with the model_id and inputs
        to get the cost for this specific request.

        Args:
            model_path: Model path (e.g. 'wavespeed-ai/flux-dev').
            inputs: Request parameters to price.

        Returns:
            Base price in USD as a Decimal.

        Raises:
            ProviderError: If the pricing API call fails.
        """
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

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a task to WaveSpeed's API.

        Forwards the request body as-is to POST /api/v3/{path}.

        Args:
            path: Model path (e.g. 'wavespeed-ai/flux-dev').
            body: Request body, forwarded without modification.

        Returns:
            WaveSpeed response dict with task ID and status.

        Raises:
            ProviderError: If the submission fails, or WaveSpeed accepted it
                without a task id to poll.
        """
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
            return result

        except ProviderError:
            raise
        except Exception as e:
            self._raise_if_key_rejected(e)
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to submit task: {e}",
            ) from e

    def _raise_if_key_rejected(self, exc: Exception) -> None:
        """Turn an upstream 401/402/403 into a 502 that hides WaveSpeed's text.

        A 402 passed on would read to x402 clients as the gateway asking for
        payment, and 401/403 would tell them our key is bad.
        """
        if isinstance(exc, ProxyError) and rejected_credentials(exc.status_code):
            logger.error("WaveSpeed rejected our key (%d): %s", exc.status_code, exc.detail)
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL) from exc

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll WaveSpeed for task completion.

        Polls GET /api/v3/predictions/{task_id} until the task
        completes, fails, or times out.

        Args:
            task_id: Task ID from submit() response.

        Returns:
            Completed task data including outputs.

        Raises:
            ProviderError: If the task fails.
            TaskTimeoutError: If polling exceeds poll_timeout.
        """
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
