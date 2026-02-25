"""Tungsten.run provider for x402gate.

Implements the BaseProvider interface for Tungsten's image generation API:
- Fixed pricing (no pricing API available)
- Task submission via POST /v1/generations
- Result polling via POST /v1/generations/track
- Image download via GET /v1/generated_images/{id}/png

Authentication uses browser session cookies (jwt + cf_clearance),
not API keys. Images are downloaded server-side and returned as base64
because the download URLs require authentication cookies.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Tungsten expects browser-like headers
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/142.0.0.0 Safari/537.36"
)
_ORIGIN = "https://tungsten.run"
_REFERER = "https://tungsten.run/"

# Image download settings
_DOWNLOAD_TIMEOUT = 60.0
_DOWNLOAD_RETRIES = 3


class TungstenProvider(BaseProvider):
    """Tungsten.run provider implementation.

    Acts as a transparent proxy to Tungsten's image generation API.
    Uses cookie-based authentication (JWT + CF clearance) instead of
    standard API keys. Downloads generated images server-side and
    returns them as base64, since image URLs require authentication.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        jwt_token: str = "",
        cf_clearance: str = "",
    ) -> None:
        super().__init__(name="tungsten", config=config)
        self._jwt_token = jwt_token or config.jwt_token
        self._cf_clearance = cf_clearance or config.cf_clearance
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers with cookie authentication."""
        cookies_parts = []
        if self._jwt_token:
            cookies_parts.append(f"jwt={self._jwt_token}")
        if self._cf_clearance:
            cookies_parts.append(f"cf_clearance={self._cf_clearance}")

        return {
            "Cookie": "; ".join(cookies_parts),
            "User-Agent": _USER_AGENT,
            "Origin": _ORIGIN,
            "Referer": _REFERER,
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Return fixed price from configuration.

        Tungsten has no pricing API — the price is configured in
        config.yaml as ``fixed_price_usd``.

        Args:
            model_path: Ignored (price is the same for all models).
            inputs: Ignored.

        Returns:
            Fixed price in USD as a Decimal.

        Raises:
            ProviderError: If fixed_price_usd is not configured.
        """
        price = self._config.fixed_price_usd
        if not price:
            raise ProviderError(
                provider=self.name,
                detail="fixed_price_usd is not configured for Tungsten provider",
                status_code=500,
            )
        result = Decimal(str(price))
        logger.info("Tungsten fixed price: $%s", result)
        return result

    async def submit(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Submit a generation task to Tungsten's API.

        Forwards the request body as-is to POST /v1/generations.
        The ``path`` from the URL is used as the endpoint path.

        Args:
            path: API path (typically 'generations').
            body: Request body with generation parameters.

        Returns:
            Dict with task info including ``id`` and ``status`` fields
            compatible with x402gate's async task detection.

        Raises:
            ProviderError: If the submission fails.
        """
        client = await self._get_client()
        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"

        try:
            response = await client.post(url, json=body, headers=self._get_headers())
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to connect to Tungsten: {e}",
                status_code=503,
            ) from e

        if response.status_code == 401:
            raise ProviderError(
                provider=self.name,
                detail=(
                    "Tungsten authentication failed (401). "
                    "JWT or CF clearance cookies may be expired."
                ),
                status_code=401,
            )

        if response.status_code >= 400:
            detail = self._parse_error(response)
            raise ProviderError(
                provider=self.name,
                detail=detail,
                status_code=response.status_code,
            )

        result = response.json()

        # Tungsten returns a list of generation objects.
        # Each item has: { "uuid": "<image_uuid>",
        #   "generation": { "uuid": "<generation_uuid>", ... } }
        # The track endpoint needs the GENERATION uuid, not the image uuid.
        if isinstance(result, list) and len(result) > 0:
            first = result[0] if isinstance(result[0], dict) else {}

            # Extract generation UUID (nested), fallback to top-level uuid
            task_id = ""
            generation = first.get("generation")
            if isinstance(generation, dict):
                task_id = generation.get("uuid", "")
            if not task_id:
                task_id = first.get("uuid", "")

            logger.info(
                "Tungsten generation submitted: %s (%d image(s))",
                task_id,
                len(result),
            )
            return {
                "data": {
                    "id": task_id,
                    "status": "processing",
                    "tungsten_raw": result,
                }
            }

        raise ProviderError(
            provider=self.name,
            detail=f"Unexpected Tungsten response format: {result}",
        )

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll Tungsten for generation completion and download images.

        Polls POST /v1/generations/track until the task completes,
        then downloads images server-side and returns them as base64.

        Args:
            task_id: Generation UUID from submit() response.

        Returns:
            Dict with ``images`` list containing base64-encoded images.

        Raises:
            ProviderError: If the generation fails.
            TaskTimeoutError: If polling exceeds poll_timeout.
        """
        from x402gate.core.proxy import TaskTimeoutError

        client = await self._get_client()
        track_url = f"{self._config.base_url.rstrip('/')}/generations/track"
        elapsed = 0
        prev_status = ""

        # Brief delay before first poll
        await asyncio.sleep(1)

        while elapsed < self._config.poll_timeout:
            try:
                response = await client.post(
                    track_url,
                    json={"uuids": [task_id]},
                    headers=self._get_headers(),
                )
            except Exception as e:
                logger.warning("Tungsten track request failed: %s", e)
                await asyncio.sleep(self._config.poll_interval)
                elapsed += self._config.poll_interval
                continue

            if response.status_code == 401:
                raise ProviderError(
                    provider=self.name,
                    detail="Tungsten cookies expired during polling",
                    status_code=401,
                )

            if response.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Track request failed: {response.text}",
                    status_code=response.status_code,
                )

            result = response.json()

            # Parse status from the list response
            status_info = None
            if isinstance(result, list) and len(result) > 0:
                status_info = result[0] if isinstance(result[0], dict) else None
            elif isinstance(result, dict):
                status_info = result

            if status_info:
                status = status_info.get("status", "")

                if status in ("completed", "succeeded", "done", "success"):
                    logger.info("Tungsten task %s completed after %ds", task_id, elapsed)
                    return await self._download_images(result)

                if status in ("failed", "error", "cancelled"):
                    reason = status_info.get("failure_reason", status)
                    raise ProviderError(
                        provider=self.name,
                        detail=f"Generation failed: {reason}",
                    )
            else:
                logger.warning(
                    "Tungsten task %s: no status in response (elapsed: %ds)",
                    task_id,
                    elapsed,
                )

            await asyncio.sleep(self._config.poll_interval)
            elapsed += self._config.poll_interval

        raise TaskTimeoutError(task_id=task_id, timeout=self._config.poll_timeout)

    async def _download_images(self, track_result: list | dict) -> dict[str, Any]:
        """Download generated images and encode them as base64.

        Image URLs on api.tungsten.run require authentication cookies,
        so we download them server-side and return base64 to the client.

        The track response contains ``image_upload_uuids`` — a list of
        UUIDs that map to download URLs:
        ``{base_url}/generated_images/{uuid}/png``

        Falls back to ``original_url_png`` for submit-response format.

        Args:
            track_result: Completed track response (list or dict).

        Returns:
            Dict with ``images`` list, each containing ``base64_png``.
        """
        client = await self._get_client()
        headers = self._get_headers()
        headers["Accept"] = "image/png,image/webp,image/*,*/*"

        base_url = self._config.base_url.rstrip("/")

        # Normalize to list
        items = track_result if isinstance(track_result, list) else [track_result]
        images: list[dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            # Strategy 1: image_upload_uuids from track response
            # Tungsten returns this as a dict: {image_uuid: upload_uuid}
            upload_uuids = item.get("image_upload_uuids")
            if upload_uuids and isinstance(upload_uuids, dict):
                for img_uuid in upload_uuids:
                    if not isinstance(img_uuid, str) or not img_uuid:
                        continue
                    download_url = f"{base_url}/generated_images/{img_uuid}/png"

                    logger.info("Downloading image %s from %s", img_uuid, download_url)
                    image_b64 = await self._download_single_image(client, download_url, headers)

                    images.append({"base64_png": image_b64})
                continue  # Skip other strategies for this item

            # Strategy 2: nested generation.images (submit response format)
            generation = item.get("generation", item)
            if isinstance(generation, dict):
                image_list = generation.get("images", [])
            else:
                image_list = item.get("images", [])

            # Strategy 3: item itself has original_url_png
            if not image_list and "original_url_png" in item:
                image_list = [item]

            for img_info in image_list:
                if not isinstance(img_info, dict):
                    continue

                url_png = img_info.get("original_url_png", "")
                url_webp = img_info.get("original_url_webp", "")
                download_url = url_png or url_webp

                if not download_url:
                    continue

                image_b64 = await self._download_single_image(client, download_url, headers)

                images.append({"base64_png": image_b64})

        if not images:
            logger.warning("No images found in Tungsten response")

        return {"images": images, "count": len(images)}

    async def _download_single_image(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
    ) -> str:
        """Download a single image and return it as base64.

        Args:
            client: HTTP client to use.
            url: Image URL to download.
            headers: Headers with authentication cookies.

        Returns:
            Base64-encoded image string.
        """
        for attempt in range(_DOWNLOAD_RETRIES):
            try:
                response = await client.get(
                    url,
                    headers=headers,
                    timeout=_DOWNLOAD_TIMEOUT,
                )
                response.raise_for_status()
                return base64.b64encode(response.content).decode("ascii")
            except Exception as e:
                if attempt < _DOWNLOAD_RETRIES - 1:
                    delay = 2**attempt
                    logger.warning(
                        "Image download attempt %d/%d failed: %s, retrying in %ds",
                        attempt + 1,
                        _DOWNLOAD_RETRIES,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise ProviderError(
                        provider=self.name,
                        detail=f"Failed to download image after {_DOWNLOAD_RETRIES} attempts: {e}",
                    ) from e
        # Unreachable, but keeps type checkers happy
        return ""  # pragma: no cover

    @staticmethod
    def _parse_error(response: httpx.Response) -> str:
        """Extract a human-readable error message from a Tungsten error response."""
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                data = data[0] if isinstance(data[0], dict) else {}
            if isinstance(data, dict):
                detail = data.get("detail", "")
                code = data.get("code")
                if code == 30100:
                    blocked = data.get("detail_struct", {}).get("blocked", [])
                    reasons = ", ".join(blocked) if blocked else "unspecified"
                    return f"Prompt blocked by safety filter: {reasons}"
                if code == 30002:
                    return "Tungsten concurrency limit reached, try again later"
                if detail:
                    return str(detail)
        except Exception:
            pass
        return response.text

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
