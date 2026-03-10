"""Social Download provider for x402gate.

Downloads media from social networks (YouTube, TikTok, Instagram,
Twitter/X, VK, Rutube, Facebook, etc.) via RapidAPI's
"Social Download All In One" service.

This is a synchronous provider — the result is returned directly
from submit() without polling. The gateway auto-detects this because
the response has no {"data": {"id": ..., "status": ...}} structure.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)


def _select_best_media(api_response: list[dict]) -> dict[str, Any] | None:
    """Pick the best downloadable media from the RapidAPI response.

    Priority:
    1. Combined (video+audio) MP4 with highest resolution.
    2. Any MP4 video with highest resolution.
    3. First available media with a URL.

    Returns a dict with url, type, extension, quality, width, height
    or None if nothing suitable is found.
    """
    if not api_response or not isinstance(api_response, list):
        return None

    first = api_response[0] if api_response else None
    if not first or not isinstance(first, dict):
        return None

    if first.get("error"):
        return None

    medias: list[dict] = first.get("medias") or []
    if not medias:
        return None

    def _resolution(m: dict) -> int:
        try:
            return int(m.get("width") or 0) * int(m.get("height") or 0)
        except (ValueError, TypeError):
            return 0

    # Prefer MP4 videos
    mp4_videos = [
        m
        for m in medias
        if m.get("type") == "video" and m.get("extension") == "mp4" and m.get("url")
    ]

    # Prefer combined (video+audio) tracks
    combined = [m for m in mp4_videos if m.get("is_audio") is True]
    candidates = combined or mp4_videos

    if candidates:
        best = max(candidates, key=_resolution)
    else:
        # Fallback: any media with a URL
        with_url = [m for m in medias if m.get("url")]
        if not with_url:
            return None
        best = with_url[0]

    return {
        "url": best["url"],
        "type": best.get("type", "video"),
        "extension": best.get("extension", "mp4"),
        "quality": best.get("quality", ""),
        "width": best.get("width"),
        "height": best.get("height"),
    }


class SocialDownloadProvider(BaseProvider):
    """Social Download provider — synchronous, fixed-price.

    Uses RapidAPI "Social Download All In One" to extract direct
    download links for media from social networks.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="socialdownload", config=config)

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Return the configured fixed price per download.

        Raises:
            ProviderError: If fixed_price_usd is not configured.
        """
        if not self._config.fixed_price_usd:
            raise ProviderError(
                provider=self.name,
                detail="fixed_price_usd is not configured for socialdownload",
            )
        return Decimal(str(self._config.fixed_price_usd))

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Call RapidAPI to get download links for the given URL.

        This is a **synchronous** provider — the full result is returned
        directly. The gateway detects this because the response does not
        contain ``{"data": {"id": ..., "status": ...}}``.

        Args:
            path: Ignored (single model: "download").
            body: Must contain ``{"url": "https://..."}``
            prepaid: Unused for this provider.

        Returns:
            Dict with source metadata and media download links.

        Raises:
            ProviderError: On validation failure or API error.
        """
        url = body.get("url", "")
        if not url:
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain 'url' field",
                status_code=400,
            )

        # Call RapidAPI
        api_url = f"{self._config.base_url.rstrip('/')}/v1/social/autolink"
        headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": "social-download-all-in-one.p.rapidapi.com",
            "x-rapidapi-key": self._config.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    api_url,
                    json={"url": url},
                    headers=headers,
                )
        except httpx.TimeoutException:
            raise ProviderError(
                provider=self.name,
                detail="RapidAPI request timed out after 30s",
                status_code=504,
            ) from None
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to call RapidAPI: {e}",
            ) from e

        if resp.status_code >= 400:
            raise ProviderError(
                provider=self.name,
                detail=f"RapidAPI error ({resp.status_code}): {resp.text[:300]}",
                status_code=resp.status_code,
            )

        api_response = resp.json()

        # RapidAPI returns a list; normalise if needed
        if isinstance(api_response, dict):
            api_response = [api_response]

        # Extract metadata from first item
        first = api_response[0] if api_response else {}
        if first.get("error"):
            raise ProviderError(
                provider=self.name,
                detail=f"RapidAPI returned error: {first.get('error')}",
            )

        best = _select_best_media(api_response)

        # Build response — return ALL medias + highlight the best one
        result: dict[str, Any] = {
            "title": first.get("title", ""),
            "author": first.get("author", ""),
            "source": first.get("source", ""),
            "duration": first.get("duration"),
            "thumbnail": first.get("thumbnail", ""),
            "medias": first.get("medias", []),
        }

        if best:
            result["best_media"] = best

        logger.info(
            "SocialDownload: %s — %s (%s)",
            result.get("source", "?"),
            result.get("title", "?")[:50],
            f"{best['width']}x{best['height']}" if best and best.get("width") else "no video",
        )

        return result

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("SocialDownloadProvider is synchronous; polling is not needed")
