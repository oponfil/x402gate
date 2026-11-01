"""Social Download provider for x402gate.

Downloads media from social networks (YouTube, TikTok, Instagram,
Twitter/X, VK, Rutube, Facebook, etc.) via RapidAPI's
"Social Download All In One" service.

This is a synchronous provider — the result is returned directly
from submit() without polling. The gateway auto-detects this because
the response has no {"data": {"id": ..., "status": ...}} structure.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.http import shared_client
from x402gate.core.proxy import (
    UPSTREAM_CREDENTIALS_DETAIL,
    UpstreamMayHaveRun,
    rejected_credentials,
)
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)


def _error_message(item: dict[str, Any]) -> str:
    """Return the human-readable text of a RapidAPI error item.

    ``error`` is sometimes the message itself and sometimes just ``true``
    next to a separate message field. With no text at all, the item is
    returned as JSON so the cause is still visible.
    """
    error = item.get("error")
    if isinstance(error, str) and error.strip():
        return error
    for key in ("message", "msg", "detail"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(item, ensure_ascii=False, default=str)[:300]


def _require_url(body: dict[str, Any], provider_name: str) -> str:
    """Return the media URL from the request body, or raise ProviderError 400.

    Shared by get_price() and submit(): get_price() runs before payment, so a
    request without a URL is turned away before it is paid for. Any scheme is
    allowed (schemeless links such as ``youtu.be/x`` work), but a non-string
    or blank value can never resolve, so it is refused too.
    """
    url = body.get("url", "")
    if not isinstance(url, str) or not url.strip():
        raise ProviderError(
            provider=provider_name,
            detail="Request body must contain 'url' field",
            status_code=400,
        )
    return url


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
            ProviderError: If fixed_price_usd is not configured, or 400 if
                the body has no ``url``.
        """
        if not self._config.fixed_price_usd:
            raise ProviderError(
                provider=self.name,
                detail="fixed_price_usd is not configured for socialdownload",
            )
        _require_url(inputs, self.name)
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
        url = _require_url(body, self.name)

        # Call RapidAPI
        api_url = f"{self._config.base_url.rstrip('/')}/v1/social/autolink"
        headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": "social-download-all-in-one.p.rapidapi.com",
            "x-rapidapi-key": self._config.api_key,
        }

        try:
            resp = await shared_client().post(
                api_url,
                json={"url": url},
                headers=headers,
                timeout=http.timeout(30.0),
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

        if rejected_credentials(resp.status_code):
            logger.error("RapidAPI rejected our key (%d): %s", resp.status_code, resp.text[:300])
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL)

        # Only 2xx ran the call. httpx does not follow redirects, so a 3xx
        # never reached the API and is refunded like a 4xx/5xx.
        if not 200 <= resp.status_code < 300:
            raise ProviderError(
                provider=self.name,
                detail=f"RapidAPI error ({resp.status_code}): {resp.text[:300]}",
                status_code=resp.status_code if resp.status_code >= 400 else 502,
            )

        # From here on RapidAPI answered 2xx: the call ran and RapidAPI billed
        # it. Every failure below is chained to UpstreamMayHaveRun so the
        # client's payment is kept rather than refunded for a call we paid for.
        try:
            return self._build_result(resp.json())
        except ProviderError as e:
            raise e from UpstreamMayHaveRun("RapidAPI billed the call")
        except Exception as e:
            logger.error("Unreadable RapidAPI response: %s: %s", type(e).__name__, e)
            raise ProviderError(
                provider=self.name,
                detail=f"RapidAPI returned an unreadable response: {resp.text[:300]}",
            ) from UpstreamMayHaveRun("RapidAPI billed the call")

    def _build_result(self, api_response: Any) -> dict[str, Any]:
        """Turn a 2xx RapidAPI body into the gateway's response.

        Raises:
            ProviderError: If RapidAPI reported an error item.
        """
        # RapidAPI returns a list; normalise if needed
        if isinstance(api_response, dict):
            api_response = [api_response]
        if not isinstance(api_response, list):
            raise TypeError(f"expected a list, got {type(api_response).__name__}")

        # Extract metadata from first item
        first = api_response[0] if api_response else {}
        if first.get("error"):
            raise ProviderError(
                provider=self.name,
                detail=f"RapidAPI returned error: {_error_message(first)}",
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
            str(result.get("title") or "?")[:50],
            f"{best['width']}x{best['height']}" if best and best.get("width") else "no video",
        )

        return result

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("SocialDownloadProvider is synchronous; polling is not needed")
