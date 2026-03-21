"""Fish Audio TTS provider for x402gate.

Synchronous provider — audio is returned directly from submit().
Pricing is calculated locally from the UTF-8 byte count of the text.

API reference: https://fish.audio/
"""

from __future__ import annotations

import base64
import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# Default pricing: $0.015 per 1 000 UTF-8 bytes.
# Can be overridden via config.yaml → providers.fishaudio.pricing
_DEFAULT_PRICE_PER_1K_BYTES = Decimal("0.015")
_ONE_THOUSAND = Decimal("1000")


class FishAudioProvider(BaseProvider):
    """Fish Audio TTS provider — synchronous, UTF-8 byte-based pricing.

    Sends text to Fish Audio, receives audio bytes, and returns them
    as base64-encoded data for the gateway JSON wrapper.

    Important: pricing is per UTF-8 byte, not per character.
    Cyrillic = 2 bytes/char, CJK = 3 bytes/char, emoji = 4 bytes/char.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="fishaudio", config=config)
        pricing = config.pricing or {}
        self._price_per_1k = Decimal(str(pricing.get("per_1k_bytes", _DEFAULT_PRICE_PER_1K_BYTES)))

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Calculate cost from UTF-8 byte count.

        Args:
            model_path: Ignored (single endpoint).
            inputs: Must contain ``text``.

        Returns:
            Base price in USD.
        """
        text = inputs.get("text", "")
        if not text:
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain non-empty 'text' field",
                status_code=400,
            )

        utf8_bytes = len(text.encode("utf-8"))
        price = Decimal(utf8_bytes) / _ONE_THOUSAND * self._price_per_1k

        logger.info(
            "FishAudio price: %d bytes (UTF-8) × $%s/1k = $%s",
            utf8_bytes,
            self._price_per_1k,
            price,
        )
        return price

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Send TTS request to Fish Audio and return audio as base64.

        Args:
            path: Ignored (single endpoint: 'tts').
            body: Must contain ``text``, optionally ``reference_id``.
            prepaid: Unused.

        Returns:
            Dict with ``audio_base64``, ``content_type``, and ``characters``.

        Raises:
            ProviderError: On validation failure or API error.
        """
        text = body.get("text", "")
        if not text:
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain non-empty 'text' field",
                status_code=400,
            )

        reference_id = body.get("reference_id", "")
        chunk_length = body.get("chunk_length", 200)
        audio_format = body.get("format", "mp3")
        latency = body.get("latency", "normal")

        api_url = f"{self._config.base_url.rstrip('/')}/v1/tts"

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        # Fish Audio API expects 'model' as an HTTP header, not a JSON body field
        model = body.get("model", "")
        if model:
            headers["model"] = model

        payload: dict[str, Any] = {
            "text": text,
            "format": audio_format,
            "chunk_length": chunk_length,
            "latency": latency,
        }
        if reference_id:
            payload["reference_id"] = reference_id

        # Forward any extra params the client sends (except model — sent as header)
        for key in ("mp3_bitrate", "opus_bitrate", "normalize", "prosody"):
            if key in body:
                payload[key] = body[key]

        try:
            async with httpx.AsyncClient(timeout=float(self._config.poll_timeout)) as client:
                resp = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException:
            raise ProviderError(
                provider=self.name,
                detail=f"FishAudio request timed out after {self._config.poll_timeout}s",
                status_code=504,
            ) from None
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to call FishAudio: {e}",
            ) from e

        if resp.status_code >= 400:
            status = 502 if resp.status_code in (401, 402, 403) else resp.status_code
            raise ProviderError(
                provider=self.name,
                detail=f"FishAudio error ({resp.status_code}): {resp.text[:500]}",
                status_code=status,
            )

        audio_b64 = base64.b64encode(resp.content).decode("ascii")
        content_type = resp.headers.get("content-type", "audio/mpeg")

        utf8_bytes = len(text.encode("utf-8"))
        logger.info(
            "FishAudio TTS: %d chars (%d UTF-8 bytes) → %d bytes audio",
            len(text),
            utf8_bytes,
            len(resp.content),
        )

        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "characters": len(text),
            "utf8_bytes": utf8_bytes,
        }

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("FishAudioProvider is synchronous; polling is not needed")
