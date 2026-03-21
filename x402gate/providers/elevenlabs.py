"""ElevenLabs TTS provider for x402gate.

Synchronous provider — audio is returned directly from submit().
Pricing is calculated locally from the character count.

API reference: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
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

# Default pricing: cost per 1 000 characters (Creator plan overage rates).
# Can be overridden via config.yaml → providers.elevenlabs.pricing
_DEFAULT_PRICE_PER_1K_STANDARD = Decimal("0.30")
_DEFAULT_PRICE_PER_1K_TURBO = Decimal("0.15")


def _is_turbo_model(model_id: str) -> bool:
    """Check if the model is a turbo/flash variant (half-price)."""
    lower = model_id.lower()
    return "turbo" in lower or "flash" in lower


class ElevenLabsProvider(BaseProvider):
    """ElevenLabs TTS provider — synchronous, character-based pricing.

    Sends text to ElevenLabs, receives audio bytes, and returns them
    as base64-encoded data so the gateway can wrap it in a JSON response.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="elevenlabs", config=config)
        pricing = config.pricing or {}
        self._price_standard = Decimal(
            str(pricing.get("per_1k_chars_standard", _DEFAULT_PRICE_PER_1K_STANDARD))
        )
        self._price_turbo = Decimal(
            str(pricing.get("per_1k_chars_turbo", _DEFAULT_PRICE_PER_1K_TURBO))
        )

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Calculate cost from character count and model type.

        Args:
            model_path: voice_id (from URL).
            inputs: Must contain ``text`` and optionally ``model_id``.

        Returns:
            Base price in USD.
        """
        text = inputs.get("text", "")
        char_count = len(text)

        if char_count == 0:
            raise ProviderError(
                provider=self.name,
                detail="Request body must contain non-empty 'text' field",
                status_code=400,
            )

        model_id = inputs.get("model_id", "")
        # Default to expensive standard rate; turbo rate only if "turbo"/"flash" in model name
        rate = self._price_turbo if _is_turbo_model(model_id) else self._price_standard

        # price = character_count / 1000 * rate_per_1k_chars
        price = Decimal(char_count) / Decimal("1000") * rate
        logger.info(
            "ElevenLabs price: %d chars × $%s/1k = $%s (model=%s)",
            char_count,
            rate,
            price,
            model_id or "default",
        )
        return price

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Send TTS request to ElevenLabs and return audio as base64.

        Args:
            path: voice_id (from URL path).
            body: Must contain ``text``, optionally ``model_id``.
            prepaid: Unused.

        Returns:
            Dict with ``audio_base64`` and ``content_type``.

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

        # voice_id comes from the URL path
        voice_id = path
        model_id = body.get("model_id", "")
        output_format = body.get("output_format", "mp3_44100_128")

        api_url = f"{self._config.base_url.rstrip('/')}/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": self._config.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        payload: dict[str, Any] = {"text": text}
        if model_id:
            payload["model_id"] = model_id

        # Forward optional voice_settings if provided
        if "voice_settings" in body:
            payload["voice_settings"] = body["voice_settings"]

        params = {"output_format": output_format}

        try:
            async with httpx.AsyncClient(timeout=float(self._config.poll_timeout)) as client:
                resp = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                    params=params,
                )
        except httpx.TimeoutException:
            raise ProviderError(
                provider=self.name,
                detail=f"ElevenLabs request timed out after {self._config.poll_timeout}s",
                status_code=504,
            ) from None
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to call ElevenLabs: {e}",
            ) from e

        if resp.status_code >= 400:
            # Remap provider auth/payment errors to 502 to avoid collision
            # with x402's 402 Payment Required
            status = 502 if resp.status_code in (401, 402, 403) else resp.status_code
            raise ProviderError(
                provider=self.name,
                detail=f"ElevenLabs error ({resp.status_code}): {resp.text[:500]}",
                status_code=status,
            )

        audio_b64 = base64.b64encode(resp.content).decode("ascii")

        content_type = resp.headers.get("content-type", "audio/mpeg")

        logger.info(
            "ElevenLabs TTS: voice=%s, model=%s, %d chars → %d bytes audio",
            voice_id,
            model_id,
            len(text),
            len(resp.content),
        )

        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "characters": len(text),
        }

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("ElevenLabsProvider is synchronous; polling is not needed")
