"""MiniMax TTS provider for x402gate.

Synchronous provider — audio is returned directly from submit().
Pricing is calculated locally from the character count.

API reference: https://www.minimaxi.com/document/T2A%20V2
"""

from __future__ import annotations

import base64
import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError, require_text

logger = logging.getLogger(__name__)

# Default pricing per 1 000 characters (official Pay-as-you-go).
# Can be overridden via config.yaml → providers.minimax.pricing
_DEFAULT_PRICE_PER_1K_TURBO = Decimal("0.06")
_DEFAULT_PRICE_PER_1K_HD = Decimal("0.10")

_ONE_THOUSAND = Decimal("1000")


def _rate_for_model(model: str) -> Decimal:
    """Return price per 1K characters based on model name.

    Defaults to the expensive HD rate.  Only uses the cheaper turbo
    rate when the model name explicitly contains ``turbo``.
    """
    if "turbo" in model.lower():
        return _DEFAULT_PRICE_PER_1K_TURBO
    return _DEFAULT_PRICE_PER_1K_HD


class MinimaxProvider(BaseProvider):
    """MiniMax TTS provider — synchronous, character-based pricing.

    Sends text to MiniMax T2A V2 API, receives audio data, and returns
    it as base64-encoded content for the gateway JSON response wrapper.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="minimax", config=config)
        pricing = config.pricing or {}
        self._price_turbo = Decimal(
            str(pricing.get("per_1k_chars_turbo", _DEFAULT_PRICE_PER_1K_TURBO))
        )
        self._price_hd = Decimal(str(pricing.get("per_1k_chars_hd", _DEFAULT_PRICE_PER_1K_HD)))

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Calculate cost from character count and model type.

        Args:
            model_path: Model name (e.g. 'speech-02-hd').
            inputs: Must contain ``text``.

        Returns:
            Base price in USD.
        """
        text = require_text(inputs, self.name)
        char_count = len(text)

        # model from body, or from URL path (e.g. "speech-2.8-hd") — never empty
        model = inputs.get("model", model_path)
        rate = _rate_for_model(model)
        price = Decimal(char_count) / _ONE_THOUSAND * rate

        logger.info(
            "MiniMax price: %d chars × $%s/1k = $%s (model=%s)",
            char_count,
            rate,
            price,
            model,
        )
        return price

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Send TTS request to MiniMax and return audio as base64.

        Args:
            path: Model path from URL (e.g. 'speech-02-hd').
            body: Must contain ``text``, optionally ``model``, ``voice_setting``.
            prepaid: Unused.

        Returns:
            Dict with ``audio_base64``, ``content_type``, and ``characters``.

        Raises:
            ProviderError: On validation failure or API error.
        """
        text = require_text(body, self.name)

        model = body.get("model", path)
        voice_setting = body.get("voice_setting", {})
        timber_weights = body.get("timber_weights", [])

        api_url = f"{self._config.base_url.rstrip('/')}/v1/t2a_v2"

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "text": text,
        }
        if voice_setting:
            payload["voice_setting"] = voice_setting
        if timber_weights:
            payload["timber_weights"] = timber_weights

        # Forward any extra params the client sends
        for key in ("audio_setting", "language_boost", "pronunciation_dict"):
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
                detail=f"MiniMax request timed out after {self._config.poll_timeout}s",
                status_code=504,
            ) from None
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to call MiniMax: {e}",
            ) from e

        if resp.status_code >= 400:
            status = 502 if resp.status_code in (401, 402, 403) else resp.status_code
            raise ProviderError(
                provider=self.name,
                detail=f"MiniMax error ({resp.status_code}): {resp.text[:500]}",
                status_code=status,
            )

        result = resp.json()

        # MiniMax T2A V2 returns {"data": {"audio": "<hex-encoded audio>"}}
        # or {"base_resp": {"status_code": 0}, "data": {"audio": "..."}}
        base_resp = result.get("base_resp", {})
        if base_resp.get("status_code", 0) != 0:
            raise ProviderError(
                provider=self.name,
                detail=f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}",
            )

        audio_hex = result.get("data", {}).get("audio", "")
        if not audio_hex:
            raise ProviderError(
                provider=self.name,
                detail="MiniMax returned no audio data",
            )

        # MiniMax returns hex-encoded audio; convert to base64 for consistency
        try:
            audio_bytes = bytes.fromhex(audio_hex)
        except ValueError:
            # Might already be base64 in newer API versions
            audio_bytes = base64.b64decode(audio_hex)

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        logger.info(
            "MiniMax TTS: model=%s, %d chars → %d bytes audio",
            model,
            len(text),
            len(audio_bytes),
        )

        # Derive content_type from audio_setting.format (default: mp3)
        audio_fmt = body.get("audio_setting", {}).get("format", "mp3")
        mime_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
            "ogg": "audio/ogg",
        }
        content_type = mime_map.get(audio_fmt, "audio/mpeg")

        return {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "characters": len(text),
        }

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("MinimaxProvider is synchronous; polling is not needed")
