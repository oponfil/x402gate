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

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.http import shared_client
from x402gate.core.proxy import (
    UPSTREAM_CREDENTIALS_DETAIL,
    UpstreamMayHaveRun,
    rejected_credentials,
)
from x402gate.providers.base import (
    BaseProvider,
    ProviderError,
    optional_str,
    raise_for_gateway_timeout,
    require_text,
)

logger = logging.getLogger(__name__)

# Default pricing per 1 000 characters (official Pay-as-you-go).
# Can be overridden via config.yaml → providers.minimax.pricing
_DEFAULT_PRICE_PER_1K_TURBO = Decimal("0.06")
_DEFAULT_PRICE_PER_1K_HD = Decimal("0.10")

_ONE_THOUSAND = Decimal("1000")

# base_resp codes for our own key or balance, sent with HTTP 200:
# 1004 authentication failed, 1008 insufficient balance, 2049 invalid API key.
_CREDENTIALS_CODES = frozenset({1004, 1008, 2049})

_MIME_BY_FORMAT = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "ogg": "audio/ogg",
}


def _check_audio_setting(inputs: dict[str, Any], provider_name: str) -> None:
    """Reject an ``audio_setting`` the response could not be built from.

    Checked in get_price(), before payment: submit() reads
    ``audio_setting.format`` only after MiniMax made and billed the audio.
    A null ``audio_setting`` or ``format`` is dropped like a missing one.

    Raises:
        ProviderError: 400 for a non-object ``audio_setting`` or a
            non-string ``format``.
    """
    audio_setting = inputs.get("audio_setting")
    if audio_setting is None:
        return
    if not isinstance(audio_setting, dict):
        raise ProviderError(
            provider=provider_name,
            detail="'audio_setting' must be an object",
            status_code=400,
        )
    fmt = audio_setting.get("format")
    if fmt is not None and not isinstance(fmt, str):
        raise ProviderError(
            provider=provider_name,
            detail="'audio_setting.format' must be a string",
            status_code=400,
        )


def _rate_for_model(
    model: str,
    turbo: Decimal = _DEFAULT_PRICE_PER_1K_TURBO,
    hd: Decimal = _DEFAULT_PRICE_PER_1K_HD,
) -> Decimal:
    """Return price per 1K characters based on model name.

    Defaults to the expensive HD rate.  Only uses the cheaper turbo
    rate when the model name explicitly contains ``turbo``.
    """
    if "turbo" in model.lower():
        return turbo
    return hd


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
        _check_audio_setting(inputs, self.name)
        char_count = len(text)

        # model from body, or from URL path (e.g. "speech-2.8-hd") — never empty
        model = optional_str(inputs, "model", self.name) or model_path
        rate = _rate_for_model(model, self._price_turbo, self._price_hd)
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

        model = optional_str(body, "model", self.name) or path
        _check_audio_setting(body, self.name)
        audio_setting = body.get("audio_setting")
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

        if audio_setting is not None:
            # A null format is left out so MiniMax applies its own mp3
            # default, which the audio/mpeg content_type below assumes.
            payload["audio_setting"] = {
                k: v for k, v in audio_setting.items() if not (k == "format" and v is None)
            }
        # Forward any extra params the client sends
        for key in ("language_boost", "pronunciation_dict"):
            if key in body:
                payload[key] = body[key]

        try:
            resp = await shared_client().post(
                api_url,
                json=payload,
                headers=headers,
                timeout=http.timeout(self._config.poll_timeout),
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

        if rejected_credentials(resp.status_code):
            logger.error("MiniMax rejected our key (%d): %s", resp.status_code, resp.text[:300])
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL)

        raise_for_gateway_timeout(resp.status_code, self.name, "MiniMax")

        # httpx does not follow redirects, so a 3xx never reached the API.
        if not 200 <= resp.status_code < 300:
            raise ProviderError(
                provider=self.name,
                detail=f"MiniMax error ({resp.status_code}): {resp.text[:500]}",
                status_code=resp.status_code if resp.status_code >= 400 else 502,
            )

        try:
            result = resp.json()
        except ValueError:
            result = None
        if not isinstance(result, dict):
            # A 2xx may have made and billed the audio: the payment is kept.
            raise ProviderError(
                provider=self.name,
                detail=f"MiniMax returned an unreadable response: {resp.text[:300]}",
            ) from UpstreamMayHaveRun("MiniMax answered 2xx")

        # MiniMax T2A V2 returns {"data": {"audio": "<hex-encoded audio>"}}
        # or {"base_resp": {"status_code": 0}, "data": {"audio": "..."}}.
        # A null or non-object part counts as missing rather than crashing.
        base_resp = result.get("base_resp")
        if not isinstance(base_resp, dict):
            base_resp = {}
        status_code = base_resp.get("status_code", 0)
        if status_code in _CREDENTIALS_CODES:
            # No audio was made, so this is refunded; the message stays private.
            logger.error(
                "MiniMax rejected our key (base_resp %s): %s",
                status_code,
                base_resp.get("status_msg"),
            )
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL)
        if status_code != 0:
            raise ProviderError(
                provider=self.name,
                detail=f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}",
            )

        data = result.get("data")
        audio_hex = data.get("audio") if isinstance(data, dict) else None
        if not isinstance(audio_hex, str) or not audio_hex:
            # MiniMax answered 2xx without reporting an error: the audio may be billed.
            raise ProviderError(
                provider=self.name,
                detail="MiniMax returned no audio data",
            ) from UpstreamMayHaveRun("MiniMax answered 2xx")

        # MiniMax returns hex-encoded audio; convert to base64 for consistency
        try:
            audio_bytes = bytes.fromhex(audio_hex)
        except ValueError:
            # Might already be base64 in newer API versions. validate=True stops
            # b64decode from dropping junk characters and serving the rest.
            try:
                audio_bytes = base64.b64decode(audio_hex, validate=True)
            except ValueError:  # binascii.Error subclasses ValueError
                raise ProviderError(
                    provider=self.name,
                    detail="MiniMax returned audio that could not be decoded",
                ) from UpstreamMayHaveRun("MiniMax answered 2xx")

        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        logger.info(
            "MiniMax TTS: model=%s, %d chars → %d bytes audio",
            model,
            len(text),
            len(audio_bytes),
        )

        # Derive content_type from audio_setting.format (default: mp3)
        audio_fmt = (audio_setting or {}).get("format") or "mp3"
        content_type = _MIME_BY_FORMAT.get(audio_fmt, "audio/mpeg")

        output: dict[str, Any] = {
            "audio_base64": audio_b64,
            "content_type": content_type,
            "characters": len(text),
            "model": model,
        }
        # MiniMax bills its own character count, which can differ from len(text).
        extra_info = result.get("extra_info")
        usage_chars = extra_info.get("usage_characters") if isinstance(extra_info, dict) else None
        if isinstance(usage_chars, int) and not isinstance(usage_chars, bool) and usage_chars >= 0:
            output["usage_characters"] = usage_chars
        return output

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Price the characters MiniMax reports it billed.

        The gateway caps a prepaid charge at the get_price() estimate.

        Returns:
            Cost in USD, or None when MiniMax sent no usage_characters.
        """
        usage_chars = result.get("usage_characters")
        if usage_chars is None:
            return None
        model = result.get("model") or body.get("model", "")
        rate = _rate_for_model(str(model), self._price_turbo, self._price_hd)
        return Decimal(usage_chars) / _ONE_THOUSAND * rate

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — this is a synchronous provider."""
        raise NotImplementedError("MinimaxProvider is synchronous; polling is not needed")
