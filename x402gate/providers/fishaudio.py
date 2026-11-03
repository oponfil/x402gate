"""Fish Audio TTS provider for x402gate.

Synchronous provider — audio is returned directly from submit().
Pricing is calculated locally from the UTF-8 byte count of the text.

API reference: https://fish.audio/
"""

from __future__ import annotations

import base64
import logging
import re
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.http import shared_client
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL, rejected_credentials
from x402gate.providers.base import (
    BaseProvider,
    ProviderError,
    optional_str,
    raise_for_gateway_timeout,
    require_text,
)

logger = logging.getLogger(__name__)

# Default pricing: $0.015 per 1 000 UTF-8 bytes.
# Can be overridden via config.yaml → providers.fishaudio.pricing
_DEFAULT_PRICE_PER_1K_BYTES = Decimal("0.015")
_ONE_THOUSAND = Decimal("1000")

# 'model' is sent as an HTTP header. A value httpx/h11 cannot encode
# (non-ASCII, CR/LF, NUL) would fail only after payment, so only plain
# model names such as "s1" or "speech-1.6" pass.
_MODEL_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")

# Sampling settings forwarded to Fish Audio, each a number from 0 to 1
# (default 0.7). A higher temperature gives a more varied intonation.
_SAMPLING_PARAMS = ("temperature", "top_p")


def _model_header(inputs: dict[str, Any], provider_name: str) -> str:
    """Return the optional ``model`` header value, ``""`` when missing or null.

    Raises:
        ProviderError: 400 for a non-string or a value that is not a plain
            model name.
    """
    model = optional_str(inputs, "model", provider_name)
    if model and not _MODEL_RE.fullmatch(model):
        raise ProviderError(
            provider=provider_name,
            detail="'model' must be a plain model name: 1-64 letters, digits, '.', '_' or '-'",
            status_code=400,
        )
    return model


def _check_sampling(inputs: dict[str, Any], provider_name: str) -> None:
    """Refuse a temperature or top_p Fish Audio would reject, before payment.

    Raises:
        ProviderError: 400 for a value that is not a number from 0 to 1.
    """
    for key in _SAMPLING_PARAMS:
        value = inputs.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ProviderError(
                provider=provider_name,
                detail=f"'{key}' must be a number from 0 to 1",
                status_code=400,
            )


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
        text = require_text(inputs, self.name)
        # model becomes an HTTP header; both are checked before payment.
        _model_header(inputs, self.name)
        optional_str(inputs, "reference_id", self.name)
        _check_sampling(inputs, self.name)

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
        text = require_text(body, self.name)
        _check_sampling(body, self.name)

        reference_id = optional_str(body, "reference_id", self.name)
        chunk_length = body.get("chunk_length", 200)
        audio_format = body.get("format", "mp3")
        latency = body.get("latency", "normal")

        api_url = f"{self._config.base_url.rstrip('/')}/v1/tts"

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        # Fish Audio API expects 'model' as an HTTP header, not a JSON body field
        model = _model_header(body, self.name)
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
        for key in _SAMPLING_PARAMS:
            if body.get(key) is not None:
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
                detail=f"FishAudio request timed out after {self._config.poll_timeout}s",
                status_code=504,
            ) from None
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to call FishAudio: {e}",
            ) from e

        if rejected_credentials(resp.status_code):
            logger.error("FishAudio rejected our key (%d): %s", resp.status_code, resp.text[:300])
            raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL)

        raise_for_gateway_timeout(resp.status_code, self.name, "FishAudio")

        # httpx does not follow redirects, so a 3xx never reached the API.
        if not 200 <= resp.status_code < 300:
            raise ProviderError(
                provider=self.name,
                detail=f"FishAudio error ({resp.status_code}): {resp.text[:500]}",
                status_code=resp.status_code if resp.status_code >= 400 else 502,
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
