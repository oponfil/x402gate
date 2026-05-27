"""Parse audio file duration for STT pricing."""

from __future__ import annotations

import io
import math
import wave

from mutagen import File as MutagenFile

from x402gate.providers.base import ProviderError


def get_audio_duration_seconds(data: bytes, fmt: str, *, provider: str = "") -> float:
    """Return audio duration in seconds from raw file bytes.

    Args:
        data: Raw audio file content.
        fmt: Format string (e.g. ``wav``, ``mp3``).
        provider: Provider name for error messages (caller-supplied).

    Returns:
        Duration in seconds (may be fractional).

    Raises:
        ProviderError: If format is unsupported or duration cannot be determined.
    """
    normalized = fmt.strip().lower().lstrip(".")
    if not normalized:
        raise ProviderError(
            provider=provider,
            detail="Missing input_audio.format",
            status_code=400,
        )

    if normalized == "wav":
        return _wav_duration(data, provider=provider)

    audio_file = MutagenFile(io.BytesIO(data))
    if audio_file is None or audio_file.info is None:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse audio format '{normalized}'",
            status_code=400,
        )

    length = getattr(audio_file.info, "length", None)
    if length is None:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot determine duration for format '{normalized}'",
            status_code=400,
        )

    duration = float(length)
    if duration <= 0:
        raise ProviderError(
            provider=provider,
            detail="Audio duration must be greater than zero",
            status_code=400,
        )
    return duration


def billing_seconds(duration: float) -> int:
    """Round duration up to whole seconds for conservative STT billing."""
    return max(1, math.ceil(duration))


def _wav_duration(data: bytes, *, provider: str = "") -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                raise ProviderError(
                    provider=provider,
                    detail="Invalid WAV sample rate",
                    status_code=400,
                )
            duration = frames / rate
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse WAV audio: {exc}",
            status_code=400,
        ) from exc

    if duration <= 0:
        raise ProviderError(
            provider=provider,
            detail="Audio duration must be greater than zero",
            status_code=400,
        )
    return duration
