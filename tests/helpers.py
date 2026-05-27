"""Shared helpers for unit tests."""

from __future__ import annotations

import base64
import io
import wave


def make_wav_bytes(duration_sec: float = 1.0, rate: int = 8000) -> bytes:
    """Build a minimal silent WAV file with the given duration."""
    frames = int(duration_sec * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def make_wav_b64(duration_sec: float = 1.0, rate: int = 8000) -> str:
    """Build a base64-encoded minimal silent WAV file."""
    return base64.b64encode(make_wav_bytes(duration_sec, rate)).decode()
