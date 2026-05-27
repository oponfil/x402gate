"""Unit tests for audio duration parsing."""

import pytest

from tests.helpers import make_wav_bytes
from x402gate.core.audio_duration import billing_seconds, get_audio_duration_seconds
from x402gate.providers.base import ProviderError


class TestGetAudioDurationSeconds:
    def test_wav_one_second(self):
        data = make_wav_bytes(1.0)
        assert get_audio_duration_seconds(data, "wav") == 1.0

    def test_wav_format_with_dot(self):
        data = make_wav_bytes(0.5, rate=16000)
        assert get_audio_duration_seconds(data, ".WAV") == 0.5

    def test_missing_format_raises(self):
        with pytest.raises(ProviderError, match="Missing input_audio.format"):
            get_audio_duration_seconds(make_wav_bytes(), "")

    def test_invalid_wav_raises(self):
        with pytest.raises(ProviderError, match="Cannot parse WAV"):
            get_audio_duration_seconds(b"not-a-wav", "wav")


class TestBillingSeconds:
    def test_rounds_up_fractional(self):
        assert billing_seconds(1.1) == 2

    def test_minimum_one_second(self):
        assert billing_seconds(0.01) == 1

    def test_whole_seconds_unchanged(self):
        assert billing_seconds(3.0) == 3
