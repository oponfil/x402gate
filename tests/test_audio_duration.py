"""Unit tests for audio duration parsing."""

import io
import random
import struct
import threading
import time
import wave
from unittest.mock import patch

import pytest
from mutagen import File as MutagenFile
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.ogg import OggPage
from mutagen.oggopus import OggOpus, OggOpusInfo
from mutagen.oggspeex import OggSpeex
from mutagen.oggvorbis import OggVorbis

from tests.helpers import make_wav_bytes
from x402gate.core import audio_duration
from x402gate.core.audio_duration import billing_seconds, get_audio_duration_seconds
from x402gate.providers.base import ProviderError


class TestGetAudioDurationSeconds:
    def test_wav_one_second(self):
        data = make_wav_bytes(1.0)
        assert get_audio_duration_seconds(data, "wav") == 1.0

    def test_wav_format_with_dot(self):
        data = make_wav_bytes(0.5, rate=16000)
        assert get_audio_duration_seconds(data, ".WAV") == 0.5

    def test_header_duration_kept_when_longer_than_content(self):
        # An honest Xing count over frames the floor counts conservatively.
        data = _xing_frame(1000) + _mp3_frame() * 230
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(1000 * _MP3_FRAME_SECONDS)

    def test_missing_format_raises(self):
        with pytest.raises(ProviderError, match="Missing input_audio.format"):
            get_audio_duration_seconds(make_wav_bytes(), "")

    def test_mutagen_parse_error_is_a_client_error(self):
        data = b"\x00\x00\x00\x10ftypM4A \x00\x00\x00\x00" + b"\x00\x00\x00\x10moov" + bytes(8)
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

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


# MPEG-1 Layer III, 128 kbps, 44.1 kHz, mono: 417-byte frames of 1152 samples.
_MP3_HEADER = b"\xff\xfb\x90\xc4"
_MP3_FRAME_SIZE = 417
_MP3_FRAME_SECONDS = 1152 / 44100


def _mp3_frame(payload: bytes = b"") -> bytes:
    body = payload.ljust(_MP3_FRAME_SIZE - 4, b"\x00")
    return _MP3_HEADER + body


def _xing_frame(frame_count: int) -> bytes:
    # Mono MPEG-1 keeps 17 bytes of side info before the Xing tag.
    return _mp3_frame(b"\x00" * 17 + b"Xing" + struct.pack(">II", 1, frame_count))


def _forge_wav_data_size(data: bytes, size: int) -> bytes:
    offset = data.index(b"data") + 4
    return data[:offset] + struct.pack("<I", size) + data[offset + 4 :]


def _aiff_bytes(seconds: float, *, claimed_frames: int) -> bytes:
    """Build a mono 16-bit 8 kHz AIFF whose COMM chunk claims ``claimed_frames``."""
    rate_8000 = b"\x40\x0b\xfa\x00\x00\x00\x00\x00\x00\x00"  # 80-bit IEEE float
    comm = struct.pack(">hIh", 1, claimed_frames, 16) + rate_8000
    samples = b"\x00\x00" * int(seconds * 8000)
    ssnd = struct.pack(">II", 0, 0) + samples
    chunks = b"COMM" + struct.pack(">I", len(comm)) + comm
    chunks += b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    return b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFF" + chunks


class TestForgedHeaderDuration:
    """Header length fields are client-controlled; the bytes set the floor."""

    def test_cbr_mp3_keeps_real_duration(self):
        frames = 230
        data = _mp3_frame() * frames
        expected = frames * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.03)

    def test_honest_xing_mp3_matches_frame_count(self):
        frames = 230
        data = _xing_frame(frames) + _mp3_frame() * frames
        expected = frames * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.03)

    def test_forged_xing_frame_count_priced_by_frames(self):
        frames = 2300  # about 60 s of audio
        data = _xing_frame(10) + _mp3_frame() * frames
        assert MutagenFile(io.BytesIO(data)).info.length < 1
        duration = get_audio_duration_seconds(data, "mp3")
        assert duration == pytest.approx(frames * _MP3_FRAME_SECONDS, abs=0.03)

    def test_frames_hidden_after_junk_still_counted(self):
        frames = 2300
        data = _xing_frame(10) + _mp3_frame() * 10 + b"\x00" * 5000 + _mp3_frame() * frames
        duration = get_audio_duration_seconds(data, "mp3")
        assert duration >= frames * _MP3_FRAME_SECONDS

    def test_forged_wav_data_chunk_priced_by_file_size(self):
        data = _forge_wav_data_size(make_wav_bytes(60.0), 16)
        with wave.open(io.BytesIO(data)) as wav_file:
            assert wav_file.getnframes() == 8
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(60.0)

    def test_normal_wav_unchanged(self):
        assert get_audio_duration_seconds(make_wav_bytes(60.0), "wav") == 60.0

    def test_forged_wav_sent_under_other_format_priced_by_file_size(self):
        data = _forge_wav_data_size(make_wav_bytes(60.0), 16)
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(60.0)

    def test_forged_aiff_frame_count_priced_by_file_size(self):
        data = _aiff_bytes(30.0, claimed_frames=8)
        assert MutagenFile(io.BytesIO(data)).info.length < 0.01
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(30.0)

    def test_honest_aiff_unchanged(self):
        data = _aiff_bytes(30.0, claimed_frames=240_000)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(30.0)

    @pytest.mark.parametrize("size", [0xFFFFFFFF, 0x7FFFFFFF], ids=["ffmpeg-pipe", "arecord"])
    def test_streaming_wav_not_priced_from_unset_size(self, size):
        # Writers to a pipe leave the RIFF and data sizes unset; no decoder
        # reads past EOF, so the PCM bytes sent are the duration.
        data = _forge_wav_data_size(make_wav_bytes(60.0, rate=16000), size)
        data = data[:4] + struct.pack("<I", size) + data[8:]
        with wave.open(io.BytesIO(data)) as wav_file:
            assert wav_file.getnframes() / 16000 > 60_000  # the trap
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(60.0)

    def test_streaming_aiff_not_priced_from_unset_size(self):
        data = _aiff_bytes(30.0, claimed_frames=0xFFFFFFFF)
        ssnd = data.index(b"SSND") + 4
        data = data[:4] + b"\xff" * 4 + data[8:ssnd] + b"\xff" * 4 + data[ssnd + 4 :]
        assert MutagenFile(io.BytesIO(data)).info.length > 500_000  # the trap
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(30.0)

    def test_lossy_container_floored_by_max_bitrate(self):
        # 10 packets (0.2 s) of 6 KB each: the bytes bound the audio higher.
        data = _ogg_opus(10, last_granule=312 + 960, packet_bytes=6000)
        expected = len(data) * 8 / 640_000
        assert expected > 0.7
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_honest_lossy_container_keeps_header_duration(self):
        data = _ogg_opus(3000, last_granule=312 + 100 * 48000)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(100.0)


def _riff(chunks: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _chunk(chunk_id: bytes, payload: bytes, *, little_endian: bool = True) -> bytes:
    size = struct.pack("<I" if little_endian else ">I", len(payload))
    return chunk_id + size + payload + b"\x00" * (len(payload) & 1)


_RATE_8000 = b"\x40\x0b\xfa\x00\x00\x00\x00\x00\x00\x00"  # 80-bit IEEE float
_AIFC_VERSION1 = 0xA2805140


def _ext80(rate: int) -> bytes:
    """Encode a whole positive sample rate as an 80-bit IEEE float."""
    shift = 64 - rate.bit_length()
    return struct.pack(">HQ", 16383 + 63 - shift, rate << shift)


def _aifc_bytes(
    compression: bytes,
    payload: bytes,
    *,
    claimed_frames: int,
    channels: int = 1,
    bits: int = 16,
    rate: bytes = _RATE_8000,
    form: bytes = b"AIFC",
    fver: int | None = _AIFC_VERSION1,
    extra: bytes = b"",
) -> bytes:
    """Build an AIFF-C (8 kHz mono 16-bit unless set) with the given compression ID."""
    comm = struct.pack(">HIH", channels, claimed_frames, bits) + rate + compression + b"\x00\x00"
    chunks = b"" if fver is None else _chunk(b"FVER", struct.pack(">I", fver), little_endian=False)
    chunks += _chunk(b"COMM", comm, little_endian=False) + extra
    chunks += _chunk(b"SSND", struct.pack(">II", 0, 0) + payload, little_endian=False)
    return b"FORM" + struct.pack(">I", 4 + len(chunks)) + form + chunks


class TestCodecChecks:
    """Formats whose byte floor would be unsound are rejected before payment."""

    def test_mp3_in_wav_sent_under_other_format_rejected(self):
        # MPEG-2.5 Layer III, 8 kbps, 8 kHz: 72-byte frames of 576 samples.
        frames = (b"\xff\xe3\x18\xc4" + b"\x00" * 68) * 1000  # 72 s of audio
        fmt = struct.pack("<HHIIHH", 0x55, 1, 8000, 1000, 65535, 0)
        data = _riff(_chunk(b"fmt ", fmt) + _chunk(b"data", frames))
        assert MutagenFile(io.BytesIO(data)).info.length < 0.01
        with pytest.raises(ProviderError, match="Cannot parse WAV"):
            get_audio_duration_seconds(data, "mp3")

    def test_wav_with_two_fmt_chunks_rejected(self):
        wav = make_wav_bytes(1.0)
        fmt_end = wav.index(b"data")
        data = wav[:fmt_end] + wav[12:fmt_end] + wav[fmt_end:]
        with pytest.raises(ProviderError, match="more than one fmt"):
            get_audio_duration_seconds(data, "wav")

    @pytest.mark.parametrize(
        "compression",
        [b"G722", b"G726", b"QDM2", b"QDMC", b"Qclp", b"SDX2", b"ADP4", b"CBD2", b"XXXX"],
    )
    def test_untimed_aifc_codec_rejected(self, compression):
        data = _aifc_bytes(compression, b"\x00" * 80_000, claimed_frames=160)
        assert MutagenFile(io.BytesIO(data)) is not None
        with pytest.raises(ProviderError, match="Unsupported AIFF audio: compression") as exc:
            get_audio_duration_seconds(data, "aiff")
        assert exc.value.status_code == 400

    def test_pcm_aifc_accepted(self):
        data = _aifc_bytes(b"sowt", b"\x00\x00" * 80_000, claimed_frames=80_000)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(10.0)

    def test_aiff_with_two_comm_chunks_rejected(self):
        data = _aiff_bytes(1.0, claimed_frames=8000)
        comm = data[12:38]
        data = data[:12] + comm + data[12:]
        with pytest.raises(ProviderError, match="more than one COMM chunk"):
            get_audio_duration_seconds(data, "aiff")


def _aiff_bitrate_seconds(data: bytes, payload: int) -> float:
    """Seconds the payload lasts at mutagen's channels x COMM bits x rate."""
    return payload * 8 / MutagenFile(io.BytesIO(data)).info.bitrate


class TestAiffCodecWidth:
    """AIFF is priced at the bytes per frame ffmpeg decodes, not COMM's sample size."""

    @pytest.mark.parametrize(
        ("compression", "width"),
        [(b"sowt", 2), (b"raw ", 1), (b"in24", 3), (b"in32", 4), (b"fl32", 4), (b"fl64", 8),
         (b"ulaw", 1), (b"alaw", 1), (b"SOWT", 2), (b"Fl32", 4), (b"FL64", 8)],
    )  # fmt: skip
    def test_fixed_width_codec_ignores_oversized_sample_size(self, compression, width):
        payload = b"\x00" * (width * 8000 * 10)
        data = _aifc_bytes(compression, payload, claimed_frames=80_000, bits=0x7FFF)
        assert _aiff_bitrate_seconds(data, len(payload)) < 0.1  # the trap
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(10.0)

    def test_stereo_fixed_width_codec(self):
        payload = b"\x00" * (2 * 3 * 8000 * 5)
        data = _aifc_bytes(b"in24", payload, claimed_frames=40_000, channels=2, bits=64)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ("bits", "width"), [(0, 1), (1, 1), (8, 1), (12, 2), (16, 2), (20, 3), (24, 3), (32, 4)]
    )
    @pytest.mark.parametrize("compression", [b"NONE", b"twos", b"none", None])
    def test_sample_size_codec_rounds_up_to_whole_bytes(self, compression, bits, width):
        payload = b"\x00" * (width * 8000 * 5)
        if compression is None:  # plain AIFF
            data = _aifc_bytes(b"", payload, claimed_frames=1, bits=bits, form=b"AIFF", fver=None)
        else:
            data = _aifc_bytes(compression, payload, claimed_frames=1, bits=bits)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(5.0)

    @pytest.mark.parametrize("compression", [b"NONE", b"twos", None])
    @pytest.mark.parametrize("bits", [33, 64, 0xFFFF])
    def test_sample_size_codec_past_32_bits_rejected(self, compression, bits):
        if compression is None:
            data = _aifc_bytes(
                b"", bytes(8000), claimed_frames=1, bits=bits, form=b"AIFF", fver=None
            )
        else:
            data = _aifc_bytes(compression, bytes(8000), claimed_frames=1, bits=bits)
        with pytest.raises(ProviderError, match=f"Unsupported AIFF audio: {bits}-bit samples"):
            get_audio_duration_seconds(data, "aiff")

    @pytest.mark.parametrize(
        ("compression", "channels", "block", "frames"),
        [(b"ima4", 1, 34, 64), (b"ima4", 2, 68, 64), (b"MAC3", 1, 2, 6), (b"MAC3", 2, 4, 6),
         (b"MAC6", 1, 1, 6), (b"MAC6", 2, 2, 6), (b"GSM ", 1, 33, 160),
         (b"GSM ", 2, 33, 160)],  # ffmpeg decodes GSM as mono whatever COMM says
    )  # fmt: skip
    def test_fixed_ratio_codec_priced_by_blocks(self, compression, channels, block, frames):
        blocks = 12_000
        data = _aifc_bytes(
            compression, b"\x00" * (block * blocks), claimed_frames=1, channels=channels
        )
        expected = blocks * frames / 8000
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(expected)

    def test_gsm_not_priced_at_sample_size(self):
        gsm_blocks = b"\x00" * 33 * 12_000  # 240 s of GSM 6.10
        data = _aifc_bytes(b"GSM ", gsm_blocks, claimed_frames=160)
        assert _aiff_bitrate_seconds(data, len(gsm_blocks)) < 30  # the trap
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(240.0)

    def test_fver_makes_plain_aiff_read_compression_type(self):
        # ffmpeg reads COMM's compression type in any FORM once an FVER
        # chunk carries AIFF-C's timestamp: this mu-law is one byte a sample.
        payload = b"\x00" * 80_000
        data = _aifc_bytes(b"ulaw", payload, claimed_frames=1, form=b"AIFF")
        assert _aiff_bitrate_seconds(data, len(payload)) == pytest.approx(5.0)  # the trap
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(10.0)

    def test_other_fver_reads_aifc_as_plain_aiff(self):
        # Any other FVER timestamp makes ffmpeg ignore the compression type.
        payload = b"\x00" * 80_000
        data = _aifc_bytes(b"ulaw", payload, claimed_frames=1, fver=0)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(5.0)
        data = _aifc_bytes(b"QDM2", payload, claimed_frames=1, fver=0)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(5.0)

    def test_aifc_comm_without_room_for_compression_read_as_plain(self):
        comm = struct.pack(">HIH", 1, 1, 16) + _RATE_8000
        chunks = _chunk(b"FVER", struct.pack(">I", _AIFC_VERSION1), little_endian=False)
        chunks += _chunk(b"COMM", comm, little_endian=False)
        chunks += _chunk(b"SSND", bytes(8) + b"\x00" * 80_000, little_endian=False)
        data = b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFC" + chunks
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(5.0)

    @pytest.mark.parametrize("size", [0, 2, 8])
    def test_fver_not_four_bytes_rejected(self, size):
        data = _aifc_bytes(b"sowt", bytes(16_000), claimed_frames=1, fver=None)
        fver = _chunk(
            b"FVER",
            struct.pack(">I", _AIFC_VERSION1).ljust(size, b"\x00")[:size],
            little_endian=False,
        )
        data = data[:12] + fver + data[12:]
        with pytest.raises(ProviderError, match="malformed FVER chunk"):
            get_audio_duration_seconds(data, "aiff")

    def test_odd_sized_comm_rejected(self):
        comm = struct.pack(">HIH", 1, 1, 16) + _RATE_8000 + b"sowt" + b"\x01x"
        chunks = _chunk(b"FVER", struct.pack(">I", _AIFC_VERSION1), little_endian=False)
        chunks += _chunk(b"COMM", comm[:-1], little_endian=False)  # 23 bytes, padded
        chunks += _chunk(b"SSND", bytes(8) + bytes(16_000), little_endian=False)
        data = b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFC" + chunks
        with pytest.raises(ProviderError, match="malformed COMM chunk"):
            get_audio_duration_seconds(data, "aiff")

    def test_xa_adpcm_chunk_rejected(self):
        # An APCM chunk switches ffmpeg to XA ADPCM (224 frames in 128 bytes).
        apcm = _chunk(b"APCM", bytes(8) + bytes(128 * 1000), little_endian=False)
        data = _aifc_bytes(b"NONE", bytes(16), claimed_frames=1, extra=apcm)
        with pytest.raises(ProviderError, match="XA ADPCM"):
            get_audio_duration_seconds(data, "aiff")

    def test_rate_read_as_ffmpeg_truncates_it(self):
        # 2**32 + 8000 Hz to mutagen; ffmpeg keeps the low 32 bits: 8000.
        payload = b"\x00\x00" * 80_000
        data = _aifc_bytes(b"sowt", payload, claimed_frames=1, rate=_ext80((1 << 32) + 8000))
        assert MutagenFile(io.BytesIO(data)).info.sample_rate > 1 << 32  # the trap
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(10.0)

    def test_rate_that_ffmpeg_reads_negative_rejected(self):
        data = _aifc_bytes(b"sowt", bytes(16_000), claimed_frames=1, rate=_ext80((1 << 31) + 8000))
        with pytest.raises(ProviderError, match="no channels or sample rate"):
            get_audio_duration_seconds(data, "aiff")

    def test_forged_high_rate_clamped(self):
        payload = b"\x00\x00" * 153_600  # 0.1 s at 1.536 MHz
        data = _aifc_bytes(b"sowt", payload, claimed_frames=1, rate=_ext80(1_536_000))
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(0.2)

    def test_zero_channels_rejected(self):
        data = _aifc_bytes(b"sowt", bytes(16_000), claimed_frames=1, channels=0)
        with pytest.raises(ProviderError, match="no channels or sample rate"):
            get_audio_duration_seconds(data, "aiff")


_KSDATAFORMAT_SUBTYPE_PCM = bytes.fromhex("0100000000001000800000aa00389b71")


def _wav_extensible(
    pcm: bytes, *, channels: int, bits: int, valid_bits: int, mask: int, rate: int = 8000
) -> bytes:
    align = channels * ((bits + 7) // 8)
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, channels, rate, rate * align, align, bits, 22,
                      valid_bits, mask) + _KSDATAFORMAT_SUBTYPE_PCM  # fmt: skip
    return _riff(_chunk(b"fmt ", fmt) + _chunk(b"data", pcm))


class TestWavDecodedWidth:
    """WAV is priced at the narrowest frame the wave module or ffmpeg reads."""

    def test_extensible_valid_bits_narrower_than_container(self):
        # wave sizes samples at 32 bits; ffmpeg decodes 16-bit samples.
        pcm = b"\x00\x00" * 80_000
        data = _wav_extensible(pcm, channels=1, bits=32, valid_bits=16, mask=0x4)
        with wave.open(io.BytesIO(data)) as wav_file:
            assert wav_file.getsampwidth() == 4  # the trap
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("channels", "mask"),
        [(2, 0x4), (2, 0x80000000), (6, 0x3), (4, 0x7), (8, 0x4), (1, 0x3), (2, 0x7)],
    )
    def test_channel_mask_not_priced(self, channels, mask):
        # ffmpeg ignores a dwChannelMask whose bit count differs from
        # nChannels and decodes nChannels, so the price stays at nChannels.
        frame = channels * 2
        data = _wav_extensible(bytes(frame * 8000 * 3), channels=channels, bits=16,
                               valid_bits=16, mask=mask)  # fmt: skip
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(3.0)

    def test_channel_mask_ignored_with_valid_bits_narrowing(self):
        # Valid bits still narrow the sample; the mismatched mask changes nothing.
        data = _wav_extensible(bytes(2 * 4 * 8000 * 3), channels=2, bits=32,
                               valid_bits=16, mask=0x4)  # fmt: skip
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(6.0)

    @pytest.mark.parametrize(
        ("channels", "bits", "valid_bits", "mask"),
        [(2, 16, 16, 0x3), (2, 24, 20, 0x3), (6, 24, 24, 0x3F), (2, 16, 0, 0), (1, 32, 32, 0)],
    )
    def test_honest_extensible_unchanged(self, channels, bits, valid_bits, mask):
        frame = channels * ((bits + 7) // 8)
        data = _wav_extensible(bytes(frame * 8000 * 3), channels=channels, bits=bits,
                               valid_bits=valid_bits, mask=mask)  # fmt: skip
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(3.0)

    def test_oversized_block_align_and_byte_rate_ignored(self):
        fmt = struct.pack("<HHIIHH", 1, 1, 8000, 0xFFFFFFFF, 0xFFFF, 16)
        data = _riff(_chunk(b"fmt ", fmt) + _chunk(b"data", b"\x00\x00" * 80_000))
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(10.0)

    def test_forged_high_rate_clamped(self):
        fmt = struct.pack("<HHIIHH", 1, 1, 1_536_000, 3_072_000, 2, 16)
        data = _riff(_chunk(b"fmt ", fmt) + _chunk(b"data", b"\x00\x00" * 153_600))
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(0.2)


class TestTrailingMetadata:
    def test_wav_trailing_metadata_chunk_not_billed(self):
        wav = make_wav_bytes(10.0, rate=16000)
        data = _riff(wav[12:] + _chunk(b"id3 ", b"\x07" * 500_000))
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(10.0)

    def test_wav_second_data_chunk_still_billed(self):
        wav = make_wav_bytes(10.0, rate=16000)
        data = _riff(wav[12:] + _chunk(b"data", b"\x00" * 320_000))
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(20.0, abs=0.01)

    def test_mp3_cover_art_not_counted_as_frames(self):
        art = random.Random(0).randbytes(1_000_000)
        apic = b"\x00image/png\x00\x03\x00" + art
        frame = b"APIC" + struct.pack(">I", len(apic)) + b"\x00\x00" + apic
        size = len(frame)
        synchsafe = bytes((size >> shift) & 0x7F for shift in (21, 14, 7, 0))
        frames = 2300
        data = b"ID3\x03\x00\x00" + synchsafe + frame + _mp3_frame() * frames
        expected = frames * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.1)


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full_box(kind: bytes, payload: bytes) -> bytes:
    return _box(kind, b"\x00\x00\x00\x00" + payload)


def _trak(
    handler: bytes,
    entry: bytes,
    sizes: list[int],
    timescale: int,
    duration: int,
    *,
    track_id: int = 0,
    sample_size: int = 0,
    count: int = 0,
    stts: list[tuple[int, int]] = (),
    chunks: tuple[int, list[tuple[int, int]]] | None = None,
    entry_count: int = 1,
    stsz_box: bytes | None = None,
    offsets: list[int] | None = None,
) -> bytes:
    """A trak; ``sample_size`` and ``count`` give a constant-size stsz instead
    of ``sizes``, ``chunks`` a chunk count and its stsc (first chunk, samples),
    ``offsets`` its chunk offsets (zeros otherwise), ``entry_count`` the stsd
    count over ``entry`` and ``stsz_box`` its own sizes box."""
    mdhd = _full_box(b"mdhd", struct.pack(">IIII", 0, 0, timescale, duration) + b"\x00" * 4)
    hdlr = _full_box(b"hdlr", b"\x00" * 4 + handler + b"\x00" * 13)
    stsd = _full_box(b"stsd", struct.pack(">I", entry_count) + entry)
    if sample_size:
        stsz = _full_box(b"stsz", struct.pack(">II", sample_size, count))
    else:
        table = struct.pack(f">{len(sizes)}I", *sizes)
        stsz = _full_box(b"stsz", struct.pack(">II", 0, len(sizes)) + table)
    if stsz_box is not None:
        stsz = stsz_box
    stts_table = b"".join(struct.pack(">II", *run) for run in stts)
    tables = _full_box(b"stts", struct.pack(">I", len(stts)) + stts_table) + stsz
    if chunks is not None:
        chunk_count, runs = chunks
        stsc = b"".join(struct.pack(">III", first, samples, 1) for first, samples in runs)
        tables += _full_box(b"stsc", struct.pack(">I", len(runs)) + stsc)
        offsets = offsets or [0] * chunk_count
        tables += _full_box(b"stco", struct.pack(f">I{chunk_count}I", chunk_count, *offsets))
    stbl = _box(b"stbl", stsd + tables)
    mdia = _box(b"mdia", mdhd + hdlr + _box(b"minf", stbl))
    tkhd = _full_box(b"tkhd", b"\x00" * 8 + struct.pack(">I", track_id) + b"\x00" * 68)
    return _box(b"trak", tkhd + mdia)


def _sound_fields(rate: int, *, channels: int = 1, bits: int = 16) -> bytes:
    fields = struct.pack(">HHHHI", channels, bits, 0, 0, rate << 16)
    return b"\x00" * 6 + b"\x00\x01" + b"\x00" * 8 + fields


_CHAN = _full_box(b"chan", bytes(20))  # mutagen wants a child box after the fields


def _aac_entry(*, oti: int = 0x40, asc: bytes | None = None, dsi: bool = True) -> bytes:
    """mp4a entry for mono AAC-LC at 16 kHz (sampling frequency index 8)."""
    if asc is None:
        asc = (2 << 11 | 8 << 7 | 1 << 3).to_bytes(2, "big")
    dsi = bytes([0x05, len(asc)]) + asc if dsi else b""
    dcd = bytes([0x04, 13 + len(dsi), oti, 0x15]) + b"\x00" * 11 + dsi
    esd = bytes([0x03, 3 + len(dcd)]) + b"\x00\x01\x00" + dcd + b"\x06\x01\x02"
    return _box(b"mp4a", _sound_fields(16000) + _full_box(b"esds", esd))


def _mp4(traks: list[bytes], mdat_size: int, *, extra: bytes = b"", mvex: bytes = b"") -> bytes:
    mvhd = _full_box(b"mvhd", struct.pack(">IIII", 0, 0, 1000, 0) + b"\x00" * 80)
    ftyp = _box(b"ftyp", b"M4A \x00\x00\x00\x00M4A mp42isom")
    moov = _box(b"moov", mvhd + b"".join(traks) + mvex)
    return ftyp + moov + extra + _box(b"mdat", b"\x00" * mdat_size)


_AAC_FRAMES_60S = 16000 * 60 // 1024


def _mp4_with_samples(
    entry: bytes,
    samples: list[bytes],
    *,
    per_chunk: int = 1,
    gap: bytes = b"",
    handler: bytes = b"soun",
    timescale: int = 48000,
    extra_traks: tuple[bytes, ...] = (),
) -> bytes:
    """An MP4 whose chunk table points at ``samples`` in mdat, ``per_chunk``
    to a chunk, each chunk after ``gap`` (bytes no sample covers)."""
    pieces, positions, pos = [], [], 0
    for index in range(0, len(samples), per_chunk):
        group = b"".join(samples[index : index + per_chunk])
        positions.append(pos + len(gap))
        pieces += [gap, group]
        pos += len(gap) + len(group)
    sizes = [len(sample) for sample in samples]

    def build(base: int) -> bytes:
        trak = _trak(
            handler, entry, sizes, timescale, timescale,
            chunks=(len(positions), [(1, per_chunk)]),
            offsets=[base + position for position in positions],
        )  # fmt: skip
        return _mp4([*extra_traks, trak], 0)[:-8]  # without its empty mdat

    head = build(0)
    return build(len(head) + 8) + _box(b"mdat", b"".join(pieces))


def _trex(track_id: int, default_size: int) -> bytes:
    return _full_box(b"trex", struct.pack(">IIIII", track_id, 1, 0, default_size, 0))


def _traf(track_id: int, sizes: list[int] | None = None, *, count: int = 0) -> bytes:
    """traf whose trun lists ``sizes``, or ``count`` samples of the default size."""
    tfhd = _box(b"tfhd", b"\x00\x02\x00\x00" + struct.pack(">I", track_id))
    if sizes is None:
        trun = _box(b"trun", b"\x00\x00\x00\x01" + struct.pack(">Ii", count, 0))
    else:
        table = struct.pack(f">{len(sizes)}I", *sizes)
        trun = _box(b"trun", b"\x00\x00\x02\x01" + struct.pack(">Ii", len(sizes), 0) + table)
    return _box(b"traf", tfhd + trun)


def _moof(trafs: bytes) -> bytes:
    return _box(b"moof", _full_box(b"mfhd", struct.pack(">I", 1)) + trafs)


class TestMp4Floor:
    def test_video_track_not_billed_as_audio(self):
        audio = _trak(b"soun", _aac_entry(), [40] * _AAC_FRAMES_60S, 16000, 960_000)
        video = _trak(b"vide", _box(b"avc1", b"\x00" * 78), [5000] * 1800, 30, 1800)
        data = _mp4([video, audio], 40 * _AAC_FRAMES_60S + 5000 * 1800)
        assert len(data) * 8 / 640_000 > 100  # the old whole-file floor
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(60.0)

    def test_forged_mdhd_priced_by_aac_frame_count(self):
        audio = _trak(b"soun", _aac_entry(), [6] * _AAC_FRAMES_60S, 16000, 16000)
        data = _mp4([audio], 6 * _AAC_FRAMES_60S)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        # 960 samples per frame is the shortest AAC frame: within 6.25 %.
        assert get_audio_duration_seconds(data, "m4a") >= 56.0

    def test_forged_amr_duration_priced_by_frame_count(self):
        # 60 s of 12.2k AMR-NB: 3000 frames of 32 bytes, 10 frames a sample.
        data = _mp4_with_samples(_AMR_NB_ENTRY, [_AMR_NB_122K * 10] * 300, timescale=8000)
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(60.0)

    def test_mov_data_tracks_not_billed_as_audio(self):
        # A phone .mov: sound, video, timecode, metadata (mebx) and subtitles.
        audio = _trak(b"soun", _aac_entry(), [40] * _AAC_FRAMES_60S, 16000, 960_000)
        video = _trak(b"vide", _box(b"avc1", b"\x00" * 78), [5000] * 600, 30, 600)
        timecode = _trak(b"tmcd", _box(b"tmcd", b"\x00" * 26), [4], 30, 600)
        meta = _trak(b"meta", _box(b"mebx", b"\x00" * 8), [2000] * 1000, 30, 600)
        text = _trak(b"sbtl", _box(b"tx3g", b"\x00" * 38), [500] * 100, 1000, 20_000)
        traks = [video, audio, timecode, meta, text]
        data = _mp4(traks, 40 * _AAC_FRAMES_60S + 5000 * 600 + 4 + 2000 * 1000 + 500 * 100)
        assert len(data) * 8 / 640_000 > 60  # the whole-file fallback
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(60.0)

    def test_audio_codec_under_other_handler_still_billed(self):
        # ffmpeg decodes an mp4a track as audio whatever its handler says.
        decoy = _trak(b"soun", _aac_entry(), [40] * 16, 16000, 16000)
        hidden = _trak(b"meta", _aac_entry(), [6] * _AAC_FRAMES_60S, 16000, 16000)
        data = _mp4([decoy, hidden], 40 * 16 + 6 * _AAC_FRAMES_60S)
        assert get_audio_duration_seconds(data, "m4a") >= 56.0

    def test_second_audio_track_not_billed_again(self):
        # An iPhone .mov with a spatial audio track, or a second language.
        audio = _trak(b"soun", _aac_entry(), [40] * _AAC_FRAMES_60S, 16000, 960_000)
        dub = _trak(b"soun", _aac_entry(), [40] * _AAC_FRAMES_60S, 16000, 960_000)
        spatial = _trak(
            b"soun", _box(b"apac", _sound_fields(48000)), [4000] * 980, 48000, 2_880_000
        )
        data = _mp4([audio, dub, spatial], 1000)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(60.0)

    def test_unlisted_audio_codec_under_data_handler_billed(self):
        # ffmpeg decodes '.mp2' as audio under a 'meta' handler too.
        decoy = _trak(b"soun", _aac_entry(), [40] * 16, 16000, 16000)
        entry = _box(b".mp2", _sound_fields(48000))
        samples = [bytes(1000)] * 960
        data = _mp4_with_samples(entry, samples, handler=b"meta", extra_traks=(decoy,))
        # Every MP2 packet decodes to at least 1152 samples at up to 48 kHz.
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(960 * 1152 / 48000)

    def test_quicktime_pcm_counted_in_frames(self):
        # QuickTime PCM: stsz sample size 1 makes each sample one PCM frame
        # (4 bytes of 16-bit stereo here), and mdhd claims 1 s of 60.
        entry = _box(b"sowt", _sound_fields(48000, channels=2) + _CHAN)
        frames = 48000 * 60
        audio = _trak(
            b"soun", entry, [], 48000, 48000,
            sample_size=1, count=frames, stts=[(frames, 1)], chunks=(60, [(1, 48000)]),
        )  # fmt: skip
        data = _mp4([audio], 1000)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(60.0)

    def test_forged_stsz_count_priced_by_chunk_table(self):
        # With every stts duration 1, ffmpeg reads the frames stsc lists and
        # ignores the stsz count.
        entry = _box(b"sowt", _sound_fields(48000, channels=2) + _CHAN)
        audio = _trak(
            b"soun", entry, [], 48000, 48000,
            sample_size=1, count=1, stts=[(1, 1)], chunks=(60, [(1, 48000)]),
        )  # fmt: skip
        data = _mp4([audio], 1000)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(60.0)

    @pytest.mark.parametrize(
        ("codec", "bits", "children"),
        [
            (b"fpcm", 32, _full_box(b"pcmC", bytes([1, 32]))),
            (b"NONE", 16, _CHAN),
            (b"ms\x00\x01", 16, _CHAN),
            (b"ipcm", 24, _full_box(b"pcmC", bytes([0, 24]))),
        ],
        ids=["fpcm", "NONE", "ms-pcm", "ipcm"],
    )
    def test_honest_pcm_track_priced_at_real_duration(self, codec, bits, children):
        entry = _box(codec, _sound_fields(48000, channels=2, bits=bits) + children)
        frame_bytes = 2 * bits // 8
        audio = _trak(b"soun", entry, [], 48000, 480_000, sample_size=frame_bytes, count=480_000)
        data = _mp4([audio], 1000)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(10.0)

    def test_fragmented_mp4_priced_from_audio_fragments(self):
        audio = _trak(b"soun", _aac_entry(), [], 16000, 0, track_id=1)
        video = _trak(b"vide", _box(b"avc1", b"\x00" * 78), [], 30, 0, track_id=2)
        mvex = _box(b"mvex", _trex(1, 40) + _trex(2, 0))
        half = _AAC_FRAMES_60S // 2
        first = _moof(_traf(1, [40] * half) + _traf(2, [5000] * 300))
        # The second fragment's sizes come from the trex default.
        second = _moof(_traf(1, count=_AAC_FRAMES_60S - half) + _traf(2, [5000] * 300))
        mdat_size = 40 * _AAC_FRAMES_60S + 5000 * 600
        data = _mp4([audio, video], mdat_size, extra=first + second, mvex=mvex)
        assert len(data) * 8 / 640_000 > 37  # the whole-file fallback
        expected = _AAC_FRAMES_60S * 960 / 16000  # shortest AAC frame per sample
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(expected)

    def test_fragment_without_sample_sizes_rejected(self):
        audio = _trak(b"soun", _aac_entry(), [], 16000, 16000, track_id=1)
        data = _mp4([audio], 800_000, extra=_moof(_traf(1, count=100)))
        with pytest.raises(ProviderError, match="sample sizes unknown") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    def test_fragment_with_no_audio_samples_not_billed_by_file_size(self):
        # mdat bytes no sample table or trun points at are never decoded.
        audio = _trak(b"soun", _aac_entry(), [], 16000, 16000, track_id=1)
        data = _mp4([audio], 800_000, extra=_box(b"moof", _full_box(b"mfhd", b"\x00" * 4)))
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(1.0)


def _ogg_opus(
    packets: int,
    *,
    last_granule: int,
    junk: bytes = b"",
    packet_bytes: int = 20,
    comment: bytes = b"",
    serial: int = 5,
) -> bytes:
    """Ogg Opus with 20 ms CELT packets, 50 per page, and a chosen last granule."""
    pages = []
    head = OggPage()
    head.serial, head.sequence, head.position, head.first = serial, 0, 0, True
    head.packets = [b"OpusHead\x01\x01" + struct.pack("<HI", 312, 48000) + b"\x00\x00\x00"]
    tags = OggPage()
    tags.serial, tags.sequence, tags.position = serial, 1, 0
    tags.packets = [b"OpusTags" + b"\x00" * 8 + comment]
    pages += [head.write(), tags.write()]
    for index in range(0, packets, 50):
        page = OggPage()
        page.serial, page.sequence = serial, 2 + index // 50
        page.packets = [bytes([19 << 3]) + b"\x55" * (packet_bytes - 1)] * min(50, packets - index)
        page.position = 312 + (index + len(page.packets)) * 960
        page.last = index + 50 >= packets
        if page.last:
            page.position = last_granule
        pages.append(page.write() + (junk if index == 0 else b""))
    return b"".join(pages)


class TestOggOpusFloor:
    def test_forged_granule_priced_by_packets(self):
        data = _ogg_opus(3000, last_granule=312 + 48000)
        assert isinstance(MutagenFile(io.BytesIO(data)), OggOpus)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0, abs=0.05)

    def test_honest_file_keeps_header_duration(self):
        data = _ogg_opus(3000, last_granule=312 + 3000 * 960)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0)

    def test_fake_page_header_does_not_hide_pages(self):
        # A capture pattern with a bad CRC claiming a full page of segments.
        junk = b"OggS" + b"\x00" * 22 + b"\xff" + b"\xff" * 255
        data = _ogg_opus(3000, last_granule=312 + 48000, junk=junk)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0, abs=0.05)


def _flac_crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc << 1 ^ 0x07 if crc & 0x80 else crc << 1) & 0xFF
    return crc


def _utf8_number(number: int) -> bytes:
    if number < 0x80:
        return bytes([number])
    return bytes([0xC0 | number >> 6, 0x80 | number & 0x3F])


def _flac_frame(number: int, body: bytes, *, block: int = 4096, crc_ok: bool = True) -> bytes:
    """Mono 16-bit 44.1 kHz frame header; the body starts with a subframe header."""
    if block == 4096:
        header = b"\xff\xf8\xc9\x08" + _utf8_number(number)
    else:  # block size code 7: explicit 16-bit size
        header = b"\xff\xf8\x79\x08" + _utf8_number(number) + struct.pack(">H", block - 1)
    crc = _flac_crc8(header) ^ (0 if crc_ok else 0x55)
    return header + bytes([crc]) + body


def _flac(frames: bytes, *, total_samples: int, padding: bytes = b"") -> bytes:
    """FLAC with a mono 16-bit 44.1 kHz STREAMINFO claiming ``total_samples``."""
    packed = 44100 << 44 | 0 << 41 | 15 << 36 | total_samples
    streaminfo = struct.pack(">HH", 4096, 4096) + bytes(6) + packed.to_bytes(8, "big") + bytes(16)
    blocks = bytes([0x00]) + len(streaminfo).to_bytes(3, "big") + streaminfo
    blocks += bytes([0x81]) + len(padding).to_bytes(3, "big") + padding  # last: PADDING
    return b"fLaC" + blocks + frames


def _flac_frames(count: int, *, rng: random.Random | None = None, **kwargs) -> bytes:
    """``count`` frames of 1000 bytes; noise bodies when ``rng`` is given."""

    def body() -> bytes:
        return b"\x00" + (rng.randbytes(999) if rng else bytes(999))

    return b"".join(_flac_frame(n, body(), **kwargs) for n in range(count))


class TestFlacFloor:
    def test_forged_total_samples_priced_by_frame_count(self):
        data = _flac(_flac_frames(1000), total_samples=44100)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        assert len(data) * 8 / 4_608_000 < 2  # the old byte floor
        expected = 1000 * 4096 / 44100
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(expected)

    def test_honest_flac_unchanged(self):
        frames = _flac_frames(999) + _flac_frame(999, b"\x00" * 999, block=1000)
        total = 999 * 4096 + 1000
        data = _flac(frames, total_samples=total)
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(total / 44100)

    def test_noisy_frame_bodies_not_overcounted(self):
        total = 2000 * 4096
        data = _flac(_flac_frames(2000, rng=random.Random(1)), total_samples=total)
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(total / 44100, abs=0.2)

    def test_headers_with_bad_crc_not_counted(self):
        data = _flac(_flac_frames(1000, crc_ok=False), total_samples=44100)
        byte_floor = len(data) * 8 / 4_608_000  # no frames counted: only the old floor
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(byte_floor)

    def test_chance_header_counted_at_8khz_or_faster(self):
        # Header bytes found by chance in audio data may claim 65536 samples
        # at 1 Hz (rate code 13); counted at 8 kHz they add at most 8.2 s.
        chance = b"\xff\xf8\x7d\x08\x00\xff\xff\x00\x01"
        chance += bytes([_flac_crc8(chance)]) + b"\x00"
        total = 100 * 4096
        data = _flac(_flac_frames(100) + chance, total_samples=total)
        expected = total / 44100 + 65536 / 8000
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(expected)

    def test_dense_valid_headers_rejected(self):
        # One-sample frames packed 8 bytes apart: no encoder writes these, and
        # scanning 25 MB of them took seconds before payment.
        header = b"\xff\xf8\x69\x08\x00\x00"
        frame = header + bytes([_flac_crc8(header)]) + b"\x00"
        data = _flac(frame * 20_000, total_samples=44100)
        with pytest.raises(ProviderError, match="too many frame headers") as exc:
            get_audio_duration_seconds(data, "flac")
        assert exc.value.status_code == 400

    def test_dense_false_syncs_rejected(self):
        data = _flac(b"\xff\xf8\xc9\x08\x00\x00\x00" * 20_000, total_samples=44100)
        with pytest.raises(ProviderError, match="too many frame headers"):
            get_audio_duration_seconds(data, "flac")

    def test_frames_inside_metadata_block_not_counted(self):
        # Decoders skip a metadata block by its length, frames inside included.
        data = _flac(b"", total_samples=44100, padding=_flac_frames(1000))
        byte_floor = len(data) * 8 / 4_608_000  # no frames counted: only the old floor
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(byte_floor)


def _ogg_stream(
    serial: int, headers: list[list[bytes]], audio: list[bytes], *, samples: int, last_granule: int
) -> bytes:
    """Ogg pages: one page per header group, then audio 50 packets a page."""
    pages = []
    for sequence, packets in enumerate(headers):
        page = OggPage()
        page.serial, page.sequence, page.position = serial, sequence, 0
        page.first = sequence == 0
        page.packets = packets
        pages.append(page.write())
    for index in range(0, len(audio), 50):
        page = OggPage()
        page.serial, page.sequence = serial, len(headers) + index // 50
        page.packets = audio[index : index + 50]
        page.position = (index + len(page.packets)) * samples
        page.last = index + 50 >= len(audio)
        if page.last:
            page.position = last_granule
        pages.append(page.write())
    return b"".join(pages)


class _BitWriter:
    """LSB-first bit packer, as Vorbis headers are written."""

    def __init__(self) -> None:
        self.value = 0
        self.size = 0

    def put(self, value: int, bits: int) -> "_BitWriter":
        self.value |= value << self.size
        self.size += bits
        return self

    def bytes(self) -> bytes:
        return self.value.to_bytes((self.size + 7) // 8, "little")


def _vorbis_setup() -> bytes:
    """A setup header with one of each part and two modes: 0 short, 1 long.

    Its three codebooks take the plain, sparse and ordered length layouts.
    """
    w = _BitWriter().put(2, 8)  # three codebooks
    w.put(0x564342, 24).put(1, 16).put(2, 24).put(0, 1).put(0, 1).put(0, 5).put(0, 5)
    w.put(0, 4)  # no lookup table
    w.put(0x564342, 24).put(2, 16).put(4, 24).put(0, 1).put(1, 1)  # sparse
    for used in (1, 0, 1, 0):
        w.put(used, 1)
        if used:
            w.put(1, 5)
    w.put(1, 4).put(0, 32).put(0, 32).put(3, 4).put(0, 1)  # lookup type 1, 4-bit values
    w.put(0, 2 * 4)  # lookup1_values(4 entries, 2 dimensions) = 2
    w.put(0x564342, 24).put(1, 16).put(4, 24).put(1, 1).put(1, 5).put(4, 3)  # ordered
    w.put(0, 4)
    w.put(0, 6).put(0, 16)  # one time domain transform (a placeholder)
    w.put(0, 6).put(1, 16).put(1, 5).put(0, 4).put(0, 3).put(0, 2).put(0, 8)  # floor 1
    w.put(0, 2).put(1, 4).put(0, 1)  # multiplier, range bits, one X value
    w.put(0, 6).put(0, 16).put(0, 24).put(0, 24).put(0, 24).put(0, 6).put(0, 8)  # residue
    w.put(0, 3).put(0, 1)  # cascade
    w.put(0, 6).put(0, 16).put(0, 1).put(0, 1).put(0, 2).put(0, 8).put(0, 8).put(0, 8)
    w.put(1, 6)  # two modes
    for blockflag in (0, 1):
        w.put(blockflag, 1).put(0, 16).put(0, 16).put(0, 8)
    w.put(1, 1)  # framing bit
    return b"\x05vorbis" + w.bytes()


def _ogg_vorbis(
    packets: int, *, last_granule: int, setup: bytes | None = None, first_byte: int = 0
) -> bytes:
    """44.1 kHz Vorbis, 256/2048 blocks, 40-byte audio packets.

    The default setup header does not parse, so packets fall back to the
    short block; with ``_vorbis_setup()``, ``first_byte`` 0x02 picks mode 1
    (long blocks: 1024 samples a packet).
    """
    ident = b"\x01vorbis" + struct.pack("<IBIiiiB", 0, 1, 44100, 0, 64000, 0, 0xB8) + b"\x01"
    comment = b"\x03vorbis" + struct.pack("<II", 0, 0) + b"\x01"
    if setup is None:
        setup = b"\x05vorbis" + b"\x00" * 20
    audio = [bytes([first_byte]) + b"\x55" * 39] * packets
    return _ogg_stream(
        7, [[ident], [comment, setup]], audio, samples=1024, last_granule=last_granule
    )


def _ogg_speex(packets: int, *, last_granule: int, extra_headers: int = 0) -> bytes:
    """16 kHz wideband Speex, one 320-sample frame per 20-byte packet."""
    fields = (1, 80, 16000, 1, 4, 1, -1, 320, 0, 1, extra_headers, 0, 0)
    header = b"Speex   " + b"1.2.0".ljust(20, b"\x00") + struct.pack("<13i", *fields)
    comment = struct.pack("<II", 0, 0)
    audio = [b"\x1e" + b"\x55" * 19] * packets
    return _ogg_stream(9, [[header], [comment]], audio, samples=320, last_granule=last_granule)


class TestOggVorbisSpeexFloor:
    def test_forged_vorbis_granule_priced_by_packet_count(self):
        data = _ogg_vorbis(3000, last_granule=44100)
        assert isinstance(MutagenFile(io.BytesIO(data)), OggVorbis)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        assert len(data) * 8 / 640_000 < 2  # the old byte floor
        # Half a 256-sample short block per packet, first and last page left out.
        expected = (3000 - 100) * 128 / 44100
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_forged_vorbis_granule_priced_by_block_sizes(self):
        data = _ogg_vorbis(3000, last_granule=44100, setup=_vorbis_setup(), first_byte=0x02)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        # Long blocks after long blocks: 1024 samples a packet, first and last
        # page left out (the real duration is 3000 * 1024 samples).
        expected = (3000 - 100) * 1024 / 44100
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_vorbis_short_blocks_from_mode_table(self):
        data = _ogg_vorbis(3000, last_granule=44100, setup=_vorbis_setup(), first_byte=0x00)
        expected = (3000 - 100) * 128 / 44100
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_vorbis_setup_trailing_bytes_ignored(self):
        # Decoders stop at the framing bit; bytes after it that look like a
        # one-mode, short-block table must not change the count.
        setup = _vorbis_setup() + _BitWriter().put(0, 41).put(0, 6).put(1, 1).bytes()
        data = _ogg_vorbis(3000, last_granule=44100, setup=setup, first_byte=0x02)
        expected = (3000 - 100) * 1024 / 44100
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_honest_vorbis_with_mode_table_keeps_header_duration(self):
        data = _ogg_vorbis(3000, last_granule=3000 * 1024, setup=_vorbis_setup(), first_byte=0x02)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(3000 * 1024 / 44100)

    def test_honest_vorbis_keeps_header_duration(self):
        data = _ogg_vorbis(3000, last_granule=3000 * 1024)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(3000 * 1024 / 44100)

    def test_forged_speex_granule_priced_by_packet_count(self):
        data = _ogg_speex(1500, last_granule=16000)
        assert isinstance(MutagenFile(io.BytesIO(data)), OggSpeex)
        assert MutagenFile(io.BytesIO(data)).info.length == pytest.approx(1.0)
        expected = (1500 - 100) * 320 / 16000
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(expected)

    def test_forged_speex_extra_headers_do_not_hide_audio(self):
        data = _ogg_speex(1500, last_granule=16000, extra_headers=0x7FFFFFFF)
        assert get_audio_duration_seconds(data, "ogg") >= (1500 - 104) * 320 / 16000

    def test_honest_speex_keeps_header_duration(self):
        data = _ogg_speex(1500, last_granule=1500 * 320)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(30.0)


class TestOggWorkBudget:
    def test_dense_empty_packets_rejected(self):
        pages = []
        for sequence in range(2000):
            page = OggPage()
            page.serial, page.sequence, page.position = 5, 2 + sequence, 0
            page.packets = [b""] * 255
            pages.append(page.write())
        data = _ogg_opus(50, last_granule=312 + 48000) + b"".join(pages)
        with pytest.raises(ProviderError, match="too many pages or packets") as exc:
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    def test_dense_fake_page_headers_rejected(self):
        # Each fake header claims a 64 KB page, so each CRC check reads 64 KB.
        junk = b"OggS" + b"\x00" * 22 + b"\xff" + b"\xff" * 255 + b"\x00"
        data = _ogg_opus(50, last_granule=312 + 48000) + junk * 2000
        with pytest.raises(ProviderError, match="too many invalid pages"):
            get_audio_duration_seconds(data, "ogg")

    def test_opus_with_a_packet_on_every_page_accepted(self):
        # 6 kbps Opus, one 15-byte packet a page, as some recorders write it.
        pages = []
        for sequence in range(60_000):
            page = OggPage()
            page.serial, page.sequence = 5, 2 + sequence
            page.position = 312 + (sequence + 1) * 960
            page.packets = [bytes([19 << 3]) + b"\x55" * 14]
            pages.append(page.write())
        data = _ogg_opus(0, last_granule=0) + b"".join(pages)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(1200.0, abs=0.05)


class TestOggOpusPreSkip:
    def test_fake_streams_pre_skip_does_not_cancel_real_audio(self):
        fakes = []
        for serial in range(100, 160):
            page = OggPage()
            page.serial, page.position, page.first, page.last = serial, 0, True, True
            page.packets = [b"OpusHead\x01\x01" + struct.pack("<HI", 65535, 48000) + bytes(3)]
            fakes.append(page.write())
        data = _ogg_opus(3000, last_granule=312 + 48000) + b"".join(fakes)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0, abs=0.05)


def _id3(payload: bytes = b"") -> bytes:
    size = len(payload)
    synchsafe = bytes((size >> shift) & 0x7F for shift in (21, 14, 7, 0))
    return b"ID3\x04\x00\x00" + synchsafe + payload


def _quiet_24bit_pcm(count: int, *, little_endian: bool, seed: int = 1) -> bytes:
    """Mono 24-bit noise at -50 dBFS."""
    rng = random.Random(seed)
    sigma = (1 << 23) * 10 ** (-50 / 20)
    samples = [round(rng.gauss(0, sigma)) for _ in range(count)]
    packed = bytearray(struct.pack(f"{'<' if little_endian else '>'}{count}i", *samples))
    del packed[3 if little_endian else 0 :: 4]  # the sign-extension byte
    return bytes(packed)


def _wav_24bit(pcm: bytes) -> bytes:
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 48000 * 3, 3, 24)
    return _riff(_chunk(b"fmt ", fmt) + _chunk(b"data", pcm))


def _aiff_24bit(pcm: bytes) -> bytes:
    rate_48000 = b"\x40\x0e\xbb\x80\x00\x00\x00\x00\x00\x00"  # 80-bit IEEE float
    comm = struct.pack(">hIh", 1, len(pcm) // 3, 24) + rate_48000
    chunks = _chunk(b"COMM", comm, little_endian=False)
    chunks += _chunk(b"SSND", bytes(8) + pcm, little_endian=False)
    return b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFF" + chunks


class TestContainerAfterId3:
    """ffmpeg skips leading ID3v2 tags and demuxes what follows; so does pricing."""

    def test_id3_before_ogg_with_hidden_mp3_frames_priced_as_ogg(self):
        data = _id3() + _ogg_opus(3000, last_granule=312 + 48000, comment=_mp3_frame() * 5)
        assert isinstance(MutagenFile(io.BytesIO(data)), MP3)  # the trap
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(60.0, abs=0.05)

    def test_id3_before_flac_priced_by_frames(self):
        data = _id3() + _flac(_flac_frames(1000), total_samples=44100)
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(1000 * 4096 / 44100)

    def test_id3_before_wav_priced_as_wav(self):
        data = _id3(b"\x00" * 100) + _forge_wav_data_size(make_wav_bytes(60.0), 16)
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(60.0, rel=0.001)

    def test_stacked_id3_and_padding_before_mp3(self):
        frames = 230
        data = _id3() + _id3() + b"\x00" * 64 + _mp3_frame() * frames
        expected = frames * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.03)

    def test_padding_between_stacked_tags_before_mp3(self):
        data = _id3() + b"\x00" * 100 + _id3() + _mp3_frame() * 230
        expected = 230 * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.03)

    def test_bytes_between_tag_and_first_frame_rejected(self):
        # ffmpeg resyncs past them, but frames that do not open the stream
        # do not make its probe pick MP3 over a stream found further in.
        data = _id3() + b"\x01\x02\x03" + _mp3_frame() * 230
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(data, "mp3")
        assert exc.value.status_code == 400

    @pytest.mark.parametrize(
        "prefix",
        [
            _id3() + _id3(),
            _id3() + b"\x00",
            _id3() + b"\x00" * 100 + _id3(),
            _id3(bytes(1 << 20)),
        ],
        ids=["stacked-tags", "padding", "padding-between-tags", "tag-past-probe-buffer"],
    )
    def test_container_ffmpeg_probe_cannot_reach_rejected(self, prefix):
        # ffmpeg's probe skips one tag that fits its 1 MiB buffer, and every
        # container probe but MP3's wants its magic right after it; here the
        # MP3 probe would win on MPEG frames hidden in the Ogg stream.
        data = prefix + _ogg_opus(3000, last_granule=312 + 48000, comment=_mp3_frame() * 150)
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    def test_tag_within_probe_buffer_priced_as_container(self):
        data = _id3(bytes(900_000)) + _ogg_opus(3000, last_granule=312 + 48000)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0, abs=0.05)

    def test_mpeg_frames_behind_id3_and_flac_counted(self):
        # 150 chained MP3 frames (3.9 s) in the PADDING block of 0.9 s of
        # FLAC: ffmpeg's MP3 probe outscores FLAC's on them.
        data = _id3() + _flac(_flac_frames(10), total_samples=40960, padding=_mp3_frame() * 150)
        expected = 150 * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(expected, abs=0.03)

    def test_mpeg_frames_behind_id3_and_adts_counted(self):
        data = _id3() + _adts_frames(100) + _mp3_frame() * 150
        assert get_audio_duration_seconds(data, "aac") >= 150 * _MP3_FRAME_SECONDS

    def test_short_mpeg_run_behind_id3_and_flac_not_counted(self):
        data = _id3() + _flac(_flac_frames(10), total_samples=40960, padding=_mp3_frame() * 6)
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(10 * 4096 / 44100)

    def test_mpeg_frames_behind_id3_and_ogg_not_counted(self):
        # The Ogg probe (100) always beats MP3's (at most 51).
        data = _id3() + _ogg_opus(50, last_granule=312 + 48000, comment=_mp3_frame() * 150)
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(1.0)

    @pytest.mark.parametrize("container", ["wav", "aiff"])
    def test_quiet_24bit_pcm_behind_id3_not_overpriced(self, container):
        # Negative samples of quiet 24-bit PCM open with 0xFF, so MPEG sync
        # matches are dense; ffmpeg still probes the file as WAV or AIFF.
        pcm = _quiet_24bit_pcm(48000 * 5, little_endian=container == "wav")
        body = _wav_24bit(pcm) if container == "wav" else _aiff_24bit(pcm)
        assert audio_duration._mpeg_frames_seconds(body) > 20  # the trap
        data = _id3(bytes(90)) + body
        assert get_audio_duration_seconds(data, container) == pytest.approx(5.0)

    def test_dense_flac_behind_id3_not_overpriced(self):
        # Verbatim frames of quiet 24-bit PCM: 12 KB per 93 ms frame.
        frames = b"".join(
            _flac_frame(n, b"\x00" + _quiet_24bit_pcm(4096, little_endian=False, seed=n))
            for n in range(50)
        )
        body = _flac(frames, total_samples=50 * 4096)
        assert audio_duration._mpeg_frames_seconds(body) > 2 * 50 * 4096 / 44100  # the trap
        data = _id3() + body
        # A chance FLAC header in the noise may add a few ms, as it would bare.
        expected = 50 * 4096 / 44100
        assert get_audio_duration_seconds(data, "flac") == pytest.approx(expected, abs=0.05)

    def test_too_many_id3_tags_rejected(self):
        data = _id3() * 17 + _mp3_frame() * 10
        with pytest.raises(ProviderError, match="too many ID3v2 tags") as exc:
            get_audio_duration_seconds(data, "mp3")
        assert exc.value.status_code == 400

    def test_few_mpeg_frames_before_another_stream_rejected(self):
        # 3 MP3 frames, then a stream ffmpeg would probe as something else.
        data = _mp3_frame() * 3 + _adts_frames(500)
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(data, "mp3")
        assert exc.value.status_code == 400

    def test_short_mp3_whose_frames_chain_to_the_end_accepted(self):
        data = _mp3_frame() * 3
        expected = 3 * _MP3_FRAME_SECONDS
        assert get_audio_duration_seconds(data, "mp3") == pytest.approx(expected, abs=0.01)

    def test_mp3_tags_are_not_parsed(self):
        # mutagen's ID3 parser takes minutes on a tag of many tiny frames.
        frame = b"TXXX" + struct.pack(">I", 3) + b"\x00\x00\x00a\x00"
        data = _id3(frame * 1000) + _mp3_frame() * 230
        with patch.object(ID3, "load", side_effect=AssertionError("ID3 parsed")):
            assert get_audio_duration_seconds(data, "mp3") == pytest.approx(
                230 * _MP3_FRAME_SECONDS, abs=0.03
            )


def _ogg_flac_page(serial: int) -> bytes:
    page = OggPage()
    page.serial, page.sequence, page.position, page.first = serial, 0, 0, True
    page.packets = [b"\x7fFLAC\x01\x00\x00\x01fLaC" + bytes(38)]
    return page.write()


def _ogg_page(serial: int, packet: bytes) -> bytes:
    page = OggPage()
    page.serial, page.sequence, page.position, page.first = serial, 0, 0, True
    page.packets = [packet]
    return page.write()


class TestContainerAllowlist:
    """Containers and codecs without a content floor are refused."""

    def test_ogg_flac_rejected(self):
        with pytest.raises(ProviderError, match="Only Opus, Vorbis and Speex") as exc:
            get_audio_duration_seconds(_ogg_flac_page(3), "ogg")
        assert exc.value.status_code == 400

    def test_ogg_flac_stream_beside_opus_rejected(self):
        data = _ogg_opus(3000, last_granule=312 + 48000) + _ogg_flac_page(3)
        with pytest.raises(ProviderError, match="Only Opus, Vorbis and Speex"):
            get_audio_duration_seconds(data, "ogg")

    def test_ogm_directshow_stream_beside_opus_rejected(self):
        # ffmpeg's old OGM mapping carries any WAVE codec (here 8-bit PCM).
        head = b"\x01Direct Show Samples embedded in Ogg".ljust(96, b"\x00")
        head += struct.pack("<I", 0x05589F81) + bytes(40)
        data = _ogg_opus(3000, last_granule=312 + 48000) + _ogg_page(6, head)
        with pytest.raises(ProviderError, match="Only Opus, Vorbis and Speex") as exc:
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    def test_unknown_ogg_stream_rejected(self):
        data = _ogg_opus(3000, last_granule=312 + 48000) + _ogg_page(6, b"\x01newcodec" + bytes(30))
        with pytest.raises(ProviderError, match="Only Opus, Vorbis and Speex"):
            get_audio_duration_seconds(data, "ogg")

    def test_video_stream_beside_opus_priced(self):
        data = _ogg_opus(3000, last_granule=312 + 48000) + _ogg_page(6, b"\x80theora" + bytes(40))
        assert get_audio_duration_seconds(data, "ogg") == pytest.approx(60.0, abs=0.05)

    def test_ogg_first_page_failing_ffmpeg_probe_rejected(self):
        # Header flags above 7 fail ffmpeg's Ogg probe, so MP3 could win it.
        data = bytearray(_ogg_opus(3000, last_granule=312 + 48000))
        data[5] |= 0x08
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(bytes(data), "ogg")
        assert exc.value.status_code == 400

    @pytest.mark.parametrize(
        "data",
        [
            b"\x30\x26\xb2\x75\x8e\x66\xcf\x11" + bytes(1000),  # ASF / WMA
            b"\x0b\x77" + bytes(1000),  # AC-3
            b"ADIF" + bytes(1000),
            b"MPCK" + bytes(1000),  # Musepack
            b"wvpk" + bytes(1000),  # WavPack
        ],
        ids=["asf", "ac3", "adif", "musepack", "wavpack"],
    )
    def test_unpriced_containers_rejected(self, data):
        with pytest.raises(ProviderError, match="Cannot parse audio format") as exc:
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    def test_webm_rejected_with_its_name(self):
        with pytest.raises(ProviderError, match="WebM/Matroska audio is not supported") as exc:
            get_audio_duration_seconds(b"\x1a\x45\xdf\xa3" + bytes(1000), "webm")
        assert exc.value.status_code == 400


class TestMalformedInput:
    """Exceptions mutagen raises on crafted bytes are a 400, never a 500."""

    def test_empty_ogg_page_before_head_rejected(self):
        empty = OggPage()
        empty.serial, empty.first = 5, True
        data = empty.write() + _ogg_opus(50, last_granule=312 + 48000)
        with pytest.raises(ProviderError) as exc:
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("error", [IndexError("list index"), struct.error("unpack")])
    def test_mutagen_errors_are_client_errors(self, error):
        data = _ogg_opus(50, last_granule=312 + 48000)
        with (
            patch.object(OggOpusInfo, "_post_tags", side_effect=error),
            pytest.raises(ProviderError, match="Cannot parse audio format") as exc,
        ):
            get_audio_duration_seconds(data, "ogg")
        assert exc.value.status_code == 400

    def test_mpeg_reader_error_is_a_client_error(self):
        with (
            patch.object(audio_duration, "MPEGInfo", side_effect=IndexError("list index")),
            pytest.raises(ProviderError, match="Cannot parse audio format") as exc,
        ):
            get_audio_duration_seconds(_mp3_frame() * 10, "mp3")
        assert exc.value.status_code == 400


_AAC_60S_SIZES = [40] * _AAC_FRAMES_60S
_OPUS_ENTRY = _box(
    b"Opus",
    _sound_fields(48000) + _box(b"dOps", bytes([0, 1]) + struct.pack(">HIhB", 312, 48000, 0, 0)),
)
_AAC_60S_FLOOR = _AAC_FRAMES_60S * 960 / 16000  # shortest AAC frame per sample


def _aac_60s(entry: bytes | None = None, **kwargs) -> bytes:
    """60 s of 16 kHz AAC whose mdhd claims 1 s."""
    entry = _aac_entry() if entry is None else entry
    audio = _trak(b"soun", entry, _AAC_60S_SIZES, 16000, 16000, **kwargs)
    return _mp4([audio], 40 * _AAC_FRAMES_60S)


class TestMp4FailsClosed:
    """The MP4 sample-count bound holds on malformations, or the file is refused."""

    def test_truncated_trailing_box_ignored(self):
        data = _aac_60s() + b"\x00\x00\x00\x10free"
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(_AAC_60S_FLOOR)

    def test_truncated_trailing_moof_rejected(self):
        data = _aac_60s() + b"\x00\x00\x00\x40moof" + bytes(8)
        with pytest.raises(ProviderError, match="Cannot parse") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    def test_box_after_counted_sample_entry_ignored(self):
        data = _aac_60s(_aac_entry() + _box(b"free", bytes(8)))
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(_AAC_60S_FLOOR)

    def test_identical_sample_entries_priced(self):
        data = _aac_60s(_aac_entry() * 2, entry_count=2)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(_AAC_60S_FLOOR)

    def test_different_sample_entries_rejected(self):
        entries = _aac_entry() + _box(b".mp2", _sound_fields(48000))
        with pytest.raises(ProviderError, match="different sample descriptions"):
            get_audio_duration_seconds(_aac_60s(entries, entry_count=2), "m4a")

    def test_esds_without_decoder_config_bounded_at_entry_rate(self):
        data = _aac_60s(_aac_entry(dsi=False))
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(_AAC_60S_FLOOR)

    def test_mp3_in_mp4_without_sample_data_bounded_by_packet_count(self):
        samples = [bytes(40)] * _AAC_FRAMES_60S
        data = _mp4_with_samples(_aac_entry(oti=0x6B), samples, per_chunk=50)
        expected = _AAC_FRAMES_60S * 576 / 48000
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(expected)

    def test_opus_in_mp4_without_toc_lengths_bounded_by_packet_count(self):
        # 2.5 ms CELT packets: the shortest an Opus packet decodes to.
        samples = [bytes([16 << 3]) + bytes(9)] * _AAC_FRAMES_60S
        data = _mp4_with_samples(_OPUS_ENTRY, samples, per_chunk=50)
        expected = _AAC_FRAMES_60S * 0.0025
        assert get_audio_duration_seconds(data, "mp4") == pytest.approx(expected)

    def test_untimed_aac_object_type_rejected(self):
        # AAC object type 36 (ALS) through the escape code: not timed here.
        asc = ((31 << 14 | 4 << 8 | 8 << 4 | 1) << 5).to_bytes(3, "big")
        with pytest.raises(ProviderError, match="object type not supported"):
            get_audio_duration_seconds(_aac_60s(_aac_entry(asc=asc)), "m4a")

    @pytest.mark.parametrize("field", [4, 8, 16])
    def test_compact_sample_sizes_counted(self, field):
        count = _AAC_FRAMES_60S
        if field == 4:
            table = bytes([0x77]) * (count // 2) + (b"\x70" if count & 1 else b"")
        elif field == 8:
            table = bytes([40]) * count
        else:
            table = struct.pack(f">{count}H", *[40] * count)
        stz2 = _full_box(b"stz2", bytes(3) + bytes([field]) + struct.pack(">I", count) + table)
        data = _aac_60s(stsz_box=stz2)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(_AAC_60S_FLOOR)

    def test_compressed_movie_rejected(self):
        audio = _trak(b"soun", _aac_entry(), _AAC_60S_SIZES, 16000, 16000)
        data = _mp4([_box(b"cmov", bytes(8)), audio], 40 * _AAC_FRAMES_60S)
        with pytest.raises(ProviderError, match="compressed movie"):
            get_audio_duration_seconds(data, "mov")


_MPEG25_FRAME = b"\xff\xe3\x18\xc0" + bytes(68)  # MPEG-2.5 L3 8 kbps 8 kHz: 72 B, 72 ms
_MP3_ENTRY = _box(b".mp3", _sound_fields(8000) + _CHAN)


def _alac_entry(*, cookie_rate: int = 44100, entry_rate: int = 44100, wave: bool = False) -> bytes:
    config = struct.pack(">IBBBBBBHIII", 4096, 0, 16, 40, 10, 14, 1, 255, 0, 0, cookie_rate)
    cookie = _full_box(b"alac", config)
    if wave:
        cookie = _box(b"wave", _box(b"frma", b"alac") + cookie)
    return _box(b"alac", _sound_fields(entry_rate) + cookie)


def _ima4_entry(rate: int, *, channels: int = 1, codec: bytes = b"ima4") -> bytes:
    """Version 1 sound description, as ffmpeg's mov muxer writes for ima4."""
    fields = bytearray(_sound_fields(rate, channels=channels))
    fields[8:10] = b"\x00\x01"
    extension = struct.pack(">IIII", 64, 34, 34 * channels, 2)
    return _box(codec, bytes(fields) + extension + _CHAN)


_AMR_NB_ENTRY = _box(b"samr", _sound_fields(8000) + _box(b"damr", b"\x00" * 9))
_AMR_WB_ENTRY = _box(b"sawb", _sound_fields(16000) + _box(b"damr", b"\x00" * 9))
_AMR_NB_122K = b"\x3c" + bytes(31)  # frame type 7, quality bit set: 32 bytes
_AMR_NB_NO_DATA = b"\x7c"  # frame type 15: the mode byte alone, 20 ms
_AMR_WB_2385K = b"\x44" + bytes(60)  # frame type 8: 61 bytes


class TestMp4Amr:
    """AMR in MP4 is timed from its frames, 20 ms each, whatever their size."""

    def test_no_data_frames_priced_by_count(self):
        # 60 s of 1-byte NO_DATA frames: the 12.2k byte cap would say 1.9 s.
        data = _mp4_with_samples(_AMR_NB_ENTRY, [_AMR_NB_NO_DATA * 10] * 300, timescale=8000)
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(60.0)

    def test_mixed_frame_types_counted(self):
        sample = (_AMR_NB_122K + _AMR_NB_NO_DATA * 4) * 2
        data = _mp4_with_samples(_AMR_NB_ENTRY, [sample] * 500, per_chunk=25, timescale=8000)
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(500 * 10 / 50)

    def test_amr_wb_frames_counted(self):
        sample = _AMR_WB_2385K * 5 + b"\x7c" * 5
        data = _mp4_with_samples(_AMR_WB_ENTRY, [sample] * 300, timescale=16000)
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(60.0)

    def test_each_sample_walked_on_its_own(self):
        # A 12.2k mode byte alone in one sample would swallow the next
        # sample's 31 NO_DATA frames in a walk over the joined bytes; the
        # decoder starts afresh at every packet.
        samples = [b"\x3c", _AMR_NB_NO_DATA * 31] * 100
        data = _mp4_with_samples(_AMR_NB_ENTRY, samples, per_chunk=10, timescale=8000)
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(100 * 32 / 50)

    def test_walk_stops_past_a_day(self):
        data = _mp4_with_samples(
            _AMR_NB_ENTRY, [_AMR_NB_NO_DATA * 5000] * 1000, per_chunk=100, timescale=8000
        )
        assert get_audio_duration_seconds(data, "3gp") == pytest.approx(86400.0, abs=0.1)

    def test_amr_read_by_chunk_rejected(self):
        # stts durations of 1 make ffmpeg cut chunks into packets itself.
        audio = _trak(
            b"soun", _AMR_NB_ENTRY, [32] * 100, 8000, 8000,
            stts=[(100, 1)], chunks=(10, [(1, 10)]),
        )  # fmt: skip
        with pytest.raises(ProviderError, match="AMR read by chunk") as exc:
            get_audio_duration_seconds(_mp4([audio], 3200), "3gp")
        assert exc.value.status_code == 400

    def test_fragmented_amr_rejected(self):
        data = _mp4_with_samples(_AMR_NB_ENTRY, [_AMR_NB_122K] * 10, timescale=8000)
        data += _moof(_traf(0, [32] * 10))
        with pytest.raises(ProviderError, match="fragmented MPEG, Opus or AMR"):
            get_audio_duration_seconds(data, "3gp")


class TestMp4UntimedCodecs:
    """Codecs ffmpeg decodes from MP4 but no table here times are refused."""

    @pytest.mark.parametrize(
        ("codec", "name"),
        [
            (b"Qclp", "QCELP"),
            (b"sevc", "EVRC"),
            (b"spex", "Speex"),
            (b"ilbc", "iLBC"),
            (b"dtsc", "DTS"),
            (b"mlpa", "TrueHD"),
            (b"sac3", "AC-3"),
            (b"QDM2", "QDesign"),
        ],
    )
    def test_untimed_codec_rejected(self, codec, name):
        # 1-byte blank frames priced at the 640 kbps cap: 50x short or worse.
        data = _mp4_with_samples(_box(codec, _sound_fields(8000) + _CHAN), [b"\x00"] * 3000)
        with pytest.raises(ProviderError, match=f"{name} audio is not supported") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    @pytest.mark.parametrize("tag", [b"ms\x00\x02", b"ms\x00\x11", b"TS\x00\x31"])
    def test_wrapped_wave_codec_rejected(self, tag):
        # ADPCM and GSM through QuickTime's WAVE wrapper.
        data = _mp4_with_samples(_box(tag, _sound_fields(8000) + _CHAN), [bytes(65)] * 100)
        with pytest.raises(ProviderError, match="WAVE format 0x00") as exc:
            get_audio_duration_seconds(data, "mov")
        assert exc.value.status_code == 400

    def test_wrapped_mp3_tag_only_under_ms_is_timed(self):
        data = _mp4_with_samples(_box(b"TS\x00\x55", _sound_fields(8000) + _CHAN), [bytes(72)])
        with pytest.raises(ProviderError, match="WAVE format 0x0055"):
            get_audio_duration_seconds(data, "mov")

    @pytest.mark.parametrize(
        ("codec", "frames"),
        [(b"ima4", 1000 * 64), (b"MAC3", 17_000 * 6), (b"MAC6", 34_000 * 6), (b"agsm", 1030 * 160)],
    )
    def test_fixed_ratio_codec_timed_from_sample_bytes(self, codec, frames):
        # 34,000 bytes of mono packets: the whole blocks they hold.
        data = _mp4_with_samples(_box(codec, _sound_fields(8000) + _CHAN), [bytes(34)] * 1000)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(frames / 8000)

    def test_ffmpeg_ima4_stereo_priced_at_real_duration(self):
        # ffmpeg's mov muxer: stts deltas of 64, 68-byte (2 x 34) packets.
        audio = _trak(
            b"soun", _ima4_entry(8000, channels=2), [], 8000, 8000,
            sample_size=68, count=7500, stts=[(7500, 64)],
        )  # fmt: skip
        assert get_audio_duration_seconds(_mp4([audio], 1000), "mov") == pytest.approx(60.0)

    def test_fragmented_ima4_timed_from_fragment_bytes(self):
        # frag_keyframe+empty_moov: every packet is in a fragment; the
        # 640 kbps byte cap would say 8.8 s.
        audio = _trak(b"soun", _ima4_entry(22050), [], 22050, 0, track_id=1)
        data = _mp4([audio], 34 * 20672, extra=_moof(_traf(1, [34] * 20672)))
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(20672 * 64 / 22050)

    def test_fragments_after_chunk_table_counted(self):
        fields = bytearray(_sound_fields(8000))
        fields[8:10] = b"\x00\x01"
        entry = _box(b"ima4", bytes(fields) + struct.pack(">IIII", 1, 34, 34, 2) + _CHAN)
        audio = _trak(
            b"soun", entry, [34] * 7500, 8000, 8000,
            stts=[(7500, 1)], chunks=(60, [(1, 8000)]), track_id=1,
        )  # fmt: skip
        data = _mp4([audio], 1000, extra=_moof(_traf(1, [34] * 7500)))
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(120.0)

    def test_gsm_blocks_not_multiplied_by_channels(self):
        # ffmpeg decodes GSM as mono whatever the entry says.
        entry = _box(b"agsm", _sound_fields(8000, channels=2) + _CHAN)
        audio = _trak(b"soun", entry, [], 8000, 8000, sample_size=33, count=3000)
        assert get_audio_duration_seconds(_mp4([audio], 1000), "mov") == pytest.approx(60.0)

    def test_misaligned_packets_priced_at_the_blocks_their_bytes_hold(self):
        # 50-byte packets decode to one 34-byte block each (8 s); the
        # blocks in all their bytes together are never fewer.
        audio = _trak(b"soun", _ima4_entry(8000), [50] * 1000, 8000, 8000)
        expected = 50_000 // 34 * 64 / 8000
        assert get_audio_duration_seconds(_mp4([audio], 1000), "mov") == pytest.approx(expected)

    @pytest.mark.parametrize("codec", [b"ima4", b"MAC6", b"agsm"])
    def test_fixed_ratio_codec_without_channels_rejected(self, codec):
        entry = _box(codec, _sound_fields(8000, channels=0) + _CHAN)
        data = _mp4_with_samples(entry, [bytes(34)] * 100)
        with pytest.raises(ProviderError, match="without channels or a rate") as exc:
            get_audio_duration_seconds(data, "mov")
        assert exc.value.status_code == 400

    def test_fixed_ratio_codec_sized_in_frames_without_chunk_table_rejected(self):
        audio = _trak(b"soun", _ima4_entry(8000), [], 8000, 8000, sample_size=1, count=1000)
        with pytest.raises(ProviderError, match="sized in frames") as exc:
            get_audio_duration_seconds(_mp4([audio], 1000), "mov")
        assert exc.value.status_code == 400

    def test_ima4_priced_from_chunk_table(self):
        # Old QuickTime ADPCM: stts durations of 1, stsc counting PCM frames
        # (7,500 packets of 64). A version 1 entry claiming 1 frame a packet
        # does not send it to the byte cap (3.2 s here).
        fields = bytearray(_sound_fields(8000))
        fields[8:10] = b"\x00\x01"  # sound description version 1
        entry = _box(b"ima4", bytes(fields) + struct.pack(">IIII", 1, 34, 34, 2) + _CHAN)
        audio = _trak(
            b"soun", entry, [34] * 7500, 8000, 8000,
            stts=[(7500, 1)], chunks=(60, [(1, 8000)]),
        )  # fmt: skip
        assert get_audio_duration_seconds(_mp4([audio], 1000), "mov") == pytest.approx(60.0)

    def test_undecodable_entry_keeps_byte_cap(self):
        # A codec ffmpeg has no decoder for is bounded at the lossy cap.
        entry = _box(b"apac", _sound_fields(48000) + _CHAN)
        audio = _trak(b"soun", entry, [8000] * 600, 48000, 48000)
        data = _mp4([audio], 1000)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(600 * 8000 * 8 / 640_000)


class TestMp4PacketHeaders:
    """Codecs whose packets say how long they are are timed from their bytes."""

    def test_opus_packets_timed_from_toc(self):
        # Code 3 packets of six 20 ms CELT frames: 120 ms, 48x the shortest.
        samples = [bytes([19 << 3 | 3, 6]) + bytes(88)] * 1000
        data = _mp4_with_samples(_OPUS_ENTRY, samples, per_chunk=50)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(120.0)

    @pytest.mark.parametrize("entry", [_aac_entry(oti=0x6B, dsi=False), _MP3_ENTRY])
    def test_mpeg_frames_counted(self, entry):
        data = _mp4_with_samples(entry, [_MPEG25_FRAME] * 1000, per_chunk=50)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(72.0)

    def test_mp1_entry_frames_counted(self):
        # ffmpeg decodes frames of any layer under '.mp1'; mdhd claims 1 s.
        entry = _box(b".mp1", _sound_fields(8000) + _CHAN)
        data = _mp4_with_samples(entry, [_MPEG25_FRAME] * 1000, per_chunk=50)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(72.0)

    def test_mp1_layer1_priced_at_real_duration(self):
        # MPEG-1 Layer I, 32 kbps, 32 kHz: 48-byte frames of 384 samples.
        layer1 = b"\xff\xff\x18\xc0" + bytes(44)
        entry = _box(b".mp1", _sound_fields(32000) + _CHAN)
        data = _mp4_with_samples(entry, [layer1] * 1000, per_chunk=50)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(1000 * 384 / 32000)

    def test_mp1_without_sample_data_bounded_by_packet_count(self):
        # A Layer I frame is 384 samples at up to 48 kHz, the shortest.
        entry = _box(b".mp1", _sound_fields(48000) + _CHAN)
        data = _mp4_with_samples(entry, [bytes(40)] * 12_000, per_chunk=100)
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(12_000 * 384 / 48000)

    def test_fragmented_mp1_rejected(self):
        data = _mp4_with_samples(_box(b".mp1", _sound_fields(8000) + _CHAN), [_MPEG25_FRAME])
        data += _moof(_traf(0, [72] * 10))
        with pytest.raises(ProviderError, match="fragmented MPEG, Opus or AMR"):
            get_audio_duration_seconds(data, "mov")

    def test_mpeg_header_between_chunks_hides_nothing(self):
        # A 1440-byte frame header before each chunk would swallow its 20
        # real frames in a walk over the file; the demuxer never reads it.
        fake = b"\xff\xe3\xe0\xc0"
        data = _mp4_with_samples(_MP3_ENTRY, [_MPEG25_FRAME * 20] * 50, gap=fake)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(72.0)

    def test_fragmented_mpeg_rejected(self):
        data = _mp4_with_samples(_MP3_ENTRY, [_MPEG25_FRAME] * 10)
        data += _moof(_traf(0, [72] * 10))
        with pytest.raises(ProviderError, match="fragmented MPEG, Opus or AMR") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    def test_chunks_reaching_past_the_file_rejected(self):
        # 1000 chunks of 10 KB over the same bytes of a 20 KB file.
        audio = _trak(
            b"soun", _MP3_ENTRY, [10_000] * 1000, 8000, 8000,
            chunks=(1000, [(1, 1)]), offsets=[0] * 1000,
        )  # fmt: skip
        with pytest.raises(ProviderError, match="samples reach past the file"):
            get_audio_duration_seconds(_mp4([audio], 20_000), "m4a")

    @pytest.mark.parametrize("codec", [b"ac-3", b"ec-3"])
    def test_ac3_and_eac3_rejected(self, codec):
        # ffmpeg splits every AC-3 frame out of a sample: no packet bound holds.
        data = _mp4_with_samples(_box(codec, _sound_fields(48000) + _CHAN), [bytes(96)] * 10)
        with pytest.raises(ProviderError, match="AC-3 audio is not supported") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    def test_flac_frames_counted(self):
        entry = _box(b"fLaC", _sound_fields(44100) + _full_box(b"dfLa", bytes(38)))
        samples = [_flac_frame(0, bytes(990))] * 1000
        data = _mp4_with_samples(entry, samples, per_chunk=100, timescale=44100)
        assert len(data) * 8 / 4_608_000 < 2  # the byte floor alone
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(1000 * 4096 / 44100)

    def test_dense_flac_frames_rejected(self):
        entry = _box(b"fLaC", _sound_fields(44100) + _full_box(b"dfLa", bytes(38)))
        samples = [_flac_frame(0, bytes(4))] * 20_000  # 10-byte frames of silence
        data = _mp4_with_samples(entry, samples, per_chunk=1000, timescale=44100)
        with pytest.raises(ProviderError, match="too many frame headers"):
            get_audio_duration_seconds(data, "m4a")

    @pytest.mark.parametrize("wave", [False, True], ids=["iso", "quicktime-wave"])
    def test_alac_bounded_by_cookie_frame_length(self, wave):
        # 16-byte packets of silence, each decoding to 4096 samples.
        data = _mp4_with_samples(_alac_entry(wave=wave), [bytes(16)] * 10_000, timescale=44100)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(10_000 * 4096 / 44100)

    @pytest.mark.parametrize("cookie_rate", [22050, 48000, 96000])
    def test_alac_priced_at_the_cookie_rate(self, cookie_rate):
        # ffmpeg and AudioToolbox take the cookie's rate over the entry's,
        # where Apple's writer leaves 44100 at any rate.
        data = _mp4_with_samples(_alac_entry(cookie_rate=cookie_rate), [bytes(16)] * 10_000)
        expected = 10_000 * 4096 / cookie_rate
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(expected)

    def test_apple_alac_priced_at_mdhd_duration(self):
        # 60 s at 48 kHz, as afconvert writes it: 704 packets of 4096 frames
        # (the last one short), entry rate 44100, mdhd at the real rate.
        audio = _trak(
            b"soun", _alac_entry(cookie_rate=48000), [16] * 704, 48000, 48000 * 60,
            chunks=(1, [(1, 704)]),
        )  # fmt: skip
        data = _mp4([audio], 16 * 704)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(60.0, abs=0.1)

    def test_alac_cookie_without_rate_takes_the_entry_rate(self):
        entry = _alac_entry(cookie_rate=0, entry_rate=32000)
        data = _mp4_with_samples(entry, [bytes(16)] * 10_000)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(10_000 * 4096 / 32000)

    def test_alac_forged_cookie_rate_clamped(self):
        data = _mp4_with_samples(_alac_entry(cookie_rate=10_000_000), [bytes(16)] * 10_000)
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(10_000 * 4096 / 768_000)

    def test_alac_without_any_rate_rejected(self):
        entry = _alac_entry(cookie_rate=0, entry_rate=0)
        with pytest.raises(ProviderError, match="ALAC sample rate unknown"):
            get_audio_duration_seconds(_mp4_with_samples(entry, [bytes(16)] * 10), "m4a")

    def test_alac_without_cookie_rejected(self):
        entry = _box(b"alac", _sound_fields(44100) + _CHAN)
        with pytest.raises(ProviderError, match="without a magic cookie"):
            get_audio_duration_seconds(_mp4_with_samples(entry, [bytes(16)] * 10), "m4a")


class TestStructureBudget:
    """Containers of millions of tiny structures are refused before parsing."""

    def test_wav_with_many_chunks_before_data_rejected(self):
        wav = make_wav_bytes(1.0)
        data = _riff(wav[12:36] + _chunk(b"JUNK", b"") * 10_001 + wav[36:])
        for fmt in ("wav", "mp3"):
            with pytest.raises(ProviderError, match="too many chunks") as exc:
                get_audio_duration_seconds(data, fmt)
            assert exc.value.status_code == 400

    def test_wav_with_many_chunks_after_data_billed_to_eof(self):
        wav = make_wav_bytes(10.0, rate=16000)
        data = _riff(wav[12:] + _chunk(b"JUNK", b"") * 20_000)
        assert get_audio_duration_seconds(data, "wav") == pytest.approx(10.0 + 160_000 / 32_000)

    def test_aiff_with_many_chunks_rejected(self):
        aiff = _aiff_bytes(1.0, claimed_frames=8000)
        chunks = aiff[12:] + _chunk(b"JUNK", b"", little_endian=False) * 10_001
        data = b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFF" + chunks
        with pytest.raises(ProviderError, match="too many chunks"):
            get_audio_duration_seconds(data, "aiff")

    def test_mp4_with_many_boxes_rejected(self):
        data = _aac_60s() + _box(b"free", b"") * 100_001
        with pytest.raises(ProviderError, match="too many boxes") as exc:
            get_audio_duration_seconds(data, "m4a")
        assert exc.value.status_code == 400

    def test_flac_with_many_metadata_blocks_rejected(self):
        flac = _flac(_flac_frames(10), total_samples=44100)
        streaminfo_end = 4 + 4 + 34
        empty_padding = bytes([0x01, 0, 0, 0])
        data = flac[:streaminfo_end] + empty_padding * 10_001 + flac[streaminfo_end:]
        with pytest.raises(ProviderError, match="too many metadata blocks"):
            get_audio_duration_seconds(data, "flac")

    def test_busy_parser_slots_refuse_with_429_without_waiting(self):
        slots = threading.BoundedSemaphore(1)
        holder = threading.Thread(target=slots.acquire)
        holder.start()
        holder.join()
        started = time.monotonic()
        with (
            patch.object(audio_duration, "_PARSE_SLOTS", slots),
            pytest.raises(ProviderError, match="retry shortly") as exc,
        ):
            get_audio_duration_seconds(make_wav_bytes(1.0), "wav")
        # Unpaid clients can fill the slots: not a provider failure (5xx).
        assert exc.value.status_code == 429
        # Callers queue on the event loop; a worker thread never waits here.
        assert time.monotonic() - started < 1.0

    def test_parser_slot_released_after_an_error(self):
        slots = threading.BoundedSemaphore(1)
        with patch.object(audio_duration, "_PARSE_SLOTS", slots):
            for _ in range(2):
                with pytest.raises(ProviderError) as exc:
                    get_audio_duration_seconds(b"not audio", "mp3")
                assert exc.value.status_code == 400


def _adts_frame(payload: int = 33, *, index: int = 8, blocks: int = 1) -> bytes:
    """Mono AAC-LC ADTS frame at the sampling index's rate (8: 16 kHz)."""
    size = 7 + payload
    header = bytes(
        [
            0xFF,
            0xF1,
            1 << 6 | index << 2,
            1 << 6 | size >> 11 & 0x03,
            size >> 3 & 0xFF,
            (size & 0x07) << 5 | 0x1F,
            0xFC | (blocks - 1),
        ]
    )
    return header + bytes(payload)


def _adts_frames(count: int, **kwargs) -> bytes:
    return _adts_frame(**kwargs) * count


class TestAdtsAac:
    def test_adts_priced_by_frame_count(self):
        data = _adts_frames(937)
        assert get_audio_duration_seconds(data, "aac") == pytest.approx(937 * 1024 / 16000)

    def test_adts_after_id3(self):
        data = _id3(b"\x00" * 50) + _adts_frames(937)
        assert get_audio_duration_seconds(data, "aac") == pytest.approx(937 * 1024 / 16000)

    def test_adts_raw_data_blocks_counted(self):
        data = _adts_frames(100, blocks=4, index=3)  # 48 kHz, 4096 samples a frame
        assert get_audio_duration_seconds(data, "aac") == pytest.approx(100 * 4096 / 48000)

    def test_adts_junk_between_frames_skipped(self):
        data = _adts_frames(500) + b"\x00" * 1000 + _adts_frames(500)
        assert get_audio_duration_seconds(data, "aac") == pytest.approx(1000 * 1024 / 16000)

    def test_dense_adts_headers_rejected(self):
        with pytest.raises(ProviderError, match="too many frame headers") as exc:
            get_audio_duration_seconds(_adts_frames(20_000, payload=0), "aac")
        assert exc.value.status_code == 400
