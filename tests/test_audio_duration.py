"""Unit tests for audio duration parsing."""

import io
import random
import struct
import wave
from unittest.mock import MagicMock, patch

import pytest
from mutagen import File as MutagenFile
from mutagen.ogg import OggPage
from mutagen.oggopus import OggOpus
from mutagen.oggspeex import OggSpeex
from mutagen.oggvorbis import OggVorbis

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

    def test_mutagen_format_duration(self):
        mock_info = MagicMock()
        mock_info.length = 2.5
        mock_file = MagicMock()
        mock_file.info = mock_info
        with patch("x402gate.core.audio_duration.MutagenFile", return_value=mock_file):
            assert get_audio_duration_seconds(b"fake-mp3-bytes", "mp3") == 2.5

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

    def test_lossy_container_floored_by_max_bitrate(self):
        mock_file = MagicMock(spec=OggOpus)
        mock_file.info = MagicMock(length=1.0, codec="")
        data = b"\x00" * 800_000  # 6.4 Mbit / 640 kbps = 10 s
        with patch("x402gate.core.audio_duration.MutagenFile", return_value=mock_file):
            assert get_audio_duration_seconds(data, "ogg") == pytest.approx(10.0)

    def test_honest_lossy_container_keeps_header_duration(self):
        mock_file = MagicMock(spec=OggOpus)
        mock_file.info = MagicMock(length=100.0, codec="")
        data = b"\x00" * 800_000  # 64 kbps over 100 s
        with patch("x402gate.core.audio_duration.MutagenFile", return_value=mock_file):
            assert get_audio_duration_seconds(data, "ogg") == 100.0


def _riff(chunks: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _chunk(chunk_id: bytes, payload: bytes, *, little_endian: bool = True) -> bytes:
    size = struct.pack("<I" if little_endian else ">I", len(payload))
    return chunk_id + size + payload + b"\x00" * (len(payload) & 1)


def _aifc_bytes(compression: bytes, payload: bytes, *, claimed_frames: int) -> bytes:
    """Build a mono 16-bit 8 kHz AIFF-C with the given compression ID."""
    rate_8000 = b"\x40\x0b\xfa\x00\x00\x00\x00\x00\x00\x00"
    comm = struct.pack(">hIh", 1, claimed_frames, 16) + rate_8000 + compression + b"\x00\x00"
    chunks = _chunk(b"FVER", struct.pack(">I", 0xA2805140), little_endian=False)
    chunks += _chunk(b"COMM", comm, little_endian=False)
    chunks += _chunk(b"SSND", struct.pack(">II", 0, 0) + payload, little_endian=False)
    return b"FORM" + struct.pack(">I", 4 + len(chunks)) + b"AIFC" + chunks


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

    def test_compressed_aifc_rejected(self):
        gsm_blocks = b"\x00" * 33 * 12_000  # 240 s of GSM 6.10
        data = _aifc_bytes(b"GSM ", gsm_blocks, claimed_frames=160)
        assert MutagenFile(io.BytesIO(data)) is not None
        with pytest.raises(ProviderError, match="uncompressed PCM AIFF"):
            get_audio_duration_seconds(data, "aiff")

    def test_pcm_aifc_accepted(self):
        data = _aifc_bytes(b"sowt", b"\x00\x00" * 80_000, claimed_frames=80_000)
        assert get_audio_duration_seconds(data, "aiff") == pytest.approx(10.0)

    def test_aiff_with_two_comm_chunks_rejected(self):
        data = _aiff_bytes(1.0, claimed_frames=8000)
        comm = data[12:38]
        data = data[:12] + comm + data[12:]
        with pytest.raises(ProviderError, match="uncompressed PCM AIFF"):
            get_audio_duration_seconds(data, "aiff")


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
) -> bytes:
    """A trak; ``sample_size`` and ``count`` give a constant-size stsz instead
    of ``sizes``, ``chunks`` a chunk count and its stsc (first chunk, samples)."""
    mdhd = _full_box(b"mdhd", struct.pack(">IIII", 0, 0, timescale, duration) + b"\x00" * 4)
    hdlr = _full_box(b"hdlr", b"\x00" * 4 + handler + b"\x00" * 13)
    stsd = _full_box(b"stsd", struct.pack(">I", 1) + entry)
    if sample_size:
        stsz = _full_box(b"stsz", struct.pack(">II", sample_size, count))
    else:
        table = struct.pack(f">{len(sizes)}I", *sizes)
        stsz = _full_box(b"stsz", struct.pack(">II", 0, len(sizes)) + table)
    stts_table = b"".join(struct.pack(">II", *run) for run in stts)
    tables = _full_box(b"stts", struct.pack(">I", len(stts)) + stts_table) + stsz
    if chunks is not None:
        chunk_count, runs = chunks
        stsc = b"".join(struct.pack(">III", first, samples, 1) for first, samples in runs)
        tables += _full_box(b"stsc", struct.pack(">I", len(runs)) + stsc)
        tables += _full_box(b"stco", struct.pack(">I", chunk_count) + bytes(4 * chunk_count))
    stbl = _box(b"stbl", stsd + tables)
    mdia = _box(b"mdia", mdhd + hdlr + _box(b"minf", stbl))
    tkhd = _full_box(b"tkhd", b"\x00" * 8 + struct.pack(">I", track_id) + b"\x00" * 68)
    return _box(b"trak", tkhd + mdia)


def _sound_fields(rate: int, *, channels: int = 1, bits: int = 16) -> bytes:
    fields = struct.pack(">HHHHI", channels, bits, 0, 0, rate << 16)
    return b"\x00" * 6 + b"\x00\x01" + b"\x00" * 8 + fields


_CHAN = _full_box(b"chan", bytes(20))  # mutagen wants a child box after the fields


def _aac_entry() -> bytes:
    """mp4a entry for mono AAC-LC at 16 kHz (sampling frequency index 8)."""
    asc = (2 << 11 | 8 << 7 | 1 << 3).to_bytes(2, "big")
    dsi = bytes([0x05, len(asc)]) + asc
    dcd = bytes([0x04, 13 + len(dsi), 0x40, 0x15]) + b"\x00" * 11 + dsi
    esd = bytes([0x03, 3 + len(dcd)]) + b"\x00\x01\x00" + dcd + b"\x06\x01\x02"
    return _box(b"mp4a", _sound_fields(16000) + _full_box(b"esds", esd))


def _mp4(traks: list[bytes], mdat_size: int, *, extra: bytes = b"", mvex: bytes = b"") -> bytes:
    mvhd = _full_box(b"mvhd", struct.pack(">IIII", 0, 0, 1000, 0) + b"\x00" * 80)
    ftyp = _box(b"ftyp", b"M4A \x00\x00\x00\x00M4A mp42isom")
    moov = _box(b"moov", mvhd + b"".join(traks) + mvex)
    return ftyp + moov + extra + _box(b"mdat", b"\x00" * mdat_size)


_AAC_FRAMES_60S = 16000 * 60 // 1024


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

    def test_forged_amr_duration_bounded_by_largest_frame(self):
        entry = _box(b"samr", _sound_fields(8000) + _box(b"damr", b"\x00" * 9))
        # 60 s of 12.2k AMR-NB: 3000 frames of 32 bytes, 10 frames a sample.
        audio = _trak(b"soun", entry, [320] * 300, 8000, 8000)
        data = _mp4([audio], 320 * 300)
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
        hidden = _trak(b"meta", _box(b".mp2", _sound_fields(48000)), [1000] * 960, 48000, 48000)
        data = _mp4([decoy, hidden], 40 * 16 + 1000 * 960)
        # 60 s of 128 kbps MP2, at the 640 kbps cap for codecs not timed here.
        assert get_audio_duration_seconds(data, "mov") == pytest.approx(12.0)

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

    def test_fragment_without_sample_sizes_falls_back_to_file_size(self):
        audio = _trak(b"soun", _aac_entry(), [], 16000, 16000, track_id=1)
        data = _mp4([audio], 800_000, extra=_moof(_traf(1, count=100)))
        assert get_audio_duration_seconds(data, "m4a") >= 800_000 * 8 / 640_000

    def test_fragment_with_no_audio_samples_not_billed_by_file_size(self):
        # mdat bytes no sample table or trun points at are never decoded.
        audio = _trak(b"soun", _aac_entry(), [], 16000, 16000, track_id=1)
        data = _mp4([audio], 800_000, extra=_box(b"moof", _full_box(b"mfhd", b"\x00" * 4)))
        assert get_audio_duration_seconds(data, "m4a") == pytest.approx(1.0)


def _ogg_opus(packets: int, *, last_granule: int, junk: bytes = b"") -> bytes:
    """Ogg Opus with 20 ms CELT packets, 50 per page, and a chosen last granule."""
    pages = []
    head = OggPage()
    head.serial, head.sequence, head.position, head.first = 5, 0, 0, True
    head.packets = [b"OpusHead\x01\x01" + struct.pack("<HI", 312, 48000) + b"\x00\x00\x00"]
    tags = OggPage()
    tags.serial, tags.sequence, tags.position = 5, 1, 0
    tags.packets = [b"OpusTags" + b"\x00" * 8]
    pages += [head.write(), tags.write()]
    for index in range(0, packets, 50):
        page = OggPage()
        page.serial, page.sequence = 5, 2 + index // 50
        page.packets = [bytes([19 << 3]) + b"\x55" * 19] * min(50, packets - index)
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
