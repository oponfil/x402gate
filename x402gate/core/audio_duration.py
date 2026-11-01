"""Parse audio file duration for STT pricing.

Container headers (an MP3 Xing/VBRI frame count, a WAV data chunk size, an
MP4 or Ogg duration field) are client-controlled, while the provider bills
the audio it actually decodes. Every duration is therefore raised to an
independent lower bound taken from the bytes themselves, so a forged header
cannot price a long file as a short one.

The bound is tight where the format allows it: MPEG and FLAC frames are
counted, PCM is sized from its sample chunk (in MP4, from the frames its
sample and chunk tables give), AAC and AMR in MP4 from the sample table (or
fragment runs), Opus in Ogg from each packet's TOC byte and Vorbis from each
packet's block size. Speex in Ogg is bounded from its packet count, other MP4
codecs from their sample bytes at the family's top bitrate.

All of this runs before payment, so every scan has a work budget: input that
would take far more work than any genuine file is rejected, not walked.
"""

from __future__ import annotations

import array
import io
import math
import operator
import re
import struct
import sys
import wave
import zlib
from dataclasses import dataclass
from itertools import accumulate, compress, repeat

from mutagen import File as MutagenFile
from mutagen import MutagenError
from mutagen.aac import AAC
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggopus import OggOpus
from mutagen.oggspeex import OggSpeex
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from x402gate.providers.base import ProviderError

# Highest average bitrate a genuine file of each family can have. Bytes / cap
# is then never above the real duration, so honest files are not overcharged.
_MPEG_MAX_BPS = 448_000  # top MPEG frame bitrate (Layer I; Layer III is 320k)
_LOSSY_MAX_BPS = 640_000  # above Opus (510k), stereo AAC (576k), Vorbis q10
_LOSSLESS_MAX_BPS = 4_608_000  # 96 kHz / 24-bit stereo PCM
_LOSSY_TYPES = (AAC, MP4, OggOpus, OggSpeex, OggVorbis)
# A frame count past this stops the scan: 25 MB cannot hold a day of real
# (not digitally silent) lossless audio, and counting on only costs CPU.
_MAX_COUNTED_SECONDS = 24 * 3600

# AIFF-C compression IDs that store plain PCM, the only ones the
# channels x bits x rate byte floor is valid for.
_AIFC_PCM = frozenset(
    (b"NONE", b"sowt", b"twos", b"raw ", b"in24", b"in32", b"fl32", b"FL32", b"fl64", b"FL64")
)

# MP4 sample entry codecs that carry uncompressed or lossless audio.
_MP4_LOSSLESS_CODECS = frozenset(
    (b"alac", b"fLaC", b"lpcm", b"sowt", b"twos", b"in24", b"in32", b"fl32", b"fl64", b"raw ",
     b"ipcm", b"fpcm", b"NONE", b"ms\x00\x01", b"ms\x00\x03", b"TS\x00\x01", b"TS\x00\x03")
)  # fmt: skip
# Sample entries of data, subtitle and video tracks. ffmpeg decodes any track
# that is not 'vide' as audio when its entry is an audio codec, so a track
# under another handler is skipped only when all its entries are listed here.
# None of them is an audio tag ('raw ' is both, so it is not listed).
_MP4_NON_AUDIO_ENTRIES = frozenset(
    (b"tmcd", b"mebx", b"tx3g", b"text", b"c608", b"c708", b"wvtt", b"stpp", b"sbtt", b"stxt",
     b"dfxp", b"gpmd", b"camm", b"mp4s", b"metx", b"mett", b"urim", b"uri ", b"rtp ", b"srtp",
     b"rrtp", b"fdp ", b"avc1", b"avc2", b"avc3", b"avc4", b"hvc1", b"hev1", b"dvh1", b"dvhe",
     b"dvav", b"dva1", b"vvc1", b"vvi1", b"av01", b"vp08", b"vp09", b"mp4v", b"encv", b"resv",
     b"jpeg", b"mjpa", b"mjpb", b"png ", b"apch", b"apcn", b"apcs", b"apco", b"ap4h", b"ap4x",
     b"s263", b"h263")
)  # fmt: skip
# Bytes per sample ffmpeg reads for each fixed-width PCM sample entry; twos,
# sowt, NONE, raw and the LPCM entries take theirs from the sample size field.
_MP4_PCM_WIDTHS = {b"in24": 3, b"in32": 4, b"fl32": 4, b"fl64": 8, b"ulaw": 1, b"alaw": 1}
# WAVE format tags in QuickTime 'ms'/'TS' entries that are PCM, by the entry
# whose sample layout they share.
_MP4_WAV_PCM = {1: b"sowt", 3: b"fl32", 6: b"alaw", 7: b"ulaw"}
# PCM frames per packet ffmpeg assumes for version 0 entries of these codecs.
_MP4_V0_FRAMES_PER_PACKET = {b"MAC3": 6, b"MAC6": 6, b"ima4": 64, b"agsm": 160}
_MP4_MAX_RATE = 768_000  # highest real PCM rate; a forged higher one is clamped
# Largest AMR frame (mode byte included) per 20 ms: 32 bytes for AMR-NB
# 12.2k, 62 for AMR-WB 23.85k. Every sample byte is frame data.
_AMR_MAX_BPS = {b"samr": 32 * 8 * 50, b"sawb": 62 * 8 * 50}
# Bytes of AudioSampleEntry fields before its child boxes, by QuickTime
# sound description version (0 is also the ISO layout).
_MP4_SOUND_FIELDS = {0: 28, 1: 44, 2: 64}
# MPEG-4 object types for AAC in esds, and the audio object types that code
# 1024 (or 960) samples per frame.
_AAC_OBJECT_TYPES = frozenset((0x40, 0x66, 0x67, 0x68))
_AAC_LONG_FRAME_AOTS = frozenset((1, 2, 3, 4, 5, 29))
_AAC_RATES = (
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350
)  # fmt: skip

_OGG_BIT_REVERSE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))
_ZERO_CRC = bytes(4)
_OGG_HEAD_BYTES = 72  # packet bytes kept: up to the Speex header's extra_headers
_VORBIS_MAX_SETUP = 1 << 20  # setup headers are a few KB; parsing is linear in size
# Work units of the Ogg walk: a page costs about as much as 16 packets, one
# that completes no packet (an empty page) more. Real streams stay under half
# a unit per byte, even 6 kbps Opus with a packet on every page.
_OGG_PAGE_COST, _OGG_EMPTY_PAGE_COST, _OGG_SLOW_PACKET_COST = 16, 64, 8

# MPEG audio frame tables, keyed by (is MPEG-1, layer); index 0 (free format)
# and 15 (invalid) are excluded by the sync pattern below.
_MPEG_KBPS = {
    (True, 1): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (True, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (True, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (False, 1): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (False, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (False, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
# Keyed by the header's version bits: 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5.
_MPEG_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mpeg_sync_pattern() -> re.Pattern[bytes]:
    """Match only fully valid MPEG audio frame headers.

    Rejecting reserved versions, layers, bitrates and sample rates in the
    regex keeps the scan in C even over junk bytes.
    """
    second = [
        0xE0 | version << 3 | layer << 1 | crc
        for version in (0, 2, 3)
        for layer in (1, 2, 3)
        for crc in (0, 1)
    ]
    third = [
        index << 4 | rate << 2 | low
        for index in range(1, 15)
        for rate in range(3)
        for low in range(4)
    ]
    return re.compile(
        b"\xff["
        + b"".join(re.escape(bytes([v])) for v in second)
        + b"]["
        + b"".join(re.escape(bytes([v])) for v in third)
        + b"]"
    )


_MPEG_SYNC = _mpeg_sync_pattern()

# FLAC frame header tables (RFC 9639, 9.1.1 and 9.1.2): samples per block and
# sample rate by code. Block size codes 6/7 and rate codes 12-14 read their
# value from bytes after the coded number; rate code 0 means STREAMINFO's.
_FLAC_BLOCK_SIZES = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608} | {
    code: 256 << (code - 8) for code in range(8, 16)
}
_FLAC_RATES = (0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000)
# Frames are counted at no less than the lowest table rate. A header found by
# chance in audio data may claim 1 Hz (up to 18 h for one header); audio
# really sampled below 8 kHz is no use for STT, so no bypass opens.
_FLAC_MIN_COUNT_RATE = 8000


def _flac_sync_pattern() -> re.Pattern[bytes]:
    """Match the fixed part of a FLAC frame header a decoder accepts.

    Sync with a zero reserved bit, no reserved block size (0) or sample rate
    (15) code, no reserved channel (11-15) or sample size (3) code, a zero
    reserved bit, and a valid UTF-8 lead byte for the coded number.
    """
    rates = [block << 4 | rate for block in range(1, 16) for rate in range(15)]
    layout = [channels << 4 | size << 1 for channels in range(11) for size in range(8) if size != 3]
    leads = [*range(0x80), *range(0xC0, 0xFF)]
    return re.compile(
        b"\xff[\xf8\xf9]"
        + b"".join(b"[" + b"".join(re.escape(bytes([v])) for v in values) + b"]"
                   for values in (rates, layout, leads))
    )  # fmt: skip


_FLAC_SYNC = _flac_sync_pattern()


def _crc8_table() -> bytes:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc << 1 ^ 0x07 if crc & 0x80 else crc << 1) & 0xFF
        table.append(crc)
    return bytes(table)


_FLAC_CRC8 = _crc8_table()  # CRC-8, polynomial 0x07, over the frame header


def _opus_toc_tables() -> tuple[tuple[float, ...], bytes]:
    """Seconds per Opus packet by TOC byte (0 for code 3), and a code-3 mask."""
    seconds = []
    for toc in range(256):
        config = toc >> 3
        if config < 12:  # SILK
            frame = (0.01, 0.02, 0.04, 0.06)[config & 3]
        elif config < 16:  # hybrid
            frame = (0.01, 0.02)[config & 1]
        else:  # CELT
            frame = (0.0025, 0.005, 0.01, 0.02)[config & 3]
        seconds.append((frame, 2 * frame, 2 * frame, 0.0)[toc & 3])
    return tuple(seconds), bytes(toc & 3 == 3 for toc in range(256))


_OPUS_TOC_SECONDS, _OPUS_CODE3 = _opus_toc_tables()
_BITS_LSB_FIRST = tuple(f"{byte:08b}"[::-1].encode() for byte in range(256))


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

    try:
        audio_file = MutagenFile(io.BytesIO(data))
    except MutagenError:
        audio_file = None  # malformed client bytes are a 400, not a 500
    if audio_file is None or audio_file.info is None:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse audio format '{normalized}'",
            status_code=400,
        )

    if isinstance(audio_file, WAVE):
        # A WAV under another format name gets the same PCM-only parse as
        # 'wav': a compressed codec (MP3 or GSM in WAV) has no fixed byte
        # rate, and mutagen takes its length from client-set header fields.
        return _wav_duration(data, provider=provider)
    if isinstance(audio_file, AIFF):
        _require_pcm_aiff(data, audio_file.info, provider=provider)

    length = getattr(audio_file.info, "length", None)
    if length is None:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot determine duration for format '{normalized}'",
            status_code=400,
        )

    duration = max(float(length), _content_floor_seconds(data, audio_file, provider=provider))
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
    # The wave module reads the last fmt chunk before the data, ffmpeg the
    # first; with two of them the priced format may not be the decoded one.
    if sum(chunk_id == b"fmt " for chunk_id, _, _ in _iff_chunks(data, little_endian=True)) > 1:
        raise ProviderError(
            provider=provider,
            detail="Cannot parse WAV audio: more than one fmt chunk",
            status_code=400,
        )
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
            byte_rate = wav_file.getnchannels() * wav_file.getsampwidth() * rate
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse WAV audio: {exc}",
            status_code=400,
        ) from exc

    # The data chunk size is client-controlled; price at least the PCM bytes
    # actually sent, since a decoder can read them all.
    duration = max(duration, _pcm_floor_seconds(data, b"data", byte_rate, little_endian=True))
    if duration <= 0:
        raise ProviderError(
            provider=provider,
            detail="Audio duration must be greater than zero",
            status_code=400,
        )
    return duration


def _require_pcm_aiff(data: bytes, info: object, *, provider: str = "") -> None:
    """Reject AIFF audio the PCM byte floor cannot bound.

    A compressed AIFF-C codec (GSM, IMA ADPCM, mu-law) packs more audio into
    each byte than channels x bits x rate, and with two COMM chunks mutagen
    and the decoder may read different formats.
    """
    chunks = _iff_chunks(data, little_endian=False)
    comms = [pos for chunk_id, pos, _ in chunks if chunk_id == b"COMM"]
    valid = len(comms) == 1 and int(getattr(info, "bitrate", 0) or 0) > 0
    if valid and data[8:12] == b"AIFC":
        valid = data[comms[0] + 18 : comms[0] + 22] in _AIFC_PCM
    if not valid:
        raise ProviderError(
            provider=provider,
            detail="Only uncompressed PCM AIFF audio is supported",
            status_code=400,
        )


def _content_floor_seconds(data: bytes, audio_file: object, *, provider: str = "") -> float:
    """Lower bound on the decodable duration that ignores header length fields."""
    info = audio_file.info  # type: ignore[attr-defined]
    if isinstance(audio_file, MP3):
        # Count the real frames; the byte floor covers free-format frames
        # that the frame scan cannot size.
        audio_bytes = len(data) - _id3v2_size(data)
        return max(_mpeg_frames_seconds(data), audio_bytes * 8 / _MPEG_MAX_BPS)
    if isinstance(audio_file, AIFF):
        # mutagen's bitrate here is channels x bits x rate, the PCM byte rate
        # a decoder reads the sample data at (compressed AIFF-C is rejected).
        byte_rate = int(getattr(info, "bitrate", 0) or 0) // 8
        return _pcm_floor_seconds(data, b"SSND", byte_rate, little_endian=False)
    if isinstance(audio_file, MP4):
        return _mp4_floor_seconds(data, info)
    lossy = isinstance(audio_file, _LOSSY_TYPES) and getattr(info, "codec", "") != "alac"
    floor = len(data) * 8 / (_LOSSY_MAX_BPS if lossy else _LOSSLESS_MAX_BPS)
    if isinstance(audio_file, FLAC):
        rate = int(getattr(info, "sample_rate", 0) or 0)
        floor = max(floor, _flac_frames_seconds(data, rate, provider=provider))
    if data[:4] == b"OggS":
        floor = max(floor, _ogg_packets_seconds(data, provider=provider))
    return floor


def _pcm_floor_seconds(
    data: bytes, chunk_id: bytes, byte_rate: int, *, little_endian: bool
) -> float:
    """Seconds of PCM in the sample chunk, counted from the bytes sent.

    The chunk's size is trusted only when well-formed chunks follow it
    exactly to the end of the file and none of them holds more samples, so
    trailing metadata (LIST, id3 cover art) is not billed. Otherwise every
    byte from the chunk's start to EOF counts, which covers a forged size.
    """
    if byte_rate <= 0:
        return 0.0
    chunk = next(
        (
            (pos, size)
            for found, pos, size in _iff_chunks(data, little_endian=little_endian)
            if found == chunk_id
        ),
        None,
    )
    if chunk is None:
        return len(data) / byte_rate
    start, size = chunk
    end = start + size
    if not (size and _chunks_tile_to_eof(data, end + (size & 1), little_endian=little_endian)):
        end = len(data)
    if chunk_id == b"SSND":
        start += 8  # AIFF SSND payload opens with offset and block size words
    return max(0, min(end, len(data)) - start) / byte_rate


def _iff_chunks(data: bytes, *, little_endian: bool):
    """Yield (id, payload offset, declared size) for each top-level RIFF/AIFF chunk."""
    size_format = "<I" if little_endian else ">I"
    pos = 12  # past the RIFF/FORM header and its WAVE/AIFF form type
    while pos + 8 <= len(data):
        (size,) = struct.unpack_from(size_format, data, pos + 4)
        yield data[pos : pos + 4], pos + 8, size
        pos += 8 + size + (size & 1)


def _chunks_tile_to_eof(data: bytes, pos: int, *, little_endian: bool) -> bool:
    """True when named, non-sample chunks run from ``pos`` exactly to EOF."""
    size_format = "<I" if little_endian else ">I"
    pad = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        # Printable IDs keep raw PCM (zero runs read as empty chunks) out.
        if chunk_id in (b"data", b"SSND") or not all(0x20 <= b <= 0x7E for b in chunk_id):
            return False
        (size,) = struct.unpack_from(size_format, data, pos + 4)
        pad = size & 1
        pos += 8 + size + pad
    # The last chunk may omit its pad byte.
    return pos == len(data) or (pad == 1 and pos == len(data) + 1)


def _id3v2_size(data: bytes) -> int:
    """Return the size of a leading ID3v2 tag (0 when absent), capped at the data."""
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size = 10 + (
        (data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | data[9] & 0x7F
    )
    if data[5] & 0x10:
        size += 10  # footer
    return min(size, len(data))


def _mpeg_frames_seconds(data: bytes) -> float:
    """Sum the duration of every MPEG audio frame in the data.

    Walks frame to frame and resyncs past junk the way decoders do, so a
    forged Xing/VBRI frame count or padding between frames hides nothing.
    """
    total = 0.0
    # Decoders skip the ID3v2 tag, so cover art in it cannot hold frames.
    pos = _id3v2_size(data)
    first = True
    while (match := _MPEG_SYNC.search(data, pos)) is not None:
        start = match.start()
        b1, b2 = data[start + 1], data[start + 2]
        version = (b1 >> 3) & 3
        layer = 4 - ((b1 >> 1) & 3)
        mpeg1 = version == 3
        bitrate = _MPEG_KBPS[(mpeg1, layer)][b2 >> 4] * 1000
        rate = _MPEG_RATES[version][(b2 >> 2) & 3]
        padding = (b2 >> 1) & 1
        if layer == 1:
            samples = 384
            frame_size = (12 * bitrate // rate + padding) * 4
        else:
            samples = 1152 if layer == 2 or mpeg1 else 576
            frame_size = samples // 8 * bitrate // rate + padding
        # A leading Xing/Info/VBRI frame carries the tag, not audio.
        head = data[start : start + 40]
        if not (first and (b"Xing" in head or b"Info" in head or b"VBRI" in head)):
            total += samples / rate
        first = False
        pos = start + frame_size
    return total


def _flac_frames_seconds(data: bytes, stream_rate: int, *, provider: str = "") -> float:
    """Sum the blocks of every FLAC frame header a decoder would accept.

    STREAMINFO's total_samples is client-set, so the frames themselves set
    the bound. Every header that passes the decoder's checks (reserved bits,
    codes, coded number, CRC-8, first subframe header) is counted, including
    overlapping ones: skipping past a header by a guessed frame length would
    let a crafted header that overlaps a real frame's sync hide that frame.
    The checks pass about three false headers per 100 MB of random bytes, and
    each is counted at 8 kHz or faster, so honest files are not overcharged
    by more than a rounding error.

    Genuine frames average kilobytes, and random audio bytes match the sync
    pattern about once per 160 KB, so more matches than one per 64 bytes
    means crafted input; it is rejected rather than scanned (stopping early
    instead would let a junk prefix hide the real frames after it).
    """
    total = 0.0
    budget = len(data) // 64 + 10_000
    pos = _flac_audio_start(data)
    while (match := _FLAC_SYNC.search(data, pos)) is not None:
        budget -= 1
        if budget < 0:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse FLAC audio: too many frame headers",
                status_code=400,
            )
        start = match.start()
        pos = start + 1
        frame = _flac_frame_header(data, start, stream_rate)
        if frame is not None:
            total += frame[0] / frame[1]
            if total > _MAX_COUNTED_SECONDS:
                break
    return total


def _flac_audio_start(data: bytes) -> int:
    """Offset of the first byte after the FLAC metadata blocks.

    Decoders skip each block by its declared length, so cover art in a
    PICTURE block cannot hold frames.
    """
    pos = _id3v2_size(data)
    if data[pos : pos + 4] != b"fLaC":
        return 0
    pos += 4
    while pos + 4 <= len(data):
        last = data[pos] & 0x80
        pos += 4 + int.from_bytes(data[pos + 1 : pos + 4], "big")
        if last:
            break
    return min(pos, len(data))


def _flac_frame_header(data: bytes, start: int, stream_rate: int) -> tuple[int, int] | None:
    """Return (samples, counting rate) of the frame header at ``start``, or None.

    The fixed bytes were checked by the sync pattern.
    """
    head = data[start : start + 18]  # longest header plus the subframe byte
    codes = head[2]
    lead = head[4]
    pos = 5 + (8 - (~lead & 0xFF).bit_length() - 1 if lead >= 0x80 else 0)
    if len(head) < pos or any(byte & 0xC0 != 0x80 for byte in head[5:pos]):
        return None  # truncated or bad UTF-8 continuation bytes
    block_code, rate_code = codes >> 4, codes & 0x0F
    extra = {6: 1, 7: 2}.get(block_code, 0) + {12: 1, 13: 2, 14: 2}.get(rate_code, 0)
    if len(head) < pos + extra + 2:
        return None
    if block_code == 6:
        samples = head[pos] + 1
        pos += 1
    elif block_code == 7:
        samples = (head[pos] << 8 | head[pos + 1]) + 1
        pos += 2
    else:
        samples = _FLAC_BLOCK_SIZES[block_code]
    if rate_code == 12:
        rate = head[pos] * 1000
        pos += 1
    elif rate_code in (13, 14):
        rate = (head[pos] << 8 | head[pos + 1]) * (10 if rate_code == 14 else 1)
        pos += 2
    else:
        rate = _FLAC_RATES[rate_code] if rate_code else stream_rate
    crc = 0
    for byte in head[:pos]:
        crc = _FLAC_CRC8[crc ^ byte]
    # The first subframe header has a zero pad bit and no reserved type
    # (2-7, 13-31); decoders reject the frame otherwise.
    subframe = head[pos + 1] >> 1
    valid_type = subframe < 2 or 8 <= subframe <= 12 or subframe >= 32
    if crc != head[pos] or subframe >= 64 or not valid_type or rate <= 0:
        return None
    return samples, max(rate, _FLAC_MIN_COUNT_RATE)


@dataclass
class _Mp4Track:
    """What one MP4 audio track's tables say the decoder will read."""

    codec: bytes
    frame_seconds: float | None  # AAC: the shortest frame the decoder outputs
    rate: float  # decode sample rate: stsd's, or mdhd's timescale without one
    frame_bytes: int  # PCM bytes per frame (all channels); 0 when not PCM
    count: int = 0  # coded samples (packets)
    sample_bytes: int = 0
    frames: int | None = None  # PCM frames, when the samples are sized in frames


def _mp4_floor_seconds(data: bytes, info: object) -> float:
    """Lower bound from the sample tables of the MP4 audio tracks.

    The decoder reads exactly the samples a track's stsz (or, in a
    fragmented movie, its trun boxes) lists, or for uncompressed QuickTime
    audio the frames its chunk table lists, so they set the bound rather than
    the forgeable mdhd duration, and the bytes of video, timecode, metadata
    or subtitle tracks are not billed as audio. The longest track sets the
    price: the provider transcribes one audio stream, so a second language
    or a spatial audio track is not billed again. Only a movie with no audio
    track, a compressed moov, or a layout this walk cannot follow (malformed
    boxes, fragment sample sizes given nowhere) falls back to the whole file
    at the top bitrate: nothing smaller is then known to hold all the audio.
    """
    try:
        tracks = _mp4_audio_tracks(data)
    except (ValueError, IndexError, struct.error):
        tracks = None
    if tracks:
        return max(_mp4_track_floor(track) for track in tracks)
    lossless = getattr(info, "codec", "") == "alac"
    return len(data) * 8 / (_LOSSLESS_MAX_BPS if lossless else _LOSSY_MAX_BPS)


def _mp4_track_floor(track: _Mp4Track) -> float:
    """Seconds one audio track's samples must decode to."""
    if track.codec in _AMR_MAX_BPS:
        floor = track.sample_bytes * 8 / _AMR_MAX_BPS[track.codec]
    else:
        cap = _LOSSLESS_MAX_BPS if track.codec in _MP4_LOSSLESS_CODECS else _LOSSY_MAX_BPS
        floor = track.sample_bytes * 8 / cap
        if track.frame_seconds is not None:
            floor = max(floor, track.count * track.frame_seconds)
    rate = min(track.rate, _MP4_MAX_RATE)
    if track.frames is None or rate <= 0:
        return floor
    # PCM frames at the decode rate are its exact duration; the byte floor
    # would overcharge PCM above the lossless cap (8 channels, 24-bit).
    seconds = track.frames / rate
    return seconds if track.frame_bytes else max(seconds, floor)


def _mp4_audio_tracks(data: bytes) -> list[_Mp4Track] | None:
    """Return the sample totals of each audio track.

    A track is audio when its handler is 'soun', or when it is not 'vide'
    and not every sample entry is a known data, subtitle or video type:
    ffmpeg decodes an audio entry as audio whatever the handler says, and an
    unknown entry is billed rather than trusted. Samples in fragments
    (moof/traf/trun) are added to their track's moov sample table. None for
    a compressed movie. Raises ValueError on malformed boxes or unknown
    fragment sizes.
    """
    top = list(_mp4_boxes(data, 0, len(data)))
    moov = next(((start, end) for kind, start, end in top if kind == b"moov"), None)
    if moov is None:
        return None
    tracks: list[_Mp4Track] = []
    track_ids: list[int] = []
    audio_ids: dict[int, _Mp4Track] = {}
    trex_sizes: dict[int, int] = {}
    for kind, start, end in _mp4_boxes(data, *moov):
        if kind == b"cmov":
            return None
        if kind == b"mvex":
            for found, payload, _ in _mp4_boxes(data, start, end):
                if found == b"trex":
                    track_id, default_size = struct.unpack_from(">I8xI", data, payload + 4)
                    trex_sizes[track_id] = default_size
            continue
        if kind != b"trak":
            continue
        tkhd, _ = _mp4_child(data, start, end, b"tkhd")
        track_id = struct.unpack_from(">I", data, tkhd + (20 if data[tkhd] == 1 else 12))[0]
        track_ids.append(track_id)
        mdia = _mp4_child(data, start, end, b"mdia")
        hdlr, _ = _mp4_child(data, *mdia, b"hdlr")
        handler = data[hdlr + 8 : hdlr + 12]
        if handler == b"vide":
            continue
        stbl = _mp4_child(data, *_mp4_child(data, *mdia, b"minf"), b"stbl")
        stsd = _mp4_child(data, *stbl, b"stsd")
        if handler != b"soun" and all(
            entry in _MP4_NON_AUDIO_ENTRIES
            for entry, _, _ in _mp4_boxes(data, stsd[0] + 8, stsd[1])
        ):
            continue  # timecode, metadata, subtitle, hint and other data tracks
        mdhd, _ = _mp4_child(data, *mdia, b"mdhd")
        timescale = struct.unpack_from(">I", data, mdhd + (20 if data[mdhd] == 1 else 12))[0]
        track = _mp4_track(data, stbl, *stsd, timescale)
        tracks.append(track)
        audio_ids[track_id] = track
    fragments = [(start, end) for kind, start, end in top if kind == b"moof"]
    if fragments and len(set(track_ids)) != len(track_ids):
        raise ValueError("duplicate MP4 track_ID")  # a traf could not be matched
    for moof in fragments:
        for kind, start, end in _mp4_boxes(data, *moof):
            if kind == b"traf":
                _mp4_add_fragment(data, start, end, audio_ids, trex_sizes)
    return tracks


def _mp4_track(
    data: bytes, stbl: tuple[int, int], stsd: int, stsd_end: int, timescale: int
) -> _Mp4Track:
    """Read one audio track's sound description and sample tables.

    ffmpeg reads a QuickTime sample of size 1 as one PCM frame (the sound
    description, not stsz, sizes it), and when stts gives every sample a
    duration of 1 it reads the frames the chunk table lists, ignoring the
    stsz count. Both are counted in frames, as the decoder plays them.
    """
    codec, frame_seconds, rate, frame_bytes, packet_frames = _mp4_sound_entry(data, stsd, stsd_end)
    track = _Mp4Track(codec, frame_seconds, rate if rate > 0 else timescale, frame_bytes)
    stsz, stsz_end = _mp4_child(data, *stbl, b"stsz")
    sample_size, count = struct.unpack_from(">II", data, stsz + 4)
    if sample_size:
        sample_bytes = sample_size * count
    else:
        if stsz + 12 + 4 * count > stsz_end:
            raise ValueError("truncated stsz")
        sample_bytes = _sum_u32_be(data, stsz + 12, count)
    chunk_frames = _mp4_chunk_frames(data, stbl)
    if sample_size == 1:
        track.frames = max(count, chunk_frames)
        track.sample_bytes = max(count, chunk_frames)  # one byte a sample, either way
    elif frame_bytes:
        track.frames = max(sample_bytes // frame_bytes, chunk_frames)
        track.count, track.sample_bytes = count, sample_bytes
    elif packet_frames > 1 and chunk_frames:
        track.frames = chunk_frames  # sized per packet from the sound description
        track.count, track.sample_bytes = count, sample_bytes
    else:
        track.count = count
        track.sample_bytes = max(sample_bytes, chunk_frames * sample_size)
    return track


def _mp4_chunk_frames(data: bytes, stbl: tuple[int, int]) -> int:
    """Samples ffmpeg reads through stsc/stco for old uncompressed audio, else 0.

    It does so only when stts holds a single entry of duration 1. With a
    malformed stsc (first chunks not 1, 2.. ascending) every chunk is taken
    at the largest per-chunk count, which only an invalid file can exceed.
    """
    try:
        stts, _ = _mp4_child(data, *stbl, b"stts")
        if struct.unpack_from(">I4xI", data, stts + 4) != (1, 1):
            return 0
        stsc, stsc_end = _mp4_child(data, *stbl, b"stsc")
        try:
            stco, _ = _mp4_child(data, *stbl, b"stco")
        except ValueError:
            stco, _ = _mp4_child(data, *stbl, b"co64")
    except (ValueError, struct.error):
        return 0  # no chunk table: ffmpeg builds no packets this way
    (chunks,) = struct.unpack_from(">I", data, stco + 4)
    (entries,) = struct.unpack_from(">I", data, stsc + 4)
    if stsc + 8 + 12 * entries > stsc_end:
        raise ValueError("truncated stsc")
    if not chunks or not entries:
        return 0
    values = array.array("I", data[stsc + 8 : stsc + 8 + 12 * entries])
    if sys.byteorder == "little":
        values.byteswap()
    firsts, counts = values[0::3], values[1::3]
    end = chunks + 1
    if firsts[0] != 1 or min(counts) < 1 or not all(map(operator.lt, firsts, firsts[1:])):
        return max(counts) * chunks
    starts = list(map(min, firsts, repeat(end)))
    spans = map(operator.sub, [*starts[1:], end], starts)
    return sum(map(operator.mul, counts, spans))


def _mp4_add_fragment(
    data: bytes, start: int, end: int, audio_ids: dict[int, _Mp4Track], trex_sizes: dict[int, int]
) -> None:
    """Add the samples of one traf to its audio track, if it belongs to one."""
    tfhd, _ = _mp4_child(data, start, end, b"tfhd")
    flags = int.from_bytes(data[tfhd + 1 : tfhd + 4], "big")
    (track_id,) = struct.unpack_from(">I", data, tfhd + 4)
    track = audio_ids.get(track_id)
    if track is None:
        return  # a video or data track, or one no trak declares (not decoded)
    pos = tfhd + 8
    pos += 8 if flags & 0x01 else 0  # base_data_offset
    pos += 4 if flags & 0x02 else 0  # sample_description_index
    pos += 4 if flags & 0x08 else 0  # default_sample_duration
    default_size = (
        struct.unpack_from(">I", data, pos)[0] if flags & 0x10 else trex_sizes.get(track_id)
    )
    for kind, payload, box_end in _mp4_boxes(data, start, end):
        if kind != b"trun":
            continue
        flags = int.from_bytes(data[payload + 1 : payload + 4], "big")
        (count,) = struct.unpack_from(">I", data, payload + 4)
        pos = payload + 8 + (4 if flags & 0x01 else 0) + (4 if flags & 0x04 else 0)
        # Per-sample duration, size, flags and composition offset, 4 bytes each.
        fields = bin(flags & 0xF00).count("1")
        if pos + 4 * fields * count > box_end:
            raise ValueError("truncated trun")
        if flags & 0x200:
            sample_bytes = _sum_u32_be(
                data, pos, count, stride=fields, index=1 if flags & 0x100 else 0
            )
        elif default_size is not None:
            sample_bytes = count * default_size
        else:
            raise ValueError("MP4 fragment sample sizes unknown")
        if track.frame_bytes:
            track.frames = (track.frames or 0) + sample_bytes // track.frame_bytes
        track.count += count
        track.sample_bytes += sample_bytes


def _sum_u32_be(data: bytes, pos: int, count: int, *, stride: int = 1, index: int = 0) -> int:
    """Sum every ``stride``-th big-endian uint32 from ``pos``.

    An array keeps a table of millions of sizes in 4 bytes per entry, where
    struct.unpack would build a Python int for each.
    """
    values = array.array("I", data[pos : pos + 4 * stride * count])
    if sys.byteorder == "little":
        values.byteswap()
    return sum(values[index::stride])


def _mp4_boxes(data: bytes, start: int, end: int):
    """Yield (type, payload start, box end) for the boxes in ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            (size,) = struct.unpack_from(">Q", data, pos + 8)
            header = 16
        elif size == 0:
            size = end - pos  # runs to the end of the parent
        if size < header or pos + size > end:
            raise ValueError("bad MP4 box size")
        yield kind, pos + header, pos + size
        pos += size


def _mp4_child(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int]:
    """Return the payload span of the first ``kind`` box, or raise ValueError."""
    for found, payload, box_end in _mp4_boxes(data, start, end):
        if found == kind:
            return payload, box_end
    raise ValueError(f"missing MP4 {kind!r} box")


def _mp4_sound_entry(
    data: bytes, start: int, end: int
) -> tuple[bytes, float | None, float, int, int]:
    """Read the track's single sound description as ffmpeg does.

    Returns the codec, the shortest AAC frame in seconds, the sample rate
    (0 when unknown), PCM bytes per frame (0 when not PCM) and PCM frames per
    coded packet (0 when unknown).
    """
    (count,) = struct.unpack_from(">I", data, start + 4)
    entries = list(_mp4_boxes(data, start + 8, end))
    if count != 1 or len(entries) != 1:
        return b"", None, 0.0, 0, 0  # several sample descriptions: not bounded
    codec, entry, entry_end = entries[0]
    version = struct.unpack_from(">H", data, entry + 8)[0]
    fields = _MP4_SOUND_FIELDS.get(version)
    if fields is None or entry + fields > entry_end:
        return codec, None, 0.0, 0, 0
    channels, bits = struct.unpack_from(">HH", data, entry + 16)
    rate: float = struct.unpack_from(">I", data, entry + 24)[0] >> 16
    packet_frames = _MP4_V0_FRAMES_PER_PACKET.get(codec, 0)
    if version == 1:
        (packet_frames,) = struct.unpack_from(">I", data, entry + 28)
    elif version == 2:
        rate, channels, bits = struct.unpack_from(">dI4xI", data, entry + 32)
        (packet_frames,) = struct.unpack_from(">I", data, entry + 60)
        if not 0 < rate < 1e9:
            rate = 0.0  # NaN, infinite or negative
    frame_seconds = None
    pcm_bits = bits
    if codec in (b"mp4a", b"ipcm", b"fpcm"):
        for kind, payload, box_end in _mp4_boxes(data, entry + fields, entry_end):
            if kind == b"esds" and codec == b"mp4a":
                frame_seconds = _aac_frame_seconds(data, payload, box_end)
            elif kind == b"pcmC" and payload + 6 <= box_end:
                pcm_bits = data[payload + 5]  # ISO PCM sample size, as ffmpeg reads it
    return codec, frame_seconds, rate, _mp4_pcm_width(codec, pcm_bits) * channels, packet_frames


def _mp4_pcm_width(codec: bytes, bits: int) -> int:
    """Bytes per PCM sample ffmpeg reads for this sample entry, 0 when not PCM."""
    if codec[:2] in (b"ms", b"TS"):  # QuickTime's wrapper for a WAVE format tag
        codec = _MP4_WAV_PCM.get(int.from_bytes(codec[2:], "big"), b"")
    if codec in (b"twos", b"sowt", b"NONE"):
        return {8: 1, 24: 3, 32: 4}.get(bits, 2)  # 16-bit unless set otherwise
    if codec == b"raw ":
        return 2 if bits == 16 else 1
    if codec in (b"lpcm", b"ipcm", b"fpcm"):
        return bits // 8 if bits in (8, 16, 24, 32, 64) else 0
    return _MP4_PCM_WIDTHS.get(codec, 0)


def _aac_frame_seconds(data: bytes, start: int, end: int) -> float | None:
    """Read the AudioSpecificConfig in an esds box into a frame duration.

    The first sampling rate in the config is the AAC core rate, also for
    HE-AAC, so 960 samples at it is the shortest frame the decoder outputs.
    """
    pos = start + 4  # version and flags
    tag, pos, _ = _mp4_descriptor(data, pos)
    if tag != 0x03:  # ES_Descriptor
        return None
    flags = data[pos + 2]
    pos += 3
    if flags & 0x80:
        pos += 2
    if flags & 0x40:
        pos += 1 + data[pos]
    if flags & 0x20:
        pos += 2
    tag, pos, _ = _mp4_descriptor(data, pos)
    if tag != 0x04 or data[pos] not in _AAC_OBJECT_TYPES:  # DecoderConfigDescriptor
        return None
    tag, pos, size = _mp4_descriptor(data, pos + 13)
    if tag != 0x05 or size < 2 or pos + size > end:  # DecoderSpecificInfo
        return None
    config = int.from_bytes(data[pos : pos + 6].ljust(6, b"\x00"), "big")
    shift = 43  # 48 bits read, 5-bit audio object type first
    aot = config >> shift
    if aot == 31:
        shift -= 6
        aot = 32 + (config >> shift & 0x3F)
    index = config >> (shift - 4) & 0xF
    if index == 15:
        rate = config >> (shift - 28) & 0xFFFFFF
    elif index < len(_AAC_RATES):
        rate = _AAC_RATES[index]
    else:
        return None
    if aot not in _AAC_LONG_FRAME_AOTS or rate <= 0:
        return None
    return 960 / rate


def _mp4_descriptor(data: bytes, pos: int) -> tuple[int, int, int]:
    """Return (tag, payload position, size) of an MPEG-4 descriptor."""
    tag = data[pos]
    pos += 1
    size = 0
    for _ in range(4):
        byte = data[pos]
        pos += 1
        size = size << 7 | byte & 0x7F
        if not byte & 0x80:
            break
    return tag, pos, size


def _ogg_pages(data: bytes, *, provider: str = ""):
    """Yield (serial, starts, lengths, spans, fast) per valid Ogg page.

    ``starts`` and ``lengths`` describe the packets completed on the page;
    a packet begun on an earlier page starts there. ``spans`` lists each
    packet's (start, end) byte ranges, or is None when every packet lies
    whole on this page. ``fast`` is False when the page took a per-segment
    walk. Pages are found as demuxers find them: by capture pattern and a
    valid CRC, so a fake page header cannot hide the pages after it. Each
    candidate's CRC covers up to 64 KB, so junk dense with capture patterns
    is rejected once the CRC work passes twice the file.
    """
    reversed_data = memoryview(data.translate(_OGG_BIT_REVERSE))
    crc_budget = 2 * len(data) + (1 << 20)
    pending: dict[int, list] = {}  # serial -> [start, length, spans] of an open packet
    pos = data.find(b"OggS")
    while pos >= 0 and pos + 27 <= len(data):
        segments = data[pos + 26]
        body = pos + 27 + segments
        lacing = data[pos + 27 : body]
        end = body + sum(lacing)
        valid = len(lacing) == segments and end <= len(data)
        if valid:
            crc_budget -= end - pos
            if crc_budget < 0:
                raise ProviderError(
                    provider=provider,
                    detail="Cannot parse Ogg audio: too many invalid pages",
                    status_code=400,
                )
            # Ogg's CRC (MSB-first, poly 0x04C11DB7, no inversion) is the
            # reflection of zlib's CRC-32 over bit-reversed bytes once zlib's
            # inversions are undone, so the stored CRC is reflected instead.
            # The CRC field itself counts as zero.
            crc = zlib.crc32(reversed_data[pos : pos + 22], 0xFFFFFFFF)
            crc = zlib.crc32(reversed_data[pos + 26 : end], zlib.crc32(_ZERO_CRC, crc))
            valid = crc ^ 0xFFFFFFFF == int.from_bytes(reversed_data[pos + 22 : pos + 26], "big")
        if not valid:
            pos = data.find(b"OggS", pos + 1)
            continue
        (serial,) = struct.unpack_from("<I", data, pos + 14)
        if serial not in pending and b"\xff" not in lacing:
            if segments == 1:
                yield serial, [body], lacing, None, True
            else:
                starts = list(accumulate(lacing[:-1], initial=body)) if lacing else []
                yield serial, starts, lacing, None, True
        else:
            starts = []
            lengths: list[int] = []
            spans: list[list[tuple[int, int]]] = []
            packet = pending.pop(serial, None)
            for value in lacing:
                if packet is None:
                    packet = [body, 0, []]
                packet[1] += value
                ranges = packet[2]
                if ranges and ranges[-1][1] == body:
                    ranges[-1] = (ranges[-1][0], body + value)
                else:
                    ranges.append((body, body + value))
                body += value
                if value < 255:
                    starts.append(packet[0])
                    lengths.append(packet[1])
                    spans.append(ranges)
                    packet = None
            if packet is not None:
                pending[serial] = packet
            yield serial, starts, lengths, spans, False
        pos = data.find(b"OggS", end)


def _ogg_packets_seconds(data: bytes, *, provider: str = "") -> float:
    """Lower bound on the audio the Opus, Vorbis and Speex packets encode.

    The decoder outputs audio for every packet whatever the granule
    positions say, so a forged last granule hides nothing. Opus packets are
    timed exactly from their TOC byte, Vorbis packets from their block sizes
    (a quarter of the previous and of the current block each), Speex packets
    as frames_per_packet frames of their mode. For Vorbis and Speex the
    packets on a stream's first and last audio pages are left out, covering
    a start or end trim the granules may ask for. A stream's codec comes from
    its first packet, as the demuxer decides it; other codecs get no bound.
    """
    streams: dict[int, _OpusStream | _VorbisStream | _SpeexStream | None] = {}
    budget = len(data) // 2 + 100_000
    for serial, starts, lengths, spans, fast in _ogg_pages(data, provider=provider):
        budget -= _OGG_PAGE_COST if starts else _OGG_EMPTY_PAGE_COST
        budget -= len(lengths) if fast else _OGG_SLOW_PACKET_COST // 2 * len(lengths)
        if serial in streams:
            stream = streams[serial]
            if stream is not None:
                budget -= stream.add_page(data, starts, lengths, spans, provider)
        elif starts:
            head = data[starts[0] : starts[0] + min(lengths[0], _OGG_HEAD_BYTES)]
            stream = streams[serial] = _ogg_stream(head)
            if stream is not None:
                budget -= stream.add_page(
                    data, starts[1:], lengths[1:], spans and spans[1:], provider
                )
        if budget < 0:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse Ogg audio: too many pages or packets",
                status_code=400,
            )
    return sum(stream.floor() for stream in streams.values() if stream is not None)


def _ogg_stream(head: bytes) -> _OpusStream | _VorbisStream | _SpeexStream | None:
    """Start the stream whose first packet is ``head``, or None for other codecs.

    Each stream's add_page takes the packets completed on one page and
    returns the extra work units they took; floor gives its lower bound.
    """
    if head.startswith(b"OpusHead"):
        return _OpusStream(struct.unpack_from("<H", head, 10)[0] if len(head) >= 12 else 0)
    if head.startswith(b"\x01vorbis") and len(head) >= 29:
        version, channels, rate = struct.unpack_from("<IBI", head, 7)
        short, long = head[28] & 0x0F, head[28] >> 4
        if not version and channels and rate and 6 <= short <= long <= 13:
            return _VorbisStream(channels, rate, 1 << short, 1 << long)
        return None  # decoders reject the stream
    if head.startswith(b"Speex   ") and (speex := _speex_packet_seconds(head)):
        return _SpeexStream(*speex)
    return None


class _OpusStream:
    """Opus packet durations, with the comment header (second packet) skipped."""

    def __init__(self, pre_skip: int) -> None:
        self.pre_skip = pre_skip
        self.comment = True
        self.seconds = 0.0
        self.last: bytes = b""  # the last packet's first bytes

    def add_page(self, data: bytes, starts, lengths, spans, provider: str) -> int:
        if self.comment and starts:
            self.comment = False
            starts, lengths = starts[1:], lengths[1:]
        if len(starts) == 1:  # one packet a page is common: keep it cheap
            if lengths[0]:
                self.last = data[starts[0] : starts[0] + min(lengths[0], 2)]
                seconds = _OPUS_TOC_SECONDS[self.last[0]]
                self.seconds += seconds or _opus_packet_seconds(self.last)  # 0: code 3
            return 0
        # Zero-length packets carry no TOC byte and are skipped.
        tocs = bytes(map(data.__getitem__, compress(starts, lengths)))
        if not tocs:
            return 0
        self.seconds += sum(map(_OPUS_TOC_SECONDS.__getitem__, tocs))
        work = 0
        if tocs.translate(_OPUS_CODE3).count(1):
            # Code 3 gives its frame count in the second byte.
            work = _OGG_SLOW_PACKET_COST * len(tocs)
            for start, length in zip(starts, lengths, strict=True):
                if length and data[start] & 3 == 3:
                    self.seconds += _opus_packet_seconds(data[start : start + min(length, 2)])
        last = next(i for i in range(len(lengths) - 1, -1, -1) if lengths[i])
        self.last = data[starts[last] : starts[last] + min(lengths[last], 2)]
        return work

    def floor(self) -> float:
        # Allow for the pre-skip and an end trim of up to the last packet; a
        # pre-skip only ever trims its own stream.
        trim = self.pre_skip / 48000 + _opus_packet_seconds(self.last)
        return max(0.0, self.seconds - trim)


class _SpeexStream:
    """Speex audio packets per page, after the comment and extra headers."""

    def __init__(self, seconds: float, headers: int) -> None:
        self.seconds = seconds
        self.headers = headers
        self.pages: list[int] = []

    def add_page(self, data: bytes, starts, lengths, spans, provider: str) -> int:
        packets = len(lengths) - lengths.count(0)  # empty packets carry nothing
        skipped = min(self.headers, packets)
        self.headers -= skipped
        if packets > skipped:
            self.pages.append(packets - skipped)
        return 0

    def floor(self) -> float:
        return sum(self.pages[1:-1]) * self.seconds


class _VorbisStream:
    """Vorbis samples per page, from each audio packet's mode and block size.

    The decoder outputs (previous block + this block) / 4 samples for every
    packet after the first, the previous block being the one it last decoded,
    so the sum is exact once the setup header's mode table is known. Without
    one (a setup the parse rejects, which decoders reject too) each packet is
    counted at half the short block, the least it can yield.
    """

    def __init__(self, channels: int, rate: int, short: int, long: int) -> None:
        self.channels = channels
        self.rate = rate
        self.blocks = (short, long)
        self.modes: list[int] | None = None
        self.setup_seen = False
        self.previous: int | None = None
        self.pages: list[int] = []

    def add_page(self, data: bytes, starts, lengths, spans, provider: str) -> int:
        samples = 0
        audio = 0
        modes = self.modes
        for index, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            if not length:
                continue
            first = data[start]
            if first & 1:  # a header packet, never decoded as audio
                if not self.setup_seen and data[start : start + 7] == b"\x05vorbis":
                    self.setup_seen = True
                    if length > _VORBIS_MAX_SETUP:
                        raise ProviderError(
                            provider=provider,
                            detail="Cannot parse Vorbis audio: setup header too large",
                            status_code=400,
                        )
                    ranges = spans[index] if spans else [(start, start + length)]
                    packet = b"".join(data[s:e] for s, e in ranges)
                    modes = self.modes = _vorbis_modes(packet, self.channels)
                continue
            audio += 1
            if modes is None:
                samples += self.blocks[0] // 2
                continue
            mode = (first >> 1) & ((1 << (len(modes) - 1).bit_length()) - 1)
            if mode >= len(modes):
                continue  # the decoder drops a packet with an invalid mode
            block = modes[mode]
            if self.previous is not None:
                samples += (self.blocks[self.previous] + self.blocks[block]) // 4
            self.previous = block
        if audio:
            self.pages.append(samples)
        return _OGG_SLOW_PACKET_COST * audio

    def floor(self) -> float:
        return sum(self.pages[1:-1]) / self.rate


class _Bits:
    """LSB-first bit reader over a packet, as Vorbis packs its headers."""

    def __init__(self, data: bytes) -> None:
        self.bits = b"".join(map(_BITS_LSB_FIRST.__getitem__, data))
        self.pos = 0

    def read(self, count: int) -> int:
        end = self.pos + count
        if end > len(self.bits):
            raise ValueError("Vorbis header truncated")
        value = int(self.bits[self.pos : end][::-1], 2) if count else 0
        self.pos = end
        return value

    def skip(self, count: int) -> None:
        self.pos += count
        if self.pos > len(self.bits):
            raise ValueError("Vorbis header truncated")


def _vorbis_modes(packet: bytes, channels: int) -> list[int] | None:
    """Return each mode's block flag from a Vorbis setup header, or None.

    Walks the whole header forward as the decoder does (Vorbis I spec,
    4.2.4), since the mode table at its end can only be found reliably that
    way: trailing bytes, which decoders ignore, would fool a backward scan.
    Returns None only where ffmpeg's decoder also rejects the header (it
    checks more than this walk, never less), so a crafted setup cannot force
    the weaker short-block bound on a stream that still decodes.
    """
    try:
        bits = _Bits(packet)
        bits.skip(56)  # packet type and "vorbis"
        for _ in range(bits.read(8) + 1):
            _vorbis_skip_codebook(bits)
        for _ in range(bits.read(6) + 1):
            if bits.read(16):
                return None  # time domain transforms are reserved
        for _ in range(bits.read(6) + 1):
            if not _vorbis_skip_floor(bits):
                return None
        for _ in range(bits.read(6) + 1):
            if bits.read(16) > 2:
                return None
            bits.skip(72)  # begin, end, partition size
            classifications = bits.read(6) + 1
            bits.skip(8)  # classbook
            cascades = []
            for _ in range(classifications):
                low = bits.read(3)
                cascades.append((bits.read(5) if bits.read(1) else 0) << 3 | low)
            bits.skip(8 * sum(bin(cascade).count("1") for cascade in cascades))
        mappings = bits.read(6) + 1
        for _ in range(mappings):
            if bits.read(16):
                return None
            submaps = bits.read(4) + 1 if bits.read(1) else 1
            if bits.read(1):
                bits.skip(2 * (bits.read(8) + 1) * (channels - 1).bit_length())
            if bits.read(2):
                return None
            if submaps > 1:
                bits.skip(4 * channels)
            bits.skip(24 * submaps)
        modes = []
        for _ in range(bits.read(6) + 1):
            modes.append(bits.read(1))
            bits.skip(32)  # window and transform types: ffmpeg does not check them
            if bits.read(8) >= mappings:
                return None
        return modes if bits.read(1) else None  # framing bit
    except (ValueError, OverflowError):
        return None


def _vorbis_skip_codebook(bits: _Bits) -> None:
    if bits.read(24) != 0x564342:
        raise ValueError("bad Vorbis codebook sync")
    dimensions = bits.read(16)
    entries = bits.read(24)
    if bits.read(1):  # ordered: runs of equal lengths
        length = bits.read(5) + 1
        current = 0
        while current < entries:
            if length > 32:
                raise ValueError("Vorbis codeword longer than 32 bits")
            current += bits.read((entries - current).bit_length())
            length += 1
        if current > entries:
            raise ValueError("Vorbis codebook overrun")
    elif bits.read(1):  # sparse: a flag per entry, then a 5-bit length if used
        match = re.compile(rb"(?:0|1[01]{5}){%d}" % entries).match(bits.bits, bits.pos)
        if match is None:
            raise ValueError("Vorbis header truncated")
        bits.pos = match.end()
    else:
        bits.skip(5 * entries)
    lookup = bits.read(4)
    if lookup > 2:
        raise ValueError("reserved Vorbis lookup type")
    if lookup:
        bits.skip(64)  # minimum and delta values
        value_bits = bits.read(4) + 1
        bits.read(1)  # sequence flag
        values = _lookup1_values(entries, dimensions) if lookup == 1 else entries * dimensions
        bits.skip(values * value_bits)


def _lookup1_values(entries: int, dimensions: int) -> int:
    """Largest r with r ** dimensions <= entries (Vorbis I spec, 9.2.3)."""
    if not dimensions:
        raise ValueError("Vorbis codebook with no dimensions")
    root = int(entries ** (1 / dimensions))
    while (root + 1) ** dimensions <= entries:
        root += 1
    while root and root**dimensions > entries:
        root -= 1
    return root


def _vorbis_skip_floor(bits: _Bits) -> bool:
    kind = bits.read(16)
    if kind == 0:
        bits.skip(54)  # order, rate, bark map size, amplitude bits and offset
        bits.skip(8 * (bits.read(4) + 1))
        return True
    if kind != 1:
        return False
    classes = [bits.read(4) for _ in range(bits.read(5))]
    dimensions = []
    for _ in range(max(classes) + 1 if classes else 0):
        dimensions.append(bits.read(3) + 1)
        subclasses = bits.read(2)
        if subclasses:
            bits.skip(8)  # master book
        bits.skip(8 << subclasses)
    bits.skip(2)  # multiplier
    range_bits = bits.read(4)
    bits.skip(range_bits * sum(dimensions[c] for c in classes))
    return True


def _speex_packet_seconds(head: bytes) -> tuple[float, int] | None:
    """Audio per Speex packet and the header packets after this one.

    The frame size comes from the mode (160 samples narrowband, doubled for
    each wider band), as the decoder takes it, not from the header field.
    """
    if len(head) < 72:
        return None
    rate, mode = struct.unpack_from("<Ii", head, 36)
    frames, extra = struct.unpack_from("<iI", head, 64)
    if not rate or mode not in (0, 1, 2):
        return None
    # extra_headers is client-set and ffmpeg ignores it; capping it keeps a
    # huge value from passing every audio packet off as a header.
    return max(frames, 1) * (160 << mode) / rate, 1 + min(extra, 4)


def _opus_packet_seconds(head: bytes) -> float:
    """Duration of one Opus packet from its TOC byte (RFC 6716, 3.1)."""
    if not head:
        return 0.0
    config = head[0] >> 3
    if config < 12:  # SILK
        frame = (0.01, 0.02, 0.04, 0.06)[config & 3]
    elif config < 16:  # hybrid
        frame = (0.01, 0.02)[config & 1]
    else:  # CELT
        frame = (0.0025, 0.005, 0.01, 0.02)[config & 3]
    code = head[0] & 3
    if code == 0:
        frames = 1
    elif code < 3:
        frames = 2
    else:
        frames = head[1] & 0x3F if len(head) > 1 else 0
    # Packets over 120 ms are invalid; capping keeps the bound conservative.
    return min(frames * frame, 0.12)
