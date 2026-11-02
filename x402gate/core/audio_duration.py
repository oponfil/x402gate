"""Parse audio file duration for STT pricing.

Container headers (an MP3 Xing/VBRI frame count, a WAV data chunk size, an
MP4 or Ogg duration field) are client-controlled, while the provider bills
the audio it actually decodes. Every duration is therefore raised to an
independent lower bound taken from the bytes themselves, so a forged header
cannot price a long file as a short one.

The bound is tight where the format allows it: MPEG and FLAC frames are
counted, PCM is sized from its sample chunk at the width the decoder reads
(in MP4, from the frames its sample and chunk tables give), AAC in MP4
from the sample table (or fragment runs), Opus from each packet's TOC byte
(in MP4 too) and Vorbis from each packet's block size. MPEG audio, AMR and FLAC in MP4 have their
frames counted, ALAC is bounded by its packet count, Speex in Ogg by its
packet count. MP4 codecs ffmpeg decodes that none of this times are
refused; entries it cannot decode are bounded by their sample bytes at the
lossy family's top bitrate.
ADTS AAC is timed from its frame headers, which is all it has.

The container is read from the bytes, never from the format name or from
mutagen's guess: ffmpeg skips leading ID3v2 tags and probes what follows,
so the gateway does too, and prices only the containers listed above.
mutagen supplies the header duration through its stream-info readers, never
its tag parsers, whose cost a crafted tag can blow up.

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
import threading
import wave
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate, compress, islice, repeat

from mutagen.aiff import AIFFInfo
from mutagen.flac import StreamInfo as FlacStreamInfo
from mutagen.mp3 import MPEGInfo
from mutagen.mp4 import Atoms, MP4Info
from mutagen.oggopus import OggOpusInfo
from mutagen.oggspeex import OggSpeexInfo
from mutagen.oggvorbis import OggVorbisInfo

from x402gate.providers.base import ProviderError

# Highest average bitrate a genuine file of each family can have. Bytes / cap
# is then never above the real duration, so honest files are not overcharged.
_MPEG_MAX_BPS = 448_000  # top MPEG frame bitrate (Layer I; Layer III is 320k)
_LOSSY_MAX_BPS = 640_000  # above Opus (510k), stereo AAC (576k), Vorbis q10
_LOSSLESS_MAX_BPS = 4_608_000  # 96 kHz / 24-bit stereo PCM
# A frame count past this stops the scan: 25 MB cannot hold a day of real
# (not digitally silent) lossless audio, and counting on only costs CPU.
_MAX_COUNTED_SECONDS = 24 * 3600

# Structures a genuine file has few of. Parsers build an object for each, so
# millions of empty ones (RIFF/AIFF chunks, FLAC metadata blocks, MP4 boxes)
# cost a gigabyte before any audio is read; such files are rejected. An hour
# of fragmented MP4 cut every second has about 7,200 top-level boxes.
_MAX_CHUNKS = 10_000  # RIFF/AIFF chunks, FLAC metadata blocks
_MAX_MP4_BOXES = 100_000  # in mutagen's container boxes, which it parses whole
_MAX_MP4_SAMPLE_ENTRIES = 16
_MAX_ID3_TAGS = 16  # stacked ID3v2 tags; writers stack two at most
# ffmpeg's format probe reads at most this much, and skips a leading ID3v2
# tag only when the tag and 16 bytes more fit in what it read.
_PROBE_BUF_MAX = 1 << 20
# Unpaid parses run in worker threads; at most this many at once, so
# concurrent price requests cannot run the process out of memory. Callers
# queue for a slot on the event loop (openrouter's STT pricing slots); a parse
# that still finds them all busy is refused at once, so no thread waits.
_PARSE_SLOTS = threading.BoundedSemaphore(2)
# Chained frames from the first byte that make a decoder's probe pick MPEG
# audio (ffmpeg: 7) or ADTS (3) over any format it finds further in.
_MPEG_PROBE_FRAMES = 7
_ADTS_PROBE_FRAMES = 3
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"  # Matroska / WebM
_NUL_RUN = re.compile(rb"\x00*")
_SUPPORTED_FORMATS = "wav, mp3, flac, ogg (Opus, Vorbis, Speex), m4a/mp4/mov, aiff or aac (ADTS)"

# Bytes per sample ffmpeg decodes for each AIFF-C compression type it reads
# as PCM, whatever COMM's sample size claims. Keyed upper-case: ffmpeg falls
# back to a case-blind match. 'NONE' and 'twos' (None) take their width from
# the sample size, as plain AIFF does.
_AIFC_PCM_WIDTHS = {
    b"NONE": None, b"TWOS": None, b"SOWT": 2, b"RAW ": 1, b"IN24": 3, b"IN32": 4,
    b"FL32": 4, b"FL64": 8, b"ALAW": 1, b"ULAW": 1,
}  # fmt: skip
# Fixed-ratio AIFF-C codecs, by the MP4 entry ffmpeg decodes the same way
# (block bytes in _MP4_BLOCK_BYTES, frames in _MP4_V0_FRAMES_PER_PACKET).
# Every other type (G.722, G.726, QDesign, QCELP, SDX2, ...) is refused.
_AIFC_BLOCK_CODECS = {b"IMA4": b"ima4", b"MAC3": b"MAC3", b"MAC6": b"MAC6", b"GSM ": b"agsm"}
_AIFC_VERSION1 = 0xA2805140  # FVER timestamp of AIFF-C; any other reads as plain AIFF

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
_MAX_PCM_RATE = 768_000  # highest real PCM rate; a forged higher one is clamped
# Largest AMR frame (mode byte included) per 20 ms: 32 bytes for AMR-NB
# 12.2k, 62 for AMR-WB 23.85k. Every sample byte is frame data.
_AMR_MAX_BPS = {b"samr": 32 * 8 * 50, b"sawb": 62 * 8 * 50}
# Bytes of an AMR frame, its mode byte included, by the frame type in bits
# 3-6 of that byte; every frame decodes to 20 ms. A NO_DATA frame is the mode
# byte alone, so a sample of them holds 32x (AMR-WB 61x) the audio the byte
# cap allows. Reserved types are taken at 1 byte too, the least any decoder
# reads, so the walk never steps over a frame a decoder would play.
_AMR_FRAME_BYTES = {
    b"samr": (13, 14, 16, 18, 20, 21, 27, 32, 6, 1, 1, 1, 1, 1, 1, 1),
    b"sawb": (18, 24, 33, 37, 41, 47, 51, 59, 61, 6, 1, 1, 1, 1, 1, 1),
}
_AMR_FRAME_LENGTHS = {
    codec: bytes(sizes[byte >> 3 & 0x0F] for byte in range(256))
    for codec, sizes in _AMR_FRAME_BYTES.items()
}
_AMR_FRAMES_PER_SECOND = 50
# Bytes of AudioSampleEntry fields before its child boxes, by QuickTime
# sound description version (0 is also the ISO layout).
_MP4_SOUND_FIELDS = {0: 28, 1: 44, 2: 64}
# Box types a movie may open with, and the boxes mutagen parses every child
# of (this module walks the same ones, and mvex); 'meta' has 4 bytes first.
_MP4_TOP_BOXES = frozenset(
    (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"uuid", b"styp", b"sidx")
)
_MP4_CONTAINERS = frozenset(
    (b"moov", b"udta", b"trak", b"mdia", b"meta", b"ilst", b"stbl", b"minf", b"moof", b"traf",
     b"mvex")
)  # fmt: skip
# MPEG-4 object types for AAC and for MPEG-1/2 audio in esds, and the audio
# object types that code 1024 (or 960) samples per frame and 512 (or 480).
_AAC_OBJECT_TYPES = frozenset((0x40, 0x66, 0x67, 0x68))
_MPEG_AUDIO_OBJECT_TYPES = frozenset((0x69, 0x6B))
_AAC_LONG_FRAME_AOTS = frozenset((1, 2, 3, 4, 5, 29))
_AAC_LOW_DELAY_AOTS = frozenset((23, 39))
# Shortest packet other fixed-frame codecs decode to: a Layer III frame
# holds at least 576 samples, Layer II 1152 and Layer I 384, at up to 48 kHz;
# an Opus packet 2.5 ms. The packets' own headers give their real length (an
# Opus packet up to 120 ms, an MPEG-2.5 frame 72 ms), and ffmpeg's MPEG
# decoders play frames of any layer under any of these entries, so these
# tracks are also timed from their sample bytes; see _mp4_track.
_MPEG_MIN_FRAME_SECONDS = 576 / 48000
_MP4_MIN_FRAME_SECONDS = {
    b".mp3": _MPEG_MIN_FRAME_SECONDS,
    b"mp3 ": _MPEG_MIN_FRAME_SECONDS,
    b"ms\x00\x55": _MPEG_MIN_FRAME_SECONDS,
    b".mp2": 1152 / 48000,
    b".mp1": 384 / 48000,
    b"Opus": 120 / 48000,
}
_MP4_MPEG_CODECS = frozenset((b".mp3", b"mp3 ", b"ms\x00\x55", b".mp2", b".mp1"))
_MP4_SCANNED_CODECS = _MP4_MPEG_CODECS | {b"Opus"} | _AMR_FRAME_BYTES.keys()
# ffmpeg parses AC-3 and E-AC-3 in MP4 into every frame a sample holds, of up
# to 6 blocks at 16-48 kHz each; no sample-table bound holds, so they are refused.
# So are the other codecs ffmpeg decodes from MP4 that no table here times:
# speech codecs whose silence frames are a byte or a few (QCELP, EVRC, SMV,
# Speex, iLBC, Nellymoser), DTS (down to 12 kbps), lossless TrueHD and the
# rest, which the 640 kbps byte cap would price up to 100x short. Sample
# entries ffmpeg cannot decode keep that cap: no decoder plays them.
_MP4_REFUSED_CODECS = {
    b"ac-3": "AC-3", b"sac3": "AC-3", b"ec-3": "E-AC-3", b"ac-4": "AC-4",
    b"dtsc": "DTS", b"dtsh": "DTS", b"dtsl": "DTS", b"dtse": "DTS", b"dtsx": "DTS",
    b"dtsy": "DTS", b"DTS ": "DTS", b"mlpa": "TrueHD", b"mha1": "MPEG-H", b"mha2": "MPEG-H",
    b"mhm1": "MPEG-H", b"mhm2": "MPEG-H", b"iamf": "IAMF", b"Qclp": "QCELP", b"Qclq": "QCELP",
    b"sqcp": "QCELP", b"sevc": "EVRC", b"ssmv": "SMV", b"spex": "Speex", b"SPXN": "Speex",
    b"ilbc": "iLBC", b"nmos": "Nellymoser", b"NELL": "Nellymoser", b"QDM2": "QDesign",
    b"QDMC": "QDesign",
}  # fmt: skip
# Fixed-ratio codecs (MACE, IMA ADPCM, GSM): bytes of one coded block, and
# whether the entry's channel count multiplies it (ffmpeg decodes GSM as mono
# whatever the entry says). Each block decodes to the frames listed above,
# and the decoder outputs only whole blocks however the packets split the
# bytes, so packets from a sample table or a fragment are timed from their
# bytes; old QuickTime audio read by chunk is timed from its chunk table.
# The 640 kbps byte cap would price them 20-50x short.
_MP4_BLOCK_BYTES = {
    b"ima4": (34, True),
    b"MAC3": (2, True),
    b"MAC6": (1, True),
    b"agsm": (33, False),
}
# Samples (and chunks) of a track whose bytes are read one by one: 7 hours
# of 26 ms MPEG frames. Real files have far fewer.
_MAX_MP4_SCANNED_SAMPLES = 1_000_000
_AAC_RATES = (
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350
)  # fmt: skip

_HIGH_NIBBLE = bytes(byte >> 4 for byte in range(256))
_LOW_NIBBLE = bytes(byte & 0x0F for byte in range(256))
_OGG_BIT_REVERSE = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))
_ZERO_CRC = bytes(4)
_OGG_HEAD_BYTES = 72  # packet bytes kept: up to the Speex header's extra_headers
_VORBIS_MAX_SETUP = 1 << 20  # setup headers are a few KB; parsing is linear in size
# Work units of the Ogg walk: a page costs about as much as 16 packets, one
# that completes no packet (an empty page) more. Real streams stay under half
# a unit per byte, even 6 kbps Opus with a packet on every page.
_OGG_PAGE_COST, _OGG_EMPTY_PAGE_COST, _OGG_SLOW_PACKET_COST = 16, 64, 8
# The Ogg codecs priced here, by the first packet's magic, with mutagen's
# reader for their header duration; and the heads of the video, subtitle and
# index streams ffmpeg knows. Any other stream is refused, even beside a
# priced one: FLAC, PCM, CELT and OGM audio (the DirectShow mapping carries
# any WAVE codec) are decoded with no packet bound here.
_OGG_INFOS = (
    (b"OpusHead", OggOpusInfo),
    (b"\x01vorbis", OggVorbisInfo),
    (b"Speex   ", OggSpeexInfo),
)
_OGG_NON_AUDIO_HEADS = (
    b"\x80theora", b"\x80kate", b"fishead\x00", b"fisbone\x00", b"BBCD\x00", b"KW-DIRAC",
    b"OVP80", b"\x01video", b"\x01text",
)  # fmt: skip
_OGG_ALLOWED_HEADS = tuple(magic for magic, _ in _OGG_INFOS) + _OGG_NON_AUDIO_HEADS

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
# ADTS: a 12-bit sync. A stream must open with a standard header (layer 0),
# but ffmpeg's parser ignores the layer bits after that, and so does the walk.
_ADTS_START = re.compile(b"\xff[\xf0\xf1\xf8\xf9]")
_ADTS_SYNC = re.compile(b"\xff[\xf0-\xff]")

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
        ProviderError: If format is unsupported or duration cannot be determined
            (400), or if too many files are being priced at once (429).
    """
    normalized = fmt.strip().lower().lstrip(".")
    if not normalized:
        raise ProviderError(
            provider=provider,
            detail="Missing input_audio.format",
            status_code=400,
        )

    if not _PARSE_SLOTS.acquire(False):  # non-blocking
        # Unpaid clients can fill the slots, so this is a 4xx: a 5xx would
        # count against the provider's health on the public dashboard.
        raise ProviderError(
            provider=provider,
            detail="Too many audio files are being priced; retry shortly",
            status_code=429,
        )
    try:
        duration = _container_duration(data, normalized, provider)
    finally:
        _PARSE_SLOTS.release()
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


def _container_duration(data: bytes, fmt: str, provider: str) -> float:
    """Price the container the bytes hold, whatever the format name says.

    A leading ID3v2 tag makes mutagen call any file MP3, while ffmpeg skips
    the tag and demuxes what follows (an Ogg, FLAC or WAV stream), so the
    container is read from the bytes after the tags. Only containers with a
    content floor are priced; anything else is refused.

    Behind a tag ffmpeg's MP3 probe, which scores at most 51, can tie or
    beat FLAC's 50, ADTS's or that of an MP4 opening with a low-scoring box
    when the bytes hold long runs of chained MPEG frames, so those runs are
    priced too. WAV (99), AIFF and Ogg (100) always win, and their samples
    are not read as MPEG frames: chance sync bytes in dense PCM overcharge.
    """
    start, probed = _audio_start(data, provider)
    body = data[start:] if start else data
    if probed and (duration := _probed_container_duration(body, fmt, provider)) is not None:
        if not start or _outscores_mp3_probe(body):
            return duration
        return max(duration, _mpeg_chained_seconds(body))
    if _MPEG_SYNC.match(body):
        return _mp3_duration(body, fmt, provider)
    if probed and _ADTS_START.match(body):
        duration = _adts_duration(body, fmt, provider)
        return max(duration, _mpeg_chained_seconds(body)) if start else duration
    if body[:4] == _EBML_MAGIC:
        raise ProviderError(
            provider=provider,
            detail=f"WebM/Matroska audio is not supported; send {_SUPPORTED_FORMATS}",
            status_code=400,
        )
    if fmt == "wav" and not start:
        return _wav_duration(body, provider=provider)  # says what is wrong with it
    raise _unparsable(fmt, provider)


def _probed_container_duration(body: bytes, fmt: str, provider: str) -> float | None:
    """Price the WAV, AIFF, FLAC, Ogg or MP4 file whose magic opens ``body``, else None."""
    magic = body[:4]
    if magic == b"RIFF":
        # Any WAV gets the PCM-only parse: a compressed codec (MP3 or GSM in
        # WAV) has no fixed byte rate, and its header length is client-set.
        _check_chunk_count(body, little_endian=True, provider=provider)
        return _wav_duration(body, provider=provider)
    if magic == b"FORM":
        return _aiff_duration(body, fmt, provider)
    if magic == b"fLaC":
        return _flac_duration(body, fmt, provider)
    if magic == b"OggS":
        return _ogg_duration(body, fmt, provider)
    if body[4:8] in _MP4_TOP_BOXES:
        return _mp4_duration(body, fmt, provider)
    return None


def _outscores_mp3_probe(body: bytes) -> bool:
    """True for a WAV, AIFF or Ogg file whose header ffmpeg's probe takes over MP3."""
    form = body[8:12]
    return (
        (body[:4] == b"RIFF" and form == b"WAVE")
        or (body[:4] == b"FORM" and form in (b"AIFF", b"AIFC"))
        or (body[:4] == b"OggS" and body[4:5] == b"\x00" and body[5:6] <= b"\x07")
    )


def _unparsable(fmt: str, provider: str) -> ProviderError:
    return ProviderError(
        provider=provider,
        detail=f"Cannot parse audio format '{fmt}'; send {_SUPPORTED_FORMATS}",
        status_code=400,
    )


def _audio_start(data: bytes, provider: str) -> tuple[int, bool]:
    """Offset past the leading ID3v2 tags and any zero padding after them,
    and whether a container other than MPEG audio may be priced there.

    Decoders skip stacked tags (Windows Media Player writes two) and zero
    padding before, between and after them. ffmpeg's format probe, though,
    skips one tag, and only one that fits its probe buffer; every container
    probe but MP3's wants its magic right after it. Behind more tags, any
    padding or a larger tag the MP3 probe wins whenever the bytes hold
    enough MPEG frames, so only MPEG audio is priced there.
    """
    pos = tags = 0
    padded = False
    while size := _id3v2_size(data, pos):
        tags += 1
        if tags > _MAX_ID3_TAGS:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse audio: too many ID3v2 tags",
                status_code=400,
            )
        end = _NUL_RUN.match(data, pos + size).end()
        padded |= end != pos + size
        pos = end
    if not tags:
        return 0, True
    return pos, tags == 1 and not padded and pos + 16 < min(len(data), _PROBE_BUF_MAX)


def _stream_info(parse: Callable[[], object], fmt: str, provider: str) -> tuple[object, float]:
    """Run a mutagen stream-info reader; return it and the duration it reads.

    mutagen raises IndexError, struct.error and more besides its own errors
    on crafted bytes; all of them are the client's malformed file, a 400.
    """
    try:
        info = parse()
    except Exception as exc:
        raise _unparsable(fmt, provider) from exc
    length = getattr(info, "length", None)
    if length is None:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot determine duration for format '{fmt}'",
            status_code=400,
        )
    return info, float(length)


def _mp3_duration(body: bytes, fmt: str, provider: str) -> float:
    if not _mpeg_stream_starts(body):
        raise _unparsable(fmt, provider)
    _, length = _stream_info(lambda: MPEGInfo(io.BytesIO(body)), fmt, provider)
    # Count the real frames; the byte floor covers free-format frames that
    # the frame scan cannot size.
    return max(length, _mpeg_frames_seconds(body), len(body) * 8 / _MPEG_MAX_BPS)


def _aiff_duration(body: bytes, fmt: str, provider: str) -> float:
    _check_chunk_count(body, little_endian=False, provider=provider)
    _stream_info(lambda: AIFFInfo(io.BytesIO(body)), fmt, provider)
    # The COMM frame count is client-set and may run past EOF, as in a
    # stream; the decoder plays the SSND bytes sent, so they set the price,
    # at the byte rate the decoder reads them at.
    byte_rate = _aiff_byte_rate(body, provider=provider)
    return _pcm_floor_seconds(body, b"SSND", byte_rate, little_endian=False)


def _flac_duration(body: bytes, fmt: str, provider: str) -> float:
    # STREAMINFO must be the first metadata block, as decoders require.
    if len(body) < 42 or body[4] & 0x7F or int.from_bytes(body[5:8], "big") < 34:
        raise _unparsable(fmt, provider)
    info, length = _stream_info(lambda: FlacStreamInfo(body[8:42]), fmt, provider)
    rate = int(getattr(info, "sample_rate", 0) or 0)
    floor = _flac_frames_seconds(body, rate, provider=provider)
    return max(length, floor, len(body) * 8 / _LOSSLESS_MAX_BPS)


def _ogg_duration(body: bytes, fmt: str, provider: str) -> float:
    if body[4:5] != b"\x00" or body[5:6] > b"\x07":
        # ffmpeg's Ogg probe wants version 0 and known flags on the first
        # page; without them it may take the file for MP3 instead.
        raise _unparsable(fmt, provider)
    first = 27 + body[26] if len(body) > 26 else len(body)
    head = body[first : first + 8]
    info_type = next((info for magic, info in _OGG_INFOS if head.startswith(magic)), None)
    if info_type is None:
        raise ProviderError(
            provider=provider,
            detail="Only Opus, Vorbis and Speex audio in Ogg is supported",
            status_code=400,
        )
    # The budgeted packet walk runs first, so a crafted stream is refused
    # before mutagen's search for the last page walks it.
    floor = max(_ogg_packets_seconds(body, provider=provider), len(body) * 8 / _LOSSY_MAX_BPS)

    def parse() -> object:
        fileobj = io.BytesIO(body)
        info = info_type(fileobj)
        info._post_tags(fileobj)  # the last granule, as mutagen's Ogg types read it
        return info

    return max(_stream_info(parse, fmt, provider)[1], floor)


def _mp4_duration(body: bytes, fmt: str, provider: str) -> float:
    _check_mp4_box_count(body, provider)

    def parse() -> object:
        fileobj = io.BytesIO(body)
        return MP4Info(Atoms(fileobj), fileobj)

    _, length = _stream_info(parse, fmt, provider)
    return max(length, _mp4_floor_seconds(body, provider=provider))


def _adts_duration(body: bytes, fmt: str, provider: str) -> float:
    if not _adts_stream_starts(body):
        raise _unparsable(fmt, provider)
    return _adts_frames_seconds(body, provider=provider)


def _check_chunk_count(data: bytes, *, little_endian: bool, provider: str) -> None:
    """Reject a RIFF/AIFF file with more chunks than any genuine one.

    mutagen's AIFF reader builds an object for every top-level chunk, and
    the AIFF checks walk them all as ffmpeg does. WAV readers stop at the
    data chunk, whose size streaming writers often leave unset, so only the
    chunks before it count.
    """
    for count, (chunk_id, _, _) in enumerate(_iff_chunks(data, little_endian=little_endian), 1):
        if count > _MAX_CHUNKS:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse audio: too many chunks",
                status_code=400,
            )
        if little_endian and chunk_id == b"data":
            return


def _wav_duration(data: bytes, *, provider: str = "") -> float:
    # The wave module reads the last fmt chunk before the data, ffmpeg the
    # first; with two of them the priced format may not be the decoded one.
    # Both stop looking for one at the data chunk.
    fmt_chunks = 0
    fmt = b""
    for chunk_id, pos, size in _iff_chunks(data, little_endian=True):
        if chunk_id == b"data":
            break
        if chunk_id == b"fmt ":
            fmt_chunks += 1
            fmt = data[pos : pos + min(size, 40)]
    if fmt_chunks > 1:
        raise ProviderError(
            provider=provider,
            detail="Cannot parse WAV audio: more than one fmt chunk",
            status_code=400,
        )
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            rate = wav_file.getframerate()
            if rate <= 0:
                raise ProviderError(
                    provider=provider,
                    detail="Invalid WAV sample rate",
                    status_code=400,
                )
            channels, width = wav_file.getnchannels(), wav_file.getsampwidth()
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse WAV audio: {exc}",
            status_code=400,
        ) from exc

    width = _wav_decoded_width(fmt, width)
    byte_rate = channels * width * min(rate, _MAX_PCM_RATE)
    # The data chunk size is client-controlled, and streaming writers leave
    # it at 0xFFFFFFFF; price the PCM bytes actually sent, since a decoder
    # reads them all and nothing past EOF. A size that fits the file is what
    # the floor counts already.
    duration = _pcm_floor_seconds(data, b"data", byte_rate, little_endian=True)
    if duration <= 0:
        raise ProviderError(
            provider=provider,
            detail="Audio duration must be greater than zero",
            status_code=400,
        )
    return duration


def _wav_decoded_width(fmt: bytes, width: int) -> int:
    """Bytes per sample to price a PCM WAV at, never more than ffmpeg reads.

    The wave module sizes samples from wBitsPerSample, the container size.
    For WAVE_FORMAT_EXTENSIBLE ffmpeg decodes at wValidBitsPerSample when it
    is set; a narrower sample means more frames in the same bytes, so the
    smaller width is priced. dwChannelMask is not priced: ffmpeg takes the
    layout from it only when its bit count equals nChannels and otherwise
    ignores it, so the decoded channel count is always nChannels. A block
    align or byte rate claim plays no part either: the PCM decoder sizes
    frames from its codec and channels alone.
    """
    if fmt[:2] == b"\xfe\xff" and len(fmt) >= 24:
        valid_bits = struct.unpack_from("<H", fmt, 18)[0]
        if valid_bits:
            width = min(width, (valid_bits + 7) // 8)
    return width


def _aiff_byte_rate(data: bytes, *, provider: str = "") -> float:
    """Bytes a second ffmpeg decodes an AIFF file's sample data at.

    ffmpeg takes the sample width from the codec, not from COMM's sample
    size: plain AIFF, and AIFF-C 'NONE' and 'twos', round the size up to 1-4
    bytes and refuse a larger one; the other PCM types have a fixed width;
    the fixed-ratio codecs decode whole blocks of a set length. An FVER
    chunk whose timestamp is not AIFF-C's makes ffmpeg read COMM as plain
    AIFF, so it does here too. Refused: two COMM chunks (mutagen and the
    decoder may read different ones), an FVER chunk not 4 bytes long or an
    odd-sized COMM (no writer makes either, and ffmpeg's chunk walk past
    them need not match this one), an 'APCM' chunk (it switches ffmpeg to
    XA ADPCM, whatever COMM says) and any codec no table here times.
    """
    form = data[8:12]
    version = _AIFC_VERSION1 if form == b"AIFC" else 0
    comm = None
    for chunk_id, pos, size in _iff_chunks(data, little_endian=False):
        if chunk_id == b"COMM":
            if comm is not None:
                raise _unsupported_aiff("more than one COMM chunk", provider)
            comm = pos, size, version
        elif chunk_id == b"FVER":
            if size != 4:
                raise _unsupported_aiff("malformed FVER chunk", provider)
            if comm is None:
                version = int.from_bytes(data[pos : pos + 4], "big")
        elif chunk_id == b"APCM":
            raise _unsupported_aiff("XA ADPCM", provider)
    if form not in (b"AIFF", b"AIFC") or comm is None:
        raise _unsupported_aiff("no COMM chunk", provider)
    pos, size, version = comm
    if size < 18 or size & 1 or pos + 18 > len(data):
        raise _unsupported_aiff("malformed COMM chunk", provider)
    channels, bits = struct.unpack_from(">H4xH", data, pos)
    rate = min(_aiff_rate(data[pos + 8 : pos + 18]), _MAX_PCM_RATE)
    if not channels or rate <= 0:
        raise _unsupported_aiff("no channels or sample rate", provider)
    tag = b"NONE"  # plain AIFF, or a COMM with no room for a compression type
    if version == _AIFC_VERSION1 and size >= 22:
        tag = data[pos + 18 : pos + 22].upper()
    if tag in _AIFC_BLOCK_CODECS:
        codec = _AIFC_BLOCK_CODECS[tag]
        block, per_channel = _MP4_BLOCK_BYTES[codec]
        block_bytes = block * channels if per_channel else block
        return rate * block_bytes / _MP4_V0_FRAMES_PER_PACKET[codec]
    width = _AIFC_PCM_WIDTHS.get(tag, 0)
    if width is None:
        width = next((n for n, top in ((1, 8), (2, 16), (3, 24), (4, 32)) if bits <= top), 0)
    if not width:
        name = tag.decode("ascii", "backslashreplace")
        reason = f"{bits}-bit samples" if tag in (b"NONE", b"TWOS") else f"compression '{name}'"
        raise _unsupported_aiff(reason, provider)
    return rate * width * channels


def _aiff_rate(extended: bytes) -> int:
    """Sample rate ffmpeg reads from COMM's 80-bit float, 0 when it refuses it.

    ffmpeg does not mask the sign bit, rounds a fraction to nearest, and
    keeps the low 32 bits of the result as a C int; mutagen's float reading
    differs for crafted values, so the decoder's is mirrored here.
    """
    exponent = int.from_bytes(extended[:2], "big") - 16383 - 63
    mantissa = int.from_bytes(extended[2:10], "big")
    if not -63 <= exponent <= 63:
        return 0
    if exponent >= 0:
        rate = mantissa << exponent
    else:
        rate = ((mantissa + (1 << (-exponent - 1))) & 0xFFFF_FFFF_FFFF_FFFF) >> -exponent
    rate &= 0xFFFF_FFFF
    return rate if rate < 1 << 31 else 0  # a negative int: ffmpeg refuses it


def _unsupported_aiff(reason: str, provider: str) -> ProviderError:
    return ProviderError(
        provider=provider,
        detail=f"Unsupported AIFF audio: {reason}",
        status_code=400,
    )


def _pcm_floor_seconds(
    data: bytes, chunk_id: bytes, byte_rate: float, *, little_endian: bool
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
    for _ in range(_MAX_CHUNKS):
        if pos + 8 > len(data):
            break
        chunk_id = data[pos : pos + 4]
        # Printable IDs keep raw PCM (zero runs read as empty chunks) out.
        if chunk_id in (b"data", b"SSND") or not all(0x20 <= b <= 0x7E for b in chunk_id):
            return False
        (size,) = struct.unpack_from(size_format, data, pos + 4)
        pad = size & 1
        pos += 8 + size + pad
    else:
        return False  # more chunks than a genuine file has: bill to EOF
    # The last chunk may omit its pad byte.
    return pos == len(data) or (pad == 1 and pos == len(data) + 1)


def _id3v2_size(data: bytes, pos: int = 0) -> int:
    """Return the size of an ID3v2 tag at ``pos`` (0 when absent), capped at the data."""
    if len(data) < pos + 10 or data[pos : pos + 3] != b"ID3":
        return 0
    size = 10 + (
        (data[pos + 6] & 0x7F) << 21
        | (data[pos + 7] & 0x7F) << 14
        | (data[pos + 8] & 0x7F) << 7
        | data[pos + 9] & 0x7F
    )
    if data[pos + 5] & 0x10:
        size += 10  # footer
    return min(size, len(data) - pos)


def _mpeg_frames_seconds(data: bytes, *, skip_id3: bool = True) -> float:
    """Sum the duration of every MPEG audio frame in the data.

    Walks frame to frame and resyncs past junk the way decoders do, so a
    forged Xing/VBRI frame count or padding between frames hides nothing.
    """
    total = 0.0
    # Decoders skip a file's ID3v2 tag, so cover art in it cannot hold
    # frames; MP4 sample data has no tag to skip.
    pos = _id3v2_size(data) if skip_id3 else 0
    first = True
    while (match := _MPEG_SYNC.search(data, pos)) is not None:
        start = match.start()
        samples, rate, frame_size = _mpeg_frame(data, start)
        # A leading Xing/Info/VBRI frame carries the tag, not audio.
        head = data[start : start + 40]
        if not (first and (b"Xing" in head or b"Info" in head or b"VBRI" in head)):
            total += samples / rate
        first = False
        pos = start + frame_size
    return total


def _mpeg_chained_seconds(data: bytes) -> float:
    """Sum the MPEG frames that sit in runs of chained frames, each at least
    as long as the run that makes ffmpeg's MP3 probe pick a file.

    A run ends at the first offset that is not a frame header, and the
    search goes on past it, as the probe's does. Chance sync bytes in PCM or
    FLAC data rarely chain more than a few frames, so they are not summed.
    """
    total = 0.0
    pos = 0
    while (match := _MPEG_SYNC.search(data, pos)) is not None:
        run, seconds, end = 0, 0.0, match.start()
        while end < len(data) and _MPEG_SYNC.match(data, end):
            samples, rate, frame_size = _mpeg_frame(data, end)
            run += 1
            seconds += samples / rate
            end += frame_size
        if run >= _MPEG_PROBE_FRAMES:
            total += seconds
        pos = end + 1
    return total


def _mpeg_frame(data: bytes, start: int) -> tuple[int, int, int]:
    """(samples, sample rate, bytes) of the frame whose header _MPEG_SYNC matched at ``start``."""
    b1, b2 = data[start + 1], data[start + 2]
    version = (b1 >> 3) & 3
    layer = 4 - ((b1 >> 1) & 3)
    mpeg1 = version == 3
    bitrate = _MPEG_KBPS[(mpeg1, layer)][b2 >> 4] * 1000
    rate = _MPEG_RATES[version][(b2 >> 2) & 3]
    padding = (b2 >> 1) & 1
    if layer == 1:
        return 384, rate, (12 * bitrate // rate + padding) * 4
    samples = 1152 if layer == 2 or mpeg1 else 576
    return samples, rate, samples // 8 * bitrate // rate + padding


def _mpeg_stream_starts(data: bytes) -> bool:
    """True when MPEG frames chain from the first byte, as an MP3 stream starts.

    A few frames hidden in another stream's metadata (an ADTS or AC-3 stream
    with MPEG frames inside) do not make ffmpeg's probe pick MP3; this many
    chained frames at the start do, so the frame count is what it decodes.
    """
    pos = 0
    for _ in range(_MPEG_PROBE_FRAMES):
        if pos >= len(data):
            return True  # a short file: every frame chains to the end
        if not _MPEG_SYNC.match(data, pos):
            return False
        pos += _mpeg_frame(data, pos)[2]
    return True


def _adts_header(data: bytes, start: int) -> tuple[float, int] | None:
    """(seconds, frame bytes) of an ADTS header at ``start`` ffmpeg accepts, or None.

    Each frame holds raw_data_blocks + 1 blocks of 1024 samples at its
    sampling index's rate (the AAC core rate, also for HE-AAC).
    """
    head = data[start : start + 7]
    if len(head) < 7:
        return None
    index = head[2] >> 2 & 0x0F
    size = (head[3] & 0x03) << 11 | head[4] << 3 | head[5] >> 5
    if index >= len(_AAC_RATES) or size < 7:
        return None
    return ((head[6] & 0x03) + 1) * 1024 / _AAC_RATES[index], size


def _adts_stream_starts(data: bytes) -> bool:
    """True when ADTS frames chain from the first byte, as an AAC stream starts."""
    pos = 0
    for _ in range(_ADTS_PROBE_FRAMES):
        if pos >= len(data):
            return True
        frame = _adts_header(data, pos) if _ADTS_SYNC.match(data, pos) else None
        if frame is None:
            return False
        pos += frame[1]
    return True


def _adts_frames_seconds(data: bytes, *, provider: str = "") -> float:
    """Sum the frames of an ADTS stream as ffmpeg's parser splits them.

    The parser takes each frame's length from its header and resyncs byte
    by byte past anything that is not a header, so the walk does the same.
    Genuine frames are tens of bytes or more; input with more headers than
    one per 32 bytes is crafted and rejected rather than walked.
    """
    total = 0.0
    budget = len(data) // 32 + 10_000
    pos = 0
    while (match := _ADTS_SYNC.search(data, pos)) is not None:
        budget -= 1
        if budget < 0:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse AAC audio: too many frame headers",
                status_code=400,
            )
        start = match.start()
        frame = _adts_header(data, start)
        if frame is None:
            pos = start + 1
            continue
        total += frame[0]
        pos = start + frame[1]
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
    pos = _flac_audio_start(data, provider=provider)
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


def _flac_audio_start(data: bytes, *, provider: str = "") -> int:
    """Offset of the first byte after the FLAC metadata blocks.

    Decoders skip each block by its declared length, so cover art in a
    PICTURE block cannot hold frames. A file has a handful of blocks; one
    with thousands of empty ones is crafted and rejected.
    """
    pos = _id3v2_size(data)
    if data[pos : pos + 4] != b"fLaC":
        return 0
    pos += 4
    blocks = 0
    while pos + 4 <= len(data):
        blocks += 1
        if blocks > _MAX_CHUNKS:
            raise ProviderError(
                provider=provider,
                detail="Cannot parse FLAC audio: too many metadata blocks",
                status_code=400,
            )
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
    # Seconds each packet is billed at: for AAC, MPEG and Opus the shortest
    # the decoder outputs, for ALAC the longest (all but the last fill it).
    frame_seconds: float | None
    rate: float  # decode sample rate: stsd's, or mdhd's timescale without one
    frame_bytes: int  # PCM bytes per frame (all channels); 0 when not PCM
    block_bytes: int = 0  # bytes per coded block of a fixed-ratio codec, else 0
    count: int = 0  # coded samples (packets)
    sample_bytes: int = 0
    frames: int | None = None  # PCM frames, when the samples are sized in frames
    content_seconds: float = 0.0  # MPEG and Opus: timed from the sample bytes


def _mp4_floor_seconds(data: bytes, *, provider: str = "") -> float:
    """Lower bound from the sample tables of the MP4 audio tracks.

    The decoder reads exactly the samples a track's stsz or stz2 (or, in a
    fragmented movie, its trun boxes) lists, or for uncompressed QuickTime
    audio the frames its chunk table lists, so they set the bound rather than
    the forgeable mdhd duration, and the bytes of video, timecode, metadata
    or subtitle tracks are not billed as audio. The longest track sets the
    price: the provider transcribes one audio stream, so a second language
    or a spatial audio track is not billed again. A movie this walk cannot
    follow (malformed boxes, a compressed moov, fragment sample sizes given
    nowhere, an AAC object type it cannot time) or with no audio track is
    refused: no byte count is then known to bound the audio it decodes.
    """
    try:
        tracks = _mp4_audio_tracks(data)
    except (ValueError, IndexError, OverflowError, struct.error) as exc:
        raise ProviderError(
            provider=provider,
            detail=f"Cannot parse MP4 audio: {exc}",
            status_code=400,
        ) from exc
    if not tracks:
        raise ProviderError(
            provider=provider,
            detail="Cannot parse MP4 audio: no audio track",
            status_code=400,
        )
    floor = max(_mp4_track_floor(track) for track in tracks)
    flac_rates = [track.rate for track in tracks if track.codec == b"fLaC"]
    if flac_rates:
        # Each FLAC sample is a frame with a checked header, and the scan
        # counts every one in the file, wherever the sample table puts it.
        stream_rate = max(1, int(min(flac_rates)))
        floor = max(floor, _flac_frames_seconds(data, stream_rate, provider=provider))
    return floor


def _mp4_track_floor(track: _Mp4Track) -> float:
    """Seconds one audio track's samples must decode to."""
    if track.codec in _AMR_MAX_BPS:
        floor = track.sample_bytes * 8 / _AMR_MAX_BPS[track.codec]
    else:
        cap = _LOSSLESS_MAX_BPS if track.codec in _MP4_LOSSLESS_CODECS else _LOSSY_MAX_BPS
        floor = track.sample_bytes * 8 / cap
        if track.frame_seconds is not None:
            floor = max(floor, track.count * track.frame_seconds)
    floor = max(floor, track.content_seconds)
    rate = min(track.rate, _MAX_PCM_RATE)
    if track.frames is None or rate <= 0:
        return floor
    # PCM frames at the decode rate are its exact duration; the byte floor
    # would overcharge PCM above the lossless cap (8 channels, 24-bit).
    seconds = track.frames / rate
    return seconds if track.frame_bytes else max(seconds, floor)


def _mp4_audio_tracks(data: bytes) -> list[_Mp4Track]:
    """Return the sample totals of each audio track.

    A track is audio when its handler is 'soun', or when it is not 'vide'
    and not every sample entry is a known data, subtitle or video type:
    ffmpeg decodes an audio entry as audio whatever the handler says, and an
    unknown entry is billed rather than trusted. Samples in fragments
    (moof/traf/trun) are added to their track's moov sample table. Raises
    ValueError on malformed boxes, a compressed movie or unknown fragment
    sizes.
    """
    moov = None
    fragments = []
    for kind, start, end in _mp4_boxes(data, 0, len(data), cut_tail=True):
        if kind == b"moov" and moov is None:
            moov = (start, end)
        elif kind == b"moof":
            fragments.append((start, end))
    if moov is None:
        raise ValueError("no moov box")
    tracks: list[_Mp4Track] = []
    track_ids: list[int] = []
    audio_ids: dict[int, _Mp4Track] = {}
    trex_sizes: dict[int, int] = {}
    for kind, start, end in _mp4_boxes(data, *moov):
        if kind == b"cmov":
            raise ValueError("compressed movie")
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
        entries = _mp4_sample_entries(data, *_mp4_child(data, *stbl, b"stsd"))
        if handler != b"soun" and all(entry in _MP4_NON_AUDIO_ENTRIES for entry, _, _ in entries):
            continue  # timecode, metadata, subtitle, hint and other data tracks
        mdhd, _ = _mp4_child(data, *mdia, b"mdhd")
        timescale = struct.unpack_from(">I", data, mdhd + (20 if data[mdhd] == 1 else 12))[0]
        track = _mp4_track(data, stbl, entries, timescale)
        tracks.append(track)
        audio_ids[track_id] = track
    if fragments and len(set(track_ids)) != len(track_ids):
        raise ValueError("duplicate MP4 track_ID")  # a traf could not be matched
    for moof in fragments:
        for kind, start, end in _mp4_boxes(data, *moof):
            if kind == b"traf":
                _mp4_add_fragment(data, start, end, audio_ids, trex_sizes)
    return tracks


def _mp4_track(
    data: bytes, stbl: tuple[int, int], entries: list[tuple[bytes, int, int]], timescale: int
) -> _Mp4Track:
    """Read one audio track's sound description and sample tables.

    ffmpeg reads a QuickTime sample of size 1 as one PCM frame (the sound
    description, not stsz, sizes it), and when stts gives every sample a
    duration of 1 it reads the frames the chunk table lists, ignoring the
    stsz count. Both are counted in frames, as the decoder plays them.
    Fixed-ratio codecs read packet by packet are counted in whole blocks.
    """
    codec, frame_seconds, rate, frame_bytes, packet_frames, channels = _mp4_sound_entry(
        data, entries
    )
    track = _Mp4Track(codec, frame_seconds, rate if rate > 0 else timescale, frame_bytes)
    sample_size, count, sample_bytes = _mp4_sample_sizes(data, stbl)
    chunk_frames = _mp4_chunk_frames(data, stbl)
    if chunk_frames and codec in _AMR_FRAME_BYTES:
        # ffmpeg then cuts each chunk into packets of its own size, so the
        # frame walk below would not see the packets the decoder gets.
        raise ValueError("AMR read by chunk is not supported")
    if codec in _MP4_BLOCK_BYTES:
        block, per_channel = _MP4_BLOCK_BYTES[codec]
        # ffmpeg refuses these with no channels; without a rate no bound holds.
        if channels <= 0 or track.rate <= 0:
            raise ValueError(f"{codec.decode()} audio without channels or a rate")
        if not chunk_frames and sample_size == 1:
            raise ValueError(f"{codec.decode()} audio sized in frames is not supported")
        track.block_bytes = block * channels if per_channel else block
    if sample_size == 1:
        track.frames = max(count, chunk_frames)
        track.sample_bytes = max(count, chunk_frames)  # one byte a sample, either way
    elif frame_bytes:
        track.frames = max(sample_bytes // frame_bytes, chunk_frames)
        track.count, track.sample_bytes = count, sample_bytes
    elif chunk_frames and (packet_frames > 1 or track.block_bytes):
        track.frames = chunk_frames  # sized per packet from the sound description
        track.count, track.sample_bytes = count, sample_bytes
    elif track.block_bytes:
        track.frames = _mp4_block_frames(track, sample_bytes)
        track.count, track.sample_bytes = count, sample_bytes
    else:
        track.count = count
        track.sample_bytes = max(sample_bytes, chunk_frames * sample_size)
    if codec in _MP4_SCANNED_CODECS:
        samples, sizes = _mp4_samples(data, stbl)
        if codec == b"Opus":
            track.content_seconds = _opus_samples_seconds(samples, sizes)
        elif codec in _AMR_FRAME_BYTES:
            track.content_seconds = _amr_samples_seconds(samples, sizes, codec)
        else:
            track.content_seconds = _mpeg_frames_seconds(samples, skip_id3=False)
    return track


def _mp4_block_frames(track: _Mp4Track, sample_bytes: int) -> int:
    """PCM frames a fixed-ratio codec's packets of ``sample_bytes`` decode to at most.

    A packet decodes to its whole blocks, so the blocks in all the packets'
    bytes together are never fewer.
    """
    return sample_bytes // track.block_bytes * _MP4_V0_FRAMES_PER_PACKET[track.codec]


def _mp4_sample_sizes(data: bytes, stbl: tuple[int, int]) -> tuple[int, int, int]:
    """Return (constant sample size or 0, sample count, sample bytes) from stsz or stz2."""
    try:
        stsz, stsz_end = _mp4_child(data, *stbl, b"stsz")
    except ValueError:
        return _mp4_compact_sizes(data, *_mp4_child(data, *stbl, b"stz2"))
    sample_size, count = struct.unpack_from(">II", data, stsz + 4)
    if sample_size:
        return sample_size, count, sample_size * count
    if stsz + 12 + 4 * count > stsz_end:
        raise ValueError("truncated stsz")
    return 0, count, _sum_u32_be(data, stsz + 12, count)


def _mp4_compact_sizes(data: bytes, start: int, end: int) -> tuple[int, int, int]:
    """Read a stz2 table of 4, 8 or 16-bit sample sizes, as ffmpeg supports."""
    values = _mp4_compact_table(data, start, end)
    return 0, len(values), sum(values)


def _mp4_compact_table(data: bytes, start: int, end: int) -> Sequence[int]:
    """Each sample's size from a stz2 table of 4, 8 or 16-bit fields."""
    field = data[start + 7]
    (count,) = struct.unpack_from(">I", data, start + 8)
    table = start + 12
    size = {4: (count + 1) // 2, 8: count, 16: 2 * count}.get(field)
    if size is None or table + size > end:
        raise ValueError("bad stz2")
    packed = data[table : table + size]
    if field == 16:
        values = array.array("H", packed)
        if sys.byteorder == "little":
            values.byteswap()
        return values
    if field == 8:
        return packed
    nibbles = bytearray(2 * size)
    nibbles[0::2] = packed.translate(_HIGH_NIBBLE)
    nibbles[1::2] = packed.translate(_LOW_NIBBLE)
    return bytes(nibbles[:count])  # without the pad nibble


def _mp4_size_table(data: bytes, stbl: tuple[int, int]) -> Sequence[int]:
    """Each sample's size, from stsz or stz2."""
    try:
        stsz, stsz_end = _mp4_child(data, *stbl, b"stsz")
    except ValueError:
        sizes = _mp4_compact_table(data, *_mp4_child(data, *stbl, b"stz2"))
    else:
        sample_size, count = struct.unpack_from(">II", data, stsz + 4)
        if count > _MAX_MP4_SCANNED_SAMPLES:
            raise ValueError("too many MP4 samples")
        if sample_size:
            return array.array("I", [sample_size]) * count
        if stsz + 12 + 4 * count > stsz_end:
            raise ValueError("truncated stsz")
        sizes = array.array("I", data[stsz + 12 : stsz + 12 + 4 * count])
        if sys.byteorder == "little":
            sizes.byteswap()
    if len(sizes) > _MAX_MP4_SCANNED_SAMPLES:
        raise ValueError("too many MP4 samples")
    return sizes


def _mp4_samples(data: bytes, stbl: tuple[int, int]) -> tuple[bytearray, Sequence[int]]:
    """The track's sample bytes in decode order, and the size of each sample.

    The demuxer reads each chunk's samples back to back from its stco or
    co64 offset and never the bytes between chunks, so a frame header put
    there cannot hide the frames after it from this walk. Chunks that
    together reach past the file's size (overlapping) or past its end, and
    a malformed stsc, are refused.
    """
    sizes = _mp4_size_table(data, stbl)
    stsc, stsc_end = _mp4_child(data, *stbl, b"stsc")
    try:
        stco, stco_end = _mp4_child(data, *stbl, b"stco")
        typecode, width = "I", 4
    except ValueError:
        stco, stco_end = _mp4_child(data, *stbl, b"co64")
        typecode, width = "Q", 8
    (chunks,) = struct.unpack_from(">I", data, stco + 4)
    (entries,) = struct.unpack_from(">I", data, stsc + 4)
    if stco + 8 + width * chunks > stco_end or stsc + 8 + 12 * entries > stsc_end:
        raise ValueError("truncated MP4 chunk table")
    if chunks > _MAX_MP4_SCANNED_SAMPLES:
        raise ValueError("too many MP4 chunks")
    offsets = array.array(typecode, data[stco + 8 : stco + 8 + width * chunks])
    values = array.array("I", data[stsc + 8 : stsc + 8 + 12 * entries])
    if sys.byteorder == "little":
        offsets.byteswap()
        values.byteswap()
    firsts, counts = values[0::3], values[1::3]
    if entries and (firsts[0] != 1 or not all(map(operator.lt, firsts, firsts[1:]))):
        raise ValueError("malformed stsc")
    starts = [min(first, chunks + 1) for first in firsts]
    per_chunk = array.array("I")
    spans = map(operator.sub, [*starts[1:], chunks + 1], starts)
    for count, span in zip(counts, spans, strict=True):
        per_chunk.extend(repeat(count, span))
    ends = array.array("Q", accumulate(sizes, initial=0))
    view = memoryview(data)
    samples = bytearray()  # appended chunk by chunk: a list of a million slices is 60 MB
    sample = 0
    for offset, count in zip(offsets, per_chunk, strict=False):  # stsc may list fewer
        if sample >= len(sizes):
            break
        last = min(sample + count, len(sizes))
        length = ends[last] - ends[sample]
        if offset + length > len(data) or len(samples) + length > len(data):
            raise ValueError("MP4 samples reach past the file")
        samples += view[offset : offset + length]
        sample = last
    return samples, sizes[:sample]


def _opus_samples_seconds(samples: bytearray, sizes: Sequence[int]) -> float:
    """Sum the Opus packets laid back to back in ``samples``, timed from their TOC bytes.

    In a multistream packet every stream lasts as long as the first, whose
    TOC byte opens the packet. Empty packets carry no audio.
    """
    tocs = bytes(map(samples.__getitem__, compress(accumulate(sizes, initial=0), sizes)))
    seconds = sum(map(_OPUS_TOC_SECONDS.__getitem__, tocs))
    if tocs.translate(_OPUS_CODE3).count(1):
        # Code 3 gives its frame count in the second byte.
        starts = compress(accumulate(sizes, initial=0), sizes)
        for start, toc in zip(starts, tocs, strict=True):
            if toc & 3 == 3:
                seconds += _opus_packet_seconds(samples[start : start + 2])
    return seconds


def _amr_samples_seconds(samples: bytearray, sizes: Sequence[int], codec: bytes) -> float:
    """Count the 20 ms AMR frames in each sample, sized from their mode bytes.

    The decoder reads a packet's frames back to back and starts afresh at
    the next packet, so each sample is walked on its own; a frame the
    sample's end cuts short still counts. The walk stops past a day.
    """
    lengths = samples.translate(_AMR_FRAME_LENGTHS[codec])
    limit = _MAX_COUNTED_SECONDS * _AMR_FRAMES_PER_SECOND
    frames = pos = 0
    for size in sizes:
        end = pos + size
        while pos < end and frames <= limit:
            pos += lengths[pos]
            frames += 1
        pos = end
    return frames / _AMR_FRAMES_PER_SECOND


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
    if track.codec in _MP4_SCANNED_CODECS:
        raise ValueError("fragmented MPEG, Opus or AMR audio is not supported")
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
        elif track.block_bytes:
            track.frames = (track.frames or 0) + _mp4_block_frames(track, sample_bytes)
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


def _mp4_boxes(data: bytes, start: int, end: int, *, cut_tail: bool = False):
    """Yield (type, payload start, box end) for the boxes in ``data[start:end]``.

    With ``cut_tail`` a last box cut off by the end of the file ends the
    walk, as a demuxer stops at EOF; a cut-off moov or moof is still an
    error, since samples it lists could point at data earlier in the file.
    """
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
            if cut_tail and size >= header and kind not in (b"moov", b"moof"):
                return
            raise ValueError("bad MP4 box size")
        yield kind, pos + header, pos + size
        pos += size


def _check_mp4_box_count(data: bytes, provider: str) -> None:
    """Reject a movie with more boxes than any genuine one.

    mutagen builds an object for every box inside its container boxes, and
    this module walks the same ones, so millions of empty boxes would take
    a gigabyte before payment. Malformed sizes end the count; the parse
    after it rejects them.
    """
    budget = _MAX_MP4_BOXES
    spans = [(0, len(data))]
    while spans:
        start, end = spans.pop()
        pos = start
        while pos + 8 <= end:
            size, kind = struct.unpack_from(">I4s", data, pos)
            header = 8
            if size == 1 and pos + 16 <= end:
                (size,) = struct.unpack_from(">Q", data, pos + 8)
                header = 16
            elif size == 0:
                size = end - pos
            if size < header:
                break
            budget -= 1
            if budget < 0:
                raise ProviderError(
                    provider=provider,
                    detail="Cannot parse MP4 audio: too many boxes",
                    status_code=400,
                )
            if kind in _MP4_CONTAINERS:
                spans.append((pos + header + (4 if kind == b"meta" else 0), min(pos + size, end)))
            pos += size


def _mp4_child(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int]:
    """Return the payload span of the first ``kind`` box, or raise ValueError."""
    for found, payload, box_end in _mp4_boxes(data, start, end):
        if found == kind:
            return payload, box_end
    raise ValueError(f"missing MP4 {kind!r} box")


def _mp4_sample_entries(data: bytes, start: int, end: int) -> list[tuple[bytes, int, int]]:
    """The sample entries stsd counts; boxes after them are ignored, as ffmpeg does."""
    (count,) = struct.unpack_from(">I", data, start + 4)
    if not 0 < count <= _MAX_MP4_SAMPLE_ENTRIES:
        raise ValueError(f"{count} sample descriptions")
    entries = list(islice(_mp4_boxes(data, start + 8, end), count))
    if len(entries) != count:
        raise ValueError("truncated stsd")
    return entries


def _mp4_sound_entry(
    data: bytes, entries: list[tuple[bytes, int, int]]
) -> tuple[bytes, float | None, float, int, int, int]:
    """Read the track's sound description as ffmpeg does.

    Returns the codec ffmpeg decodes, the seconds each packet is billed at
    (None when the codec has no fixed frame), the sample rate (0 when
    unknown), PCM bytes per frame (0 when not PCM), PCM frames per coded
    packet (0 when unknown) and the channel count (0 when unknown).
    Several entries are priced only when they all read the same: ffmpeg
    switches between them by chunk, and different ones could hide a cheaper
    reading behind the first.
    """
    parsed = [_mp4_sound_fields(data, *entry) for entry in entries]
    if any(fields != parsed[0] for fields in parsed[1:]):
        raise ValueError("different sample descriptions")
    return parsed[0]


def _mp4_sound_fields(
    data: bytes, codec: bytes, entry: int, entry_end: int
) -> tuple[bytes, float | None, float, int, int, int]:
    """Read one sample entry; see _mp4_sound_entry."""
    if codec in _MP4_REFUSED_CODECS:
        raise ValueError(f"{_MP4_REFUSED_CODECS[codec]} audio is not supported")
    wav_tag = int.from_bytes(codec[2:], "big")
    wav_wrapped = codec[:2] in (b"ms", b"TS") and codec not in _MP4_MPEG_CODECS
    if wav_wrapped and wav_tag not in _MP4_WAV_PCM:
        # QuickTime's wrapper for a WAVE codec: ADPCM, GSM, G.723 and the
        # other non-PCM ones are low-rate codecs no table here times.
        raise ValueError(f"WAVE format 0x{wav_tag:04x} audio is not supported")
    frame_seconds = _MP4_MIN_FRAME_SECONDS.get(codec)
    version = struct.unpack_from(">H", data, entry + 8)[0]
    fields = _MP4_SOUND_FIELDS.get(version)
    if fields is None or entry + fields > entry_end:
        if codec in (b"mp4a", b"alac"):
            raise ValueError(f"unreadable {codec.decode()} sample entry")
        return codec, frame_seconds, 0.0, 0, 0, 0
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
    pcm_bits = bits
    children = entry + fields
    if codec == b"mp4a":
        frame_seconds = _aac_min_frame_seconds(rate)  # no esds: AAC at this rate
        decoded = codec
        for kind, payload, box_end in _mp4_boxes(data, children, entry_end):
            if kind == b"esds":
                decoded, frame_seconds = _esds_frame_seconds(data, payload, box_end, rate)
        if frame_seconds is None:
            raise ValueError("MP4 audio object type not supported")
        codec = decoded
    elif codec in (b"ipcm", b"fpcm"):
        for kind, payload, box_end in _mp4_boxes(data, children, entry_end):
            if kind == b"pcmC" and payload + 6 <= box_end:
                pcm_bits = data[payload + 5]  # ISO PCM sample size, as ffmpeg reads it
    elif codec == b"alac":
        frame_seconds = _alac_frame_seconds(data, children, entry_end, rate)
    elif codec == b"fLaC":
        rate = _dfla_rate(data, children, entry_end) or rate
    frame_bytes = _mp4_pcm_width(codec, pcm_bits) * channels
    return codec, frame_seconds, rate, frame_bytes, packet_frames, channels


def _alac_frame_seconds(data: bytes, start: int, end: int, entry_rate: float) -> float:
    """Longest packet an ALAC track decodes to, from its magic cookie.

    The decoder outputs at most the cookie's frameLength samples a packet,
    and every packet an encoder writes but the last holds exactly that many.
    It runs at the cookie's sample rate (ffmpeg and AudioToolbox take it
    over the entry's, where Apple's writer leaves 44100 at any rate), or at
    the entry's when the cookie gives none.
    """
    cookie = None
    for kind, payload, box_end in _mp4_boxes(data, start, end):
        if kind == b"wave":  # QuickTime keeps the cookie inside a wave box
            cookie = next(
                ((pos, stop) for found, pos, stop in _mp4_boxes(data, payload, box_end)
                 if found == b"alac"),
                None,
            )  # fmt: skip
        elif kind == b"alac":
            cookie = payload, box_end
        if cookie is not None:
            break
    if cookie is None or cookie[0] + 28 > cookie[1]:
        raise ValueError("ALAC sample entry without a magic cookie")
    frame_length, cookie_rate = struct.unpack_from(">I16xI", data, cookie[0] + 4)
    rate = cookie_rate or entry_rate
    if rate <= 0:
        raise ValueError("ALAC sample rate unknown")
    return frame_length / min(rate, _MAX_PCM_RATE)


def _dfla_rate(data: bytes, start: int, end: int) -> int:
    """STREAMINFO's sample rate in a fLaC entry's dfLa box, 0 when absent."""
    for kind, payload, box_end in _mp4_boxes(data, start, end):
        if kind == b"dfLa" and payload + 21 <= box_end:
            # Version and flags, a metadata block header, then STREAMINFO.
            return int.from_bytes(data[payload + 18 : payload + 21], "big") >> 4
    return 0


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


def _esds_frame_seconds(
    data: bytes, start: int, end: int, entry_rate: float
) -> tuple[bytes, float | None]:
    """The codec an mp4a track decodes, and its shortest frame, from its esds.

    The first sampling rate in the AudioSpecificConfig is the AAC core rate,
    also for HE-AAC, so 960 samples at it is the shortest frame (480 for the
    low-delay object types). MPEG-1/2 audio is reported as '.mp3', with
    frames of at least 576 samples at up to 48 kHz. AAC with no usable
    config is decoded at the sample entry's rate. The frame is None for
    object types this module does not time.
    """
    pos = start + 4  # version and flags
    tag, pos, _ = _mp4_descriptor(data, pos)
    if tag != 0x03:  # ES_Descriptor
        return b"mp4a", _aac_min_frame_seconds(entry_rate)
    flags = data[pos + 2]
    pos += 3
    if flags & 0x80:
        pos += 2
    if flags & 0x40:
        pos += 1 + data[pos]
    if flags & 0x20:
        pos += 2
    tag, pos, _ = _mp4_descriptor(data, pos)
    if tag != 0x04:  # DecoderConfigDescriptor
        return b"mp4a", _aac_min_frame_seconds(entry_rate)
    if data[pos] in _MPEG_AUDIO_OBJECT_TYPES:
        return b".mp3", _MPEG_MIN_FRAME_SECONDS
    if data[pos] not in _AAC_OBJECT_TYPES:
        return b"mp4a", None
    tag, pos, size = _mp4_descriptor(data, pos + 13)
    if tag != 0x05 or size < 2 or pos + size > end:  # DecoderSpecificInfo
        return b"mp4a", _aac_min_frame_seconds(entry_rate)
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
        return b"mp4a", _aac_min_frame_seconds(entry_rate)
    if rate <= 0:
        return b"mp4a", _aac_min_frame_seconds(entry_rate)
    if aot in _AAC_LONG_FRAME_AOTS:
        return b"mp4a", 960 / rate
    if aot in _AAC_LOW_DELAY_AOTS:
        return b"mp4a", 480 / rate
    return b"mp4a", None


def _aac_min_frame_seconds(rate: float) -> float:
    """Shortest AAC frame at a sample entry's rate, held to AAC's 7.35-96 kHz."""
    return 960 / min(max(rate, 7350), 96000)


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
    its first packet, as the demuxer decides it; a stream that is not one of
    these codecs or a known video, subtitle or index stream is refused.
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
            if not head.startswith(_OGG_ALLOWED_HEADS):
                raise ProviderError(
                    provider=provider,
                    detail="Only Opus, Vorbis and Speex audio in Ogg is supported",
                    status_code=400,
                )
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
