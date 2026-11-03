"""OpenRouter provider for x402gate.

Implements the BaseProvider interface for OpenRouter's API:
- Pricing via GET /api/v1/models (chat, embedding, and transcription models)
- Chat completion via POST /api/v1/chat/completions (synchronous)
- Embeddings via POST /api/v1/embeddings (synchronous)
- Speech-to-text via POST /api/v1/audio/transcriptions (synchronous)
- No polling needed (responses are synchronous)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterator
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from x402gate.core import http
from x402gate.core.audio_duration import (
    billing_seconds,
    get_audio_duration_seconds,
)
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import (
    _GATEWAY_TIMEOUT_STATUSES,
    UpstreamMayHaveRun,
    rejected_credentials,
    safe_to_resend,
)
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# A billed POST is sent again only when the connection never opened.
_CONNECT_RETRIES = 2
_CONNECT_RETRY_DELAY = 2.0

# After a failed background refresh the stale catalog stays in use and the
# next refresh starts no sooner than this, so an outage of the models API
# does not start a fetch on every request.
_CATALOG_RETRY_S = 60.0

# A price served from the app's price cache may come from the catalog a
# refresh then replaced (a cached price lives 60 s by default, and verify and
# settle run in between), so for this long the endpoint cap of a request
# without the rates get_price stored is never above the replaced rates.
_CAP_GRACE_S = 180.0

# A 5xx body that reports a timeout ("Gateway Time-out", "upstream timed
# out") means the request was forwarded and may have run.
_TIMEOUT_BODY = re.compile(r"timed?[\s_-]?out", re.IGNORECASE)

# Chat text is weighed per character class, in units of 1/270 token, so every
# rate below is an exact integer and sums never round. Chat prompts in ASCII,
# especially on DeepSeek, bill close to 2.8 characters per token, so plain
# ASCII stays at 2.7 per token. Denser text is weighed so it cannot be priced
# below what it bills:
# - Digits: Qwen, DeepSeek, Gemini and Llama 2 style tokenizers split numbers
#   into single digits, one token each.
# - 2-byte UTF-8 letters (Cyrillic, Greek, accented Latin, Hebrew, Arabic):
#   Russian prose is about 0.3 tokens per character on o200k/Gemini and
#   0.45-0.6 on cl100k/Claude class tokenizers. 0.8 keeps about the same
#   headroom over the cl100k class as the ASCII rate has over English, and a
#   string of random letters, mostly one token each, gains at most 1.25x.
# - 3-byte UTF-8 (CJK, kana, Hangul, Indic, Thai, typographic punctuation):
#   about 1 token per character, up to 2-3 for rare ones; 1.5 each.
# - 4-byte UTF-8 (emoji, CJK Extension B and up): 2-3 tokens on byte-level
#   BPE; 3 each.
# - Characters no vocabulary holds bill one token per UTF-8 byte: ASCII
#   control codes, combining marks (stackable without limit), private use,
#   and rare blocks (CJK Extension A, Yi, tag characters, variation
#   selectors). Normal Russian, Greek, CJK, kana, Hangul and emoji text uses
#   them seldom or never; the one common case, the emoji selector U+FE0F,
#   adds 1.5 tokens per emoji that carries it.
# Other rare 2-byte letters (IPA, Armenian) can still bill up to 2 tokens
# against 0.8. That 2.5x bound is the price of not overcharging Cyrillic.
_UNITS_PER_TOKEN = 270
_ASCII_UNITS = 100  # 2.7 characters per token
_DIGIT_UNITS = 270  # 1 token
_TWO_BYTE_UNITS = 216  # 0.8 token
_THREE_BYTE_UNITS = 405  # 1.5 tokens
_FOUR_BYTE_UNITS = 810  # 3 tokens
_ASCII_CONTROL_UNITS = 270  # 1 byte
_COMBINING_UNITS = 540  # 2 bytes
_RARE_THREE_BYTE_UNITS = 810  # 3 bytes
_RARE_FOUR_BYTE_UNITS = 1080  # 4 bytes
_DIGITS = "0123456789"
_ASCII_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Combining marks of 2-byte scripts: generic, Cyrillic, Hebrew points,
# Arabic harakat and Quranic marks, Syriac, Thaana and NKo.
_COMBINING_MARKS = re.compile(
    r"[\u0300-\u036f\u0483-\u0489\u0591-\u05c7\u0610-\u061a\u064b-\u065f\u0670"
    r"\u06d6-\u06ed\u0730-\u074a\u07a6-\u07b0\u07eb-\u07f3]"
)
# Combining marks (extended, supplement, for symbols, half marks), CJK
# Extension A, Yi, private use and variation selectors.
_RARE_THREE_BYTE = re.compile(
    r"[\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\u3400-\u4dbf\ua000-\ua4cf"
    r"\ue000-\uf8ff\ufe00-\ufe0f\ufe20-\ufe2f]"
)
# Tag characters, variation selectors supplement, supplementary private use.
_RARE_FOUR_BYTE = re.compile(r"[\U000e0000-\U000e01ef\U000f0000-\U0010ffff]")

# OpenRouter duration STT sets pricing.completion to 0. Token-priced
# transcribers set completion above 0. pricing.prompt is USD per second,
# except values at or above this, which are the hourly rate (MAI-Transcribe).
_HOURLY_STT_PRICE = Decimal("0.01")

# Pricing a transcription decodes up to 25 MB of base64 and parses the audio
# in a worker thread, before any payment. At most this many run at once, and
# a request that finds them busy for _STT_PRICING_WAIT_S gets a 429, so
# unpaid price requests cannot fill the default executor or the memory.
# audio_duration's own parse slots refuse at once as a backstop.
_STT_PRICING_SLOTS = 2
_STT_PRICING_WAIT_S = 10.0
# One semaphore per event loop: it binds to the loop that first waits on it
# (tests run each case in a new loop).
_stt_slots: asyncio.Semaphore | None = None
_stt_slots_loop: asyncio.AbstractEventLoop | None = None


# The only paths x402gate prices. Anything else would be billed by the wrong
# formula (or not at all), so it is refused before payment.
_CHAT = "chat"
_EMBEDDINGS = "embeddings"
_TRANSCRIPTION = "transcription"
_PATH_KINDS = {
    "chat/completions": _CHAT,
    "embeddings": _EMBEDDINGS,
    "audio/transcriptions": _TRANSCRIPTION,
}

# Token allowance per image part. It covers the largest common per-image
# bills: Claude high-resolution images (~4,784 tokens at 2576 px), GPT-4.1
# nano / mini and o4-mini patch multipliers (~2,500-3,800), and Gemini
# 768 px tiles at 258 tokens each (~4,100 for a 3k x 3k image). GPT-4o high
# detail stays under 1,500. Qwen-VL dynamic resolution can go higher (~16k)
# but those models are cheap enough that the gap is a fraction of a cent.
# The image bytes themselves are not counted: a base64 data URL would
# otherwise be priced as hundreds of thousands of text tokens.
_IMAGE_INPUT_TOKENS = 4800

# OpenRouter's provider.max_price takes prompt and completion in USD per
# million tokens.
_MILLION = Decimal("1000000")

# Content parts the estimate can price. Files go through OpenRouter's PDF
# parser (billed per page), so they are refused.
_TEXT_PARTS = frozenset({"text"})
_IMAGE_PARTS = frozenset({"image_url"})
_INLINE_PARTS = frozenset({"input_audio"})  # base64 data, counted by size
# A video's length is unknown up front, so a request with one is priced at
# the model's whole context window: the model cannot bill more input tokens
# than that. Only OpenRouter takes video, and only on a prepaid balance,
# which gets the unused part of that hold back (see _PREPAID_ONLY).
_VIDEO_PARTS = frozenset({"video_url"})

# Body key get_price sets, with the reason as its value, on a request that
# only a prepaid balance may pay for. The app refuses an x402 payment for it.
_PREPAID_ONLY = "_prepaid_only"
_VIDEO_NEEDS_PREPAID = (
    "Video input is accepted only with a prepaid balance. A video is priced at the "
    "model's whole context window, and a prepaid balance gets back everything above "
    "the actual cost after the response, which a per-request x402 payment cannot. "
    "Top up with POST /v1/topup and sign the request with the X-PREPAID-* headers."
)

# Exa results are billed per result, so the count the client may ask for is
# capped. Native search engines bill by the provider's own rates and are not
# priced, so the web plugin is pinned to Exa.
_MAX_WEB_RESULTS = 10
_WEB_ENGINE = "exa"

# Perplexity Sonar and OpenAI search-preview models search on their own on
# every request, usually once, at the catalog's pricing.web_search. No
# request field bounds the count, so each completion is priced at this many
# searches. Other models list a web_search rate too, but it is the price of
# native search when a request asks for it, which _check_request refuses, so
# it is never charged. Deep research models run dozens of searches and long
# reasoning, which nothing in the request bounds, so they are refused.
_NATIVE_SEARCHES = 5
_DEEP_RESEARCH = "deep-research"
_SEARCH_PREVIEW = "-search-preview"

# A "~vendor/family-latest" model is an alias that OpenRouter redirects to
# the newest model of the family, billed at that model's price. The alias's
# own catalog price is not that price (~deepseek/deepseek-flash-latest lists
# $0.035/M input and bills up to $0.30/M), so an alias is priced at the
# highest rates among the catalog models of its family: same vendor, every
# word of the alias name. Variants (":free", ":batch") are other products.
_ALIAS_PREFIX = "~"
_ALIAS_SUFFIX = "-latest"

# Tool types the model only writes a call for, which the client then runs:
# billed as tokens. Every other type (web_search, web_search_preview,
# openrouter:web_search, Anthropic's web_search_20250305, code_interpreter,
# file_search...) runs on the provider's side, as often as the model likes,
# and bills per use, so it is refused before payment. A tool without a type
# is a function tool only when it has a "function" object; Gemini's typeless
# {"google_search": {}} is a search.
_TOKEN_TOOL_TYPES = frozenset({"function", "custom"})

# Audio input tokens per second of audio, billed at the catalog "audio" rate
# (or the input rate, when higher). Gemini bills 32 per second, Qwen-Omni 25,
# Voxtral and Phi-4 12.5 and OpenAI gpt-4o audio about 10; a model fed raw
# Whisper encoder frames would bill 50, the most any of them reaches.
_AUDIO_TOKENS_PER_SECOND = 50
# Raw PCM has no header to time it from. Its length at 8 kHz mono, the
# lowest common sample rate, bounds its duration from above.
_PCM_BYTES_PER_SECOND = {"pcm16": 16_000, "pcm24": 24_000}

# Anthropic bills a cache write with a 1-hour TTL at 2x the input rate, above
# the 5-minute write rate (1.25x) the catalogs list as input_cache_write.
_SHORT_CACHE_TTLS = (None, "5m")
_LONG_CACHE_WRITE = Decimal(2)

# pricing.overrides entries change the rates under a condition. An entry
# whose only condition is min_prompt_tokens is a long-context tier: once the
# prompt reaches it, the whole request bills at its rates. Any other key
# outside these rate fields is a condition (utc_start, utc_end, utc_days, or
# one this code does not know).
_TIER_KEY = "min_prompt_tokens"
_OVERRIDE_RATE_FIELDS = frozenset(
    {
        "prompt",
        "completion",
        "internal_reasoning",
        "input_cache_read",
        "input_cache_write",
        "input_cache_write_1h",
        "image",
        "audio",
        "input_audio_cache",
        "web_search",
        "request",
    }
)


def _text_units(text: str) -> int:
    """Weight of one chat text in 1/270 token units (see _UNITS_PER_TOKEN).

    Counted with str/bytes builtins rather than a Python loop per character,
    since a body can carry megabytes of text and is weighed more than once.
    """
    digits = sum(map(text.count, _DIGITS))
    controls = len(_ASCII_CONTROLS.findall(text))
    units = digits * _DIGIT_UNITS + controls * _ASCII_CONTROL_UNITS
    if text.isascii():
        return units + (len(text) - digits - controls) * _ASCII_UNITS
    # Split the rest by UTF-8 length. surrogatepass keeps a lone surrogate,
    # which JSON allows, from raising here.
    ascii_chars = len(text.encode("ascii", "ignore"))
    four_byte = len(text.encode("utf-16-le", "surrogatepass")) // 2 - len(text)
    two_or_three = len(text) - ascii_chars - four_byte
    three_byte = (
        len(text.encode("utf-8", "surrogatepass")) - ascii_chars - 4 * four_byte - 2 * two_or_three
    )
    two_byte = two_or_three - three_byte
    combining = len(_COMBINING_MARKS.findall(text))
    rare_three = len(_RARE_THREE_BYTE.findall(text))
    rare_four = len(_RARE_FOUR_BYTE.findall(text))
    return (
        units
        + (ascii_chars - digits - controls) * _ASCII_UNITS
        + (two_byte - combining) * _TWO_BYTE_UNITS
        + combining * _COMBINING_UNITS
        + (three_byte - rare_three) * _THREE_BYTE_UNITS
        + rare_three * _RARE_THREE_BYTE_UNITS
        + (four_byte - rare_four) * _FOUR_BYTE_UNITS
        + rare_four * _RARE_FOUR_BYTE_UNITS
    )


def _embedding_input_tokens(text_input: Any) -> int:
    """Token estimate of an embeddings ``input``: a string, or a list of them.

    Text is weighed like chat text, so CJK, emoji and digits cannot be
    priced below their bill. An integer is one token id, and a list of
    integers a pre-tokenized input of that many tokens.
    """
    items = text_input if isinstance(text_input, list) else [text_input]
    units = 0
    for item in items:
        if isinstance(item, int) and not isinstance(item, bool):
            units += _UNITS_PER_TOKEN
        elif isinstance(item, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in item
        ):
            units += len(item) * _UNITS_PER_TOKEN
        else:
            units += _value_units(item)
    return max(units // _UNITS_PER_TOKEN, 1)


def _value_units(value: Any) -> int:
    """Token units the model reads for one request value.

    Numbers and flags are settings (temperature, max_tokens), not text.
    Anything structured (tools, response_format schemas, tool_calls) is
    counted as compact JSON.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return 0
    if isinstance(value, str):
        return _text_units(value)
    return _text_units(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _too_deep(provider_name: str) -> ProviderError:
    """A 400 for a body json.loads accepted but json.dumps cannot recurse through."""
    return ProviderError(
        provider=provider_name,
        detail="Request body is nested too deeply",
        status_code=400,
    )


def _part_type(part: dict[str, Any], provider_name: str, *, video: bool = False) -> Any:
    """A content part's type, refusing types the estimate cannot price.

    Video parts pass only with ``video`` (OpenRouter); NanoGPT refuses them.
    """
    part_type = part.get("type", "text" if "text" in part else None)
    if not isinstance(part_type, (str, type(None))):
        # A list or object type is unhashable, so it cannot be looked up below.
        raise ProviderError(
            provider=provider_name,
            detail="Content part type must be a string",
            status_code=400,
        )
    if part_type in _TEXT_PARTS or part_type in _IMAGE_PARTS or part_type in _INLINE_PARTS:
        return part_type
    if video and part_type in _VIDEO_PARTS:
        return part_type
    supported = (
        "text, image_url, input_audio or video_url" if video else ("text, image_url or input_audio")
    )
    raise ProviderError(
        provider=provider_name,
        detail=(
            f"Content part type '{part_type}' is not supported because x402gate "
            f"cannot price it before payment. Use {supported}."
        ),
        status_code=400,
    )


def _content_parts(content: Any) -> list[Any] | None:
    """A message's content as a list of parts, or None for plain content."""
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return content
    return None


def _content_size(content: Any, provider_name: str, *, video: bool = False) -> tuple[int, int]:
    """Return (token units, images) for a message's content.

    A video part adds nothing: a request with one is priced at the whole
    context window (see ``chat_video_parts``).
    """
    parts = _content_parts(content)
    if parts is None:
        return _value_units(content), 0
    units = 0
    images = 0
    for part in parts:
        if not isinstance(part, dict):
            units += _value_units(part)
            continue
        part_type = _part_type(part, provider_name, video=video)
        if part_type in _TEXT_PARTS:
            units += _value_units(part.get("text"))
        elif part_type in _IMAGE_PARTS:
            images += 1
        elif part_type in _VIDEO_PARTS:
            continue
        else:
            # The base64 audio is priced by its duration (see
            # chat_audio_inputs), not as text. Anything else the part carries
            # is weighed as text.
            units += _value_units({k: v for k, v in part.items() if k != "input_audio"})
            audio = part.get("input_audio")
            if isinstance(audio, dict):
                units += _value_units({k: v for k, v in audio.items() if k != "data"})
            else:
                units += _value_units(audio)
    return units, images


def chat_audio_inputs(body: dict[str, Any]) -> list[Any]:
    """The ``input_audio`` values of a chat body's content parts.

    Their audio is not in the text estimate of ``chat_input_size``: a caller
    that accepts audio prices it from its duration and the model's audio
    rate, or refuses it. Shared with NanoGPT.
    """
    found: list[Any] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return found
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in _content_parts(message.get("content")) or ():
            if isinstance(part, dict) and part.get("type") in _INLINE_PARTS:
                found.append(part.get("input_audio"))
    return found


def chat_video_parts(body: dict[str, Any]) -> int:
    """Number of video content parts in a chat body."""
    count = 0
    messages = body.get("messages")
    if not isinstance(messages, list):
        return count
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in _content_parts(message.get("content")) or ():
            if isinstance(part, dict) and part.get("type") in _VIDEO_PARTS:
                count += 1
    return count


def reject_server_tools(body: dict[str, Any], provider_name: str) -> None:
    """Refuse tools and options that make the provider search the web itself.

    A web search tool in ``tools`` (or any other tool the provider runs) and
    xAI Live Search (``search_parameters``) bill per search or per source,
    as many as the model decides, which nothing in the request bounds.
    Function tools are plain tokens and pass. Shared with NanoGPT.
    """
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise ProviderError(
                provider=provider_name, detail="tools must be a list", status_code=400
            )
        for tool in tools:
            if not isinstance(tool, dict):
                raise ProviderError(
                    provider=provider_name, detail="Each tool must be an object", status_code=400
                )
            tool_type = tool.get("type", "function" if "function" in tool else None)
            if not isinstance(tool_type, str) or tool_type not in _TOKEN_TOOL_TYPES:
                raise ProviderError(
                    provider=provider_name,
                    detail=(
                        f"Tool type '{tool_type}' is not supported: the provider runs it "
                        "and bills each use, which x402gate cannot price before payment. "
                        "Only function tools are accepted."
                    ),
                    status_code=400,
                )
    search = body.get("search_parameters")
    if search is not None and not (isinstance(search, dict) and search.get("mode") == "off"):
        raise ProviderError(
            provider=provider_name,
            detail=(
                "search_parameters (xAI Live Search) is not supported: each source is "
                "billed and the count is not known up front. Leave it out or set "
                'mode to "off".'
            ),
            status_code=400,
        )


def check_content_parts(body: dict[str, Any], provider_name: str, *, video: bool = False) -> None:
    """Refuse content parts the estimate cannot price, as chat_input_size does.

    Checks the part types without weighing any text, which on a body of
    megabytes takes tens of milliseconds of event loop time per pass. Shared
    with NanoGPT, which passes no ``video`` and so refuses video parts.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in _content_parts(message.get("content")) or ():
            if isinstance(part, dict):
                _part_type(part, provider_name, video=video)


def chat_input_size(
    body: dict[str, Any], provider_name: str, *, video: bool = False
) -> tuple[int, int]:
    """Return (text tokens, images) of everything a chat model reads as input.

    Message text and every other non-numeric field the client sends (tools,
    functions, tool_choice, prompt, response_format, tool_calls, system...)
    are counted, so the reservation cannot be dodged by moving the prompt
    out of ``messages[].content``. Only ``model`` and gateway-internal keys
    are skipped, the base64 data of audio parts, which is priced by its
    duration (``chat_audio_inputs``), and video parts, which ``video``
    allows and the caller prices at the context window. Unknown content part
    types and nesting too deep to serialize raise a 400 ProviderError.
    """
    try:
        return _chat_input_size(body, provider_name, video)
    except RecursionError as exc:
        raise _too_deep(provider_name) from exc


def _chat_input_size(body: dict[str, Any], provider_name: str, video: bool) -> tuple[int, int]:
    units = 0
    images = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                units += _value_units(message)
                continue
            for key, value in message.items():
                if key == "role":
                    continue
                if key == "content":
                    part_units, part_images = _content_size(value, provider_name, video=video)
                    units += part_units
                    images += part_images
                else:
                    units += _value_units(value)
    else:
        units += _value_units(messages)
    for key, value in body.items():
        if key in ("model", "messages") or key.startswith("_"):
            continue
        units += _value_units(value)
    return max(units // _UNITS_PER_TOKEN, 1), images


def estimate_chat_input_tokens(body: dict[str, Any], provider_name: str) -> int:
    """Conservative input-token estimate for an OpenAI-style chat body.

    Shared with NanoGPT, which takes the same request format.
    """
    text_tokens, images = chat_input_size(body, provider_name)
    return text_tokens + images * _IMAGE_INPUT_TOKENS


def _nested_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Every dict nested in ``value``, itself included.

    Walked with a stack rather than recursion: json.loads accepts nesting
    thousands deep, past what a recursive walk survives.
    """
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def has_nested_key(value: Any, key: str) -> bool:
    """True when ``key`` appears in any nested dict of ``value``."""
    return any(key in current for current in _nested_dicts(value))


def long_cache_ttl(value: Any) -> bool:
    """True when a ``cache_control`` in ``value`` asks for a TTL above 5 minutes.

    Any ttl other than missing or "5m" counts, so an unknown value is priced
    as the dearer write. Shared with NanoGPT.
    """
    for current in _nested_dicts(value):
        control = current.get("cache_control")
        if isinstance(control, dict) and control.get("ttl") not in _SHORT_CACHE_TTLS:
            return True
    return False


def check_json_encodable(body: dict[str, Any], provider_name: str) -> None:
    """Refuse a body that parses but cannot be sent on as JSON.

    json.loads accepts NaN, Infinity and lone surrogate escapes (\\ud800),
    while httpx encodes the upstream body with allow_nan=False as UTF-8, so
    such a request would be paid for and then fail on every send. Checked
    before payment with a walk, not json.dumps, which on a 25 MB body takes
    tens of milliseconds of event loop time. Shared with NanoGPT.
    """
    stack: list[Any] = [v for k, v in body.items() if not k.startswith("_")]
    stack.extend(k for k in body if not k.startswith("_"))
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ProviderError(
                    provider=provider_name,
                    detail="NaN and Infinity are not valid JSON numbers",
                    status_code=400,
                )
        elif isinstance(value, str) and not value.isascii():
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ProviderError(
                    provider=provider_name,
                    detail="Request body has a lone surrogate (\\ud800-\\udfff) in a string",
                    status_code=400,
                ) from exc


def _named_models(body: dict[str, Any]) -> list[str]:
    """The string model ids in ``model`` and ``models``, unvalidated."""
    fallbacks = body.get("models")
    named = [body.get("model"), *(fallbacks if isinstance(fallbacks, list) else ())]
    return [model_id for model_id in named if isinstance(model_id, str)]


def _online_variant(model_id: str) -> bool:
    """True for an ``:online`` model id, which turns on OpenRouter web search.

    OpenRouter runs it with the model's native search engine where there is
    one, at that provider's per-search rates, so it is refused.
    """
    return "online" in model_id.lower().split(":")[1:]


def _alias_family(alias_id: str, models: Catalog) -> list[str]:
    """Catalog models a ``~`` alias may redirect to; empty for any other model."""
    vendor, _, name = alias_id.removeprefix(_ALIAS_PREFIX).partition("/")
    if not alias_id.startswith(_ALIAS_PREFIX) or not name:
        return []
    words = set(name.removesuffix(_ALIAS_SUFFIX).split("-"))
    family = []
    for model_id in models:
        if model_id.startswith(_ALIAS_PREFIX) or ":" in model_id:
            continue
        model_vendor, _, model_name = model_id.partition("/")
        if model_vendor == vendor and words <= set(model_name.split("-")):
            family.append(model_id)
    return family


def _searches_every_request(model_id: str) -> bool:
    """True for a model that runs billed web searches on every request."""
    lowered = model_id.lower().lstrip("~")
    return lowered.startswith("perplexity/") or _SEARCH_PREVIEW in lowered


def _override_rate(override: dict[str, Any], field: str) -> Decimal | None:
    """One rate of a pricing.overrides entry, or None when it has no usable one."""
    raw = override.get(field)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        rate = Decimal(str(raw))
    except InvalidOperation:
        return None
    return rate if rate.is_finite() and rate >= 0 else None


def _with_override(rates: dict[str, Decimal], override: dict[str, Any]) -> dict[str, Decimal]:
    """``rates`` with the rates of one pricing.overrides entry in their place.

    A rate the entry leaves out keeps the base rate, except the cache write,
    audio and reasoning rates, which scale with its prompt and completion
    rates (Gemini's tier lists no cache write).
    """
    changed = dict(rates)
    for key in ("prompt", "completion", "image", "request"):
        rate = _override_rate(override, key)
        if rate is not None:
            changed[key] = rate
    for key, field, basis in (
        ("cache_write", "input_cache_write", "prompt"),
        ("audio", "audio", "prompt"),
        ("reasoning", "internal_reasoning", "completion"),
    ):
        rate = _override_rate(override, field)
        if rate is None and rates[basis] > 0:
            rate = rates[key] * changed[basis] / rates[basis]
        elif rate is None:
            rate = max(rates[key], changed[basis])
        changed[key] = rate
    changed["cache_write"] = max(changed["prompt"], changed["cache_write"])
    return changed


def _tiered_rates(
    rates: dict[str, Decimal], pricing: dict[str, Any], input_tokens: int | Decimal
) -> dict[str, Decimal]:
    """Rates of one model for a prompt of ``input_tokens``, from pricing.overrides.

    Of the long-context tiers the prompt reaches, the one with the highest
    min_prompt_tokens replaces the base rates. Any other entry (a UTC time
    window, or a condition this code does not know) may apply at some moment
    of the call, so each rate is the higher of the two. Callers apply this
    per model, before taking the max over ``models`` fallbacks, so one
    tiered candidate never raises another's rate.
    """
    overrides = pricing.get("overrides")
    if not isinstance(overrides, list):
        return rates
    tier: dict[str, Any] | None = None
    tier_floor: int | float = -1
    others = []
    for override in overrides:
        if not isinstance(override, dict):
            continue
        floor = override.get(_TIER_KEY)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            floor = None
        if floor is not None and floor > input_tokens:
            continue  # this prompt is below the entry's threshold
        if floor is not None and set(override) - _OVERRIDE_RATE_FIELDS == {_TIER_KEY}:
            if floor > tier_floor:
                tier, tier_floor = override, floor
        else:
            others.append(override)
    result = _with_override(rates, tier) if tier is not None else dict(rates)
    for override in others:
        window = _with_override(rates, override)
        result = {key: max(rate, window[key]) for key, rate in result.items()}
    return result


def _reasoning_tokens(usage: dict[str, Any], completion_tokens: Decimal) -> Decimal:
    """Reasoning tokens inside ``completion_tokens``, as the usage reports them."""
    details = usage.get("completion_tokens_details")
    raw = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return Decimal(0)
    return min(Decimal(raw), completion_tokens)


def _billed_audio_tokens(
    usage: dict[str, Any],
    prompt_tokens: Decimal,
    body: dict[str, Any],
    audio_ceiling: Decimal | None = None,
) -> Decimal:
    """Audio tokens inside ``prompt_tokens``, for the catalog cost fallback.

    As ``prompt_tokens_details.audio_tokens`` reports them. A body with audio
    parts whose usage reports none has ``audio_ceiling`` of the prompt priced
    as audio: the audio tokens get_price held for the timed parts, so the
    audio costs no less than the hold assumed and the text beside it bills
    at the prompt rate. Without a ceiling (the audio could not be timed
    again), the whole prompt is priced as audio, the dearer of the two
    rates; the app never charges more than the hold. So is the whole prompt
    of a body with video parts: a video's sound track bills as audio, and
    nothing times it.
    """
    reported = _reported_audio_tokens(usage)
    if reported > 0:
        return min(reported, prompt_tokens)
    if chat_video_parts(body):
        return prompt_tokens
    if not chat_audio_inputs(body):
        return Decimal(0)
    if audio_ceiling is None:
        return prompt_tokens
    return min(max(audio_ceiling, Decimal(0)), prompt_tokens)


def _reported_audio_tokens(usage: dict[str, Any]) -> Decimal:
    """``prompt_tokens_details.audio_tokens`` of a usage, or 0."""
    details = usage.get("prompt_tokens_details")
    raw = details.get("audio_tokens") if isinstance(details, dict) else None
    return Decimal(raw) if not isinstance(raw, bool) and isinstance(raw, int) else Decimal(0)


async def held_audio_tokens(
    usage: dict[str, Any],
    body: dict[str, Any],
    measure: Callable[[list[Any]], Awaitable[int]],
    provider_name: str,
) -> Decimal | None:
    """Audio tokens get_price held for a chat body's audio parts, timed again.

    The ceiling ``_billed_audio_tokens`` needs when the usage reports no
    audio tokens. None when it is not needed (no audio parts, or the usage
    reports them) or the timing fails now; the whole prompt is then priced
    as audio, as before.
    """
    audio_inputs = chat_audio_inputs(body)
    if not audio_inputs or _reported_audio_tokens(usage) > 0:
        return None
    try:
        seconds = await measure(audio_inputs)
    except Exception as exc:
        logger.warning(
            "%s could not time the chat audio again for the actual cost: %s",
            provider_name,
            exc,
        )
        return None
    return Decimal(seconds * _AUDIO_TOKENS_PER_SECOND)


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _budget_value(value: Any, field: str, provider_name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError(
            provider=provider_name,
            detail=f"{field} must be a non-negative integer",
            status_code=400,
        )
    return value


def reasoning_budget(body: dict[str, Any], provider_name: str) -> int:
    """Thinking tokens the request asks for.

    ``reasoning.max_tokens`` (OpenRouter) and ``thinking.budget_tokens``
    (Anthropic style). Rejects a malformed value. Shared with NanoGPT.
    """
    budget = 0
    for key, field in (("reasoning", "max_tokens"), ("thinking", "budget_tokens")):
        options = body.get(key)
        if isinstance(options, dict):
            budget = max(budget, _budget_value(options.get(field), f"{key}.{field}", provider_name))
    return budget


def output_ceiling(
    body: dict[str, Any],
    default_max_tokens: int,
    provider_name: str,
    *,
    extra_budget: int = 0,
) -> int:
    """Output tokens one completion may bill, thinking included.

    Thinking counts inside max_tokens, and Anthropic requires max_tokens
    above the budget. So a cap above the budget bounds the whole output,
    while a cap at or below it is raised upstream to fit the budget, and
    then the budget is billed on top. Each cap that is forwarded is checked
    this way, since the upstream may honour either of them:

    - ``max_tokens``: the client's value, raised to ``default_max_tokens``
      by submit. Left unset by the client, the injected default always has
      the budget added, as the upstream may raise it to fit.
    - ``max_completion_tokens``: when the client sent a positive integer.

    ``extra_budget`` is a budget given outside the body (NanoGPT's
    ``:thinking:<n>`` model suffix); the larger of the budgets counts.
    Shared with NanoGPT.
    """
    budget = max(reasoning_budget(body, provider_name), extra_budget)

    def billed(cap: int) -> int:
        return cap if cap > budget else cap + budget

    client_max = _positive_int_or_none(body.get("max_tokens"))
    if client_max is None:
        ceilings = [default_max_tokens + budget]
    else:
        ceilings = [billed(max(client_max, default_max_tokens))]
    completion_cap = _positive_int_or_none(body.get("max_completion_tokens"))
    if completion_cap is not None:
        ceilings.append(billed(completion_cap))
    return max(ceilings)


def reject_non_text_output(body: dict[str, Any], provider_name: str) -> None:
    """Refuse image or audio output, which models bill above the text completion rate.

    Shared with NanoGPT, which takes the same request format.
    """
    modalities = body.get("modalities")
    if modalities is not None and (
        not isinstance(modalities, list) or any(m != "text" for m in modalities)
    ):
        raise ProviderError(
            provider=provider_name,
            detail=(
                "Only text output is priced by x402gate. Image and audio output are "
                'billed at other rates; send modalities ["text"] or leave it out.'
            ),
            status_code=400,
        )
    if body.get("audio") is not None:
        raise ProviderError(
            provider=provider_name,
            detail="Audio output is not priced by x402gate. Send the request without 'audio'.",
            status_code=400,
        )


def parse_billed_json(resp: httpx.Response, provider_name: str) -> dict[str, Any]:
    """Parse a 2xx response body that the provider has already billed.

    A body that is not a JSON object (an SSE stream, an HTML error page)
    means the call ran and we cannot tell what came back. The error is
    chained to a RemoteProtocolError so ``outcome_unknown()`` keeps the
    payment instead of refunding a job the provider charged us for.
    """
    try:
        result = resp.json()
    except ValueError as exc:
        reason = f"response body is not JSON: {exc}"
    else:
        if isinstance(result, dict):
            return result
        reason = f"response body is JSON {type(result).__name__}, not an object"
    unreadable = httpx.RemoteProtocolError(reason)
    raise ProviderError(
        provider=provider_name,
        detail=f"Provider returned {resp.status_code} but the {reason}",
        status_code=502,
    ) from unreadable


def _usage_amount(value: Any) -> Decimal:
    """A usage count or cost as a Decimal; 0 when it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return Decimal(0)
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return Decimal(0)
    return amount if amount.is_finite() else Decimal(0)


def raise_for_error_body(result: dict[str, Any], provider_name: str, label: str) -> None:
    """Refuse a 2xx body that carries only an error object.

    OpenRouter commits a 200 (keep-alive whitespace) before the model
    answers, so a model that fails after routing arrives as ``{"error":
    {"code": 502, ...}}`` with no choices. A body whose usage reports a
    positive cost or token count was billed and is returned so that usage
    is charged; tokens count even at cost 0, which BYOK reports. Zeroed
    usage bills nothing. Otherwise the provider reported the failure
    itself: a known failure, refunded, with the status from ``error.code``,
    unless that code or the message is a timeout, which may have run, as in
    ``_may_have_run``. Shared with NanoGPT.
    """
    error = result.get("error")
    if not error or any(key in result for key in ("choices", "data", "text")):
        return
    usage = result.get("usage")
    if isinstance(usage, dict) and any(
        _usage_amount(usage.get(key)) > 0
        for key in ("cost", "prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return
    code = error.get("code") if isinstance(error, dict) else None
    status = code if isinstance(code, int) and 400 <= code < 600 else 502
    message = error.get("message") if isinstance(error, dict) else error
    may_have_run = status in _GATEWAY_TIMEOUT_STATUSES or (
        status >= 500 and _TIMEOUT_BODY.search(str(message)) is not None
    )
    exc = ProviderError(
        provider=provider_name,
        detail=f"{label} API error: {json.dumps(error, ensure_ascii=False, default=str)}",
        # As in post_billed_request: a key or credit error is not the client's.
        status_code=502 if rejected_credentials(status) else status,
    )
    if may_have_run:
        raise exc from UpstreamMayHaveRun(f"{label} reported error {status} in a 2xx body")
    raise exc


def _may_have_run(resp: httpx.Response) -> bool:
    """True when an error response leaves open whether the call ran upstream.

    A gateway timeout or reset (504, 520, 524), or another 5xx whose body
    reports a timeout, came after the request was forwarded, as in
    proxy_request. Other errors were reported by the provider: known failures.
    """
    if resp.status_code in _GATEWAY_TIMEOUT_STATUSES:
        return True
    return resp.status_code >= 500 and _TIMEOUT_BODY.search(resp.text) is not None


async def post_billed_request(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    request_timeout: float,
    provider_name: str,
    label: str,
) -> httpx.Response:
    """POST a call that bills when it runs and return its 2xx response.

    Mirrors proxy_request for a non-idempotent call. Only a connection that
    never opened (connect error, connect or pool timeout) is sent again:
    once any HTTP response came back the call may have run, and billed, so
    resending it could pay twice. Errors are left for ``outcome_unknown()``
    to read from the exception chain. A read or write timeout, a dropped
    connection and a gateway timeout status keep the payment; an error
    status the provider reported and a connection that never opened refund
    it. Shared with NanoGPT.
    """
    for attempt in range(1 + _CONNECT_RETRIES):
        try:
            # The pooled client skips a TCP and TLS handshake per call.
            resp = await http.shared_client().post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                # The pool is shared by every provider: when it is full, fail
                # fast (never sent, so retried, then refunded) instead of
                # waiting the whole read timeout for a free connection.
                timeout=httpx.Timeout(
                    request_timeout,
                    connect=min(request_timeout, http.CONNECT_TIMEOUT_S),
                    pool=min(request_timeout, http.CONNECT_TIMEOUT_S),
                ),
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            never_sent = isinstance(exc, httpx.ConnectError) or safe_to_resend(exc)
            if never_sent and attempt < _CONNECT_RETRIES:
                delay = _CONNECT_RETRY_DELAY * (2**attempt)
                logger.warning(
                    "%s %s (attempt %d/%d), retrying in %.1fs",
                    label,
                    type(exc).__name__,
                    attempt + 1,
                    1 + _CONNECT_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if isinstance(exc, httpx.ConnectError):
                raise ProviderError(
                    provider=provider_name,
                    detail=f"Failed to submit request: {exc}",
                ) from exc
            # The timeout stays in __context__, which outcome_unknown() reads.
            raise ProviderError(
                provider=provider_name,
                detail=f"{label} request timed out after {request_timeout:g}s",
                status_code=504,
            ) from None
        except Exception as exc:
            raise ProviderError(
                provider=provider_name,
                detail=f"Failed to submit request: {exc}",
            ) from exc

        if 200 <= resp.status_code < 300:
            return resp
        # httpx does not follow redirects, so a 3xx never reached the model.
        # A 401/402/403 is about the gateway's key or credit (or, for 403, a
        # moderation flag), so it must not reach an x402 client as a payment
        # request; the upstream text is kept.
        status = resp.status_code
        if status < 400 or rejected_credentials(status):
            status = 502
        error = ProviderError(
            provider=provider_name,
            detail=f"{label} API error: {resp.text}",
            status_code=status,
        )
        if _may_have_run(resp):
            raise error from UpstreamMayHaveRun(f"{label} returned {resp.status_code}")
        raise error

    raise AssertionError("unreachable: the last attempt returns or raises")


def _is_stt_request(path: str, inputs: dict[str, Any]) -> bool:
    return "transcription" in path.lower() or "input_audio" in inputs


def _is_duration_based_stt(pricing: dict[str, Any]) -> bool:
    """True when the catalog prices this STT model by audio duration."""
    prompt = Decimal(str(pricing.get("prompt", "0")))
    completion = Decimal(str(pricing.get("completion", "0")))
    return prompt > 0 and completion == 0


def _duration_stt_cost(seconds: int, prompt_price: Decimal) -> Decimal:
    """Price a duration-based transcription from the catalog prompt rate."""
    per_second = (
        prompt_price / Decimal("3600") if prompt_price >= _HOURLY_STT_PRICE else prompt_price
    )
    return _apply_cost_floor(Decimal(seconds) * per_second)


def _stt_pricing_slots() -> asyncio.Semaphore:
    """Return the STT pricing semaphore for the running event loop."""
    global _stt_slots, _stt_slots_loop
    loop = asyncio.get_running_loop()
    if _stt_slots is None or _stt_slots_loop is not loop:
        _stt_slots = asyncio.Semaphore(_STT_PRICING_SLOTS)
        _stt_slots_loop = loop
    return _stt_slots


async def in_audio_pricing_slot(measure: Callable[[], int], provider_name: str) -> int:
    """Run an audio measurement in a worker thread, in an STT pricing slot.

    The thread holds the slot until it ends, even when the request is
    cancelled first: a thread cannot be stopped, so releasing its slot
    early would let more decodes run than the limit allows. The slots are
    shared by every provider that times audio before payment (NanoGPT).
    """
    slots = _stt_pricing_slots()
    try:
        await asyncio.wait_for(slots.acquire(), _STT_PRICING_WAIT_S)
    except TimeoutError:
        # Unpaid clients can fill the slots, so this is a 4xx: a 5xx would
        # count against the provider's health on the public dashboard.
        raise ProviderError(
            provider=provider_name,
            detail="Too many audio files are being priced; retry shortly",
            status_code=429,
        ) from None
    work = asyncio.ensure_future(asyncio.to_thread(measure))

    def finished(task: asyncio.Future[int]) -> None:
        slots.release()
        if not task.cancelled():
            task.exception()  # retrieved, so an abandoned failure is not logged

    work.add_done_callback(finished)
    return await asyncio.shield(work)


def _extract_stt_audio(
    inputs: dict[str, Any],
    provider_name: str,
    *,
    max_audio_bytes: int,
) -> tuple[bytes, str]:
    input_audio = inputs.get("input_audio")
    if not isinstance(input_audio, dict):
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio object",
            status_code=400,
        )

    data_b64 = input_audio.get("data", "")
    fmt = input_audio.get("format", "")
    if not data_b64:
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio.data (base64-encoded audio)",
            status_code=400,
        )
    if not fmt:
        raise ProviderError(
            provider=provider_name,
            detail="Missing input_audio.format",
            status_code=400,
        )

    try:
        audio_bytes = base64.b64decode(data_b64, validate=True)
    except Exception as exc:
        raise ProviderError(
            provider=provider_name,
            detail="Invalid base64 in input_audio.data",
            status_code=400,
        ) from exc

    if len(audio_bytes) > max_audio_bytes:
        max_mb = max_audio_bytes // 1024 // 1024
        raise ProviderError(
            provider=provider_name,
            detail=(
                f"Audio file too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum: {max_mb} MB"
            ),
            status_code=413,
        )

    return audio_bytes, str(fmt)


def check_audio_length(seconds: int, max_seconds: int, provider_name: str) -> int:
    """Refuse audio longer than ``max_seconds`` before payment; return ``seconds``.

    A few dollars of transcription fit in one upload at a low bitrate, and no
    real request needs hours of audio, so long files get a 400.
    """
    if seconds > max_seconds:
        raise ProviderError(
            provider=provider_name,
            detail=(
                f"Audio too long ({_audio_length(seconds)}). "
                f"Maximum: {_audio_length(max_seconds)} per request"
            ),
            status_code=400,
        )
    return seconds


def _audio_length(seconds: int) -> str:
    """Seconds as "45 s", "60 min" or "61.5 min", for error messages."""
    if seconds < 60:
        return f"{seconds} s"
    if seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds / 60:.1f} min"


def _chat_audio_seconds(
    audio_inputs: list[Any], provider_name: str, *, max_audio_bytes: int, max_audio_seconds: int
) -> int:
    """Billing seconds of all the audio parts of a chat request.

    Each part is decoded and timed like a transcription upload, so a forged
    header cannot price long audio as short. Raw PCM (``pcm16``, ``pcm24``)
    is timed from its length at 8 kHz mono. Runs in a worker thread, as
    ``_stt_billing_seconds`` does.
    """
    seconds = 0
    for audio in audio_inputs:
        data, fmt = _extract_stt_audio(
            {"input_audio": audio}, provider_name, max_audio_bytes=max_audio_bytes
        )
        pcm_rate = _PCM_BYTES_PER_SECOND.get(fmt.strip().lower())
        if pcm_rate is not None:
            duration = len(data) / pcm_rate
        else:
            duration = get_audio_duration_seconds(data, fmt, provider=provider_name)
        seconds += billing_seconds(duration)
    return check_audio_length(seconds, max_audio_seconds, provider_name)


def _reported_cost(usage: dict[str, Any]) -> Decimal | None:
    """OpenRouter's own charge from ``usage.cost``, when usable.

    BYOK requests report only OpenRouter's fee there, not the inference the
    upstream key paid for, so they fall back to the catalog computation.
    """
    raw = usage.get("cost")
    if usage.get("is_byok") or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        cost = Decimal(str(raw))
    except InvalidOperation:
        return None
    if not cost.is_finite() or cost < 0:
        return None
    return cost


def _steers_endpoints(routing: dict[str, Any]) -> bool:
    """True when the client's provider preferences narrow or pick the endpoints.

    The catalog lists one price per model, and some endpoints charge more.
    OpenRouter's default routing load-balances across live endpoints,
    weighted by inverse square of price, and falls back when one fails. So
    even default traffic sometimes lands on an endpoint above the catalog
    rate, and the gateway absorbs that difference; a cap there would cost
    the fallback that keeps requests alive. (Sending sort "price" would
    bound it without losing fallback, at some latency; left to the owner.)

    Preferences that remove or reorder endpoints shift traffic to the
    pricier ones on every request, not only now and then, so they get the
    cap: order, only, ignore, sort, quantizations, require_parameters true,
    allow_fallbacks false, data_collection "deny", zdr true, throughput or
    latency preferences, and any key not known to be neutral. If no endpoint
    fits under the cap, OpenRouter refuses with a 4xx before anything runs,
    which is a known failure and refunded.

    Neutral: the client's own max_price, unset or empty values, OpenRouter's
    defaults (allow_fallbacks true, data_collection "allow") and a false
    boolean filter (require_parameters false, zdr false), which filters
    nothing.
    """
    for key, value in routing.items():
        if key == "max_price" or value is None:
            continue
        if isinstance(value, (list, dict, str)) and not value:
            continue
        if key == "allow_fallbacks":
            if value is True:
                continue
        elif value is False:
            continue
        if key == "data_collection" and value == "allow":
            continue
        return True
    return False


def _upstream_body(body: dict[str, Any]) -> dict[str, Any]:
    """Strip gateway-internal keys before forwarding to OpenRouter."""
    return {k: v for k, v in body.items() if not k.startswith("_")}


def _context_window(model_info: dict[str, Any]) -> int | None:
    """Largest context window a catalog entry lists, or None when it lists none.

    The entry's own context_length and its top provider's may differ; the
    larger one bounds every endpoint.
    """
    top = model_info.get("top_provider")
    lengths = [model_info.get("context_length")]
    if isinstance(top, dict):
        lengths.append(top.get("context_length"))
    valid = [
        int(length)
        for length in lengths
        if not isinstance(length, bool) and isinstance(length, (int, float)) and length > 0
    ]
    return max(valid) if valid else None


def _apply_cost_floor(cost: Decimal) -> Decimal:
    if cost < Decimal("0.000001"):
        return Decimal("0.000001")
    return cost


Catalog = dict[str, dict[str, Any]]

# Body key get_price stores the rates it priced at under, for submit's cap.
_PRICED_RATES = "_priced_rates"


class _PricedRates(dict[str, Decimal]):
    """Rates get_price priced a request at, kept in its body for submit.

    Its own type, so a client's JSON (which only makes plain dicts) cannot
    forge one under ``_PRICED_RATES`` and lift the endpoint cap.
    """


class CatalogCache:
    """Model catalogs by key, served stale while one task refreshes them.

    Only the first load of a key makes a request wait. After that an
    expired catalog is still served, and a single background task fetches a
    new one; a failed refresh keeps the old catalog and is tried again after
    ``_CATALOG_RETRY_S``. Concurrent loads of one key share one fetch. An
    empty catalog counts as a failed fetch, so it never replaces the prices.
    Shared with NanoGPT.
    """

    def __init__(
        self, ttl: float, fetch: Callable[[str], Awaitable[Catalog]], provider_name: str
    ) -> None:
        self._ttl = ttl
        self._fetch = fetch
        self._provider_name = provider_name
        self._catalogs: dict[str, Catalog] = {}
        self._loaded_at: dict[str, float] = {}
        self._refresh_at: dict[str, float] = {}
        # The catalog each refresh replaced, and when.
        self._previous: dict[str, tuple[Catalog, float]] = {}
        self._tasks: dict[str, asyncio.Task[Catalog]] = {}

    def put(self, key: str, models: Catalog) -> None:
        """Store a freshly fetched catalog."""
        now = time.time()
        old = self._catalogs.get(key)
        if old is not None:
            self._previous[key] = (old, now)
        self._catalogs[key] = models
        self._loaded_at[key] = now
        self._refresh_at[key] = now + self._ttl

    def expired(self, key: str) -> bool:
        """True when the catalog is older than the TTL or was never loaded."""
        loaded_at = self._loaded_at.get(key)
        return loaded_at is None or time.time() - loaded_at > self._ttl

    def may_fetch(self, key: str) -> bool:
        """True when ``load`` may fetch now: the catalog is expired and not
        inside the retry delay after a failed refresh (or was never loaded).
        """
        if key not in self._catalogs:
            return True
        return self.expired(key) and time.time() >= self._refresh_at[key]

    def replaced(self, key: str, within: float) -> Catalog | None:
        """The catalog a refresh replaced less than ``within`` seconds ago."""
        previous = self._previous.get(key)
        if previous is None or time.time() - previous[1] > within:
            return None
        return previous[0]

    async def get(self, key: str) -> Catalog:
        """The catalog, stale or not; waits only when it was never loaded."""
        models = self._catalogs.get(key)
        if models is None:
            return await self.load(key)
        if time.time() >= self._refresh_at[key]:
            self._start(key)
        return models

    async def load(self, key: str) -> Catalog:
        """Fetch the catalog now, joining a fetch already in flight."""
        # A request that is cancelled (client gone) must not cancel the fetch
        # other requests may be waiting on.
        return await asyncio.shield(self._start(key))

    def _start(self, key: str) -> asyncio.Task[Catalog]:
        task = self._tasks.get(key)
        # A task of another event loop (tests run each case in a new one)
        # cannot be awaited here.
        if task is None or task.done() or task.get_loop() is not asyncio.get_running_loop():
            task = asyncio.create_task(self._fetch_and_store(key))
            task.add_done_callback(lambda done: self._finished(key, done))
            self._tasks[key] = task
        return task

    async def _fetch_and_store(self, key: str) -> Catalog:
        try:
            models = await self._fetch(key)
            if not models:
                raise ProviderError(
                    provider=self._provider_name,
                    detail="The models API returned an empty catalog",
                    status_code=503,
                )
        except Exception:
            if key in self._catalogs:
                self._refresh_at[key] = time.time() + _CATALOG_RETRY_S
            raise
        self.put(key, models)
        return models

    def _finished(self, key: str, task: asyncio.Task[Catalog]) -> None:
        if self._tasks.get(key) is task:
            del self._tasks[key]
        if task.cancelled():
            return
        # Read here so a refresh nobody awaits does not log "never retrieved".
        exc = task.exception()
        if exc is not None and key in self._catalogs:
            logger.warning(
                "%s catalog refresh failed, serving the old one: %s", self._provider_name, exc
            )


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation.

    OpenRouter is an LLM aggregator with 300+ chat, embedding, and STT models
    and an OpenAI-compatible API.  Supports chat completions, embeddings, and
    speech-to-text transcriptions.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        default_max_tokens: int = 1024,
        web_search_tokens_per_result: int = 2048,
        default_web_search_max_results: int = 3,
        web_search_cost_per_result: float = 0.004,
    ) -> None:
        super().__init__(name="openrouter", config=config)
        # Keyed by the output_modalities filter: "" (chat), "embeddings",
        # "transcription".
        self._catalogs = CatalogCache(
            float(config.models_cache_ttl), self._fetch_models_list, self.name
        )
        # Family of each ~ alias, by catalog key and alias, with the catalog
        # it was found in: a refreshed catalog finds it again.
        self._families: dict[tuple[str, str], tuple[Catalog, list[str]]] = {}
        self._default_max_tokens = default_max_tokens
        self._web_search_tokens_per_result = web_search_tokens_per_result
        self._default_web_search_max_results = default_web_search_max_results
        self._web_search_cost_per_result = Decimal(str(web_search_cost_per_result))
        self._max_stt_audio_bytes = config.max_stt_audio_mb * 1024 * 1024
        self._max_stt_audio_seconds = config.max_stt_audio_seconds

    async def _fetch_model_info(
        self,
        model_id: str,
        *,
        output_modalities: str = "",
    ) -> dict[str, Any]:
        """Fetch model info (including pricing) from OpenRouter Models API.

        Looks only in the catalog of the request type. An expired catalog is
        served while it refreshes in the background, except for a model it
        lacks (perhaps added since), which waits for the fresh catalog. In
        the retry delay after a failed refresh such a model is not found
        without a fetch, so unknown ids cannot start one per request.

        Args:
            model_id: Model identifier (e.g. 'openai/gpt-4o-mini').

        Returns:
            Model info dict with 'pricing' sub-dict.

        Raises:
            ProviderError: If the model is not found or the API call fails.
        """
        models = await self._catalogs.get(output_modalities)
        if model_id not in models and self._catalogs.may_fetch(output_modalities):
            models = await self._catalogs.load(output_modalities)

        if model_id in models:
            return models[model_id]

        raise ProviderError(
            provider=self.name,
            detail=f"Model '{model_id}' not found on OpenRouter",
            status_code=404,
        )

    async def warm_up(self) -> None:
        """Load the chat, embeddings and transcription catalogs at startup."""
        results = await asyncio.gather(
            *(self._catalogs.load(key) for key in ("", _EMBEDDINGS, _TRANSCRIPTION)),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _fetch_models_list(self, output_modalities: str = "") -> Catalog:
        """Fetch one catalog from OpenRouter, keyed by model id."""
        url = f"{self._config.base_url.rstrip('/')}/models"
        if output_modalities:
            url += f"?output_modalities={output_modalities}"

        max_retries = 2
        retry_delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                resp = await http.shared_client().get(url, timeout=http.timeout(15.0))

                if resp.status_code >= 500 and attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API returned %d (attempt %d/%d), retrying in %.1fs",
                        resp.status_code,
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code >= 400:
                    raise ProviderError(
                        provider=self.name,
                        detail=f"OpenRouter Models API error: {resp.text}",
                        status_code=resp.status_code,
                    )

                models: Catalog = {}
                for model in resp.json().get("data", []):
                    mid = model.get("id", "")
                    if mid:
                        models[mid] = model
                label = f" ({output_modalities})" if output_modalities else ""
                logger.info("Cached %d models from OpenRouter%s", len(models), label)
                return models

            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        "OpenRouter Models API error: %s (attempt %d/%d), retrying in %.1fs",
                        e,
                        attempt + 1,
                        1 + max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

        raise ProviderError(
            provider=self.name,
            detail=f"Failed to fetch models list after retries: {last_error}",
            status_code=503,
        ) from last_error

    def _stt_billing_seconds(self, inputs: dict[str, Any]) -> int:
        """Billing seconds of the request audio.

        Decoding up to 25 MB of base64 and reading the duration can take a
        second on a crafted file, so get_price runs this in a thread, holding
        one of the STT pricing slots.
        """
        audio_bytes, fmt = _extract_stt_audio(
            inputs,
            self.name,
            max_audio_bytes=self._max_stt_audio_bytes,
        )
        seconds = billing_seconds(get_audio_duration_seconds(audio_bytes, fmt, provider=self.name))
        return check_audio_length(seconds, self._max_stt_audio_seconds, self.name)

    def _estimate_stt_price(
        self,
        model_id: str,
        seconds: int,
        model_info: dict[str, Any],
        prompt_price_dec: Decimal,
    ) -> Decimal:
        pricing = model_info.get("pricing", {})

        if _is_duration_based_stt(pricing):
            estimated_cost = _duration_stt_cost(seconds, prompt_price_dec)
            logger.info(
                "OpenRouter price estimate for %s: $%s (%d billing seconds, duration-based STT)",
                model_id,
                estimated_cost,
                seconds,
            )
            return _apply_cost_floor(estimated_cost)

        raise ProviderError(
            provider=self.name,
            detail=(
                f"Token-based STT model '{model_id}' is not supported because "
                "x402gate cannot guarantee an upfront maximum cost. Use a "
                "duration-priced Whisper model such as 'openai/whisper-1'."
            ),
            status_code=400,
        )

    def _error(self, detail: str, status_code: int = 400) -> ProviderError:
        return ProviderError(provider=self.name, detail=detail, status_code=status_code)

    def _check_request(self, path: str, body: dict[str, Any]) -> str:
        """Validate the path and body shape, returning the request kind.

        Used by get_price (before payment) and again by submit, so nothing
        reaches OpenRouter that was not priced by the matching formula.
        """
        kind = _PATH_KINDS.get(path.strip("/"))
        if kind is None:
            supported = ", ".join(f"/v1/openrouter/{p}" for p in _PATH_KINDS)
            raise self._error(f"Path '{path}' is not priced by x402gate. Supported: {supported}.")
        if body.get("stream"):
            # The gateway reads one JSON body; an SSE stream would be billed
            # upstream and then fail to parse here.
            raise self._error(
                "Streaming is not supported. Send the request with stream set to false."
            )
        routing = body.get("provider")
        if routing is not None and (
            not isinstance(routing, dict)
            or not isinstance(routing.get("max_price", {}), (dict, type(None)))
        ):
            # submit may merge a price cap into this object, so it must be one.
            raise self._error("provider must be an object, and provider.max_price an object")

        if kind == _EMBEDDINGS:
            text_input = body.get("input")
            if not isinstance(text_input, (str, list)) or not text_input:
                raise self._error("Embeddings require a non-empty 'input' string or list")
            if "messages" in body:
                raise self._error("Embeddings take 'input', not 'messages'")
        elif kind == _CHAT:
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise self._error("Chat completions require a non-empty 'messages' list")
            if body.get("web_search_options"):
                raise self._error(
                    "web_search_options is not supported: native web search is billed "
                    "by the model provider at rates x402gate cannot price. Use the "
                    "'web' plugin instead."
                )
            if any(_online_variant(model_id) for model_id in _named_models(body)):
                raise self._error(
                    "The :online model variant is not supported: it searches with the "
                    "model's native engine, billed at rates x402gate cannot price. Use "
                    'the model without :online and the "web" plugin instead.'
                )
            reject_server_tools(body, self.name)
            plugins = body.get("plugins")
            if plugins is not None and not isinstance(plugins, list):
                raise self._error("plugins must be a list")
            for plugin in plugins or []:
                plugin_id = plugin.get("id") if isinstance(plugin, dict) else None
                if plugin_id != "web":
                    raise self._error(
                        f"Plugin '{plugin_id}' is not supported. Only the 'web' plugin is priced."
                    )
                if plugin.get("engine", _WEB_ENGINE) != _WEB_ENGINE:
                    raise self._error(
                        "Only the Exa web search engine is priced. Remove 'engine' "
                        f"or set it to '{_WEB_ENGINE}'."
                    )
            reject_non_text_output(body, self.name)
            reasoning_budget(body, self.name)  # rejects a malformed budget
            check_content_parts(body, self.name, video=True)
        return kind

    def _web_results(self, plugin: dict[str, Any]) -> int:
        """Results the web plugin may return, as priced and as forwarded.

        A missing, non-integer or non-positive value falls back to the
        default, and the count is capped so one request cannot buy an
        unbounded number of billed results.
        """
        raw = _positive_int_or_none(plugin.get("max_results"))
        if raw is None:
            return self._default_web_search_max_results
        return min(raw, _MAX_WEB_RESULTS)

    def _candidate_models(self, inputs: dict[str, Any]) -> list[str]:
        """The primary model plus any ``models`` fallbacks OpenRouter may route to."""
        ids: list[str] = []
        model = inputs.get("model")
        if model is not None:
            if not isinstance(model, str) or not model:
                raise self._error("model must be a non-empty string")
            ids.append(model)
        fallbacks = inputs.get("models")
        if fallbacks is not None:
            if not isinstance(fallbacks, list) or not all(
                isinstance(m, str) and m for m in fallbacks
            ):
                raise self._error("models must be a list of model ids")
            ids.extend(m for m in fallbacks if m not in ids)
        if not ids:
            # Without a model OpenRouter uses the account default, which is unpriced.
            raise self._error("Missing model id")
        return ids

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Estimate maximum request cost from model pricing and request inputs.

        Supports chat completions, embeddings, and speech-to-text transcriptions.
        The formula comes from the path, never from the body shape. With a
        ``models`` fallback list each rate is the highest among the
        candidates, since OpenRouter may route to any of them. When the
        client's routing picks endpoints, submit caps the endpoint price at
        these rates, which are stored in ``inputs`` under ``_PRICED_RATES``.

        Args:
            model_path: API sub-path (e.g. 'chat/completions').
            inputs: Request body.

        Returns:
            Estimated maximum cost in USD as a Decimal.

        Raises:
            ProviderError: If the path or body is unsupported or pricing info
                is unavailable.
        """
        kind = self._check_request(model_path, inputs)
        candidates = self._candidate_models(inputs)
        label = ",".join(candidates)
        check_json_encodable(inputs, self.name)

        if kind == _TRANSCRIPTION:
            return await self._estimate_stt(candidates, inputs)

        rates = await self._max_rates(kind, candidates)
        # The app hands submit this same body; its endpoint cap uses these.
        inputs[_PRICED_RATES] = _PricedRates(
            {key: rates[key] for key in ("prompt", "completion", "request")}
        )

        if kind == _EMBEDDINGS:
            try:
                estimated_input_tokens = _embedding_input_tokens(inputs.get("input", ""))
            except RecursionError as exc:
                raise _too_deep(self.name) from exc

            rates = await self._max_rates(kind, candidates, estimated_input_tokens)
            estimated_cost = Decimal(estimated_input_tokens) * rates["prompt"] + rates["request"]
            logger.info(
                "OpenRouter price estimate for %s: $%s (~%d input tokens, embedding)",
                label,
                estimated_cost,
                estimated_input_tokens,
            )
            return _apply_cost_floor(estimated_cost)

        # Weighed once here; _check_request only checked the part types.
        estimated_input_tokens, images = chat_input_size(inputs, self.name, video=True)
        # A video bills an unknown number of tokens, up to the context window.
        context_window = None
        if chat_video_parts(inputs):
            context_window = await self._max_context_window(candidates)
            inputs[_PREPAID_ONLY] = _VIDEO_NEEDS_PREPAID
        # Audio parts are timed, not weighed as text: their base64 would
        # otherwise price as text tokens at the input rate, which is far
        # above the audio tokens they bill for most files but can be below
        # them for a near-silent one, or at an audio rate above the input rate.
        audio_inputs = chat_audio_inputs(inputs)
        audio_tokens = 0
        if audio_inputs:
            audio_tokens = await self._measure_chat_audio(audio_inputs) * _AUDIO_TOKENS_PER_SECOND

        web_search_cost = Decimal(0)
        for plugin in inputs.get("plugins") or []:
            max_results = self._web_results(plugin)
            estimated_input_tokens += max_results * self._web_search_tokens_per_result
            web_search_cost += Decimal(max_results) * self._web_search_cost_per_result

        max_tokens = output_ceiling(inputs, self._default_max_tokens, self.name)
        raw_n = inputs.get("n", 1)
        if raw_n is None:
            raw_n = 1
        if isinstance(raw_n, bool) or not isinstance(raw_n, int) or raw_n < 1:
            raise ProviderError(
                provider=self.name,
                detail="n must be a positive integer",
                status_code=400,
            )

        # The estimate is weighed to stay above the real prompt, so a prompt
        # billed at the long-context tier is priced at it. The cap above
        # keeps the listed rates, which are what OpenRouter compares.
        prompt_tokens = estimated_input_tokens + images * _IMAGE_INPUT_TOKENS + audio_tokens
        rates = await self._max_rates(
            kind, candidates, context_window if context_window is not None else prompt_tokens
        )
        web_search_cost += Decimal(_NATIVE_SEARCHES * raw_n) * rates["web_search"]
        input_price = self._input_price(rates, inputs)
        # Reasoning tokens bill inside max_tokens, at their own rate if higher.
        output_price = max(rates["completion"], rates["reasoning"])
        # The catalog may also bill each input image at a flat rate.
        image_cost = Decimal(images) * max(_IMAGE_INPUT_TOKENS * input_price, rates["image"])
        # Audio tokens bill at the catalog audio rate, or as input tokens.
        audio_cost = Decimal(audio_tokens) * max(rates["audio"], input_price)
        input_cost = Decimal(estimated_input_tokens) * input_price + image_cost + audio_cost
        if context_window is not None:
            # Every input token fits in the window, the video's included, at
            # the audio rate a video's sound track may bill at. A flat
            # per-image fee comes on top of the image's tokens.
            input_cost = max(
                input_cost,
                Decimal(context_window) * max(rates["audio"], input_price)
                + Decimal(images) * rates["image"],
            )

        estimated_cost = (
            input_cost
            + Decimal(max_tokens) * output_price * Decimal(raw_n)
            + web_search_cost
            + rates["request"]
        )

        logger.info(
            "OpenRouter price estimate for %s: $%s (~%d input tokens, %d images, "
            "%d audio tokens, %d max output tokens%s)",
            label,
            estimated_cost,
            estimated_input_tokens,
            images,
            audio_tokens,
            max_tokens,
            f", video at a {context_window}-token window" if context_window is not None else "",
        )
        return _apply_cost_floor(estimated_cost)

    @staticmethod
    def _input_price(rates: dict[str, Decimal], body: dict[str, Any]) -> Decimal:
        """Per-token input rate of a chat body.

        A cache_control breakpoint can bill the whole prompt at the write
        rate (1.25x on Anthropic), and one with a 1-hour TTL at 2x.
        """
        if not has_nested_key(body, "cache_control"):
            return rates["prompt"]
        if long_cache_ttl(body):
            return max(rates["cache_write"], _LONG_CACHE_WRITE * rates["prompt"])
        return rates["cache_write"]

    def _model_rates(self, model_id: str, model_info: dict[str, Any]) -> dict[str, Decimal]:
        """Catalog rates for one model, in USD.

        ``prompt``, ``completion``, ``reasoning`` and ``cache_write`` are per
        token (the cache write rate is never below the prompt rate),
        ``audio`` per input audio token, ``image`` per input image,
        ``web_search`` per search (0 unless the model searches on every
        request) and ``request`` per request.
        """
        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")

        if prompt_price is None or completion_price is None:
            raise ProviderError(
                provider=self.name,
                detail=f"No pricing info for model '{model_id}'",
                status_code=503,
            )

        prompt = Decimal(str(prompt_price))
        completion = Decimal(str(completion_price))
        if prompt < 0 or completion < 0:
            # openrouter/auto and other routers publish -1: the real model,
            # and so the price, is chosen after payment.
            raise self._error(
                f"Model '{model_id}' has no fixed price (it routes to another model). "
                "Pick a specific model."
            )

        web_search = max(Decimal(str(pricing.get("web_search") or 0)), Decimal(0))
        if web_search > 0 and _DEEP_RESEARCH in model_id.lower():
            raise self._error(
                f"Model '{model_id}' is not supported because x402gate cannot price "
                "it before payment: deep research runs a number of billed searches "
                "and reasoning tokens that no request field bounds."
            )

        cache_write = pricing.get("input_cache_write")
        return {
            "prompt": prompt,
            "completion": completion,
            "reasoning": max(Decimal(str(pricing.get("internal_reasoning") or 0)), Decimal(0)),
            "cache_write": max(prompt, Decimal(str(cache_write or 0))),
            "audio": max(Decimal(str(pricing.get("audio") or 0)), Decimal(0)),
            "image": max(Decimal(str(pricing.get("image") or 0)), Decimal(0)),
            "web_search": web_search if _searches_every_request(model_id) else Decimal(0),
            "request": max(Decimal(str(pricing.get("request") or 0)), Decimal(0)),
        }

    async def _max_rates(
        self, kind: str, model_ids: list[str], input_tokens: int | Decimal | None = None
    ) -> dict[str, Decimal]:
        """Highest rate of each kind among the models OpenRouter may route to.

        With ``input_tokens``, each model's rates are first set from its
        pricing.overrides for a prompt that long (see ``_tiered_rates``).
        Without, they are the listed base rates, which the endpoint cap uses.
        """
        output_modalities = "" if kind == _CHAT else kind
        per_model = []
        for model_id in model_ids:
            model_info = await self._fetch_model_info(model_id, output_modalities=output_modalities)
            per_model.append(self._rates_at(model_id, model_info, input_tokens))
            # A ~ alias bills at the price of the model it leads to.
            for member_id, member_info in await self._family(model_id, output_modalities):
                try:
                    per_model.append(self._rates_at(member_id, member_info, input_tokens))
                except ProviderError:
                    continue  # unpriceable (a router), so not where the alias leads
        return {key: max(rates[key] for rates in per_model) for key in per_model[0]}

    def _rates_at(
        self, model_id: str, model_info: dict[str, Any], input_tokens: int | Decimal | None
    ) -> dict[str, Decimal]:
        """One model's rates, tiered for a prompt of ``input_tokens`` when given."""
        rates = self._model_rates(model_id, model_info)
        if input_tokens is None:
            return rates
        return _tiered_rates(rates, model_info.get("pricing", {}), input_tokens)

    async def _family(
        self, model_id: str, output_modalities: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """The catalog entries of a ``~`` alias's family; empty for any other model."""
        if not model_id.startswith(_ALIAS_PREFIX):
            return []
        models = await self._catalogs.get(output_modalities)
        cached = self._families.get((output_modalities, model_id))
        if cached is not None and cached[0] is models:
            family = cached[1]
        else:
            family = _alias_family(model_id, models)
            self._families[(output_modalities, model_id)] = (models, family)
            if not family:
                logger.warning(
                    "OpenRouter alias %s has no family models in the catalog; "
                    "priced at its own rates",
                    model_id,
                )
        return [(member_id, models[member_id]) for member_id in family]

    async def _max_context_window(self, model_ids: list[str]) -> int:
        """Largest context window among the models OpenRouter may route to.

        A model with no window in the catalog cannot bound a video's tokens,
        so the request is refused.
        """
        windows = []
        for model_id in model_ids:
            model_info = await self._fetch_model_info(model_id)
            window = _context_window(model_info)
            if window is None:
                raise self._error(
                    f"Model '{model_id}' lists no context window, so x402gate cannot "
                    "price video input for it before payment."
                )
            windows.append(window)
            for _, member_info in await self._family(model_id, ""):
                windows.append(_context_window(member_info) or window)
        return max(windows)

    async def _measure_stt_audio(self, inputs: dict[str, Any]) -> int:
        """Billing seconds of the request audio, measured in a worker thread."""
        return await self._in_pricing_slot(lambda: self._stt_billing_seconds(inputs))

    async def _measure_chat_audio(self, audio_inputs: list[Any]) -> int:
        """Billing seconds of a chat request's audio parts, measured in a worker thread."""
        return await self._in_pricing_slot(
            lambda: _chat_audio_seconds(
                audio_inputs,
                self.name,
                max_audio_bytes=self._max_stt_audio_bytes,
                max_audio_seconds=self._max_stt_audio_seconds,
            )
        )

    async def _in_pricing_slot(self, measure: Callable[[], int]) -> int:
        """Run an audio measurement in a worker thread, in an STT pricing slot."""
        return await in_audio_pricing_slot(measure, self.name)

    async def _estimate_stt(self, model_ids: list[str], inputs: dict[str, Any]) -> Decimal:
        """Maximum cost of a transcription on any of the candidate models.

        The audio is decoded and measured once for all candidates, off the
        event loop, so a large file does not stall other requests.
        """
        priced = []
        for model_id in model_ids:
            model_info = await self._fetch_model_info(model_id, output_modalities=_TRANSCRIPTION)
            priced.append((model_id, model_info, self._model_rates(model_id, model_info)))
        seconds = await self._measure_stt_audio(inputs)
        return max(
            self._estimate_stt_price(model_id, seconds, model_info, rates["prompt"])
            for model_id, model_info, rates in priced
        )

    async def _fallback_cap_rates(self, kind: str, body: dict[str, Any]) -> dict[str, Decimal]:
        """Cap rates for a request get_price did not price in this call.

        The current rates, lowered to the rates a refresh replaced within
        ``_CAP_GRACE_S``: the price may come from the old catalog, and a
        rate raised in between must not lift the cap above what was reserved.
        """
        model_ids = self._candidate_models(body)
        rates = await self._max_rates(kind, model_ids)
        replaced = self._catalogs.replaced("" if kind == _CHAT else kind, _CAP_GRACE_S)
        if replaced is not None and all(model_id in replaced for model_id in model_ids):
            try:
                per_model = [self._model_rates(m, replaced[m]) for m in model_ids]
            except ProviderError:
                # Not priceable from the old catalog, so not priced from it.
                per_model = []
            if per_model:
                for key in ("prompt", "completion", "request"):
                    rates[key] = min(rates[key], max(r[key] for r in per_model))
        return rates

    async def _routing_with_price_cap(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """The client's provider preferences plus a max_price at the priced rates.

        Used only when ``_steers_endpoints`` says the client's routing picks
        endpoints. max_price then makes OpenRouter refuse endpoints priced
        above the catalog rate instead of billing above the reservation. A
        lower cap the client sent itself is kept.

        The cap is the rates get_price stored in the body, so a request
        priced from a catalog refreshed a moment ago is capped at the rates
        it paid. A price from the app's price cache skips get_price, and
        then ``_fallback_cap_rates`` applies.
        """
        priced = body.get(_PRICED_RATES)
        if isinstance(priced, _PricedRates):
            rates: dict[str, Decimal] = dict(priced)
        else:
            rates = await self._fallback_cap_rates(kind, body)
        cap = {
            "prompt": float(rates["prompt"] * _MILLION),
            "completion": float(rates["completion"] * _MILLION),
            "request": float(rates["request"]),
        }
        routing = dict(body.get("provider") or {})
        client_cap = dict(routing.get("max_price") or {})
        for key, value in cap.items():
            theirs = client_cap.get(key)
            if not isinstance(theirs, bool) and isinstance(theirs, (int, float)) and theirs >= 0:
                value = min(value, float(theirs))
            client_cap[key] = value
        routing["max_price"] = client_cap
        return routing

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Submit a request to OpenRouter (chat, embeddings, or STT).

        Routes to the appropriate endpoint based on path:
        - 'chat/completions' → POST /api/v1/chat/completions
        - 'embeddings' → POST /api/v1/embeddings
        - 'audio/transcriptions' → POST /api/v1/audio/transcriptions

        OpenRouter returns results synchronously (no polling needed).

        Args:
            path: API sub-path.
            body: Request body, forwarded to OpenRouter.
            prepaid: Kept for the provider interface. Chat requests are capped
                in every mode so the prepaid reservation cannot be exceeded.

        Returns:
            OpenRouter response dict.

        Raises:
            ProviderError: If the request fails.
        """
        # The gateway prices first, but checking again here keeps an unpriced
        # path, stream or plugin from ever reaching OpenRouter.
        kind = self._check_request(path, body)

        if kind == _CHAT:
            raw_max = body.get("max_tokens")
            user_max = raw_max if isinstance(raw_max, int) else 0
            if user_max < self._default_max_tokens:
                body = {**body, "max_tokens": self._default_max_tokens}
                logger.info(
                    "max_tokens %s -> %d (enforced minimum)",
                    raw_max,
                    self._default_max_tokens,
                )
            if "max_completion_tokens" in body and (
                _positive_int_or_none(body["max_completion_tokens"]) is None
            ):
                # get_price only counts a positive integer; drop anything else.
                body = {k: v for k, v in body.items() if k != "max_completion_tokens"}

            plugins = body.get("plugins") or []
            patched_plugins = [
                {**plugin, "max_results": self._web_results(plugin), "engine": _WEB_ENGINE}
                for plugin in plugins
            ]
            if patched_plugins != plugins:
                body = {**body, "plugins": patched_plugins}
                logger.info("Pinned web plugin to the priced engine and result count")

        if kind in (_CHAT, _EMBEDDINGS) and _steers_endpoints(body.get("provider") or {}):
            # The catalog is warm from get_price, so this rarely hits the network.
            body = {**body, "provider": await self._routing_with_price_cap(kind, body)}

        resp = await post_billed_request(
            f"{self._config.base_url.rstrip('/')}/{path.strip('/')}",
            _upstream_body(body),
            api_key=self._config.api_key,
            request_timeout=float(self._config.poll_timeout),
            provider_name=self.name,
            label="OpenRouter",
        )
        result = parse_billed_json(resp, self.name)
        raise_for_error_body(result, self.name, "OpenRouter")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        if "text" in result:
            logger.info(
                "OpenRouter STT completed: model=%s, seconds=%s, cost=$%s",
                body.get("model", "unknown"),
                usage.get("seconds", "?"),
                usage.get("cost", "?"),
            )
        else:
            logger.info(
                "OpenRouter request completed: model=%s, tokens=%s",
                body.get("model", "unknown"),
                usage.get("total_tokens", "?"),
            )
        return result

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Not used — OpenRouter returns results synchronously."""
        raise NotImplementedError(
            "OpenRouter returns results synchronously; polling is not needed."
        )

    async def calculate_actual_cost(
        self, body: dict[str, Any], result: dict[str, Any]
    ) -> Decimal | None:
        """Calculate actual cost from usage data in the response.

        Prefers ``usage.cost``, the amount OpenRouter actually charged. It
        already covers the provider that served the request, a ``models``
        fallback, cache writes and plugins. Without it (or for BYOK), chat
        and embeddings fall back to catalog rates for the model that served
        the request, or the priciest candidate, at the cache write rate when
        the body has a cache breakpoint.

        Args:
            body: Original request body (used for model ID).
            result: Provider response containing 'usage' dict.

        Returns:
            Actual cost in USD, or None if usage data is unavailable.
        """
        usage = result.get("usage")
        if not usage or not isinstance(usage, dict):
            return None

        is_stt = _is_stt_request("", body) or (
            "text" in result and "choices" not in result and usage.get("seconds") is not None
        )

        reported_cost = _reported_cost(usage)
        if reported_cost is not None:
            actual_cost = _apply_cost_floor(reported_cost)
            logger.info(
                "OpenRouter actual cost for %s: $%s (from usage.cost)",
                body.get("model", ""),
                actual_cost,
            )
            return actual_cost

        if is_stt:
            return await self._actual_stt_cost(body, usage)

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            return None
        completion_tokens = usage.get("completion_tokens", 0)

        # The request was validated before payment, so the body shape tells
        # embeddings (input, no messages) from chat.
        kind = _EMBEDDINGS if "input" in body and "messages" not in body else _CHAT
        try:
            candidates = self._candidate_models(body)
            # Price the model that served the request when the response names
            # one of the candidates; otherwise the most expensive of them.
            served = result.get("model")
            model_ids = [served] if served in candidates else candidates
            # Same input, reasoning and long-context rates get_price reserved.
            rates = await self._max_rates(kind, model_ids, Decimal(prompt_tokens))
        except ProviderError:
            return None

        input_price = self._input_price(rates, body)
        audio_ceiling = await held_audio_tokens(usage, body, self._measure_chat_audio, self.name)
        audio_tokens = _billed_audio_tokens(usage, Decimal(prompt_tokens), body, audio_ceiling)
        input_cost = (Decimal(prompt_tokens) - audio_tokens) * input_price + audio_tokens * max(
            rates["audio"], input_price
        )
        output_tokens = Decimal(completion_tokens)
        reasoning = _reasoning_tokens(usage, output_tokens)
        output_cost = (output_tokens - reasoning) * rates["completion"] + reasoning * max(
            rates["completion"], rates["reasoning"]
        )

        web_search_cost = Decimal(0)
        plugins = body.get("plugins") or []
        for plugin in plugins:
            if isinstance(plugin, dict) and plugin.get("id") == "web":
                max_results = self._web_results(plugin)
                web_search_cost += Decimal(max_results) * self._web_search_cost_per_result
        # The native searches get_price reserved, on a model that searches on
        # every request; the usage does not count them.
        completions = _positive_int_or_none(body.get("n")) or 1
        web_search_cost += Decimal(_NATIVE_SEARCHES * completions) * rates["web_search"]

        actual_cost = input_cost + output_cost + web_search_cost + rates["request"]
        actual_cost = _apply_cost_floor(actual_cost)

        logger.info(
            "OpenRouter actual cost for %s: $%s (prompt=%d, completion=%d tokens)",
            ",".join(model_ids),
            actual_cost,
            prompt_tokens,
            completion_tokens,
        )
        return actual_cost

    async def _actual_stt_cost(self, body: dict[str, Any], usage: dict[str, Any]) -> Decimal | None:
        """Catalog cost of a transcription when usage.cost is missing."""
        model_id = body.get("model", "")
        try:
            model_info = await self._fetch_model_info(
                model_id,
                output_modalities=_TRANSCRIPTION,
            )
        except ProviderError:
            return None

        pricing = model_info.get("pricing", {})
        prompt_price = pricing.get("prompt")
        completion_price = pricing.get("completion")
        if prompt_price is None or completion_price is None:
            return None

        prompt_price_dec = Decimal(str(prompt_price))
        completion_price_dec = Decimal(str(completion_price))

        audio_seconds = usage.get("seconds")
        if audio_seconds is not None and _is_duration_based_stt(pricing):
            seconds = billing_seconds(float(audio_seconds))
            actual_cost = _duration_stt_cost(seconds, prompt_price_dec)
            logger.info(
                "OpenRouter actual cost for %s: $%s (%d billing seconds, duration STT)",
                model_id,
                actual_cost,
                seconds,
            )
            return actual_cost

        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            return None
        completion_tokens = usage.get("completion_tokens", 0)
        return _apply_cost_floor(
            Decimal(prompt_tokens) * prompt_price_dec
            + Decimal(completion_tokens) * completion_price_dec
        )
