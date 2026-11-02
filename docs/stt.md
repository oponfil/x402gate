# Speech-to-Text (STT)

Speech-to-Text via [OpenRouter](https://openrouter.ai) — Whisper and other transcription models.

STT is **synchronous** — transcribed text is returned immediately in the JSON response. No polling required.

## Endpoint

```
POST /v1/openrouter/audio/transcriptions
```

| Resource | Link |
|---|---|
| OpenRouter STT docs | [openrouter.ai/docs/guides/overview/multimodal/stt](https://openrouter.ai/docs/guides/overview/multimodal/stt) |
| Model catalog | [openrouter.ai/models](https://openrouter.ai/models) — filter by **Transcription** |
| Whisper 1 pricing | [openrouter.ai/openai/whisper-1](https://openrouter.ai/openai/whisper-1) — $0.0001/sec ($0.006/min) |

## Request

Send JSON with base64-encoded audio (same format as [OpenRouter STT API](https://openrouter.ai/docs/guides/overview/multimodal/stt)):

```json
{
  "model": "openai/whisper-1",
  "input_audio": {
    "data": "UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQIAAAAAAA==",
    "format": "wav"
  },
  "language": "en"
}
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | ✅ | STT model slug (e.g. `openai/whisper-1`) |
| `input_audio.data` | string | ✅ | Base64-encoded audio (raw bytes, not a data URI) |
| `input_audio.format` | string | ✅ | Audio format: `wav`, `mp3`, `flac`, `m4a`, `ogg`, `aac`. WebM is not supported |
| `language` | string | | ISO-639-1 code (e.g. `en`, `ru`). Auto-detected if omitted |
| `temperature` | number | | Sampling temperature 0–1 |

> **Note:** STT uses **JSON + base64**, not multipart upload. Encode your audio file before sending.

## Response Format

```json
{
  "data": {
    "text": "Hello, this is a test of speech-to-text transcription.",
    "usage": {
      "seconds": 9.2,
      "cost": 0.000508,
      "input_tokens": 83,
      "output_tokens": 30,
      "total_tokens": 113
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Transcribed text |
| `usage.seconds` | number | Duration of input audio in seconds |
| `usage.cost` | number | Actual cost in USD (authoritative for prepaid billing) |

## Pricing

x402gate estimates cost **before** payment from audio duration and the model's published pricing. For safety, the gateway currently supports duration-priced STT models only:

| Model type | Examples | How priced |
|---|---|---|
| Supported: duration-based | `openai/whisper-1`, `openai/whisper-large-v3` | USD per **second** of audio (`pricing.prompt`) |
| Not supported: token-based | `openai/gpt-4o-transcribe`, `openai/gpt-4o-mini-transcribe` | Cannot be safely capped before payment |

**Estimate rules:**
- Audio duration is parsed from the uploaded file and rounded **up to whole seconds** (conservative estimate).
- Minimum billing unit is **1 second** — even a 0.01 s clip is estimated as 1 s.
- Maximum file size: **25 MB** (`max_stt_audio_mb`, the OpenRouter Whisper limit); a larger file gets HTTP 413 `Maximum: 25 MB`.
- Maximum length: **1 hour** per request (`max_stt_audio_seconds`); longer audio gets HTTP 400 `Audio too long (… min). Maximum: 60 min per request` before payment. Split longer recordings into parts.
- The container is identified from the file's bytes, not from `input_audio.format`. Supported: WAV (PCM), MP3, FLAC, Ogg (Opus, Vorbis or Speex only), MP4/M4A/MOV/3GP, AIFF/AIFF-C and ADTS AAC. Anything else gets HTTP 400 before payment, including WebM/Matroska, Ogg FLAC, ADIF, WMA/ASF, AC-3, Musepack, WavPack and APE. Inside MP4, codecs the gateway cannot time also get 400: AC-3, E-AC-3, AC-4, DTS, TrueHD, MPEG-H, IAMF, QCELP, EVRC, SMV, Speex, iLBC, Nellymoser, QDesign, and WAVE-wrapped codecs other than PCM, A-law, mu-law and MP3.
- The header's stated length is not trusted on its own. The gateway also counts what the file actually holds (MP3 and ADTS frames, FLAC frames, Ogg Opus/Vorbis/Speex packets, MP4 audio-track samples and packets) and prices the larger of the two, so a file whose header claims a shorter length than it plays is still priced in full. WAV and AIFF are priced from the PCM bytes actually sent, never the header length, so a streaming file whose size fields were left unset is priced correctly. WAV must be uncompressed PCM. It is priced at nChannels, whatever `dwChannelMask` says, and an extensible WAV at the narrower of its container width and `wValidBitsPerSample`. AIFF-C may hold PCM (`NONE`, `twos`, `sowt`, `raw `, `in24`, `in32`, `fl32`, `fl64`), A-law, mu-law, IMA4, MACE 3/6 or GSM; the last four are timed by their coded blocks. AIFF is priced at the sample width the decoder uses, not COMM's sample size, and its sample rate is read as ffmpeg reads it. WAV and AIFF sample rates above 768 kHz are charged as 768 kHz. HTTP 400 before payment (`Unsupported AIFF audio: <reason>`) goes to a compressed WAV, any other AIFF-C codec, PCM above 32 bits, an `APCM` chunk (XA ADPCM), an FVER chunk that is not 4 bytes, an odd-sized or second COMM chunk, no channels, a sample rate the decoder rejects, and any file that cannot be parsed.
- An MP3 must start with a valid frame right after its ID3 tags and any zero padding between them. Another container behind an ID3 tag (Ogg, FLAC, WAV, AIFF, MP4, ADTS) is accepted only with exactly one tag under about 1 MiB and no padding. WAV, AIFF and Ogg there are priced as the container alone; FLAC, MP4 and ADTS are billed at least the MPEG frames they hold in chained runs of 7 or more.
- MP4 gets HTTP 400 for a compressed `moov`, no audio track, malformed boxes, fragments without sample sizes, AAC object types it cannot time, differing or more than 16 sample descriptions, or ALAC without its magic cookie. In MP4, MPEG audio (`.mp3`, `.mp2`, `.mp1`) and AMR have their frames counted, AMR at 20 ms a frame, NO_DATA frames included (AMR read by chunk or fragmented gets 400). ALAC is timed at its magic cookie's sample rate. IMA ADPCM, MACE 3/6 and QuickTime GSM are timed from their coded bytes, and get 400 with no channels or rate, or with one-byte samples and no chunk table. Structure caps (also 400): more than 10,000 RIFF/AIFF chunks before the audio data, 10,000 FLAC metadata blocks, 100,000 MP4 boxes, 1,000,000 MP4 samples or chunks, or 16 stacked ID3v2 tags.
- At most 2 files are priced at once, base64 decoding included, shared with the audio parts of OpenRouter and NanoGPT chat requests. A request that waits more than 10 seconds for a slot gets HTTP 429 `Too many audio files are being priced; retry shortly` with `Retry-After: 2`.
- Microsoft MAI models publish an hourly rate in `pricing.prompt` (`0.10`, `0.36`). Those rates are divided by 3600. Every other duration model is already priced per second.

**After the request:** prepaid mode deducts `usage.cost` from OpenRouter when available; x402 mode settles the signed estimate (see [prepaid.md](prepaid.md)).

## Supported Models

Discover available models:

```bash
curl "https://openrouter.ai/api/v1/models?output_modalities=transcription"
```

Popular models:

| Model | Pricing | Notes |
|---|---|---|
| `openai/whisper-1` | $0.0001/sec | 50+ languages, up to 25 MB |
| `openai/whisper-large-v3` | $0.0000075/sec | Multilingual, higher quality |

## Example (x402)

```bash
# 1. Encode audio
base64 -w0 sample.wav > sample.b64

# 2. Request (expect 402, then retry with PAYMENT-SIGNATURE)
curl -X POST https://x402gate.io/v1/openrouter/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"openai/whisper-1\",\"input_audio\":{\"data\":\"$(cat sample.b64)\",\"format\":\"wav\"}}"
```

## Example (prepaid)

```python
import base64, hashlib, time, httpx
from eth_account import Account
from eth_account.messages import encode_defunct

account = Account.from_key("0x...")
pubkey = account.address
path = "audio/transcriptions"
ts = int(time.time())
msg = f"x402gate:openrouter/{path}:{ts}".encode()
sig = account.sign_message(encode_defunct(primitive=msg)).signature.hex()

with open("sample.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

resp = httpx.post(
    "https://x402gate.io/v1/openrouter/" + path,
    json={
        "model": "openai/whisper-1",
        "input_audio": {"data": audio_b64, "format": "wav"},
    },
    headers={
        "X-PREPAID-PUBKEY": pubkey,
        "X-PREPAID-SIGNATURE": sig,
        "X-PREPAID-TIMESTAMP": str(ts),
    },
)
print(resp.json()["data"]["text"])
```

## Notes

- STT is part of the **OpenRouter** managed provider — same x402 and prepaid flows as chat/embeddings.
- `max_tokens` injection (used for chat completions) does **not** apply to `audio/transcriptions` or `embeddings`.
- The container is identified in-process from its bytes; `mutagen` only reads header durations (stream info, never tags). The gateway then counts the content in-process for the floor described under Pricing.
- For real transcription quality, use meaningful audio — the config example is a minimal silent WAV for documentation only.
