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
| Whisper 1 pricing | [openrouter.ai/openai/whisper-1](https://openrouter.ai/openai/whisper-1) — $0.006/min |

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
| `input_audio.format` | string | ✅ | Audio format: `wav`, `mp3`, `flac`, `m4a`, `ogg`, `webm`, `aac` |
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
| Supported: duration-based | `openai/whisper-1`, `openai/whisper-large-v3` | USD per **minute** of audio |
| Not supported: token-based | `openai/gpt-4o-transcribe`, `openai/gpt-4o-mini-transcribe` | Cannot be safely capped before payment |

**Estimate rules:**
- Audio duration is parsed from the uploaded file and rounded **up to whole seconds** (conservative estimate).
- Minimum billing unit is **1 second** — even a 0.01 s clip is estimated as 1 s.
- Maximum file size: **25 MB** (OpenRouter Whisper limit).

**After the request:** prepaid mode deducts `usage.cost` from OpenRouter when available; x402 mode settles the signed estimate (see [prepaid.md](prepaid.md)).

## Supported Models

Discover available models:

```bash
curl "https://openrouter.ai/api/v1/models?output_modalities=transcription"
```

Popular models:

| Model | Pricing | Notes |
|---|---|---|
| `openai/whisper-1` | $0.006/min | 50+ languages, up to 25 MB |
| `openai/whisper-large-v3` | ~$0.0015/min | Multilingual, higher quality |

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
- WAV duration is parsed in-process; other formats use the `mutagen` library.
- For real transcription quality, use meaningful audio — the config example is a minimal silent WAV for documentation only.
