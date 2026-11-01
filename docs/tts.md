# TTS Providers

Text-to-Speech via [ElevenLabs](https://elevenlabs.io), [MiniMax](https://www.minimax.io), and [Fish Audio](https://fish.audio).

All three providers are **synchronous** — audio is returned immediately in the response as base64. No polling required.

## Endpoints

| Provider | Endpoint | Pricing | Best For |
|---|---|---|---|
| ElevenLabs | `POST /v1/elevenlabs/{voice_id}` | $0.30/1k chars (standard), $0.15/1k chars (turbo/flash) | Premium quality, 1000+ voices |
| MiniMax | `POST /v1/minimax/{model}` | $0.06/1k chars (turbo), $0.10/1k chars (HD) | Emotional expression, low cost |
| Fish Audio | `POST /v1/fishaudio/tts` | $0.015/1k bytes (UTF-8) | Uncensored, cheapest for ASCII |

> **Note:** Fish Audio charges per **UTF-8 byte**, not per character. Cyrillic = 2 bytes/char, emoji = 4 bytes/char. For Latin text, it's the cheapest option.

## Price Comparison (1000 Latin characters)

| Provider | Model | Price |
|---|---|---|
| Fish Audio | S2 Pro | **$0.015** |
| MiniMax | speech-2.8-turbo | **$0.06** |
| ElevenLabs | eleven_flash_v2_5 | **$0.15** |
| MiniMax | speech-2.8-hd | **$0.10** |
| ElevenLabs | eleven_v3 | **$0.30** |

## Response Format

All three providers return the same JSON structure:

```json
{
  "data": {
    "audio_base64": "//uQxAAAAAAA...",
    "content_type": "audio/mpeg",
    "characters": 100,
    "model": "eleven_v3"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `audio_base64` | string | Base64-encoded audio data |
| `content_type` | string | MIME type (`audio/mpeg`, `audio/wav`, etc.) |
| `characters` | int | Number of characters processed |
| `utf8_bytes` | int | UTF-8 byte count (Fish Audio only) |
| `usage_characters` | int | Characters MiniMax says it billed (MiniMax only, when MiniMax reports it). In prepaid mode this count at the model's rate is what gets deducted, never more than the reserved estimate. x402 payments still settle the quoted price |
| `model` | string | Model used for generation |

---

## ElevenLabs

[elevenlabs.io](https://elevenlabs.io) — Premium TTS with 1000+ pre-built voices and voice cloning.

| Resource | Link |
|---|---|
| API Docs | [elevenlabs.io/docs/api-reference/text-to-speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert) |
| Voice Library | [elevenlabs.io/voice-library](https://elevenlabs.io/voice-library) |
| Pricing | [elevenlabs.io/pricing](https://elevenlabs.io/pricing) — Starter plan ($5/mo) minimum for API access |

### Request

```
POST /v1/elevenlabs/{voice_id}
```

The `voice_id` is passed as part of the URL path (same as ElevenLabs native API). Find voice IDs at [elevenlabs.io/voice-library](https://elevenlabs.io/voice-library). The path must be a plain voice ID: 1-64 letters, digits, `_` or `-`. Anything else (a slash, `..`, a query string) gets HTTP 400 before any payment.

```json
{
  "text": "Hello, this is a test of ElevenLabs text to speech.",
  "model_id": "eleven_v3"
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ | — | Text to synthesize |
| `model_id` | string | | `eleven_v3` | Model to use (see below) |
| `output_format` | string | | `mp3_44100_128` | Audio format |
| `voice_settings` | object | | — | Optional pitch, stability, similarity settings |

### Models

| Model ID | Type | Price/1k chars | Description |
|---|---|---|---|
| `eleven_v3` | Standard | $0.30 | Latest, most expressive (70+ languages) |
| `eleven_multilingual_v2` | Standard | $0.30 | Multilingual, 29 languages |
| `eleven_flash_v2_5` | Flash | $0.15 | Ultra-low latency |
| `eleven_turbo_v2` | Turbo | $0.15 | Fast generation |
| `eleven_monolingual_v1` | Standard | $0.30 | English only |

### Example

```bash
curl -X POST https://x402gate.io/v1/elevenlabs/21m00Tcm4TlvDq8ikWAM \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "model_id": "eleven_v3"}'
```

---

## MiniMax

[minimax.io](https://www.minimax.io) — High-quality TTS with emotional expression and 300+ pre-built voices.

| Resource | Link |
|---|---|
| API Docs | [platform.minimax.io/docs/api-reference/speech-t2a-http](https://platform.minimax.io/docs/api-reference/speech-t2a-http) |
| Voice Library | [minimax.io/audio/text-to-speech](https://www.minimax.io/audio/text-to-speech) — 300+ voices, preview & copy voice_id |

### Request

```
POST /v1/minimax/{model}
```

```json
{
  "text": "Welcome to the future of AI voice synthesis.",
  "model": "speech-2.8-hd",
  "voice_setting": {"voice_id": "English_expressive_narrator"}
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ | — | Text to synthesize |
| `model` | string | | URL path | Model name (see below). Falls back to `{model}` from the URL |
| `voice_setting` | object | | — | Voice ID, speed, pitch, emotion |
| `audio_setting` | object | | — | Output format (`mp3`, `wav`, `flac`, `pcm`), sample rate, bitrate |
| `language_boost` | string | | — | Language hint, e.g. `Russian` |
| `pronunciation_dict` | object | | — | Custom pronunciations |
| `timber_weights` | array | | — | Mix several voices |

### Models

| Model | Price/1k chars | Description |
|---|---|---|
| `speech-2.8-hd` | $0.10 | Highest quality |
| `speech-2.8-turbo` | $0.06 | Low latency |

Any model whose name contains `turbo` is billed at the turbo rate; every other model is billed at the HD rate.

### Example

```bash
curl -X POST https://x402gate.io/v1/minimax/speech-2.8-hd \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "model": "speech-2.8-hd", "voice_setting": {"voice_id": "English_expressive_narrator"}}'
```

---

## Fish Audio

[fish.audio](https://fish.audio) — Uncensored TTS with voice cloning and open-source models.

| Resource | Link |
|---|---|
| API Docs | [fish.audio/docs](https://fish.audio/docs) |
| Voice Catalog | [fish.audio/app/discovery](https://fish.audio/app/discovery) — browse 100k+ community voices |
| Pricing | [fish.audio/pricing](https://fish.audio/pricing) — pay-as-you-go, $15/1M UTF-8 bytes |

### Request

```
POST /v1/fishaudio/tts
```

```json
{
  "text": "Artificial intelligence leads the way.",
  "model": "s2-pro",
  "format": "mp3",
  "reference_id": "e58b0d7efca34eb38d5c4985e378abcb",
  "prosody": {"speed": 1.1}
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ | — | Text to synthesize |
| `model` | string | | — | TTS model: `s2-pro` (newest) or `s1` (expressive) |
| `format` | string | | `mp3` | Output format: `mp3`, `wav`, `ogg`, `flac` |
| `reference_id` | string | | — | Voice ID from [fish.audio/app/discovery](https://fish.audio/app/discovery) |
| `prosody` | object | | — | `{"speed": 1.1}` — speech rate multiplier |
| `chunk_length` | int | | — | Chunk length for processing |
| `latency` | string | | `normal` | `normal` or `balanced` |

### Pricing

Fish Audio charges per **UTF-8 byte** ($15.00/1M bytes):

| Text | Chars | UTF-8 Bytes | Price |
|---|---|---|---|
| `Hello` | 5 | 5 | $0.000075 |
| `Привет` | 6 | 12 | $0.000180 |
| `こんにちは` | 5 | 15 | $0.000225 |

### Example

```bash
curl -X POST https://x402gate.io/v1/fishaudio/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "format": "mp3"}'
```

## Notes

- All providers are **synchronous** — audio is returned directly, no polling needed.
- Audio is returned as **base64** inside a JSON response (`{"data": {"audio_base64": "..."}}`).
- **Pricing safety:** if `model_id` (ElevenLabs) or `model` (MiniMax) is omitted or unrecognized, the **most expensive** tier is used for pricing. Always specify the model explicitly to get the cheaper rate.
- Maximum text length depends on the provider's API limits (typically 5000+ characters).
- Voice cloning is supported by Fish Audio via `reference_id` and by ElevenLabs via custom `voice_id`.
