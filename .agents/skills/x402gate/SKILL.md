---
name: x402gate-api
description: Access x402gate.io to use paid AI and utility APIs over x402/USDC without provider API keys. Use when you need LLMs, image or video generation, TTS, STT, file conversion, social media download, prepaid balance requests, or provider request examples.
metadata: {"openclaw":{"emoji":"$","homepage":"https://x402gate.io","os":["linux","darwin","win32"]}}
---

# x402gate Agent Skill

Use this skill to access the public gateway at `https://x402gate.io`.

This skill intentionally points agents to the hosted `x402gate.io` service. If this file is mirrored or served from another domain, continue using `https://x402gate.io` unless the skill text is explicitly customized for another deployment.

## When to Use

- Accessing paid AI APIs without managing provider-specific API keys
- Discovering supported providers, pricing, and example request bodies
- Making repeated prepaid requests with Base or Solana USDC
- Using one gateway for LLMs, image/video generation, TTS, STT, file conversion, and social media download

## Quick Start

Discover providers, endpoints, prices, and example payloads from the root manifest first:

```bash
curl -H "Accept: application/json" https://x402gate.io/
```

## 1. Capabilities & Supported Providers
1. **LLMs:** OpenRouter (300+ models, web search, embeddings, STT/Whisper), BlockRun.
2. **Image & Video:** WaveSpeed (60+ models like Flux, Sora), Tungsten (SDXL with LoRA).
3. **TTS:** ElevenLabs (premium), MiniMax (emotional), Fish Audio (uncensored).
4. **Utilities:** CloudConvert (file conversion), SocialDownload (YouTube/TikTok).

## 2. Discovery (Find endpoints and schemas)
**Do not guess endpoints or request schemas.**
Always `GET https://x402gate.io/` (with `Accept: application/json`).
The JSON manifest contains all providers, pricing, and `example_requests` showing the exact body structure.

## 3. Minimal API Flow

Use the manifest first, then query balance or proceed to top-up and prepaid usage.

```bash
# Discover providers, pricing, and example requests
curl -H "Accept: application/json" https://x402gate.io/

# Check prepaid balance
curl https://x402gate.io/v1/balance/YOUR_WALLET_ADDRESS
```

For prepaid signing details, top-up flow, and full Python examples, see `docs/prepaid.md`.

## 4. Documentation Links
If you need deep technical details on specific features:
- **Main Readme:** [github.com/oponfil/x402gate](https://github.com/oponfil/x402gate)
- **Prepaid Mode (Detailed API):** [docs/prepaid.md](https://github.com/oponfil/x402gate/blob/main/docs/prepaid.md)
- **Speech-to-Text (Whisper):** [docs/stt.md](https://github.com/oponfil/x402gate/blob/main/docs/stt.md)
- **Adding Providers / Architecture:** [docs/architecture.md](https://github.com/oponfil/x402gate/blob/main/docs/architecture.md)
- **x402 Protocol Specification:** [x402.org](https://x402.org)

## 5. Key Agent Rules
- **Pay on success**: If the provider fails (HTTP 5xx), the payment signature is never collected. Do not hesitate to use the API.
- **Do not invent endpoints**: Always refer to the discovery manifest (`GET /`).
- **File Uploads**: For `cloudconvert`, use standard `multipart/form-data`. For OpenRouter STT, send JSON with base64-encoded audio in `input_audio.data`.

## Tips

- Prefer prepaid mode for repeated requests; it avoids an on-chain transaction per call.
- Use the manifest `example_requests` as the source of truth for request bodies.
- Do not send provider-specific API keys to `x402gate`; x402 payment is the access mechanism.
- If you need a specific provider schema, follow the corresponding `provider_docs` link from the manifest.
