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
1. **LLMs:** OpenRouter (300+ models, web search, embeddings, STT/Whisper).
2. **Image & Video:** WaveSpeed (60+ models like Flux, Sora), SpicyAPI (Flux, Wan, Seedance, Kling, and other image/video models via `POST /v1/spicyapi/jobs`; chat models via `POST /v1/spicyapi/chat/completions`).
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
- **Pay on success**: If the provider reports a failure, your settled payment (Base or Solana) is credited to your prepaid balance. On Base, if the payment is still unconfirmed, the error carries `settlement_pending: true` and `pending_refund`, and the refund is credited once the transaction confirms (if the transfer already reverted or expired during the job, nothing was charged and those fields are absent); check `GET /v1/balance/{address}` later. On Base the result is returned only after your payment lands. Use a fresh payment for every request: reusing one that another request is still using returns HTTP 409. On Base, parallel requests from one wallet must be covered together: if the wallet's USDC balance is below the sum of its unsettled payments, the extra requests get HTTP 402 "Insufficient USDC balance: other payments from this wallet are still settling". Wait for earlier requests to finish (a payment whose settlement outcome is unknown counts until it confirms or its authorization expires), or use prepaid mode. On Base sign an EIP-3009 authorization; Permit2 payloads are refused with HTTP 402. If the outcome is unknown (a timeout, a dropped connection to the provider, a gateway timeout 504/520/524 on submit, or a job the provider accepted and then lost), the payment is kept and the error says `"outcome_unknown": true`. SocialDownload is charged even when RapidAPI cannot download the URL (unsupported or private), because RapidAPI bills every call it answers, so check the URL before you send it.
- **Long calls**: a call that runs longer than 20 seconds is kept open with newline heartbeats, and the status is already 200. Parse the body: a failure arrives there as error JSON with `status` and `outcome_unknown`.
- **Top-up**: a 502 with `outcome_unknown: true` credits nothing, but the payment may have gone through; check your wallet and `GET /v1/balance/{address}` before signing a new one. `"settlement_pending": true` on Solana means the balance was credited. On Base it comes with HTTP 202 and `"credited": "0"`: the `pending_credit` is added once the transaction confirms, so do not sign a new payment; check the balance later.
- **Do not invent endpoints**: Always refer to the discovery manifest (`GET /`).
- **File Uploads**: For `cloudconvert`, use standard `multipart/form-data` with one file. A 503 with `Retry-After` means the gateway is busy with other uploads; wait and retry. A 429 before payment (for example, STT audio being priced) also carries `Retry-After`. Every other provider takes JSON only and refuses multipart with 400 before payment. For OpenRouter STT, send JSON with base64-encoded audio in `input_audio.data` (wav, mp3, flac, ogg Opus/Vorbis/Speex, m4a/mp4/mov, aiff or ADTS aac; not webm).
- **Chat (OpenRouter, NanoGPT)**: Do not stream; `stream: true` returns HTTP 400. Always set `model`. Options the gateway cannot price before payment (non-text output, file or video parts, plugins other than `web`, native web search (`:online`, `search_parameters` unless `mode: "off"`), tools other than `function`/`custom`, deep research models, audio input on NanoGPT models without an audio rate, NanoGPT memory or priority tiers, NaN or Infinity values) also return HTTP 400 before any payment, so read the error and resend without them.

## Tips

- Prefer prepaid mode for repeated requests; it avoids an on-chain transaction per call.
- Use the manifest `example_requests` as the source of truth for request bodies.
- Do not send provider-specific API keys to `x402gate`; x402 payment is the access mechanism.
- If you need a specific provider schema, follow the corresponding `provider_docs` link from the manifest.
