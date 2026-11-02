# x402gate

🌐 [x402gate.io](https://x402gate.io/) | **Transparent x402 payment proxy for AI services.**

Pay per API call with USDC on **Base** or **Solana** — no accounts, no API keys, no subscriptions.

x402gate sits between your AI agent and AI service providers, adding [x402](https://x402.org) payment support. One gateway, multiple providers:

| Provider | Type | What | Endpoint |
|----------|------|------|----------|
| [OpenRouter](https://openrouter.ai) | Managed | 300+ LLMs, embeddings, STT (GPT, Claude, Whisper…) | `/v1/openrouter/...` |
| [NanoGPT](https://nano-gpt.com) | Managed | 600+ chat models, including uncensored and abliterated | `/v1/nanogpt/chat/completions` |
| [WaveSpeed AI](https://wavespeed.ai) | Managed | 60+ image/video models (Flux, Wan, Kling, Sora…) | `/v1/wavespeed/...` |
| [SpicyAPI](https://spicyapi.ai) | Managed | Image, video, and chat models (Flux, Wan, Seedance, DeepSeek, Grok…) | `/v1/spicyapi/jobs`, `/v1/spicyapi/chat/completions` |
| [CloudConvert](https://cloudconvert.com) | Managed | File conversion (200+ formats) | `/v1/cloudconvert/...` |
| [SocialDownload](https://rapidapi.com/nguyenmanhict-MuTUtGWD7K/api/social-download-all-in-one) | Managed | Download media from social networks | `/v1/socialdownload/...` |
| [ElevenLabs](https://elevenlabs.io) | Managed | Premium TTS (1000+ voices, Multilingual v2, Flash) | `/v1/elevenlabs/...` |
| [MiniMax](https://www.minimax.io) | Managed | High-quality TTS with emotional expression | `/v1/minimax/...` |
| [Fish Audio](https://fish.audio) | Managed | Uncensored TTS (Fish Speech) | `/v1/fishaudio/...` |

## How It Works

### Managed Provider (WaveSpeed)

```
Client                    x402gate                  WaveSpeed AI
  │                          │                          │
  ├─ POST /v1/wavespeed/... ─►                          │
  │                          ├─ GET price ──────────────►│
  │                          │◄─ $0.005 ────────────────┤
  │◄─── 402 ($0.00620) ─────┤  total = provider + 4% + $0.001 gas
  │                          │                          │
  ├─ POST + PAYMENT-SIGNATURE►                          │
  │                          ├─ verify (on-chain)        │
  │                          ├─ settle (on-chain)        │
  │                          ├─ forward request ────────►│
  │                          │◄─ result ────────────────┤
  │◄─── 200 + result ───────┤                          │
```

On both networks the settle step runs right after verify, before the request is forwarded. A price quote that gets a 5xx is retried; a submit that reached the provider is never sent twice (see **Retries** below). Only CloudConvert takes multipart uploads: any other managed provider answers a multipart body with HTTP 400 before payment, without reading it. WaveSpeed's `enable_sync_mode` is dropped: the gateway always polls. A JSON body (or `/v1/topup` body) with an integer longer than 40 characters gets HTTP 400 `Invalid JSON body` before pricing.

1. Client sends a request → gateway fetches the price from the provider
2. Gateway responds with **HTTP 402** and payment options for all configured networks
3. Client picks a network (Base or Solana), signs the payment, retries with `PAYMENT-SIGNATURE` header
4. Gateway verifies and settles the payment, then forwards the request and returns the result. If the settlement fails, the response is HTTP 402 and the provider is not called.

On both networks the payment is settled before the provider is called, so a task the provider reports as failed credits that amount to the payer's prepaid balance. One payment buys one request: a payment that another request or a top-up is still using gets HTTP 409. When the outcome is unknown (the task timed out, or the connection to the provider dropped after the request was sent), the payment is kept on both networks, because the provider may have run and billed the job. See [Error Handling](#error-handling).

### Chat (OpenRouter, NanoGPT)

OpenRouter takes `chat/completions`, `embeddings` and `audio/transcriptions`; NanoGPT takes `chat/completions`. The quote covers everything the model reads: messages, tools, response formats, images (a fixed allowance per image), and the reasoning budget (`reasoning.max_tokens`, `thinking.budget_tokens`, or an all-digit NanoGPT model suffix such as `:thinking:<n>` or `-thinking:<n>`), which is added on top of `max_tokens` unless `max_tokens` already exceeds it. Text is counted by character class: plain English at about 2.7 characters per token, digits, Cyrillic, CJK and emoji at denser rates, so dense text cannot be quoted below what the model bills. A request with a `cache_control` breakpoint (or, on NanoGPT, `caching` / `prompt_caching`) is quoted at the model's cache write rate for the whole prompt, which is above the normal input rate on some models (1.25x on Anthropic); a `cache_control` `ttl` other than `5m` (such as `1h`) is quoted at 2x the input rate. Output is quoted at the higher of the model's completion and internal reasoning rates. Long-context rates come from the catalogs. On OpenRouter the quote uses the model's `pricing.overrides` tier with the highest `min_prompt_tokens` the estimated prompt reaches (for example GPT-5.4+ from 272k tokens, Claude Haiku 5.5 from 100k), and any other override, such as a time window, at the higher of its rate and the base rate; models without overrides are flat. On NanoGPT it uses the rates NanoGPT lists above `highContextThresholdTokens`, and Gemini 2.5/3.x Pro prompts above 200k tokens are quoted at 2x input and 1.5x output. Models that search the web on every request (`perplexity/*`, `*-search-preview`) have 5 searches per completion added to the quote, and deep research models (`deep-research` in the id) get HTTP 400. Embeddings input is counted with the same character weights as chat. With a `models` fallback list, each rate in the quote (input, output, image, per request) is the highest that rate among the listed models, after each model's own long-context tier, since OpenRouter may route to any of them. The gateway does not stream and returns text only, so `stream: true`, non-text `modalities` or an `audio` output field, and options it cannot price before payment get HTTP 400 before any payment. These include a missing `model`, router models without a fixed price (such as `openrouter/auto`), file or video content parts, OpenRouter plugins other than `web`, `web_search_options` and web engines other than Exa, the OpenRouter `:online` model variant (use the `web` plugin), NanoGPT web search (`:online`, `web_search`, `linkup`), any tool other than a `function` or `custom` tool (such as `web_search` or `code_interpreter`, on both providers), xAI `search_parameters` unless `mode` is `"off"`, memory, scraping, YouTube transcripts and the advisor model, and NanoGPT `service_tier` values other than `auto`, `default` or `flex`. A reasoning budget that is not a non-negative integer, a model-suffix budget longer than 9 digits, NaN or Infinity, a lone surrogate (`\ud800`-`\udfff`) in a string or key, a content part `type` that is not a string, and a body nested too deeply to serialize are also refused with 400. When the request sets OpenRouter routing preferences (`provider.order`, `only`, `ignore`, `sort`, `zdr: true`, `data_collection: "deny"` and so on), the gateway adds `provider.max_price` at the models' listed rates, so a narrowed choice of endpoints cannot cost more than the quote; if no endpoint fits, the request fails before it runs and is refunded. Audio input parts (`input_audio`) are decoded and timed before payment like an STT upload, and quoted at 50 tokens per second at the higher of the model's audio and input rates; those tokens count toward the long-context tier. Raw `pcm16`/`pcm24` is timed as 8 kHz mono, audio that cannot be timed gets 400, each part over `max_stt_audio_mb` gets 413, and busy audio pricing slots give 429. On NanoGPT only models whose catalog lists `pricing.audio_input` take audio; others get 400.

### File Conversion (CloudConvert)

CloudConvert uses **multipart/form-data** for file uploads. Supports 200+ formats (DOCX→PDF, PNG→JPG, etc.) and PDF optimization. Send the file as a multipart file field (for example `file`), plus `output_format` (required for `convert`), and optionally `operation` (`convert`, the default, or `optimize`) and `input_format`. A request with no file, an unknown `operation`, or a convert without `output_format` gets HTTP 400 before any payment. One file per request and at most 20 form fields (more gets 400). The size limit (`max_upload_mb`) is checked while the body is read, so an oversized upload gets 413 even without `Content-Length`. If unpaid uploads in memory would pass their budget, the gateway answers 503 with `Retry-After: 5`; an upload that takes longer than 600 seconds to arrive gets 408.

### SpicyAPI (image, video, and chat)

Every SpicyAPI image and video model uses one endpoint. Put the catalog model id in `model` and that model's fields in `input`. The gateway quotes the exact request before payment, then runs the job. If SpicyAPI quotes more at submit than the price you paid, the job is refused with 409 (`price increased`) and refunded; retry to pay the new price.

```
POST /v1/spicyapi/jobs
```

Image:

```bash
curl -X POST https://x402gate.io/v1/spicyapi/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "alibaba/z-image-spicy/text-to-image",
    "input": {
      "prompt": "A ceramic teapot on a linen cloth, soft window light, muted palette",
      "enable_prompt_expansion": false
    }
  }'
```

Video uses the same path. The gateway polls until the clip is ready, up to 15 minutes, and sends whitespace heartbeats while it waits so the connection is not closed for being idle:

```bash
curl -X POST https://x402gate.io/v1/spicyapi/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bytedance/seedance-1.5-pro/text-to-video",
    "input": {
      "prompt": "A cat slowly walking across a sunlit room, cinematic",
      "aspect_ratio": "16:9",
      "resolution": "480p",
      "duration_seconds": 4,
      "generate_audio": false
    }
  }'
```

Model ids come from the [SpicyAPI catalog](https://spicyapi.ai). A finished job returns the file URL in `output.assets`.

Chat uses the same quote, then returns one OpenAI completion. The gateway does not stream, and the chat must finish within SpicyAPI's `poll_timeout`. Set `max_tokens`; if you omit it, the gateway uses its default cap. `reasoning_effort` changes the quote, because the model reserves thinking tokens before it writes.

```bash
curl -X POST https://x402gate.io/v1/spicyapi/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/v4.1-flash/chat",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 256,
    "reasoning_effort": "low"
  }'
```

### Transaction Summary

After each settled payment, the gateway logs a one-line summary:

```
TX wavespeed settled: $0.0062 -> cost=$0.005 gas=$0.000855 SOL profit=$0.000345 (0.0s -> 0.0s)
```

Settlement runs before the provider call, so `cost` and `profit` in this line are the pre-payment estimate. The actual cost, computed from real usage after the response, goes to the dashboard's revenue and profit figures:

- **cost** (log line) — the estimate the client paid for
- **Actual cost** (dashboard) — from the provider's reported usage (`usage.cost` or token counts)
- **Net profit** (dashboard) — calculated from the actual provider cost, not the estimate

Gas costs are fetched in real-time from the chain. Native token prices (ETH, SOL) come from CoinGecko.

### Error Handling

If the provider rejects a request (e.g. invalid parameters), the gateway returns a structured error. A prepaid hold is released; a settled x402 payment is credited to the payer's prepaid balance (`refunded_to_prepaid` in the error):

```json
{
  "error": "size does not meet the constraints.",
  "provider": "wavespeed",
  "status": 502
}
```

**Unknown outcome:** money goes back (to the prepaid balance for x402 payments) only when the provider itself reported the failure: an error status, a task marked failed, or a connection that never opened. If the gateway cannot tell how the job ended, the payment is kept, because the provider may have finished the job and billed it. That covers a task that does not finish within the provider's `poll_timeout`, and a request that timed out or dropped after it was sent. It also covers a WaveSpeed or SpicyAPI job the provider accepted and then lost (still 404 after 20 seconds of polling), a poll that is redirected or refused with 401/402/403, a submit answered with 504, 520 or 524 or with a 200 that has no readable result or task, a SpicyAPI chat that breaks off after it has started writing, runs past `poll_timeout`, or whose stream gets 504, 520 or 524, a MiniMax 200 that is not a JSON object or reports success without usable audio, and a SocialDownload call that RapidAPI answers with 2xx but an error (an unsupported or private URL), because RapidAPI bills every 2xx. An x402 payment stays settled, and in prepaid mode the reserved estimate is charged. The error says so:

```json
{
  "error": "Task timed out after 900s",
  "status": 504,
  "provider": "wavespeed",
  "task_id": "abc123",
  "outcome_unknown": true
}
```

An OpenRouter or NanoGPT 2xx that carries only an `error` object, with no usage or usage that reports no cost and no tokens, is a known failure and is refunded, unless the error is a timeout. Its status comes from `error.code` (401/402/403 and unusable codes become 502).

**Pricing errors** come before any payment, so nothing is charged. A 429 (for example, all audio parse slots busy) carries `Retry-After: 2`, and a request the gateway itself fails to price gets 400 `Could not price this request`.

**Long calls:** any managed call, synchronous (chat, TTS, STT) or polled, that runs longer than 20 seconds, settlement included, is kept open with newline heartbeats. The status is then already 200 and the `X-Timing` and `X-Prepaid-*` headers are absent; a failure arrives as the usual error JSON in the body, with `status`, `outcome_unknown` and, in prepaid mode, `prepaid`. A settlement that fails after the heartbeats started arrives the same way, with `status: 402`. Read the body, not only the status.

**Client disconnects:** a long job keeps running if the client closes the connection, even during settlement. It is charged or refunded by the same rules, as if the client had stayed, but the result cannot be delivered.

**Payment rejection reasons:** When payment verification fails (e.g. insufficient USDC, stale transaction, network mismatch), the 402 response includes the specific reason:

```json
{
  "error": "Payment verification failed: transaction_simulation_failed: Simulation failed: InstructionErrorCustom(1)",
  "status": 402
}
```

On Base only EIP-3009 authorizations are accepted; a Permit2 payload gets 402 `Payment verification failed: unsupported_payload: Permit2 is not accepted`.

**Base payment still pending:** a Base settlement that was sent but not confirmed in time still runs the job. If the provider then reports a failure, the error carries `settlement_pending: true`, `pending_refund` (the full price) and `transaction`, and the refund is credited to prepaid once the transaction's receipt shows the transfer (or at once, as `refunded_to_prepaid`, if it landed during the job). These three fields appear only while the gateway is still following the receipt: if the transfer already reverted or its authorization expired during the job, nothing was charged and the error carries none of them. A revert, an expiry or a gateway restart after the error credits nothing. Check `GET /v1/balance/{address}` later.

**Retries:** a request that starts a job is never sent twice once it reached the provider, because the first one may already be running and billing. It is retried only when the connection never opened. The exception is SpicyAPI job creation: it carries an Idempotency-Key, so after a server error or a timeout it is retried up to **2 times** with that same key, and SpicyAPI does not start a second job. A WaveSpeed or SpicyAPI price quote starts nothing, so it is also retried on a server error (502, 503, etc.), up to **2 times** with exponential backoff (2s → 4s). A WaveSpeed, SpicyAPI, OpenRouter, NanoGPT, ElevenLabs, MiniMax, Fish Audio or SocialDownload submit answered with 504, 520 or 524 (a gateway timeout in front of the provider) has an unknown outcome; the payment is kept and the client gets 504. For CloudConvert, any failure while creating the job (a 5xx, a timeout or a dropped connection) or before the whole file was uploaded is a known failure and is refunded; after the whole file was sent, a lost answer or a 504, 520 or 524 from the upload host keeps the payment (504). While a job is polled, rate limits (429) and server errors are waited out until the provider's `poll_timeout` (wall-clock time, with one last poll at the deadline); WaveSpeed and SpicyAPI also wait out a briefly missing task (404) and back off up to 30s between polls. Client errors (4xx) are never retried. An upstream 401, 402 or 403 reaches the client as 502 with a fixed message, without the upstream body (a SpicyAPI balance error as 503), so it is not mistaken for an x402 payment request. A 3xx at submit is refunded as 502.


> **Note:** On both networks USDC moves before the provider is called; if that settlement fails, the response is 402 and the client is not charged. On Solana the gateway rebroadcasts the same signed transaction until it lands. If the blockhash expires first, or the transaction lands with an error, the response is 402 and the client is not charged. If the provider then reports a failure, the error JSON includes `refunded_to_prepaid` and `prepaid_balance` — the full settled amount, credited in memory, not sent back on-chain.

### Prepaid Mode

For high-frequency usage, you can top-up a prepaid balance and skip on-chain transactions per request:

```
Client                    x402gate
  │                          │
  ├─ POST /v1/topup (402) ───►  ← pay once (Solana or Base USDC)
  ├─ POST /v1/topup + sig ──►  ← balance credited
  │                          │
  ├─ POST /v1/openrouter/... ►  ← wallet signature (no blockchain tx)
  │  X-PREPAID-PUBKEY         │  ← reserve estimate, then charge actual cost
  │  X-PREPAID-SIGNATURE      │
  │  X-PREPAID-TIMESTAMP      │
  │◄─── 200 + result ────────┤
```

Commission and gas are charged **once at top-up**. Subsequent requests deduct the provider's actual cost, up to the estimate reserved before the call. OpenRouter and NanoGPT chat use the same `max_tokens` ceiling as a direct x402 payment, so a prepaid request cannot run past the reserved amount. If the provider reports a failure, the reservation is returned to the balance. If the outcome is unknown, the reservation is charged.

Supports both **Solana** (Ed25519) and **Base/EVM** (EIP-191) wallets — auto-detected by address format.

A Base top-up must be covered by the wallet's USDC on top of its other unsettled Base payments, or it gets 402 before settling. If a top-up's settlement was sent but not confirmed, the response has `"settlement_pending": true`. On Solana the balance is credited at once. On Base the response is HTTP 202 with `"credited": "0"` and `pending_credit`, and the credit is added once the transaction's receipt shows the transfer; a reverted transfer, an authorization that expires first, or a gateway restart meanwhile credits nothing. A top-up whose settle crashed returns 502 with `outcome_unknown: true` and credits nothing. Either way, check your wallet and balance before signing a new payment.

> **⚠️ Warning:** Prepaid balances are stored in memory only and are lost on server restart. Top-up: $0.10 – $10. See [docs/prepaid.md](docs/prepaid.md) for full details.

### Service Discovery

The root endpoint `GET /` uses content negotiation:

- **Browser** (`Accept: text/html`) → human-readable HTML page with docs, examples, and links
- **AI agent / curl** (`Accept: */*` or `application/json`) → machine-readable JSON manifest

```bash
# HTML (as browser sees it)
curl -H "Accept: text/html" https://x402gate.io/

# JSON (for AI agents)
curl https://x402gate.io/
```

### Dashboard

x402gate includes a built-in dashboard at `/dashboard` with real-time provider status, request metrics, remaining provider balances, revenue tracking, network settlement stats, prepaid balance, and live logs. A balance turns red under $20, under 300 CloudConvert credits, or under 10,000 ElevenLabs characters. OpenRouter's account balance needs `OPENROUTER_MANAGEMENT_KEY`. NanoGPT shows the USD balance on its API key. MiniMax shows its dollar balance; at $0 it shows `$0 + points?` without red, because requests are then paid from Audio Package points that MiniMax does not report.

| Endpoint | Description |
|----------|-------------|
| `GET /dashboard` | HTML dashboard with auto-refresh (5s) |
| `GET /v1/stats` | JSON snapshot: uptime, per-provider metrics, provider balances, revenue/cost/profit, network stats, prepaid balance |
| `GET /v1/logs?limit=200` | Recent log entries (newest first, default 200) |

Statistics are stored **in-memory** and reset on server restart. The log buffer retains the last 1 000 entries (configurable via `MAX_LOG_ENTRIES` in `x402gate/core/stats.py`). Providers that haven't received requests in 4 hours are marked as `unknown`.

## Supported Networks

| Network | Chain ID | Token | Gas Token | Payment Overhead |
|---------|----------|-------|-----------|-----------------|
| **Base** (EVM) | `eip155:8453` | USDC | ETH | ~2.4s |
| **Solana** (SVM) | `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp` | USDC | SOL | ~0.5s |

Both networks are offered simultaneously — the client chooses which to pay on. Payment overhead is the additional latency added by on-chain verification and settlement (on top of the AI provider's generation time).

> **Security:** Both networks verify the client's on-chain USDC balance **before** running generation. Solana uses transaction simulation; Base performs an explicit `balanceOf` RPC call, which must also cover the same wallet's other Base payments that have not settled yet (a settlement with an unknown outcome counts until its receipt resolves or its authorization expires). Clients with insufficient USDC receive HTTP 402 with the specific rejection reason (e.g. `transaction_simulation_failed: InstructionErrorCustom(1)` for insufficient funds on Solana, `Insufficient USDC balance` on Base, or `Insufficient USDC balance: other payments from this wallet are still settling` when parallel Base requests exceed the balance) — no free generation is possible. Because the gateway pays the Solana fee, it also refuses transactions with more than 2 signatures, no `SetComputeUnitLimit`, a compute-unit limit above 40,000, a compute-unit price above 100,000 microlamports, or the fee payer in an instruction (402 `transaction_simulation_failed: ...`). The standard x402 client (20,000 CU at 1 microlamport) is well inside these limits.

## Quick Start

### 1. Install

```bash
git clone https://github.com/oponfil/x402gate.git
cd x402gate
pip install -r requirements-dev.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API keys and wallet addresses. See [`.env.example`](.env.example) for all available variables.

> **Note:** Facilitator wallets need a small balance of native tokens (ETH on Base, SOL on Solana) to pay gas for settlement transactions.
>
> **Startup validation:** x402gate validates `${ENV_VAR}` placeholders from active config during boot and exits immediately if any required payment secret or secret for an enabled provider is missing. Disabled providers (`enabled: false`) may keep unresolved placeholders until they are enabled. A broken active config should fail the deploy, not wait until the first user request.

### 3. Run

```bash
python -m x402gate.main
```

The gateway starts on `http://localhost:4021`. Production: [https://x402gate.io](https://x402gate.io)

### 4. Test

```bash
# List providers and their endpoints
curl https://x402gate.io/v1/providers

# Send a request to any provider (returns 402 with price)
curl -X POST https://x402gate.io/v1/{provider}/{model_path} \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

> **Tip:** Each provider has `example_request` entries in `config.yaml` (e.g. `example_request`, `example_request_2`) with working models and bodies. The HTML landing page at [https://x402gate.io](https://x402gate.io) shows ready-to-use curl examples generated from the config.

## Configuration

All settings are in `config.yaml`. Secrets use `${ENV_VAR}` interpolation, and missing referenced variables in active config fail startup immediately. Disabled providers may keep unresolved placeholders until enabled. See [docs/configuration.md](docs/configuration.md) for the full reference.

> [!IMPORTANT]
> **Changing a config parameter?** Update in **2 places**:
> 1. `config.yaml` — runtime value
> 2. `x402gate/core/config.py` — default in the `GatewayConfig` / `ProviderConfig` model
>
> The landing page (`index.html`) reads all values from config automatically via Jinja2.

## E2E Tests

Run end-to-end tests on mainnet (requires funded wallets):

```bash
# Run against local server (starts x402gate on port 4022 automatically)
python -m pytest tests/e2e/ -v -s

# Run against production
python -m pytest tests/e2e/ --prod -v -s

# Run against a custom gateway
GATEWAY_URL=https://staging.example.com/ python -m pytest tests/e2e/ -v -s
```

Tests are grouped into three categories:

- **Error & Validation** — bad params, invalid models, topup limits (no funds spent)
- **Success** — on-chain payments across all providers on Base and Solana
- **Prepaid** — top-up + multi-provider flows (EVM and Solana wallets)

Requires `BASE_E2ETEST_PRIVATE_KEY` and/or `SOLANA_E2ETEST_PRIVATE_KEY` in `.env`.

## Adding Providers

x402gate supports two provider types:

- **Managed** — x402gate handles pricing, payment verification, and settlement (e.g., WaveSpeed)
- **Passthrough** — transparent proxy for x402-native providers; the client pays that provider directly

See [docs/add-provider.md](docs/add-provider.md) for a step-by-step guide.

## Documentation

- [Architecture](docs/architecture.md) — how x402gate works under the hood
- [Prepaid Mode](docs/prepaid.md) — top-up and pay without on-chain transactions
- [Dashboard](docs/dashboard.md) — real-time metrics, logs, and revenue tracking
- [Deployment](docs/deployment.md) — Railway, Docker, VPS guides
- [Configuration](docs/configuration.md) — all config options
- [TTS Providers](docs/tts.md) — ElevenLabs, MiniMax, Fish Audio
- [STT (OpenRouter)](docs/stt.md) — Whisper and speech-to-text transcription
- [SpicyAPI](https://docs.spicyapi.ai/docs/tasks) — image and video jobs (`POST /v1/spicyapi/jobs`) and chat (`POST /v1/spicyapi/chat/completions`)
- [SocialDownload](docs/socialdownload.md) — download media from social networks
- [Adding a Provider](docs/add-provider.md) — extend with new AI services
- [AI Agent Skill](.agents/skills/x402gate/SKILL.md) — public skill for autonomous agents to use `x402gate.io` (OpenClaw, etc.)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, style guide, and PR process.

## License

[MIT](LICENSE)
