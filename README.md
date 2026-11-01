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
  │                          ├─ forward request ────────►│
  │                          ├─ settle (on-chain, Base)  │
  │                          │◄─ result ────────────────┤
  │                          ├─ wait for settlement      │
  │◄─── 200 + result ───────┤                          │
```

On Solana the settle step runs right after verify, before the request is forwarded. A price quote that gets a 5xx is retried; a submit that reached the provider is never sent twice (see **Retries** below).

1. Client sends a request → gateway fetches the price from the provider
2. Gateway responds with **HTTP 402** and payment options for all configured networks
3. Client picks a network (Base or Solana), signs the payment, retries with `PAYMENT-SIGNATURE` header
4. Gateway verifies the payment, then forwards the request and returns the result. On Base it settles while the provider runs and returns the result only once the payment has landed. On Solana it settles first, because the signed transaction expires in about a minute.

On both networks the payment is settled by the time the provider answers, so a task the provider reports as failed credits that amount to the payer's prepaid balance. One payment buys one request: a payment that another request is still using gets HTTP 409. When the outcome is unknown (the task timed out, or the connection to the provider dropped after the request was sent), the payment is kept on both networks, because the provider may have run and billed the job. See [Error Handling](#error-handling).

### Chat (OpenRouter, NanoGPT)

OpenRouter takes `chat/completions`, `embeddings` and `audio/transcriptions`; NanoGPT takes `chat/completions`. The quote covers everything the model reads: messages, tools, response formats, images (a fixed allowance per image), and the reasoning budget (`reasoning.max_tokens`, `thinking.budget_tokens` or a NanoGPT `:thinking:<n>` suffix), which is added on top of `max_tokens` unless `max_tokens` already exceeds it. Text is counted by character class: plain English at about 2.7 characters per token, digits, Cyrillic, CJK and emoji at denser rates, so dense text cannot be quoted below what the model bills. A request with a `cache_control` breakpoint (or, on NanoGPT, `caching` / `prompt_caching`) is quoted at the model's cache write rate for the whole prompt, which is above the normal input rate on some models (1.25x on Anthropic). With a `models` fallback list, each rate in the quote (input, output, image, per request) is the highest that rate among the listed models, since OpenRouter may route to any of them. The gateway does not stream and returns text only, so `stream: true`, non-text `modalities` or an `audio` output field, and options it cannot price before payment get HTTP 400 before any payment. These include a missing `model`, router models without a fixed price (such as `openrouter/auto`), file or video content parts, OpenRouter plugins other than `web`, `web_search_options` and web engines other than Exa, NanoGPT web search (`:online`, `web_search`, `linkup`), memory, scraping, YouTube transcripts and the advisor model, and NanoGPT `service_tier` values other than `auto`, `default` or `flex`. A reasoning budget that is not a non-negative integer, or a `:thinking:<n>` budget longer than 9 digits, is also refused with 400. When the request sets OpenRouter routing preferences (`provider.order`, `only`, `ignore`, `sort`, `zdr: true`, `data_collection: "deny"` and so on), the gateway adds `provider.max_price` at the quoted rates, so a narrowed choice of endpoints cannot cost more than the quote; if no endpoint fits, the request fails before it runs and is refunded.

### File Conversion (CloudConvert)

CloudConvert uses **multipart/form-data** for file uploads. Supports 200+ formats (DOCX→PDF, PNG→JPG, etc.) and PDF optimization. Send the file as a multipart file field (for example `file`), plus `output_format` (required for `convert`), and optionally `operation` (`convert`, the default, or `optimize`) and `input_format`. A request with no file, an unknown `operation`, or a convert without `output_format` gets HTTP 400 before any payment.

### SpicyAPI (image, video, and chat)

Every SpicyAPI image and video model uses one endpoint. Put the catalog model id in `model` and that model's fields in `input`. The gateway quotes the exact request before payment, then runs the job.

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

Chat uses the same quote, then returns one OpenAI completion. The gateway does not stream. Set `max_tokens`; if you omit it, the gateway uses its default cap. `reasoning_effort` changes the quote, because the model reserves thinking tokens before it writes.

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

Settlement runs before the provider call (Solana) or alongside it (Base), so `cost` and `profit` in this line are the pre-payment estimate. The actual cost, computed from real usage after the response, goes to the dashboard's revenue and profit figures:

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

**Unknown outcome:** money goes back (to the prepaid balance for x402 payments) only when the provider itself reported the failure: an error status, a task marked failed, or a connection that never opened. If the gateway cannot tell how the job ended, the payment is kept, because the provider may have finished the job and billed it. That covers a task that does not finish within the provider's `poll_timeout`, and a request that timed out or dropped after it was sent. It also covers a WaveSpeed or SpicyAPI job the provider accepted and then lost (still 404 after 20 seconds of polling), a poll that is redirected or refused with 401/402/403, a submit answered with 504, 520 or 524 or with a 200 that has no readable result or task, a SpicyAPI chat that breaks off after it has started writing, and a SocialDownload call that RapidAPI answers with 2xx but an error (an unsupported or private URL), because RapidAPI bills every 2xx. An x402 payment stays settled, and in prepaid mode the reserved estimate is charged. The error says so:

```json
{
  "error": "Task timed out after 900s",
  "status": 504,
  "provider": "wavespeed",
  "task_id": "abc123",
  "outcome_unknown": true
}
```

**Client disconnects:** a long job keeps running if the client closes the connection. It is charged or refunded by the same rules, as if the client had stayed, but the result cannot be delivered.

**Payment rejection reasons:** When payment verification fails (e.g. insufficient USDC, stale transaction, network mismatch), the 402 response includes the specific reason:

```json
{
  "error": "Payment verification failed: transaction_simulation_failed: Simulation failed: InstructionErrorCustom(1)",
  "status": 402
}
```

**Retries:** a request that starts a job is never sent twice once it reached the provider, because the first one may already be running and billing. It is retried only when the connection never opened. The exception is SpicyAPI job creation: it carries an Idempotency-Key, so after a server error or a timeout it is retried up to **2 times** with that same key, and SpicyAPI does not start a second job. A WaveSpeed or SpicyAPI price quote starts nothing, so it is also retried on a server error (502, 503, etc.), up to **2 times** with exponential backoff (2s → 4s). A WaveSpeed, SpicyAPI, OpenRouter or NanoGPT submit answered with 504, 520 or 524 (a gateway timeout in front of the provider) has an unknown outcome. For CloudConvert and SocialDownload, any 5xx is a known failure and is refunded. While a job is polled, rate limits (429) and server errors are waited out until the provider's `poll_timeout`; WaveSpeed and SpicyAPI also wait out a briefly missing task (404) and back off up to 30s between polls. Client errors (4xx) are never retried. An upstream 401, 402 or 403 reaches the client as 502 (a SpicyAPI balance error as 503), so it is not mistaken for an x402 payment request.


> **Note:** On Base, USDC moves while the provider runs, and the result is returned only if that settlement succeeded; otherwise the response is 402. On Solana, USDC moves before the provider is called. The gateway rebroadcasts that same signed transaction until it lands. If the blockhash expires first, the response is 402 and the client is not charged. If the provider then reports a failure, the error JSON includes `refunded_to_prepaid` and `prepaid_balance` — the full settled amount, credited in memory, not sent back on-chain.

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

> **Security:** Both networks verify the client's on-chain USDC balance **before** running generation. Solana uses transaction simulation; Base performs an explicit `balanceOf` RPC call, which must also cover the same wallet's other Base payments that have not settled yet. Clients with insufficient USDC receive HTTP 402 with the specific rejection reason (e.g. `transaction_simulation_failed: InstructionErrorCustom(1)` for insufficient funds on Solana, `Insufficient USDC balance` on Base, or `Insufficient USDC balance: other payments from this wallet are still settling` when parallel Base requests exceed the balance) — no free generation is possible. Because the gateway pays the Solana fee, it also refuses transactions with more than 2 signatures, no `SetComputeUnitLimit`, a compute-unit limit above 40,000, a compute-unit price above 100,000 microlamports, or the fee payer in an instruction (402 `transaction_simulation_failed: ...`). The standard x402 client (20,000 CU at 1 microlamport) is well inside these limits.

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
