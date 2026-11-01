# Architecture

x402gate is a transparent payment proxy that sits between AI agents/clients and AI service providers, enforcing payments via the [x402 protocol](https://x402.org).

## Overview

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│              │     │      x402gate         │     │   Provider   │
│  AI Agent /  │────►│                      │────►│  (WaveSpeed,  │
│   Client     │◄────│  FastAPI + x402 flow │◄────│  OpenRouter…) │
│              │     │                      │     │              │
└──────────────┘     └──────────┬───────────┘     └──────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   Facilitator     │
                     │  (Coinbase x402)  │
                     │                   │
                     │ Verify & Settle   │
                     │ on Base or Solana │
                     └───────────────────┘
```

## Request Flow

### 1. First Request (No Payment)

```
Client → POST /v1/wavespeed/{model}
           ↓
   x402gate → fetch price from WaveSpeed Pricing API
           ↓
   x402gate → apply 4% commission
           ↓
   Client ← 402 Payment Required (price: $0.00315)
```

### 2. Second Request (With Payment)

```
Client → POST /v1/wavespeed/{model} + PAYMENT-SIGNATURE header
           ↓
   x402gate → verify payment via Facilitator
           ↓
   x402gate → forward request to WaveSpeed as-is,
              settling the payment at the same time
           ↓
   x402gate → poll for async task result
           ↓
   x402gate → wait for the settlement to land
           ↓
   Client ← 200 OK + result
```

The diagram is the Base order. If the settlement fails, the client gets 402 instead of the result. On Solana, settle runs immediately after verify and before the request is forwarded.

## Components

| Component | File | Purpose |
|---|---|---|
| **Config** | `x402gate/core/config.py` | Load `config.yaml`, interpolate env vars |
| **Pricing** | `x402gate/core/pricing.py` | TTL cache, commission, USDC formatting |
| **Payment** | `x402gate/core/payment.py` | Build 402 responses, verify/settle via facilitator |
| **Prepaid** | `x402gate/core/prepaid.py` | In-memory prepaid balances, wallet signature verification |
| **Stats** | `x402gate/core/stats.py` | Dashboard statistics, log capture, revenue tracking |
| **Balances** | `x402gate/core/balances.py` | Upstream account balances shown on provider cards |
| **Proxy** | `x402gate/core/proxy.py` | Forward requests, poll async tasks |
| **Audio Duration** | `x402gate/core/audio_duration.py` | Parse audio file duration for STT pricing |
| **Provider Base** | `x402gate/providers/base.py` | Abstract provider interface |
| **WaveSpeed** | `x402gate/providers/wavespeed.py` | WaveSpeed AI (60+ image/video models) |
| **OpenRouter** | `x402gate/providers/openrouter.py` | OpenRouter aggregator (LLM, embeddings, STT) |
| **NanoGPT** | `x402gate/providers/nanogpt.py` | NanoGPT chat catalog (prices are USD per million tokens) |
| **SpicyAPI** | `x402gate/providers/spicyapi.py` | Image and video jobs (quote, create, poll) and quoted chat (`jobs/stream`) |
| **CloudConvert** | `x402gate/providers/cloudconvert.py` | File conversion (200+ formats, multipart upload) |
| **SocialDownload** | `x402gate/providers/socialdownload.py` | Social media download via RapidAPI |
| **ElevenLabs** | `x402gate/providers/elevenlabs.py` | Premium text-to-speech |
| **Fish Audio** | `x402gate/providers/fishaudio.py` | Text-to-speech with voice cloning |
| **MiniMax** | `x402gate/providers/minimax.py` | Text-to-speech with emotional expression |
| **Tungsten** | `x402gate/providers/tungsten.py` | Image generation. Disabled in config (`enabled: false`) |
| **App** | `x402gate/app.py` | FastAPI routes, lifecycle, provider registry |
| **Entry** | `x402gate/main.py` | Uvicorn server startup |

## Key Design Decisions

### Dynamic Route Registration

Provider routes (`/v1/{provider}/{path}`) are generated automatically from `config.yaml` at startup. A provider with `enabled: false` is not routed and is omitted from the public manifest. Tungsten and the BlockRun passthrough are disabled this way. Adding a passthrough provider requires only a config entry — no code changes. Adding a managed provider requires a config entry plus a provider class registered in `PROVIDER_REGISTRY`.

### Upstream Connections

Managed providers share one pooled HTTP client (`x402gate/core/http.py`), so a request reuses an open TLS connection to the provider instead of a new handshake (50-200 ms) per call. Idle connections are kept 30 seconds. Connecting and waiting for a free connection give up after 10 seconds, which means nothing was sent, so the request is refunded. The pool stores no cookies. The BlockRun passthrough has its own cookie-free pool, so one user's upstream cookies never reach another user. At startup each provider loads its catalog in the background (`warm_up`), and OpenRouter and NanoGPT keep serving an expired catalog while one background task refreshes it, so no paid request waits for a catalog.

### Transparent Proxy

The gateway forwards request bodies to providers **as-is**, with two exceptions. Before payment it checks only what it needs to price the request and refuses, with HTTP 400, anything it cannot price or that can never succeed (for chat: `stream: true`, unpriced options, a missing `model`). For chat it also adds caps that keep the cost within the quote (`max_tokens`, `provider.max_price`, web plugin limits). Keys the gateway adds itself (`_file_bytes`, `_file_name`, `_caller`) are dropped from client bodies. Otherwise it does not modify model-specific parameters. This means:

- Adding a new model requires zero gateway changes
- Request/response schemas are the provider's responsibility
- The gateway only cares about: pricing, payment, and task lifecycle

### Dynamic Pricing

Prices are fetched from the provider's pricing API at request time. A TTL cache (`price_cache_ttl` in config) reduces redundant API calls. The same cache key is used for both the 402 response and the payment verification retry.

### Settlement

Payment is verified before any work.

On Base the client signs an EIP-3009 authorization that stays valid for `max_timeout` (3600 seconds). The gateway settles it at the same time as it calls the provider, and returns the result only after the settlement succeeds. A payer who moves the USDC or cancels the authorization during the job gets a 402 instead of the result. If the provider reports a failure after the settlement landed, the amount is credited to the payer's in-memory prepaid balance, as on Solana. Settling first would add the settlement time (about 1-2 seconds) to every request; running both together adds it only to jobs that finish faster than the settlement.

One payment buys one request. Parallel requests with the same payment all pass verify, because nothing has landed on-chain yet, so `PaymentClaims` in `x402gate/core/payment.py` lets one request hold a payment at a time (keyed by payer and nonce on Base, by the transaction bytes on Solana). The others get HTTP 409. The claim lasts until the settlement finishes, plus 60 seconds on Base and 180 seconds on Solana after a success, so an RPC node that lags behind cannot let a replay through. On Base each authorization has its own nonce, so the balance check also counts the USDC that the same wallet's other unsettled requests have promised (`InFlightAmounts`); otherwise one balance could start many paid provider calls at once.

A settlement that was broadcast but whose confirmation wait failed (`settlement_pending`) is waited for once more. If it is still unconfirmed it most likely lands, so the result is returned (Base) or the job runs (Solana), and a provider failure is not credited. A settle that crashed is treated as failed for the result, but its payment stays claimed in case the transaction went out.

On Solana the client signs a transaction whose blockhash expires in about a minute. The gateway settles that transaction immediately after verify, before calling the provider. While it waits, it rebroadcasts the same signed bytes. If the blockhash expires before the transaction lands, settlement fails and the client is not charged. If the provider then reports a failure, the settled amount is credited to the payer's in-memory prepaid balance. The error JSON includes `refunded_to_prepaid` and `prepaid_balance`.

The gateway is the Solana fee payer, and the x402 SDK verifier caps only the compute-unit price, not the limit. Before signing, `SolanaSigner` therefore refuses a transaction that needs more than 2 signatures, has no `SetComputeUnitLimit`, asks for more than 40,000 compute units, bids more than 100,000 microlamports per unit, or lists the fee payer in any instruction's accounts. That bounds the priority fee to 4,000 lamports per payment. The client gets 402 `Payment verification failed: transaction_simulation_failed: <reason>`.

### Unknown Outcome

Funds go back only when the failure is known: the provider returned an error status, marked the task failed, or the connection never opened. When the gateway cannot tell how the job ended, it keeps the payment, because the provider may have run and billed the job. That happens when polling exceeds `poll_timeout`, or when a request times out or drops after it was sent. `outcome_unknown()` in `x402gate/providers/base.py` makes that call by walking the exception chain. Keeping the payment means not crediting the settled x402 payment back, or charging the prepaid reservation. The error JSON carries `"outcome_unknown": true`.

The same applies once the provider has accepted a job. For WaveSpeed and SpicyAPI, a poll that keeps getting 404 for more than 20 seconds keeps the payment; a CloudConvert 404 while polling is a known failure and is refunded. A redirect or an upstream 401/402/403 while polling also keeps the payment. So does a WaveSpeed, OpenRouter or NanoGPT submit answered with 504, 520 or 524, an OpenRouter or NanoGPT 5xx whose body reports a timeout, or an unreadable 200 (a WaveSpeed 200 with no task id, an OpenRouter or NanoGPT 2xx body that is not a JSON object). For SpicyAPI, a createTask where any attempt hit 504, 520 or 524 or timed out after it was sent keeps the payment, as does a 2xx answer that is unreadable or has no `taskId` (unless it carries a 4xx error code such as 40001 or 40201), and a chat stream that fails after it has produced output. A redirect at submit never reached the provider, so it is returned as 502 and refunded. RapidAPI bills every call it answers with 2xx, so a SocialDownload 200 that carries an error (an unsupported or private URL) is charged too. A non-`ProviderError` exception after payment is a gateway bug with an unknown outcome.

### Client Disconnects

A polled job runs in its own task, tracked in `_pending_jobs`. If the client leaves while heartbeats are going out, only the stream stops. The job finishes and is charged or refunded as if the client had stayed. On shutdown the gateway waits for these jobs before it waits for settlements.

### Custom x402 Flow

The standard `PaymentMiddlewareASGI` from the x402 library requires static route-to-price mappings defined at startup. Since our prices are dynamic per-request, we implement the x402 verify/settle protocol directly using the facilitator's HTTP API.

### Actual Cost Tracking

For token-based providers (OpenRouter chat/embeddings, NanoGPT chat), the gateway tracks both estimated and actual costs. OpenRouter catalog rates are USD per token. NanoGPT `pricing.prompt` and `pricing.completion` are USD per million tokens:

- **Estimated cost** — ceiling calculated upfront from `max_tokens × completion_price + input_tokens × prompt_price`
- **Actual cost** — for OpenRouter, the `usage.cost` OpenRouter reports, which already covers the endpoint that served the request, a `models` fallback, cache writes and plugins. For BYOK requests, or when `usage.cost` is missing, it is computed from `usage.prompt_tokens` and `usage.completion_tokens` at the catalog rates of the model that served the request (or the most expensive `models` candidate). NanoGPT is computed from real token counts, with cache reads and cache writes at their own catalog rates.

For OpenRouter STT (`audio/transcriptions`), the gateway supports duration-priced Whisper models. It parses audio length before payment, and uses `usage.cost` from OpenRouter for actual billing when available. See [stt.md](stt.md).

The x402 `exact` scheme requires settling the full signed amount, so the client always pays the ceiling price. Settlement runs before (Solana) or alongside (Base) the provider call, so the settlement log line shows the estimate. In x402 mode the actual cost is used only for the dashboard's revenue and profit stats; in prepaid mode it is what gets deducted, never more than the reservation. When x402 adds a `deferred` scheme (settle ≤ signed amount), the gateway is ready to use actual cost for settlement.
