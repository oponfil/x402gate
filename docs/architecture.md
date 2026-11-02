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

The gateway forwards request bodies to providers **as-is**, with two exceptions. Before payment it checks only what it needs to price the request and refuses, with HTTP 400, anything it cannot price or that can never succeed (for chat: `stream: true`, unpriced options, a missing `model`; for any JSON body: an integer longer than 40 characters, which gets `Invalid JSON body`, since Python parses and prints long integers in quadratic time). For chat it also adds caps that keep the cost within the quote (`max_tokens`, `provider.max_price`, web plugin limits). Keys the gateway adds itself (`_file_bytes`, `_file_name`, `_caller`, `_priced_rates`) are dropped from client bodies, a multipart body is read only for CloudConvert (any other provider answers 400 before reading it), and WaveSpeed's `enable_sync_mode` is removed so the gateway always polls (a sync request that outlives the read timeout would lose its task id). Otherwise it does not modify model-specific parameters. This means:

- Adding a new model requires zero gateway changes
- Request/response schemas are the provider's responsibility
- The gateway only cares about: pricing, payment, and task lifecycle

### Dynamic Pricing

Prices are fetched from the provider's pricing API at request time. A TTL cache (`price_cache_ttl` in config) reduces redundant API calls. The same cache key, the full SHA-256 digest of the request body, is used for both the 402 response and the payment verification retry. The cache also keeps the rates OpenRouter priced at (`_priced_rates`), so a cached price restores the `provider.max_price` cap it was priced with. Pricing fails before payment, so nothing is charged: a 429 carries `Retry-After: 2`, a gateway error while pricing gets 400 `Could not price this request`, and only a 5xx counts against the provider's health on the dashboard.

### Settlement

Payment is verified before any work.

On Base the client signs an EIP-3009 authorization that stays valid for `max_timeout` (3600 seconds). Permit2 payloads are refused at verify (`unsupported_payload: Permit2 is not accepted`). The gateway settles it at the same time as it calls the provider, and returns the result only after the settlement succeeds. A payer who moves the USDC or cancels the authorization during the job gets a 402 instead of the result. If the provider reports a failure after the settlement landed, the amount is credited to the payer's in-memory prepaid balance, as on Solana. Settling first would add the settlement time (about 1-2 seconds) to every request; running both together adds it only to jobs that finish faster than the settlement.

One payment buys one request. Parallel requests with the same payment all pass verify, because nothing has landed on-chain yet, so `PaymentClaims` in `x402gate/core/payment.py` lets one request hold a payment at a time (keyed by payer and nonce on Base, by the transaction bytes on Solana). The others get HTTP 409. The claim lasts until the settlement finishes, plus 60 seconds on Base and 180 seconds on Solana after a success, so an RPC node that lags behind cannot let a replay through. After a Base settlement with an unknown outcome it lasts until the authorization's `validBefore` plus 60 seconds (at most a day), so the same authorization cannot be sent again while its first transaction may still land. `POST /v1/topup` takes the same claims, so one payment cannot fund both a top-up and a job. On Base each authorization has its own nonce, so the balance check also counts the USDC that the same wallet's other unsettled requests have promised (`InFlightAmounts`); otherwise one balance could start many paid provider calls at once.

A settlement that was broadcast but whose confirmation wait failed (`settlement_pending`) is waited for once more. If it is still unconfirmed it most likely lands, so the result is returned (Base) or the job runs (Solana). A provider failure on Solana is then not credited. On Base a known failure is credited to prepaid once the reconciler below sees the transfer: the error carries `settlement_pending: true`, `pending_refund` (the full price) and `transaction` while the reconcile is still running, or `refunded_to_prepaid` and `prepaid_balance` if the transfer already landed during the job. If the reconcile already ended without a landing (a revert, or an authorization that expired), nothing was charged and the error carries none of these fields. An unknown outcome is never credited, and a successful or kept job's revenue is counted on its provider card when the transfer lands. A settle that crashed is treated as failed for the result, but its payment stays claimed in case the transaction went out. A Base settlement whose outcome stays unknown is then followed by `_reconcile_base` in the background: it polls the receipt by transaction hash (2 s, backing off to 60 s) until it shows the payer's USDC transfer to the gateway, a revert or other transfer, or the authorization expires. Until the task ends, however it ends, the amount stays in `InFlightAmounts`, so the same USDC cannot back another payment. A landed transfer is recorded with its real gas in USD. A lookup that fails is logged and polled again; a crash is logged. A top-up runs the same balance check before settling. A pending Solana top-up is credited at once (`"settlement_pending": true`); a pending Base top-up answers 202 with `"credited": "0"` and `pending_credit`, and is credited only when the reconciler sees the transfer. A top-up whose settle crashed credits nothing and returns 502 with `outcome_unknown: true`. On shutdown open reconciles are cancelled, since the credit they would make lives in memory anyway. Each makes one last receipt lookup, all at once and for at most 5 seconds (`_RECONCILE_LAST_LOOKUP_S`), and logs with its transaction hash whether the transfer landed, did not move the amount, or is still unresolved. A landed transfer is logged as owed to the payer only for a top-up or a job that failed with a known error; for a delivered job, or one whose outcome is unknown or that had not ended, the payment is kept and nothing is owed. A revert, an authorization that expires first, a restart, or a Base settle that crashed without a transaction hash credits nothing.

The SDK's Base receipt wait is 30 seconds, so a Base settle that stays pending is called unknown after about a minute. Every Base transaction is signed with an EIP-155 chain id (from the configured `eip155:<id>` network, otherwise read from the node once), so it cannot be replayed on another chain. Base sends are serial, from taking the nonce to the node's reply. Calls made under that send lock (resend, nonce count, send, transaction lookup) are tried once, each capped at 10 seconds. Once one settlement has held the lock for 10 seconds (`_EVM_SEND_LOCK_BUDGET_S`) it starts no new lookup or resend, though its nonce count and send still run, so a slow RPC holds the lock for at most that budget, the call in progress, a count and a send. Reads, `eth_call` and receipt polls run outside the lock, in parallel, and get one retry on a dropped connection or a 5xx, none on a read timeout; `PaymentHandler`'s own calls (gas cost receipts, reconcile receipts, the USDC balance check) use the same bounded provider instead of web3's default of 5 tries at 30 seconds. A failed send is sorted by the node's reply. Only an explicit refusal given before the node takes a transaction (`_SEND_REFUSED_ERRORS`: insufficient funds, underpriced, intrinsic gas too low, a fee cap below the base fee, an invalid sender or chain id and the like), for a transaction the node confirms it does not have, gives the nonce back and fails the send, so a failed send leaves no gap that would stall later settlements. Any other JSON-RPC error, such as a replica whose forward to the sequencer failed (`context deadline exceeded`, EOF, `internal error`), is an unknown outcome, like a timeout, a dropped connection or a 5xx. `already known` means the node holds this very transaction: the nonce is kept and the receipt wait decides. `nonce too low` or `replacement transaction underpriced` means some transaction holds the nonce: if the node has ours, it waits the same way; if the lookup gets no answer, the outcome is unknown; if the node does not have ours, it is not signed again, because the holder may be this very transfer after a load balancer retried the send: the nonce stays used and the receipt wait and reconcile decide. A real collision with another transaction therefore ends as an unknown outcome and expires after `validBefore`. A send with an unknown outcome keeps its nonce and signed bytes and is sent again before the next Base send, before a pending settle's second wait (so a transfer that never reached the node is rebroadcast by its own pending retry), and on each reconcile poll that finds no receipt. It reuses the same nonce, so it cannot pay twice, and fills that nonce if it never arrived. Such settlements end as a success, a revert or an unknown outcome, never as a known failure while the USDC moves. A resent transaction that the node reports as known, or whose nonce is held, is dropped without giving the nonce back; one the node refuses and does not have gives it back. If a given-back nonce then sits below a nonce already out, the resend before a pending settle's second wait and on each reconcile poll fills it with a 0-value transfer to the facilitator's own address (21,000 gas), so later transfers do not wait behind the gap until the next settlement. Verify and settle run on per-chain thread pools of 64 workers (`x402-base`, `x402-solana`). The x402 SDK is pinned (`x402==2.25.0` in `requirements.txt`), because `payment.py` relies on its exact settle semantics and signer internals; re-check those before upgrading.

On Solana the client signs a transaction whose blockhash expires in about a minute. The gateway settles that transaction immediately after verify, before calling the provider. While it waits, it rebroadcasts the same signed bytes. Simulation and send preflight run at Confirmed commitment. If the blockhash expires before the transaction lands, or the transaction lands with an error at confirmed or finalized (an error seen only at processed keeps the wait going), settlement fails with `transaction_failed: ...` and the client is not charged. That failure is remembered and the SDK's pending entry is dropped, so a replay of the same payload is verified again and refused instead of waiting on the old signature. If the provider then reports a failure, the settled amount is credited to the payer's in-memory prepaid balance. The error JSON includes `refunded_to_prepaid` and `prepaid_balance`.

The gateway is the Solana fee payer, and the x402 SDK verifier caps only the compute-unit price, not the limit. Before signing, `SolanaSigner` therefore refuses a transaction that needs more than 2 signatures, has no `SetComputeUnitLimit`, asks for more than 40,000 compute units, bids more than 100,000 microlamports per unit, or lists the fee payer in any instruction's accounts. That bounds the priority fee to 4,000 lamports per payment. The client gets 402 `Payment verification failed: transaction_simulation_failed: <reason>`.

### Unknown Outcome

Funds go back only when the failure is known: the provider returned an error status, marked the task failed, or the connection never opened. When the gateway cannot tell how the job ended, it keeps the payment, because the provider may have run and billed the job. That happens when polling exceeds `poll_timeout`, or when a request times out or drops after it was sent. `outcome_unknown()` in `x402gate/providers/base.py` makes that call by walking the exception chain. Keeping the payment means not crediting the settled x402 payment back, or charging the prepaid reservation. The error JSON carries `"outcome_unknown": true`.

The same applies once the provider has accepted a job. For WaveSpeed and SpicyAPI, a poll that keeps getting 404 for more than 20 seconds keeps the payment; a CloudConvert 404 while polling is a known failure and is refunded. A redirect or an upstream 401/402/403 while polling also keeps the payment. So does a WaveSpeed, OpenRouter, NanoGPT, ElevenLabs, MiniMax, Fish Audio or SocialDownload submit answered with 504, 520 or 524, an OpenRouter or NanoGPT 5xx whose body reports a timeout, or an unreadable 200 (a WaveSpeed 200 with no task id, an OpenRouter or NanoGPT 2xx body that is not a JSON object, a MiniMax 200 that is not a JSON object, or that reports success but has no audio or audio that is neither hex nor valid base64). An OpenRouter or NanoGPT 2xx that carries only an `error` object, with no usage or usage that reports no cost and no tokens, is a known failure, refunded with the status from `error.code`, unless the code or message is a timeout. For SpicyAPI, a createTask where any attempt hit 504, 520 or 524 or timed out after it was sent keeps the payment, as does a 2xx answer that is unreadable or has no `taskId` (unless it carries a 4xx error code such as 40001 or 40201), and a chat stream that fails after it has produced output, gets 504, 520 or 524, or does not finish within `poll_timeout`. A SpicyAPI quote that fails during submit (an expired cached quote, or the re-quote after a stale-quote 40901) is a known failure, even after a timeout. CloudConvert refunds any failure while creating the job and any upload failure before the whole file was sent; after a complete upload, a lost answer or a 504, 520 or 524 from the upload host keeps the payment. A redirect at submit never reached the provider, so it is returned as 502 and refunded. RapidAPI bills every call it answers with 2xx, so a SocialDownload 200 that carries an error (an unsupported or private URL) is charged too. A non-`ProviderError` exception after payment is a gateway bug with an unknown outcome.

### Client Disconnects

Every managed provider call, synchronous or polled, runs in its own task, tracked in `_pending_jobs`. A call that finishes within 20 seconds gets a normal response with its real status and headers; a longer one becomes a stream of newline heartbeats followed by the body, with status 200 from the first pad. If the client leaves while heartbeats are going out, only the stream stops. The job finishes and is charged or refunded as if the client had stayed. On shutdown the gateway waits for these jobs before it waits for settlements; `railway.json` sets `drainingSeconds` to 1020 so a redeploy leaves time for them.

Polling (`poll_result`) counts `poll_timeout` in wall-clock time, slow poll requests included, and always polls once more at the deadline, so a task that finished during the last wait is returned.

### Custom x402 Flow

The standard `PaymentMiddlewareASGI` from the x402 library requires static route-to-price mappings defined at startup. Since our prices are dynamic per-request, we implement the x402 verify/settle protocol directly using the facilitator's HTTP API.

### Actual Cost Tracking

For token-based providers (OpenRouter chat/embeddings, NanoGPT chat), the gateway tracks both estimated and actual costs. OpenRouter catalog rates are USD per token. NanoGPT `pricing.prompt` and `pricing.completion` are USD per million tokens:

- **Estimated cost** — ceiling calculated upfront from `max_tokens × completion_price + input_tokens × prompt_price`
- **Actual cost** — for OpenRouter, the `usage.cost` OpenRouter reports, which already covers the endpoint that served the request, a `models` fallback, cache writes and plugins. For BYOK requests, or when `usage.cost` is missing, it is computed from `usage.prompt_tokens` and `usage.completion_tokens` at the catalog rates of the model that served the request (or the most expensive `models` candidate), with reasoning tokens at the higher of the completion and internal reasoning rates, the `pricing.overrides` tier the real prompt reaches, and the 5 native searches per completion of a model that searches on every request. NanoGPT is computed from real token counts, with cache reads and cache writes at their own catalog rates (writes with a `ttl` other than `5m` at least 2x input) and the above-threshold rates NanoGPT lists; the Gemini Pro 2x / 1.5x tier is applied to the quote only.

For OpenRouter STT (`audio/transcriptions`), the gateway supports duration-priced Whisper models. It identifies the container from the bytes and parses audio length before payment, and uses `usage.cost` from OpenRouter for actual billing when available. See [stt.md](stt.md). Audio parts in OpenRouter and NanoGPT chat are timed the same way and quoted at 50 tokens per second; an actual cost computed from tokens (NanoGPT, or OpenRouter without `usage.cost`) bills reported `audio_tokens` at the higher of the audio and input rates. When usage reports no audio tokens, the audio parts are timed again at 50 tokens per second (the tokens the quote held), capped at the prompt; those bill at the higher of the two rates and the rest of the prompt at the input rate. If the audio cannot be timed again, the whole prompt bills as audio, never more than the quote.

The x402 `exact` scheme requires settling the full signed amount, so the client always pays the ceiling price. Settlement runs before (Solana) or alongside (Base) the provider call, so the settlement log line shows the estimate. In x402 mode the actual cost is used only for the dashboard's revenue and profit stats; in prepaid mode it is what gets deducted, never more than the reservation. When x402 adds a `deferred` scheme (settle ≤ signed amount), the gateway is ready to use actual cost for settlement.
