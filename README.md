# x402gate

🌐 [x402gate.io](https://x402gate.io/) | **Transparent x402 payment proxy for AI services.**

Pay per API call with USDC on **Base** or **Solana** — no accounts, no API keys, no subscriptions.

x402gate sits between your AI agent and AI service providers, adding [x402](https://x402.org) payment support. One gateway, multiple providers:

| Provider | Type | What | Endpoint |
|----------|------|------|----------|
| [OpenRouter](https://openrouter.ai) | Managed | 300+ LLMs (GPT, Claude, Gemini, Llama…) | `/v1/openrouter/...` |
| [WaveSpeed AI](https://wavespeed.ai) | Managed | 60+ image/video models (Flux, Wan, Kling, Sora…) | `/v1/wavespeed/...` |
| [Tungsten](https://tungsten.run) | Managed | AI image generation (SDXL, Flux, Qwen, Z-Image…) | `/v1/tungsten/...` |
| [BlockRun](https://blockrun.ai) | Passthrough | 40+ LLMs (GPT, Claude, Gemini…) | `/v1/blockrun/...` |
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
  │                          │◄─ 503? retry (2× backoff)►│
  │                          │◄─ result ────────────────┤
  │                          ├─ settle (on-chain)        │
  │◄─── 200 + result ───────┤                          │
```

1. Client sends a request → gateway fetches the price from the provider
2. Gateway responds with **HTTP 402** and payment options for all configured networks
3. Client picks a network (Base or Solana), signs the payment, retries with `PAYMENT-SIGNATURE` header
4. Gateway verifies → forwards request → waits for result → settles payment → returns output

**No payment? No settlement on failure.** Payment is only settled after the AI task completes successfully.

### Passthrough Providers (BlockRun)

For x402-native providers like [BlockRun](https://blockrun.ai), x402gate acts as a **transparent proxy** — no payment handling, no commission. The client pays the provider directly:

```
Client                    x402gate                  BlockRun
  │                          │                          │
  ├─ POST /v1/blockrun/... ──►                          │
  │                          ├─ forward request ────────►│
  │                          │◄─ 402 ($0.001) ──────────┤
  │◄─── 402 ($0.001) ───────┤  (passthrough, no markup)  │
  │                          │                          │
  ├─ POST + PAYMENT-SIGNATURE►                          │
  │                          ├─ forward + signature ────►│
  │                          │◄─ LLM response ──────────┤
  │◄─── 200 + response ─────┤                          │
```

The payment goes directly to BlockRun's wallet — x402gate earns nothing but provides a unified "single window" experience.

### File Conversion (CloudConvert)

CloudConvert uses **multipart/form-data** for file uploads. Supports 200+ formats (DOCX→PDF, PNG→JPG, etc.) and PDF optimization. See [docs/configuration.md](docs/configuration.md) for supported parameters.

### Transaction Summary

After each settled payment, the gateway logs a financial summary with both estimated and actual costs:

```
+------------------- Transaction Summary -------------------+
|  Network:                   solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp |
|  Revenue (client paid):     $0.006200 USDC                 |
|  Provider cost:            -$0.005000 USDC                 |
|  Commission (4%):           $0.000200 USDC                 |
|  Gas surcharge:             $0.001000 USDC                 |
|  Gas cost:                 -$0.000855 (0.0000100010 SOL)   |
|  --------------------------------------------------------- |
|  Net profit:                $0.000345 USDC                 |
|  Generation time:           12.0s                          |
|  Client wait time:          12.5s                          |
+-----------------------------------------------------------+
```

- **Estimated cost** — ceiling based on `max_tokens` (shown only when different from actual)
- **Provider cost** — actual cost computed from real token usage
- **Net profit** — calculated from actual provider cost, not the estimate

Gas costs are fetched in real-time from the chain. Native token prices (ETH, SOL) come from CoinGecko.

### Error Handling

If the provider rejects a request (e.g. invalid parameters), the gateway returns a structured error **without settling payment** — the client keeps their money:

```json
{
  "error": "size does not meet the constraints.",
  "provider": "wavespeed",
  "status": 502
}
```

**Payment rejection reasons:** When payment verification fails (e.g. insufficient USDC, stale transaction, network mismatch), the 402 response includes the specific reason:

```json
{
  "error": "Payment verification failed: transaction_simulation_failed: Simulation failed: InstructionErrorCustom(1)",
  "status": 402
}
```

**Retry on 5xx:** If a provider returns a server error (502, 503, etc.), x402gate automatically retries the request up to **2 times** with exponential backoff (2s → 4s) before returning the error. This prevents payment loss during brief provider outages (e.g. CDN restarts). Timeouts are also retried. Client errors (4xx) are never retried.


> **Note:** Payment is settled **only** on success (HTTP 200). On any error, the payment is not settled and no USDC is transferred.

### Prepaid Mode

For high-frequency usage, you can top-up a prepaid balance and skip on-chain transactions per request:

```
Client                    x402gate
  │                          │
  ├─ POST /v1/topup (402) ───►  ← pay once (Solana or Base USDC)
  ├─ POST /v1/topup + sig ──►  ← balance credited
  │                          │
  ├─ POST /v1/openrouter/... ►  ← wallet signature (no blockchain tx)
  │  X-PREPAID-PUBKEY         │  ← deduct actual_cost from balance
  │  X-PREPAID-SIGNATURE      │
  │  X-PREPAID-TIMESTAMP      │
  │◄─── 200 + result ────────┤
```

Commission and gas are charged **once at top-up**. Subsequent requests deduct only the provider's actual cost. In prepaid mode, `max_tokens` is not injected — models respond fully, and you pay for actual usage.

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

x402gate includes a built-in dashboard at `/dashboard` with real-time provider status, request metrics, revenue tracking, network settlement stats, prepaid balance, and live logs.

| Endpoint | Description |
|----------|-------------|
| `GET /dashboard` | HTML dashboard with auto-refresh (5s) |
| `GET /v1/stats` | JSON snapshot: uptime, per-provider metrics, revenue/cost/profit, network stats, prepaid balance |
| `GET /v1/logs?limit=200` | Recent log entries (newest first, default 200) |

Statistics are stored **in-memory** and reset on server restart. The log buffer retains the last 1 000 entries (configurable via `MAX_LOG_ENTRIES` in `x402gate/core/stats.py`). Providers that haven't received requests in 1 hour are marked as `unknown`.

## Supported Networks

| Network | Chain ID | Token | Gas Token | Payment Overhead |
|---------|----------|-------|-----------|-----------------|
| **Base** (EVM) | `eip155:8453` | USDC | ETH | ~2.4s |
| **Solana** (SVM) | `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp` | USDC | SOL | ~0.5s |

Both networks are offered simultaneously — the client chooses which to pay on. Payment overhead is the additional latency added by on-chain verification and settlement (on top of the AI provider's generation time).

> **Security:** Both networks verify the client's on-chain USDC balance **before** running generation. Solana uses transaction simulation; Base performs an explicit `balanceOf` RPC call. Clients with insufficient USDC receive HTTP 402 with the specific rejection reason (e.g. `transaction_simulation_failed: InstructionErrorCustom(1)` for insufficient funds on Solana, or `Insufficient USDC balance` on Base) — no free generation is possible.

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
- **Passthrough** — transparent proxy for x402-native providers; client pays them directly (e.g., BlockRun)

See [docs/add-provider.md](docs/add-provider.md) for a step-by-step guide.

## Documentation

- [Architecture](docs/architecture.md) — how x402gate works under the hood
- [Prepaid Mode](docs/prepaid.md) — top-up and pay without on-chain transactions
- [Dashboard](docs/dashboard.md) — real-time metrics, logs, and revenue tracking
- [Deployment](docs/deployment.md) — Railway, Docker, VPS guides
- [Configuration](docs/configuration.md) — all config options
- [TTS Providers](docs/tts.md) — ElevenLabs, MiniMax, Fish Audio
- [Tungsten](docs/tungsten.md) — Tungsten image generation provider
- [SocialDownload](docs/socialdownload.md) — download media from social networks
- [Adding a Provider](docs/add-provider.md) — extend with new AI services
- [AI Agent Skill](.agents/skills/x402gate/SKILL.md) — public skill for autonomous agents to use `x402gate.io` (OpenClaw, etc.)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, style guide, and PR process.

## License

[MIT](LICENSE)
