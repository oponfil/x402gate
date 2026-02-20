# x402gate

🌐 [x402gate.io](https://www.x402gate.io/) | **Transparent x402 payment proxy for AI services.**

Pay per API call with USDC on **Base** or **Solana** — no accounts, no API keys, no subscriptions.

x402gate sits between your AI agent and AI service providers, adding [x402](https://x402.org) payment support. One gateway, multiple providers:

| Provider | Type | What | Endpoint |
|----------|------|------|----------|
| [WaveSpeed AI](https://wavespeed.ai) | Managed | Image/video generation | `/v1/wavespeed/...` |
| [BlockRun](https://blockrun.ai) | Passthrough | 40+ LLMs (GPT, Claude, Gemini…) | `/v1/blockrun/...` |

## How It Works

### Managed Provider (WaveSpeed)

```
Client                    x402gate                  WaveSpeed AI
  │                          │                          │
  ├─ POST /v1/wavespeed/... ─►                          │
  │                          ├─ GET price ──────────────►│
  │                          │◄─ $0.005 ────────────────┤
  │◄─── 402 ($0.00625) ─────┤  total = provider + 5% + $0.001 gas
  │                          │                          │
  ├─ POST + PAYMENT-SIGNATURE►                          │
  │                          ├─ verify (on-chain)        │
  │                          ├─ forward request ────────►│
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

### Transaction Summary

After each settled payment, the gateway logs a financial summary:

```
+------------------- Transaction Summary -------------------+
|  Network:                   solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp |
|  Revenue (client paid):     $0.006000 USDC                 |
|  Provider cost:            -$0.005000 USDC                 |
|  Commission (5%):           $0.001000 USDC                 |
|  Gas cost:                 -$0.000820 (0.0000100010 SOL)   |
|  --------------------------------------------------------- |
|  Net profit:                $0.000180 USDC                 |
+-----------------------------------------------------------+
```

Gas costs are fetched in real-time from the chain. Native token prices (ETH, SOL) come from CoinGecko.

## Supported Networks

| Network | Chain ID | Token | Gas Token |
|---------|----------|-------|-----------|
| **Base** (EVM) | `eip155:8453` | USDC | ETH |
| **Solana** (SVM) | `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp` | USDC | SOL |

Both networks are offered simultaneously — the client chooses which to pay on.

## Quick Start

### 1. Install

```bash
git clone https://github.com/oponfil/x402gate.git
cd x402gate
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
WAVESPEED_API_KEY=your_key

# Base (EVM)
BASE_PAY_TO_ADDRESS=0xYourWallet
BASE_FACILITATOR_PRIVATE_KEY=your_key

# Solana
SOLANA_PAY_TO_ADDRESS=YourSolanaWallet
SOLANA_FACILITATOR_PRIVATE_KEY=your_base58_key
```

> **Note:** Facilitator wallets need a small balance of native tokens (ETH on Base, SOL on Solana) to pay gas for settlement transactions.

### 3. Run

```bash
python -m x402gate.main
```

The gateway starts on `http://localhost:4021`.

### 4. Test

```bash
# List providers
curl http://localhost:4021/v1/providers

# WaveSpeed: image generation (returns 402 with price)
curl -X POST http://localhost:4021/v1/wavespeed/wavespeed-ai/z-image/turbo \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in space"}'

# BlockRun: LLM chat (returns BlockRun's 402 with price)
curl -X POST http://localhost:4021/v1/blockrun/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 20}'
```

## Configuration

All settings are in `config.yaml`. Secrets use `${ENV_VAR}` interpolation:

| Setting | Default | Description |
|---------|---------|-------------|
| `gateway.commission` | `0.05` | Markup rate (5%) |
| `gateway.min_commission` | `0.001` | Minimum commission per request ($0.001) |
| `gateway.price_cache_ttl` | `60` | Price cache TTL in seconds |
| `payment.networks.base` | — | Base Mainnet (EVM) config |
| `payment.networks.solana` | — | Solana Mainnet (SVM) config |
| `providers.wavespeed.poll_timeout` | `300` | Max wait for AI result (seconds) |

See [docs/configuration.md](docs/configuration.md) for full reference.

## E2E Tests

Run end-to-end tests on mainnet (requires funded wallets):

```bash
# WaveSpeed on Solana
python -m pytest tests/e2e/test_solana_wavespeed.py -v -s

# BlockRun passthrough (Base)
python -m pytest tests/e2e/test_blockrun_passthrough.py -v -s
```

Requires `BASE_E2ETEST_PRIVATE_KEY` and/or `SOLANA_E2ETEST_PRIVATE_KEY` in `.env`.

## Adding Providers

x402gate supports two provider types:

- **Managed** — x402gate handles pricing, payment verification, and settlement (e.g., WaveSpeed)
- **Passthrough** — transparent proxy for x402-native providers; client pays them directly (e.g., BlockRun)

See [docs/add-provider.md](docs/add-provider.md) for a step-by-step guide.

## Documentation

- [Architecture](docs/architecture.md) — how x402gate works under the hood
- [Deployment](docs/deployment.md) — Railway, Docker, VPS guides
- [Configuration](docs/configuration.md) — all config options
- [Adding a Provider](docs/add-provider.md) — extend with new AI services

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, style guide, and PR process.

## License

[MIT](LICENSE)
