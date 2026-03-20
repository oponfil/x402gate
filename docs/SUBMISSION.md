# x402gate — The Synthesis Hackathon Submission

> **Track:** Agents that pay · Open Track · Locus Bounty
>
> **One-liner:** A transparent pay-per-request gateway that lets AI agents buy LLM completions, image generation, video, and file conversion with USDC on Base — no accounts, no API keys, no subscriptions.

## The Problem

AI agents need to pay for services. Today that means:
- A human pre-registers API keys on every provider
- Credit cards that can be revoked or surveilled
- Centralized billing platforms that can change terms or cut access

**The agent has no independent way to pay.**

## The Solution: x402gate

[x402gate](https://x402gate.io) is a **production gateway** that wraps 6 AI service providers behind the [x402 payment protocol](https://x402.org). Any agent with a USDC wallet on Base (or Solana) can:

1. **Discover** available services via `GET /` (JSON manifest) or `GET /v1/providers`
2. **Request** any AI service → receive `HTTP 402` with the USDC price
3. **Sign** an on-chain payment with its own wallet
4. **Receive** the result — payment settles only on success (escrow model)

### Providers Available Today

| Provider | What | Price Range |
|----------|------|-------------|
| OpenRouter | 300+ LLMs (GPT, Claude, Gemini, Llama…) | $0.001–$0.50 |
| WaveSpeed AI | 60+ image/video models (Flux, Wan, Kling, Sora…) | $0.005–$0.10 |
| Tungsten | AI image generation (SDXL, Flux, LoRA support) | $0.01 |
| BlockRun | 40+ LLMs with web search (passthrough, no markup) | Direct pricing |
| CloudConvert | File conversion (200+ formats) | $0.03 |
| SocialDownload | Media download from social networks | $0.005 |

### Key Design Principles

- **No payment on failure.** If the AI provider errors out, the payment is never settled. The agent keeps its USDC.
- **Transparent pricing.** 4% commission + $0.001 gas surcharge. All costs visible in the 402 response.
- **Multi-network.** Both Base (EVM) and Solana are supported simultaneously.
- **Prepaid mode.** For high-frequency agents: top-up once, skip per-request blockchain transactions.
- **Service discovery.** Machine-readable JSON at the root endpoint for autonomous agent navigation.

## Locus Integration Demo

We built a demo script (`scripts/locus_demo.py`) showing the full autonomous agent lifecycle:

1. Agent **self-registers** with [Locus](https://paywithlocus.com) → gets a Base wallet with spending limits
2. Agent **discovers** AI services on x402gate (JSON manifest)
3. Agent **requests** LLM completion → receives `402 Payment Required ($0.006 USDC)`
4. Agent **signs** on-chain payment with its Locus wallet → receives AI response in 1.7s

**Actual test run output (March 20, 2026):**
```
16:43:52 [locus-demo] 🔍 Found 6 providers
16:43:52 [locus-demo] 💳 Payment Required! $0.006338 USDC
16:43:52 [locus-demo] ✅ Payment signed in 0.01s
16:43:54 [locus-demo] 📝 AI Response (google/gemini-2.5-flash):
   "Autonomous AI agents need decentralized payment infrastructure
    like x402 to enable secure, trustless, and censorship-resistant
    transactions, fostering a truly peer-to-peer AI economy..."
16:43:54 [locus-demo] ✅ Demo Complete! Total time: 3.6s
```

**Run it yourself:**
```bash
git clone https://github.com/oponfil/x402gate.git
cd x402gate
pip install -r requirements.txt
BASE_PRIVATE_KEY=0x_your_base_wallet_key python scripts/locus_demo.py
```

## Architecture

```
Agent                      x402gate                   AI Provider
  │                          │                          │
  ├─ POST /v1/provider/... ──►                          │
  │                          ├─ GET price ──────────────►│
  │                          │◄─ $0.005 ────────────────┤
  │◄─── 402 ($0.006) ───────┤  (provider + 4% + gas)    │
  │                          │                          │
  ├─ POST + PAYMENT-SIGNATURE►                          │
  │                          ├─ verify on-chain          │
  │                          ├─ forward request ────────►│
  │                          │◄─ result ────────────────┤
  │                          ├─ settle on-chain          │
  │◄─── 200 + result ───────┤                          │
```

## On-Chain Artifacts

Every transaction through x402gate creates verifiable on-chain artifacts:
- **USDC transfers** on Base Mainnet (`eip155:8453`) and Solana Mainnet
- **Settlement transactions** visible on [BaseScan](https://basescan.org) / [Solscan](https://solscan.io)
- Gateway wallet: see live activity on the dashboard at [x402gate.io/dashboard](https://x402gate.io/dashboard)

## Human-Agent Collaboration Log

This project was built through direct collaboration between a human developer and AI coding assistants. The conversation log documenting our hackathon planning, architecture decisions, and implementation process is available in the repository.

Key collaboration moments:
- **Planning:** AI analyzed the Synthesis themes and identified "Agents that pay" as the primary fit
- **Strategy:** Human challenged the Locus integration value — led to clearer positioning
- **Insight:** Human pointed out that AI judges can't watch videos — shifted strategy to on-chain artifacts
- **Implementation:** AI wrote the demo script; human tested and approved

## Links

- 🌐 **Live Gateway:** [x402gate.io](https://x402gate.io)
- 📊 **Dashboard:** [x402gate.io/dashboard](https://x402gate.io/dashboard)
- 📦 **Source Code:** [github.com/oponfil/x402gate](https://github.com/oponfil/x402gate)
- 📄 **Demo Script:** [scripts/locus_demo.py](scripts/locus_demo.py)
- 🔗 **x402 Protocol:** [x402.org](https://x402.org)

## Tech Stack

- **Backend:** Python, FastAPI, deployed on Railway
- **Payment:** x402 protocol, USDC on Base (EVM) and Solana (SVM)
- **Testing:** End-to-end tests with real on-chain payments
- **License:** MIT
