# Architecture

x402gate is a transparent payment proxy that sits between AI agents/clients and AI service providers, enforcing payments via the [x402 protocol](https://x402.org).

## Overview

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│              │     │      x402gate         │     │   Provider   │
│  AI Agent /  │────►│                      │────►│  (WaveSpeed)  │
│   Client     │◄────│  FastAPI + x402 flow │◄────│              │
│              │     │                      │     │              │
└──────────────┘     └──────────┬───────────┘     └──────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   Facilitator     │
                     │  (Coinbase x402)  │
                     │                   │
                     │ Verify & Settle   │
                     │ on Base (USDC)    │
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
   x402gate → forward request to WaveSpeed as-is
           ↓
   x402gate → poll for async task result
           ↓
   x402gate → settle payment via Facilitator
           ↓
   Client ← 200 OK + result + PAYMENT-RESPONSE header
```

## Components

| Component | File | Purpose |
|---|---|---|
| **Config** | `x402gate/core/config.py` | Load `config.yaml`, interpolate env vars |
| **Pricing** | `x402gate/core/pricing.py` | TTL cache, commission, USDC formatting |
| **Payment** | `x402gate/core/payment.py` | Build 402 responses, verify/settle via facilitator |
| **Proxy** | `x402gate/core/proxy.py` | Forward requests, poll async tasks |
| **Provider Base** | `x402gate/providers/base.py` | Abstract provider interface |
| **WaveSpeed** | `x402gate/providers/wavespeed.py` | WaveSpeed AI implementation |
| **OpenRouter** | `x402gate/providers/openrouter.py` | OpenRouter LLM aggregator (300+ models) |
| **App** | `x402gate/app.py` | FastAPI routes, lifecycle |
| **Entry** | `x402gate/main.py` | Uvicorn server startup |

## Key Design Decisions

### Transparent Proxy

The gateway forwards request bodies to providers **as-is**. It does not parse, validate, or modify model-specific parameters. This means:

- Adding a new model requires zero gateway changes
- Request/response schemas are the provider's responsibility
- The gateway only cares about: pricing, payment, and task lifecycle

### Dynamic Pricing

Prices are fetched from the provider's pricing API at request time. A TTL cache (`price_cache_ttl` in config) reduces redundant API calls. The same cache key is used for both the 402 response and the payment verification retry.

### Settlement After Work

Payment is verified **before** starting work, but settled **after** successful completion. If the task fails or times out, the payment is never settled and the client keeps their money.

### Custom x402 Flow

The standard `PaymentMiddlewareASGI` from the x402 library requires static route-to-price mappings defined at startup. Since our prices are dynamic per-request, we implement the x402 verify/settle protocol directly using the facilitator's HTTP API.

### Actual Cost Tracking

For token-based providers (OpenRouter), the gateway tracks both estimated and actual costs:

- **Estimated cost** — ceiling calculated upfront from `max_tokens × completion_price + input_tokens × prompt_price`
- **Actual cost** — computed after response from real `usage.prompt_tokens` and `usage.completion_tokens`

The x402 `exact` scheme requires settling the full signed amount, so the client always pays the ceiling price. Actual cost is used only for financial reporting in the Transaction Summary. When x402 adds a `deferred` scheme (settle ≤ signed amount), the gateway is ready to use actual cost for settlement.
