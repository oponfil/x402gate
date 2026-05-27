# Prepaid Mode

Top-up your wallet balance once, then make unlimited API requests without on-chain transactions per request.

## How It Works

```
Client                           x402gate                      Blockchain
  │                                 │                             │
  │  1. POST /v1/topup              │                             │
  │     (no header) ───────────────►│                             │
  │◄── 402 Payment Required ───────│                             │
  │                                 │                             │
  │  2. POST /v1/topup              │                             │
  │     PAYMENT-SIGNATURE ─────────►│── verify + settle ─────────►│
  │◄── {"balance": "$0.095"} ──────│◄─ confirmed ───────────────┤
  │                                 │                             │
  │  3. POST /v1/openrouter/...     │                             │
  │     X-PREPAID-PUBKEY            │                             │
  │     X-PREPAID-SIGNATURE         │   (no blockchain tx!)       │
  │     X-PREPAID-TIMESTAMP ───────►│──► forward to provider      │
  │◄── 200 + AI response ─────────│   balance -= actual_cost     │
```

1. **Top-up**: Send a USDC payment (Solana or Base) to `/v1/topup` — commission and gas are deducted once
2. **Request**: Sign each request with your wallet key — no on-chain transaction
3. **Deduction**: Only the provider's actual cost is deducted from your balance (no repeated commission)

> **⚠️ Warning:** Balances are stored **in memory only**. They will be lost on server restart, crash, or redeployment. Top-up limits: **$0.10 – $10** (configurable via `gateway.min_prepaid_topup` / `gateway.max_prepaid_topup`). Don't top up more than you plan to use in one session.

## Supported Wallets

| Wallet Type | Address Format | Signature Scheme | Top-up Network |
|-------------|---------------|-----------------|----------------|
| **Solana** | Base58 (44 chars) | Ed25519 | Solana USDC |
| **EVM (Base)** | `0x...` (42 chars hex) | EIP-191 personal_sign | Base USDC |

The wallet type is auto-detected by address format. Both wallet types share the same balance system.

## Prepaid vs x402 (max_tokens)

In **x402 mode**, the gateway injects a default `max_tokens` (2048) into OpenRouter **chat completion** requests when the client doesn't specify one. This caps the response size to match the pre-paid estimate.

This injection does **not** apply to OpenRouter `embeddings` or duration-priced `audio/transcriptions` (STT) — those endpoints use different pricing logic.

In **prepaid mode**, `max_tokens` is **not injected** — the model responds with its full capacity. You're charged **actual token usage**, so there's no need to cap output. If you want to limit tokens, include `max_tokens` in your request explicitly.

## Endpoints

### `POST /v1/topup`

Top up your prepaid balance.

**Request body** (optional):
```json
{"amount": 0.50}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `amount` | float | `0.10` (min_prepaid_topup) | Desired top-up amount in USD ($0.10 – $10) |

**Without payment header** → validates amount, returns 402 with payment requirements for the specified amount.

**With `PAYMENT-SIGNATURE` header** → verifies the USDC payment, deducts commission (4% + $0.001 gas), and credits the remaining amount to your wallet's prepaid balance.

Response:
```json
{
  "pubkey": "YourWalletAddress",
  "credited": "0.095000",
  "balance": "0.095000",
  "warning": "Balance is stored in memory only. It will be lost on server restart."
}
```

### `GET /v1/balance/{address}`

Check prepaid balance for any wallet address. No authentication required.

```bash
# Solana wallet
curl https://x402gate.io/v1/balance/YourSolanaPubkey

# EVM wallet
curl https://x402gate.io/v1/balance/0xYourEvmAddress
```

Response:
```json
{
  "pubkey": "YourWalletAddress",
  "balance": "0.082350"
}
```

## Headers for Prepaid Requests

When using prepaid balance, send these headers **instead of** `PAYMENT-SIGNATURE`:

| Header | Description |
|--------|-------------|
| `X-PREPAID-PUBKEY` | Your wallet address (Solana base58 or EVM hex `0x...`) |
| `X-PREPAID-SIGNATURE` | Wallet signature (base58 for Solana, hex for EVM) |
| `X-PREPAID-TIMESTAMP` | Unix timestamp (integer seconds) |

### Response Header

Successful prepaid responses include the remaining balance:

| Header | Description |
|--------|-------------|
| `X-Prepaid-Balance` | Remaining prepaid balance after deduction (e.g. `0.0535824`) |

### Signing Message Format

The message to sign is: `x402gate:{path}:{timestamp}` encoded as UTF-8 bytes.

- `path` is the full API path after `/v1/`, e.g. `openrouter/chat/completions`
- `timestamp` is the current Unix timestamp (must be within `prepaid_timestamp_window` seconds of server time on arrival, default: 300s)

**Solana**: sign raw bytes with `keypair.sign_message(msg)`
**EVM**: sign with EIP-191 `personal_sign` via `eth_account.Account.sign_message(encode_defunct(msg))`

## Python Examples

### Solana Wallet

```python
import time
import httpx
from solders.keypair import Keypair

keypair = Keypair.from_base58_string("your_private_key_base58")
gateway_url = "https://x402gate.io"

# Step 1: Top-up (standard x402 flow, done once)
# ... (send PAYMENT-SIGNATURE to /v1/topup)

# Step 2: Make prepaid requests (no blockchain tx!)
path = "openrouter/chat/completions"
ts = int(time.time())
msg = f"x402gate:{path}:{ts}".encode("utf-8")
sig = keypair.sign_message(msg)

response = httpx.post(
    f"{gateway_url}/v1/{path}",
    json={
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
    headers={
        "X-PREPAID-PUBKEY": str(keypair.pubkey()),
        "X-PREPAID-SIGNATURE": str(sig),
        "X-PREPAID-TIMESTAMP": str(ts),
    },
)
print(response.json())
```

### EVM (Base) Wallet

```python
import time
import httpx
from eth_account import Account
from eth_account.messages import encode_defunct

account = Account.from_key("your_private_key_hex")
gateway_url = "https://x402gate.io"

# Step 1: Top-up (standard x402 flow via Base, done once)
# ... (send PAYMENT-SIGNATURE to /v1/topup)

# Step 2: Make prepaid requests (no blockchain tx!)
path = "openrouter/chat/completions"
ts = int(time.time())
msg = f"x402gate:{path}:{ts}".encode("utf-8")
signable = encode_defunct(msg)
signed = account.sign_message(signable)

response = httpx.post(
    f"{gateway_url}/v1/{path}",
    json={
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
    headers={
        "X-PREPAID-PUBKEY": account.address,
        "X-PREPAID-SIGNATURE": signed.signature.hex(),
        "X-PREPAID-TIMESTAMP": str(ts),
    },
)
print(response.json())
```

## Limitations

- **In-memory storage only** — balances are lost on server restart
- **Top-up limits** — min $0.10, max $10 (configurable in config.yaml)
- **No refunds** — prepaid amounts are non-refundable
- **No multi-server** — balances are local to a single server instance
