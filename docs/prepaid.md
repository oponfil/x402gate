# Prepaid Mode

Top-up your wallet balance once, then make unlimited API requests without on-chain transactions per request.

## How It Works

```
Client                           x402gate                      Solana
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

1. **Top-up**: Send a Solana USDC payment to `/v1/topup` — commission and gas are deducted once
2. **Request**: Sign each request with your wallet's Ed25519 key — no on-chain transaction
3. **Deduction**: Only the provider's base cost is deducted from your balance (no repeated commission)

> **⚠️ Warning:** Balances are stored **in memory only**. They will be lost on server restart, crash, or redeployment. Maximum single top-up: **$10** (configurable via `gateway.max_prepaid_topup`). Don't top up more than you plan to use in one session.

## Endpoints

### `POST /v1/topup`

Top up your prepaid balance.

**Without payment header** → returns 402 with payment requirements (like any managed endpoint).

**With `PAYMENT-SIGNATURE` header** → verifies the Solana USDC payment, deducts commission (4% + $0.001 gas), and credits the remaining amount to your wallet's prepaid balance.

Response:
```json
{
  "pubkey": "YourSolanaPubkey...",
  "credited": "0.095000",
  "balance": "0.095000",
  "warning": "Balance is stored in memory only. It will be lost on server restart."
}
```

### `GET /v1/balance/{pubkey}`

Check prepaid balance for any wallet address. No authentication required.

```bash
curl https://x402gate.io/v1/balance/YourSolanaPubkey
```

Response:
```json
{
  "pubkey": "YourSolanaPubkey...",
  "balance": "0.082350"
}
```

## Headers for Prepaid Requests

When using prepaid balance, send these headers **instead of** `PAYMENT-SIGNATURE`:

| Header | Description |
|--------|-------------|
| `X-PREPAID-PUBKEY` | Your Solana public key (base58) |
| `X-PREPAID-SIGNATURE` | Ed25519 signature of the signing message (base58) |
| `X-PREPAID-TIMESTAMP` | Unix timestamp (integer seconds) |

### Signing Message Format

The message to sign is: `x402gate:{path}:{timestamp}` encoded as UTF-8 bytes.

- `path` is the API path **after** `/v1/{provider}/`, e.g. `chat/completions`
- `timestamp` is the current Unix timestamp (must be within 60 seconds of server time)

## Python Example

```python
import time
import httpx
from solders.keypair import Keypair

# Your Solana wallet keypair
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

## Limitations

- **In-memory storage only** — balances are lost on server restart
- **Max top-up $10** — configurable via `gateway.max_prepaid_topup` in config.yaml
- **Solana wallets only** — uses Ed25519 (Solana's native signature scheme)
- **No refunds** — prepaid amounts are non-refundable
- **No multi-server** — balances are local to a single server instance
