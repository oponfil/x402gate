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

In **prepaid mode**, OpenRouter chat gets the same `max_tokens` ceiling. The gateway reserves that estimate on the balance before calling the provider, then charges the actual cost up to the reservation and returns the rest. A failed provider call releases the whole reservation. Set `max_tokens` yourself when you want a higher ceiling; that larger value is what gets reserved.

OpenRouter chat with a video part (`video_url`) is accepted in prepaid mode only. Its length is unknown before the call, so the reservation is the model's whole context window at the higher of its audio and input rates, plus the output ceiling: about $0.15 for a model with a 1M-token window at $0.15 per million tokens. Keep that much free on the balance for each video request in flight. After the response only the actual cost is charged (usually a fraction of a cent for a short clip) and the rest goes back. If the outcome is unknown, the whole reservation is charged (see below). An x402 payment for a video request gets HTTP 400.

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

**With `PAYMENT-SIGNATURE` header** → verifies the USDC payment, settles it on-chain, and only then deducts commission (4% + $0.001 gas) and credits the remaining amount. The signed amount must sit inside the min/max limits and, when the body includes `amount`, must equal that amount. Otherwise the response is HTTP 400 and nothing is settled. If settlement fails, the balance is not credited and the response is HTTP 402. A Base top-up is checked before settling: the wallet's USDC must cover it plus its other unsettled Base payments, or the response is HTTP 402 (`Insufficient USDC balance`, or `...: other payments from this wallet are still settling`); if the check cannot run, HTTP 503 `Payment network unavailable, retry`. If the settlement was sent but its confirmation is still pending, the response adds `"settlement_pending": true`. A Solana top-up is then credited at once. A Base top-up answers HTTP 202 with `"credited": "0"`, `pending_credit`, `transaction` and a `message`, and the credit is added once the transaction's receipt shows the transfer; a reverted transfer, an authorization that expires first, or a gateway restart meanwhile credits nothing. Do not sign a new payment while it is pending. If the settle status is unknown (it crashed, maybe after sending), nothing is credited and the response is HTTP 502 with `outcome_unknown: true`: check your wallet and balance before signing a new payment. A payment that another top-up or a job is using gets HTTP 409, and a non-finite `amount` (NaN, Infinity) or a JSON integer longer than 40 characters (`Invalid JSON body`) gets HTTP 400. Base top-ups take EIP-3009 authorizations only; Permit2 gets 402.

Response:
```json
{
  "pubkey": "YourWalletAddress",
  "credited": "0.095000",
  "balance": "0.095000",
  "warning": "Balance is stored in memory only. It will be lost on server restart."
}
```

Pending Base top-up (HTTP 202):
```json
{
  "pubkey": "0xYourEvmAddress",
  "credited": "0",
  "pending_credit": "0.095000",
  "balance": "0",
  "warning": "Balance is stored in memory only. It will be lost on server restart.",
  "settlement_pending": true,
  "transaction": "0x...",
  "message": "Do not sign a new payment; your balance is credited once the transaction confirms."
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

### Response Headers and Body

Prepaid responses report the cost of the request and the remaining balance twice: in headers and in a top-level `prepaid` object in the JSON body.

| Header | Body field | Description |
|--------|------------|-------------|
| `X-Prepaid-Cost` | `prepaid.cost` | Actual cost deducted for this request in USD (e.g. `0.0123`); `0` if nothing was deducted |
| `X-Prepaid-Balance` | `prepaid.balance` | Remaining prepaid balance after deduction (e.g. `0.0535824`) |

```json
{
  "data": { "...": "provider result" },
  "prepaid": { "cost": "0.0012", "balance": "0.9988" }
}
```

Amounts are strings in plain decimal notation, never exponent notation, rounded down to $0.00000001 (a millionth of a cent). The charge itself is rounded the same way, so `old balance − cost = new balance` holds exactly. Use `cost` rather than the difference between two balances: parallel requests from one wallet interleave their balances.

A failure the provider reported returns the full hold. Its error JSON has `"prepaid": {"cost": "0", "balance": "..."}` and the same headers. When the outcome is unknown (the task timed out, or the connection to the provider dropped after the request was sent), the job may have run and billed us, so the reserved estimate is charged. That error has `"outcome_unknown": true` and `prepaid.cost` equal to the estimate.

If the client disconnects during a long job, the job still finishes and is charged or refunded by the same rules.

**Long jobs:** any call, synchronous or polled, that runs longer than 20 seconds is kept open with newline heartbeats. Its headers go out with the first heartbeat, before the cost is known, so that response has no `X-Prepaid-*` headers and the cost is **only in the body**. The HTTP status is already 200 by then. A failure shows up as the usual error JSON, with `status` and `prepaid`. Reading the body is the reliable way for every request.

The `prepaid` object appears only on prepaid requests. Do not confuse it with `refunded_to_prepaid` and `prepaid_balance`: those appear on a failed **x402** payment (Base or Solana), when the settled amount is credited to the payer's prepaid balance.

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
- **No cash-out** — prepaid balances cannot be withdrawn back to USDC
- **x402 job refunds** — if a Base or Solana payment settled and the provider then reports a failure, the amount is credited to that wallet's prepaid balance. It is still in-memory and lost on restart. This is not an on-chain refund
- **No multi-server** — balances are local to a single server instance
