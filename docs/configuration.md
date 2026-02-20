# Configuration

x402gate is configured via `config.yaml` in the project root. Secrets are stored in `.env` and interpolated using `${VAR}` syntax.

## config.yaml Reference

### `gateway`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | int | `4021` | Server port |
| `commission` | float | `0.05` | Markup rate on provider prices (0.05 = 5%) |
| `price_cache_ttl` | int | `60` | Price cache TTL in seconds. `0` = disabled |

### `payment`

| Field | Type | Default | Description |
|---|---|---|---|
| `pay_to` | string | — | **Required.** EVM wallet address to receive USDC |
| `network` | string | `"eip155:8453"` | Blockchain network (CAIP-2). `eip155:8453` = Base Mainnet |
| `scheme` | string | `"exact"` | x402 payment scheme |
| `token_address` | string | — | **Required.** USDC contract address on the target network |
| `token_name` | string | `"USDC"` | EIP-712 domain name of the token contract |
| `token_version` | string | `"2"` | EIP-712 domain version of the token contract |
| `max_timeout` | int | `3600` | Maximum timeout for payment validity (seconds) |
| `rpc_url` | string | `"https://mainnet.base.org"` | Blockchain RPC endpoint |
| `facilitator_key` | string | — | **Required.** Private key for settling payments on-chain |

### `providers`

Each provider is a key under `providers`:

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Whether this provider is active |
| `base_url` | string | — | **Required.** Provider API base URL |
| `api_key` | string | `""` | Provider API key (use `${ENV_VAR}` syntax) |
| `poll_interval` | int | `2` | Seconds between async task status polls |
| `poll_timeout` | int | `300` | Max seconds to wait for task completion |

## Environment Variables

Store secrets in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `WAVESPEED_API_KEY` | Yes | WaveSpeed AI API key |
| `BASE_PAY_TO_ADDRESS` | Yes | EVM wallet address for receiving USDC |
| `BASE_FACILITATOR_PRIVATE_KEY` | Yes | EVM private key for on-chain settlement (needs ETH for gas) |
| `SOLANA_PAY_TO_ADDRESS` | Solana | Solana wallet address for receiving USDC (base58) |
| `SOLANA_FACILITATOR_PRIVATE_KEY` | Solana | Solana private key for settlement (base58, needs SOL for gas) |
| `E2ETEST_BASE_PRIVATE_KEY` | E2E only | Base client wallet key for E2E tests |
| `SOLANA_E2ETEST_PRIVATE_KEY` | E2E only | Solana client wallet key for E2E tests |
