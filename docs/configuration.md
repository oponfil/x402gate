# Configuration

x402gate is configured via `config.yaml` in the project root. Secrets are stored in `.env` and interpolated using `${VAR}` syntax.

## config.yaml Reference

### `gateway`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | int | `4021` | Server port |
| `commission` | float | `0.05` | Markup rate on provider prices (0.05 = 5%) |
| `gas_surcharge` | float | `0.001` | Fixed gas surcharge ($0.001) added per request |
| `default_max_tokens` | int | `1024` | Default `max_tokens` for token-based providers when client omits it |
| `price_cache_ttl` | int | `60` | Price cache TTL in seconds. `0` = disabled |

### `payment`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_timeout` | int | `3600` | Maximum timeout for payment validity (seconds) |

#### `payment.networks.<name>`

Each network is a key under `payment.networks` (e.g. `base`, `solana`):

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | — | **Required.** `"evm"` or `"svm"` |
| `network` | string | — | **Required.** CAIP-2 network id (e.g. `eip155:8453`, `solana:5eykt...`) |
| `pay_to` | string | — | **Required.** Wallet address to receive USDC |
| `token_address` | string | — | **Required.** USDC contract/mint address |
| `token_name` | string | `"USD Coin"` | EIP-712 domain name (EVM only) |
| `token_version` | string | `"2"` | EIP-712 domain version (EVM only) |
| `rpc_url` | string | — | **Required.** Blockchain RPC endpoint |
| `facilitator_key` | string | — | **Required.** Private key for on-chain settlement |

### `providers`

Each provider is a key under `providers`:

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"managed"` | `"managed"` = full x402 flow, `"passthrough"` = transparent proxy |
| `enabled` | bool | `true` | Whether this provider is active |
| `base_url` | string | — | **Required.** Provider API base URL |
| `api_key` | string | `""` | Provider API key (use `${ENV_VAR}` syntax) |
| `poll_interval` | int | `2` | Seconds between async task status polls |
| `poll_timeout` | int | `300` | Max seconds to wait for task completion |
| `docs_url` | string | `""` | Link to provider API documentation |
| `example_request` | dict | `{}` | Example request body for the landing page |

## Environment Variables

Store secrets in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `WAVESPEED_API_KEY` | Yes | WaveSpeed AI API key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `BASE_PAY_TO_ADDRESS` | Yes | EVM wallet address for receiving USDC |
| `BASE_FACILITATOR_PRIVATE_KEY` | Yes | EVM private key for on-chain settlement (needs ETH for gas) |
| `SOLANA_PAY_TO_ADDRESS` | Solana | Solana wallet address for receiving USDC (base58) |
| `SOLANA_FACILITATOR_PRIVATE_KEY` | Solana | Solana private key for settlement (base58, needs SOL for gas) |
| `BASE_E2ETEST_PRIVATE_KEY` | E2E only | Base client wallet key for E2E tests |
| `SOLANA_E2ETEST_PRIVATE_KEY` | E2E only | Solana client wallet key for E2E tests |
