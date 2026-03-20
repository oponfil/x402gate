# Configuration

x402gate is configured via `config.yaml` in the project root. Secrets are stored in `.env` and interpolated using `${VAR}` syntax.

## config.yaml Reference

### `gateway`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | int | `4021` | Server port |
| `commission` | float | `0.04` | Markup rate on provider prices (0.04 = 4%) |
| `gas_surcharge` | float | `0.001` | Fixed gas surcharge ($0.001) added per request |
| `default_max_tokens` | int | `2048` | Default `max_tokens` for token-based providers when client omits it (x402 mode only; skipped in prepaid mode) |
| `web_search_tokens_per_result` | int | `2048` | Estimated extra input tokens per web search result |
| `default_web_search_max_results` | int | `3` | Default `max_results` for web search plugins when client omits it |
| `web_search_cost_per_result` | float | `0.004` | Fixed cost per web search result ($0.004, OpenRouter Exa: $4/1000) |
| `price_cache_ttl` | int | `60` | Price cache TTL in seconds. `0` = disabled |
| `max_upload_mb` | int | `300` | Maximum file upload size in MB (all in RAM, no disk on Railway) |
| `max_prepaid_topup` | float | `10.0` | Maximum single top-up amount in USD |
| `min_prepaid_topup` | float | `0.10` | Minimum single top-up amount in USD |
| `prepaid_timestamp_window` | int | `300` | Signature validity window in seconds (checked on arrival, covers large uploads) |

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
| `fixed_price_usd` | float | `0.0` | Fixed price per request in USD (when provider has no pricing API) |
| `description` | string | `""` | Short description shown on the landing page |
| `poll_interval` | int | `2` | Seconds between async task status polls |
| `poll_timeout` | int | `600` | Max seconds to wait for task completion |
| `docs_url` | string | `""` | Link to provider API documentation |
| `example_request` | dict | `{}` | Example request for the landing page (`{model, body}`) |
| `example_request_2` | dict | `{}` | Second example request (e.g. video model) |
| `jwt_token` | string | `""` | JWT token for cookie-based auth (Tungsten only) |
| `cf_clearance` | string | `""` | Cloudflare clearance cookie (Tungsten only) |

## Environment Variables

Store secrets in `.env` (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `WAVESPEED_API_KEY` | Yes | WaveSpeed AI API key |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `TUNGSTEN_JWT_TOKEN` | Yes | Tungsten JWT cookie |
| `TUNGSTEN_CF_CLEARANCE` | Yes | Tungsten Cloudflare clearance cookie |
| `CLOUDCONVERT_API_KEY` | Yes | CloudConvert API key |
| `RAPIDAPI_KEY` | Yes | RapidAPI key for SocialDownload |
| `BASE_PAY_TO_ADDRESS` | Yes | EVM wallet address for receiving USDC |
| `BASE_FACILITATOR_PRIVATE_KEY` | Yes | EVM private key for on-chain settlement (needs ETH for gas) |
| `SOLANA_PAY_TO_ADDRESS` | Solana | Solana wallet address for receiving USDC (base58) |
| `SOLANA_FACILITATOR_PRIVATE_KEY` | Solana | Solana private key for settlement (base58, needs SOL for gas) |
| `SOLANA_RPC_URL` | Solana | Solana RPC endpoint (e.g. Helius: `https://mainnet.helius-rpc.com/?api-key=KEY`) |
| `BASE_E2ETEST_PRIVATE_KEY` | E2E only | Base client wallet key for E2E tests |
| `SOLANA_E2ETEST_PRIVATE_KEY` | E2E only | Solana client wallet key for E2E tests |
