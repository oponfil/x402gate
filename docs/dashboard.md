# Dashboard

The dashboard (`/dashboard`) provides real-time monitoring of gateway health, provider metrics, and financial performance. It auto-refreshes every 5 seconds.

## Sections

### Header

- **Uptime** — time since server start
- **Requests** — total request count across all providers (including topup)
- **Auto-refresh indicator** — green pulsing dot

### Provider Cards

One card per registered provider + a `topup` system card. Each card shows:

| Field | Description |
|---|---|
| **Status dot** | 🟢 ok, 🔴 error, ⚫ unknown (no requests or stale > 4h) |
| **Type badge** | `managed`, `passthrough`, or `system` (topup) |
| **Requests** | Total request count |
| **Success** | Success rate (%) |
| **Avg job time** | Average request latency |
| **Last activity** | Time since last request |
| **Revenue** | Total revenue (USD) |
| **Profit** | Revenue − Cost − Gas |

If a provider's last request was an error, the card shows the error message in a red banner.

### Summary Bar

Bottom-line financial totals with de-duplication (see [Financial Metrics](#financial-metrics) below):

- **Revenue** — real money received
- **Cost** — real money paid to providers
- **Gas** — total blockchain fees
- **Profit** — Revenue − Cost − Gas
- **Prepaid balance** — unspent prepaid funds across all wallets

### Network Cards

One card per blockchain network (Base, Solana). Each card shows:

| Field | Description |
|---|---|
| **Gas label** | Native token (ETH, SOL) |
| **Settlements** | Total on-chain settlement count |
| **Last activity** | Time since last settlement |
| **Revenue** | Total settled revenue on this chain |
| **Avg overhead** | Average x402 protocol overhead per request |
| **Total gas** | Total gas spent (USD) |
| **Avg gas** | Average gas per settlement (USD) |

### Live Logs

Rolling buffer of the last 1000 log entries from all `x402gate.*` loggers. Newest entries on top. Includes a **Copy All** button.

---

## Financial Metrics

### Provider Cards

Each card shows metrics scoped to that provider:

| Metric | x402 request (direct payment) | Prepaid request (from balance) |
|---|---|---|
| **Revenue** | Settlement amount (includes commission) | `actual_base_price` |
| **Cost** | `actual_base_price` (real provider cost) | `actual_base_price` |
| **Gas** | Blockchain settlement gas | 0 (no settlement) |
| **Profit** | Revenue − Cost − Gas ≈ commission − gas | 0 |

#### Topup Card

| Metric | Value |
|---|---|
| **Revenue** | Full top-up amount |
| **Cost** | `net_credit` (amount credited to balance = topup − commission − gas_surcharge) |
| **Gas** | Actual blockchain settlement gas |
| **Profit** | commission + gas_surcharge − actual_gas |

### Summary Bar (Bottom-Line Totals)

The summary bar shows **de-duplicated** financial totals to avoid double-counting between top-ups and prepaid usage:

| Metric | Formula | What it means |
|---|---|---|
| **Revenue** | Sum of card revenues **minus** prepaid usage | Real money received (topup payments + x402 direct payments) |
| **Cost** | Sum of card costs **minus** topup net_credit | Real money paid to providers |
| **Gas** | Sum of gas across all providers | Blockchain fees |
| **Profit** | Revenue − Cost − Gas | Current actual profit |
| **Prepaid balance** | Sum of all wallet balances | Unspent prepaid funds |

#### Why de-duplication is needed

When a user tops up $7, the gateway records topup revenue = $7 and cost = net_credit (~$6.72). When that balance is spent on requests, each provider records revenue = cost = actual_base_price. Without de-duplication, both the topup and the usage would be counted, inflating Revenue and Cost.

The summary bar removes:
- **Prepaid usage** from Revenue (real money came at top-up, not at each request)
- **Topup net_credit** from Cost (that money flows into prepaid balance and appears as provider costs when spent)

This ensures Profit = real income − real expenses − gas.

---

## Data Source

All stats are in-memory (`x402gate/core/stats.py`) and **reset on server restart**. Prepaid balances are also in-memory — unspent balance becomes profit after restart.

| Endpoint | Description |
|---|---|
| `GET /v1/stats` | Stats JSON (used by dashboard JS) |
| `GET /v1/logs` | Log entries JSON |
| `GET /dashboard` | HTML dashboard |
