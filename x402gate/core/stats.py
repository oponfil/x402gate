"""In-memory statistics collector for x402gate dashboard.

Tracks per-provider request counts, revenue, latency, and status.
Captures existing logs via a custom logging.Handler attached to
the x402gate logger.

All data is stored in RAM and resets on server restart.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from x402gate.core.prepaid import get_total_balance

# How long (seconds) before a provider with no requests goes "unknown"
STALE_THRESHOLD = 14400  # 4 hours
MAX_LOG_ENTRIES = 1000  # Rolling log buffer size


@dataclass
class ProviderStats:
    """Per-provider statistics."""

    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    total_revenue_usd: Decimal = Decimal("0")
    total_cost_usd: Decimal = Decimal("0")
    total_latency_s: float = 0.0  # sum, for computing average
    last_status: str = "unknown"  # "ok" | "error" | "unknown"
    last_status_at: float = 0.0  # time.time() of last request
    last_error: str = ""


@dataclass
class NetworkStats:
    """Per-blockchain-network settlement statistics."""

    total_settlements: int = 0
    total_settle_latency_s: float = 0.0
    total_gas_cost_usd: float = 0.0
    total_gas_cost_native: float = 0.0
    gas_label: str = ""  # "ETH" or "SOL"
    total_overhead_s: float = 0.0  # sum of x402 overhead per request
    overhead_count: int = 0


@dataclass
class _GlobalStats:
    """Singleton holding all gateway statistics."""

    started_at: float = field(default_factory=time.time)
    providers: dict[str, ProviderStats] = field(default_factory=dict)
    networks: dict[str, NetworkStats] = field(default_factory=dict)
    total_topups: int = 0
    total_topup_usd: Decimal = Decimal("0")
    logs: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=MAX_LOG_ENTRIES)
    )


# Module-level singleton
_stats = _GlobalStats()


# ---------------------------------------------------------------------------
# Log capture handler
# ---------------------------------------------------------------------------


class _DashboardLogHandler(logging.Handler):
    """Captures log records from x402gate loggers into the stats log buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):
            _stats.logs.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "message": self.format(record),
                }
            )


def install_log_handler() -> None:
    """Attach the dashboard log handler to the x402gate root logger.

    Call once at application startup (in lifespan).
    """
    handler = _DashboardLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("x402gate")
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init(provider_names: list[str]) -> None:
    """Initialize stats for known providers (call at startup)."""
    _stats.started_at = time.time()
    for name in provider_names:
        if name not in _stats.providers:
            _stats.providers[name] = ProviderStats()


def record_request(
    provider: str,
    latency_s: float,
    success: bool,
    error_msg: str | None = None,
) -> None:
    """Record the outcome of a provider request."""
    ps = _stats.providers.setdefault(provider, ProviderStats())
    ps.total_requests += 1
    ps.total_latency_s += latency_s
    ps.last_status_at = time.time()

    if success:
        ps.success_count += 1
        ps.last_status = "ok"
    else:
        ps.error_count += 1
        ps.last_status = "error"
        ps.last_error = error_msg or "Unknown error"


def record_revenue(
    provider: str,
    amount_usd: Decimal,
    cost_usd: Decimal,
) -> None:
    """Record revenue from a settled payment."""
    ps = _stats.providers.setdefault(provider, ProviderStats())
    ps.total_revenue_usd += amount_usd
    ps.total_cost_usd += cost_usd


def record_topup(amount_usd: Decimal) -> None:
    """Record a prepaid top-up."""
    _stats.total_topups += 1
    _stats.total_topup_usd += amount_usd


def record_settlement(
    network: str,
    settle_latency_s: float,
    gas_cost_usd: float,
    gas_cost_native: float,
    gas_label: str,
) -> None:
    """Record a blockchain settlement for per-network statistics."""
    ns = _stats.networks.setdefault(network, NetworkStats())
    ns.total_settlements += 1
    ns.total_settle_latency_s += settle_latency_s
    ns.total_gas_cost_usd += gas_cost_usd
    ns.total_gas_cost_native += gas_cost_native
    if gas_label:
        ns.gas_label = gas_label


def record_overhead(
    network: str,
    overhead_s: float,
) -> None:
    """Record x402 protocol overhead for a request on a given network.

    Overhead = client_wait_time - provider_generation_time, i.e. the
    additional delay caused by price lookup, payment verification, etc.
    """
    ns = _stats.networks.setdefault(network, NetworkStats())
    ns.total_overhead_s += overhead_s
    ns.overhead_count += 1


def _fmt(d: Decimal) -> str:
    """Format Decimal to fixed-point with 4 decimal places."""
    return f"{d:.4f}"


def get_stats() -> dict[str, Any]:
    """Return a snapshot of all statistics for the dashboard."""
    now = time.time()
    uptime_s = now - _stats.started_at

    provider_data: dict[str, Any] = {}
    for name, ps in _stats.providers.items():
        # Determine effective status (stale → unknown)
        effective_status = ps.last_status
        stale = ps.last_status_at > 0 and (now - ps.last_status_at) > STALE_THRESHOLD
        if stale or ps.total_requests == 0:
            effective_status = "unknown"

        avg_latency_s = (
            round(ps.total_latency_s / ps.total_requests, 1) if ps.total_requests > 0 else 0
        )
        success_rate = (
            round(ps.success_count / ps.total_requests * 100, 1) if ps.total_requests > 0 else 0
        )

        provider_data[name] = {
            "status": effective_status,
            "total_requests": ps.total_requests,
            "success_count": ps.success_count,
            "error_count": ps.error_count,
            "success_rate": success_rate,
            "avg_latency_s": avg_latency_s,
            "revenue_usd": _fmt(ps.total_revenue_usd),
            "cost_usd": _fmt(ps.total_cost_usd),
            "profit_usd": _fmt(ps.total_revenue_usd - ps.total_cost_usd),
            "last_error": ps.last_error,
            "last_activity_ago_s": (
                round(now - ps.last_status_at, 0) if ps.last_status_at > 0 else None
            ),
        }

    total_requests = sum(ps.total_requests for ps in _stats.providers.values())
    total_revenue = sum(ps.total_revenue_usd for ps in _stats.providers.values())
    total_cost = sum(ps.total_cost_usd for ps in _stats.providers.values())

    # Network settlement stats
    network_data: dict[str, Any] = {}
    for net_name, ns in _stats.networks.items():
        avg_settle_s = (
            round(ns.total_settle_latency_s / ns.total_settlements, 1)
            if ns.total_settlements > 0
            else 0
        )
        avg_gas_usd = (
            ns.total_gas_cost_usd / ns.total_settlements if ns.total_settlements > 0 else 0.0
        )
        avg_overhead_s = (
            round(ns.total_overhead_s / ns.overhead_count, 2)
            if ns.overhead_count > 0
            else 0
        )
        network_data[net_name] = {
            "total_settlements": ns.total_settlements,
            "avg_settle_latency_s": avg_settle_s,
            "avg_overhead_s": avg_overhead_s,
            "total_gas_cost_usd": round(ns.total_gas_cost_usd, 4),
            "avg_gas_cost_usd": round(avg_gas_usd, 4),
            "total_gas_cost_native": round(ns.total_gas_cost_native, 6),
            "gas_label": ns.gas_label,
        }

    return {
        "uptime_s": round(uptime_s, 0),
        "total_requests": total_requests,
        "total_revenue_usd": _fmt(total_revenue),
        "total_cost_usd": _fmt(total_cost),
        "total_profit_usd": _fmt(total_revenue - total_cost),
        "total_topups": _stats.total_topups,
        "total_topup_usd": _fmt(_stats.total_topup_usd),
        "total_prepaid_balance_usd": _fmt(get_total_balance()),
        "providers": provider_data,
        "networks": network_data,
    }


def get_logs(limit: int = MAX_LOG_ENTRIES) -> list[dict[str, Any]]:
    """Return the most recent log entries (newest first)."""
    all_entries = list(_stats.logs)
    if limit:
        all_entries = all_entries[-limit:]
    return list(reversed(all_entries))


def reset() -> None:
    """Clear all stats (for testing)."""
    _stats.providers.clear()
    _stats.networks.clear()
    _stats.total_topups = 0
    _stats.total_topup_usd = Decimal("0")
    _stats.logs = collections.deque(maxlen=MAX_LOG_ENTRIES)
    _stats.started_at = time.time()
