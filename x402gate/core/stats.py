"""In-memory statistics collector for x402gate dashboard.

Tracks per-provider request counts, revenue, latency, and status.
Captures existing logs via a custom logging.Handler attached to
the x402gate logger.

All data is stored in RAM and resets on server restart.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

# How long (seconds) before a provider with no requests goes "unknown"
STALE_THRESHOLD = 1800  # 30 minutes
MAX_LOG_ENTRIES = 5000  # Rolling log buffer size


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
class _GlobalStats:
    """Singleton holding all gateway statistics."""

    started_at: float = field(default_factory=time.time)
    providers: dict[str, ProviderStats] = field(default_factory=dict)
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
        try:
            _stats.logs.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "message": self.format(record),
                }
            )
        except Exception:
            pass  # Never break the application


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


def get_stats() -> dict[str, Any]:
    """Return a snapshot of all statistics for the dashboard."""
    now = time.time()
    uptime_s = now - _stats.started_at

    provider_data: dict[str, Any] = {}
    for name, ps in _stats.providers.items():
        # Determine effective status (stale → unknown)
        effective_status = ps.last_status
        if ps.last_status_at > 0 and (now - ps.last_status_at) > STALE_THRESHOLD:
            effective_status = "unknown"
        elif ps.total_requests == 0:
            effective_status = "unknown"

        avg_latency_ms = (
            round(ps.total_latency_s / ps.total_requests * 1000, 1)
            if ps.total_requests > 0
            else 0
        )
        success_rate = (
            round(ps.success_count / ps.total_requests * 100, 1)
            if ps.total_requests > 0
            else 0
        )

        provider_data[name] = {
            "status": effective_status,
            "total_requests": ps.total_requests,
            "success_count": ps.success_count,
            "error_count": ps.error_count,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency_ms,
            "revenue_usd": str(ps.total_revenue_usd),
            "cost_usd": str(ps.total_cost_usd),
            "profit_usd": str(ps.total_revenue_usd - ps.total_cost_usd),
            "last_error": ps.last_error,
            "last_activity_ago_s": (
                round(now - ps.last_status_at, 0) if ps.last_status_at > 0 else None
            ),
        }

    total_requests = sum(ps.total_requests for ps in _stats.providers.values())
    total_revenue = sum(
        ps.total_revenue_usd for ps in _stats.providers.values()
    )
    total_cost = sum(ps.total_cost_usd for ps in _stats.providers.values())

    return {
        "uptime_s": round(uptime_s, 0),
        "total_requests": total_requests,
        "total_revenue_usd": str(total_revenue),
        "total_cost_usd": str(total_cost),
        "total_profit_usd": str(total_revenue - total_cost),
        "total_topups": _stats.total_topups,
        "total_topup_usd": str(_stats.total_topup_usd),
        "providers": provider_data,
    }


def get_logs(limit: int = 200) -> list[dict[str, Any]]:
    """Return the most recent log entries (newest first)."""
    all_entries = list(_stats.logs)
    if limit:
        all_entries = all_entries[-limit:]
    return list(reversed(all_entries))


def reset() -> None:
    """Clear all stats (for testing)."""
    _stats.providers.clear()
    _stats.total_topups = 0
    _stats.total_topup_usd = Decimal("0")
    _stats.logs = collections.deque(maxlen=MAX_LOG_ENTRIES)
    _stats.started_at = time.time()
