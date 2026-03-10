"""Unit tests for the stats module."""

from decimal import Decimal
import time

from x402gate.core.stats import (
    MAX_LOG_ENTRIES,
    STALE_THRESHOLD,
    get_logs,
    get_stats,
    init,
    record_request,
    record_revenue,
    record_topup,
    reset,
)


class TestRecordRequest:
    """Tests for record_request()."""

    def setup_method(self):
        reset()
        init(["openrouter", "wavespeed"])

    def test_increments_total_requests(self):
        record_request("openrouter", 0.5, True)
        record_request("openrouter", 0.3, True)
        s = get_stats()
        assert s["providers"]["openrouter"]["total_requests"] == 2

    def test_counts_success_and_error(self):
        record_request("openrouter", 0.5, True)
        record_request("openrouter", 0.3, False, error_msg="timeout")
        record_request("openrouter", 0.2, True)
        s = get_stats()
        p = s["providers"]["openrouter"]
        assert p["success_count"] == 2
        assert p["error_count"] == 1
        assert p["success_rate"] == round(2 / 3 * 100, 1)

    def test_calculates_avg_latency(self):
        record_request("wavespeed", 1.0, True)
        record_request("wavespeed", 3.0, True)
        s = get_stats()
        assert s["providers"]["wavespeed"]["avg_latency_ms"] == 2000.0

    def test_sets_status_ok_on_success(self):
        record_request("openrouter", 0.1, True)
        s = get_stats()
        assert s["providers"]["openrouter"]["status"] == "ok"

    def test_sets_status_error_on_failure(self):
        record_request("openrouter", 0.1, False, error_msg="API down")
        s = get_stats()
        p = s["providers"]["openrouter"]
        assert p["status"] == "error"
        assert p["last_error"] == "API down"

    def test_unknown_provider_auto_created(self):
        record_request("new_provider", 0.1, True)
        s = get_stats()
        assert "new_provider" in s["providers"]
        assert s["providers"]["new_provider"]["total_requests"] == 1


class TestProviderStatus:
    """Tests for provider status transitions."""

    def setup_method(self):
        reset()
        init(["test_provider"])

    def test_initial_status_is_unknown(self):
        s = get_stats()
        assert s["providers"]["test_provider"]["status"] == "unknown"

    def test_stale_provider_becomes_unknown(self):
        record_request("test_provider", 0.1, True)
        # Manually set last_status_at to stale
        from x402gate.core.stats import _stats
        _stats.providers["test_provider"].last_status_at = (
            time.time() - STALE_THRESHOLD - 1
        )
        s = get_stats()
        assert s["providers"]["test_provider"]["status"] == "unknown"


class TestRecordRevenue:
    """Tests for record_revenue()."""

    def setup_method(self):
        reset()
        init(["openrouter"])

    def test_accumulates_revenue(self):
        record_revenue("openrouter", Decimal("0.01"), Decimal("0.008"))
        record_revenue("openrouter", Decimal("0.02"), Decimal("0.015"))
        s = get_stats()
        p = s["providers"]["openrouter"]
        assert p["revenue_usd"] == "0.03"
        assert p["cost_usd"] == "0.023"
        assert p["profit_usd"] == "0.007"

    def test_totals(self):
        record_revenue("openrouter", Decimal("0.05"), Decimal("0.03"))
        s = get_stats()
        assert s["total_revenue_usd"] == "0.05"
        assert s["total_cost_usd"] == "0.03"
        assert s["total_profit_usd"] == "0.02"


class TestRecordTopup:
    """Tests for record_topup()."""

    def setup_method(self):
        reset()

    def test_counts_topups(self):
        record_topup(Decimal("1.00"))
        record_topup(Decimal("2.50"))
        s = get_stats()
        assert s["total_topups"] == 2
        assert s["total_topup_usd"] == "3.50"


class TestLogs:
    """Tests for the log buffer."""

    def setup_method(self):
        reset()

    def test_logs_capped_at_max(self):
        from x402gate.core.stats import _stats
        for i in range(MAX_LOG_ENTRIES + 500):
            _stats.logs.append({"ts": time.time(), "level": "INFO", "message": f"msg {i}"})
        assert len(_stats.logs) == MAX_LOG_ENTRIES
        # Oldest entries should have been evicted
        logs = get_logs(limit=0)
        assert logs[-1]["message"] == f"msg 500"

    def test_get_logs_returns_newest_first(self):
        from x402gate.core.stats import _stats
        _stats.logs.append({"ts": 1.0, "level": "INFO", "message": "first"})
        _stats.logs.append({"ts": 2.0, "level": "INFO", "message": "second"})
        logs = get_logs()
        assert logs[0]["message"] == "second"
        assert logs[1]["message"] == "first"

    def test_get_logs_respects_limit(self):
        from x402gate.core.stats import _stats
        for i in range(10):
            _stats.logs.append({"ts": float(i), "level": "INFO", "message": f"msg {i}"})
        logs = get_logs(limit=3)
        assert len(logs) == 3


class TestGetStats:
    """Tests for get_stats() structure."""

    def setup_method(self):
        reset()
        init(["openrouter", "wavespeed"])

    def test_returns_complete_structure(self):
        s = get_stats()
        assert "uptime_s" in s
        assert "total_requests" in s
        assert "total_revenue_usd" in s
        assert "total_cost_usd" in s
        assert "total_profit_usd" in s
        assert "total_topups" in s
        assert "total_topup_usd" in s
        assert "providers" in s
        assert "openrouter" in s["providers"]
        assert "wavespeed" in s["providers"]

    def test_provider_has_all_fields(self):
        record_request("openrouter", 0.5, True)
        s = get_stats()
        p = s["providers"]["openrouter"]
        expected_keys = {
            "status", "total_requests", "success_count", "error_count",
            "success_rate", "avg_latency_ms", "revenue_usd", "cost_usd",
            "profit_usd", "last_error", "last_activity_ago_s",
        }
        assert expected_keys.issubset(set(p.keys()))


class TestReset:
    """Tests for reset()."""

    def test_clears_everything(self):
        init(["test"])
        record_request("test", 0.1, True)
        record_topup(Decimal("5.00"))
        reset()
        s = get_stats()
        assert s["total_requests"] == 0
        assert s["total_topups"] == 0
        assert len(s["providers"]) == 0
