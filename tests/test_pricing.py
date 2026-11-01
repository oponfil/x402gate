"""Unit tests for pricing logic."""

import hashlib
import json
from decimal import Decimal
from unittest.mock import patch

from x402gate.core.pricing import PriceCache, apply_commission, format_price_for_x402


class TestApplyCommission:
    """Tests for the apply_commission function."""

    def test_standard_commission(self):
        """4% commission on $0.003 = $0.003120."""
        result = apply_commission(Decimal("0.003"), 0.04)
        assert result == Decimal("0.003120")

    def test_zero_commission(self):
        """0% commission returns the original price."""
        result = apply_commission(Decimal("0.003"), 0.0)
        assert result == Decimal("0.003000")

    def test_high_commission(self):
        """50% commission on $1.00 = $1.500000."""
        result = apply_commission(Decimal("1.00"), 0.5)
        assert result == Decimal("1.500000")

    def test_rounds_up_to_6_decimals(self):
        """Commission result is rounded UP to 6 decimal places (USDC precision)."""
        # 0.007 * 1.04 = 0.00728 -> rounds to 0.007280
        result = apply_commission(Decimal("0.007"), 0.04)
        assert result == Decimal("0.007280")

    def test_tiny_price(self):
        """Very small prices are rounded up, never to zero."""
        result = apply_commission(Decimal("0.0000001"), 0.04)
        assert result > Decimal("0")

    def test_returns_decimal(self):
        """Result is always a Decimal."""
        result = apply_commission(Decimal("1.0"), 0.04)
        assert isinstance(result, Decimal)


class TestFormatPriceForX402:
    """Tests for x402 price string formatting."""

    def test_standard_format(self):
        """Price is formatted as $X.XXXXXX."""
        result = format_price_for_x402(Decimal("0.003150"))
        assert result == "$0.003150"

    def test_whole_dollar(self):
        """Whole dollar amount."""
        result = format_price_for_x402(Decimal("1.000000"))
        assert result == "$1.000000"

    def test_starts_with_dollar_sign(self):
        """Format always starts with $."""
        result = format_price_for_x402(Decimal("0.5"))
        assert result.startswith("$")


class TestPriceCache:
    """Tests for the TTL price cache."""

    def test_set_and_get(self):
        """Cached values can be retrieved."""
        cache = PriceCache(ttl=60)
        cache.set("model-a", {"prompt": "cat"}, Decimal("0.003"))
        result = cache.get("model-a", {"prompt": "cat"})
        assert result == Decimal("0.003")

    def test_cache_miss(self):
        """Missing keys return None."""
        cache = PriceCache(ttl=60)
        result = cache.get("model-a", {"prompt": "cat"})
        assert result is None

    def test_different_inputs_different_keys(self):
        """Different inputs produce different cache keys."""
        cache = PriceCache(ttl=60)
        cache.set("model-a", {"prompt": "cat"}, Decimal("0.003"))
        result = cache.get("model-a", {"prompt": "dog"})
        assert result is None

    def test_same_inputs_same_key(self):
        """Same model+inputs always hits the same cache entry."""
        cache = PriceCache(ttl=60)
        cache.set("model-a", {"prompt": "cat", "size": 512}, Decimal("0.005"))
        result = cache.get("model-a", {"size": 512, "prompt": "cat"})
        # Keys are sorted, so different dict order = same key
        assert result == Decimal("0.005")

    def test_disabled_cache(self):
        """TTL=0 disables caching entirely."""
        cache = PriceCache(ttl=0)
        cache.set("model-a", {"prompt": "cat"}, Decimal("0.003"))
        result = cache.get("model-a", {"prompt": "cat"})
        assert result is None

    def test_clear(self):
        """clear() removes all entries."""
        cache = PriceCache(ttl=60)
        cache.set("model-a", {}, Decimal("0.003"))
        cache.clear()
        assert cache.get("model-a", {}) is None

    def test_file_bytes_are_hashed(self):
        """Uploads with different bytes do not share a cache entry."""
        cache = PriceCache(ttl=60)
        cache.set("model-a", {"_file_bytes": b"aaa", "name": "a"}, Decimal("1"))
        assert cache.get("model-a", {"name": "a", "_file_bytes": b"aaa"}) == Decimal("1")
        assert cache.get("model-a", {"name": "a", "_file_bytes": b"bbb"}) is None
        key = PriceCache._make_key("model-a", {"_file_bytes": b"hello world" * 1000})
        assert "hello" not in key
        assert len(key) < 80


class TestPriceCacheBound:
    """Unpaid requests cannot grow the cache without limit."""

    def test_oldest_entry_is_dropped_at_the_cap(self):
        cache = PriceCache(ttl=60, max_entries=2)
        cache.set("p/a", {"x": 1}, Decimal("1"))
        cache.set("p/a", {"x": 2}, Decimal("2"))
        cache.set("p/a", {"x": 3}, Decimal("3"))
        assert cache.get("p/a", {"x": 1}) is None
        assert cache.get("p/a", {"x": 2}) == Decimal("2")
        assert cache.get("p/a", {"x": 3}) == Decimal("3")

    def test_reset_entry_moves_to_newest(self):
        cache = PriceCache(ttl=60, max_entries=2)
        cache.set("p/a", {"x": 1}, Decimal("1"))
        cache.set("p/a", {"x": 2}, Decimal("2"))
        cache.set("p/a", {"x": 1}, Decimal("1"))
        cache.set("p/a", {"x": 3}, Decimal("3"))
        assert cache.get("p/a", {"x": 1}) == Decimal("1")
        assert cache.get("p/a", {"x": 2}) is None


class TestPriceCacheKey:
    """A request hashes its body once and reuses the key."""

    def test_key_round_trip(self):
        cache = PriceCache(ttl=60)
        key = cache.key("p/a", {"prompt": "cat"})
        cache.set_by_key(key, Decimal("0.5"))
        assert cache.get_by_key(key) == Decimal("0.5")
        # The dict wrappers use the same key.
        assert cache.get("p/a", {"prompt": "cat"}) == Decimal("0.5")

    def test_disabled_cache_does_not_hash(self):
        """With TTL 0 no key is computed, so a large upload is never hashed."""
        cache = PriceCache(ttl=0)
        with patch("x402gate.core.pricing.hashlib.sha256") as sha:
            key = cache.key("p/a", {"_file_bytes": b"x" * 10})
        assert key is None
        sha.assert_not_called()
        cache.set_by_key(key, Decimal("1"))
        assert cache.get_by_key(key) is None

    def test_file_is_hashed_once_per_key(self):
        cache = PriceCache(ttl=60)
        upload = {"_file_bytes": b"audio" * 1000, "model": "m"}
        with patch("x402gate.core.pricing.hashlib.sha256", wraps=hashlib.sha256) as sha:
            key = cache.key("p/stt", upload)
            cache.get_by_key(key)
            cache.set_by_key(key, Decimal("1"))
        # One for the file, one for the small JSON around it.
        assert sha.call_count == 2

    def test_raw_body_key(self):
        """Equal raw bytes share a price; any byte difference does not."""
        cache = PriceCache(ttl=60)
        raw = json.dumps({"prompt": "cat"}).encode()
        cache.set_by_key(cache.key("p/a", {"prompt": "cat"}, raw), Decimal("2"))
        assert cache.get_by_key(cache.key("p/a", {"prompt": "cat"}, raw)) == Decimal("2")
        assert cache.get_by_key(cache.key("p/a", {}, raw + b" ")) is None
        assert cache.get_by_key(cache.key("p/b", {"prompt": "cat"}, raw)) is None
        # Raw keys and parsed-body keys never share an entry.
        assert cache.get("p/a", {"prompt": "cat"}) is None

    def test_raw_key_is_short(self):
        key = PriceCache._make_key("p/a", {}, b'{"image": "' + b"A" * 100_000 + b'"}')
        assert "AAAA" not in key
        assert len(key) < 80
