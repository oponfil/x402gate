"""Unit tests for pricing logic."""

from decimal import Decimal

from x402gate.core.pricing import PriceCache, apply_commission, format_price_for_x402


class TestApplyCommission:
    """Tests for the apply_commission function."""

    def test_standard_commission(self):
        """5% commission on $0.003 = $0.003150."""
        result = apply_commission(Decimal("0.003"), 0.05)
        assert result == Decimal("0.003150")

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
        # 0.007 * 1.05 = 0.00735 -> rounds to 0.007350
        result = apply_commission(Decimal("0.007"), 0.05)
        assert result == Decimal("0.007350")

    def test_tiny_price(self):
        """Very small prices are rounded up, never to zero."""
        result = apply_commission(Decimal("0.0000001"), 0.05)
        assert result > Decimal("0")

    def test_returns_decimal(self):
        """Result is always a Decimal."""
        result = apply_commission(Decimal("1.0"), 0.05)
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
