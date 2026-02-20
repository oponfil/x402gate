"""Pricing logic for x402gate.

Fetches dynamic prices from providers and applies commission markup.
Includes TTL-based caching to avoid redundant pricing API calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from decimal import ROUND_UP, Decimal
from typing import Any


class PriceCache:
    """Simple TTL cache for pricing lookups.

    Keys are (model_id, inputs_hash) tuples. Expired entries are
    lazily evicted on the next get/set call.
    """

    def __init__(self, ttl: int = 60) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[Decimal, float]] = {}

    @staticmethod
    def _make_key(model_id: str, inputs: dict[str, Any]) -> str:
        """Create a deterministic cache key from model_id and inputs."""
        inputs_json = json.dumps(inputs, sort_keys=True, default=str)
        inputs_hash = hashlib.sha256(inputs_json.encode()).hexdigest()[:16]
        return f"{model_id}:{inputs_hash}"

    def get(self, model_id: str, inputs: dict[str, Any]) -> Decimal | None:
        """Return cached price if it exists and hasn't expired."""
        if self._ttl <= 0:
            return None
        key = self._make_key(model_id, inputs)
        entry = self._store.get(key)
        if entry is None:
            return None
        price, timestamp = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return price

    def set(self, model_id: str, inputs: dict[str, Any], price: Decimal) -> None:
        """Store a price in the cache with the current timestamp."""
        if self._ttl <= 0:
            return
        key = self._make_key(model_id, inputs)
        self._store[key] = (price, time.monotonic())

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()


def apply_commission(
    base_price: Decimal, commission_rate: float, gas_surcharge: float = 0.0
) -> Decimal:
    """Apply commission markup to a base price.

    Args:
        base_price: The provider's base cost in USD.
        commission_rate: Commission as a decimal (e.g., 0.05 for 5%).
        gas_surcharge: Fixed gas surcharge in USD always added on top (e.g., 0.001).

    Returns:
        Final price rounded up to 6 decimal places (USDC precision).
    """
    commission = base_price * Decimal(str(commission_rate))
    gas_fee = Decimal(str(gas_surcharge)) if gas_surcharge > 0 else Decimal("0")
    final = base_price + commission + gas_fee
    # Round up to 6 decimal places (USDC has 6 decimals)
    return final.quantize(Decimal("0.000001"), rounding=ROUND_UP)


def format_price_for_x402(price: Decimal) -> str:
    """Format a Decimal price as an x402-compatible string.

    The x402 protocol expects prices as strings like "$0.00315".

    Args:
        price: Price in USD as a Decimal.

    Returns:
        Price string in x402 format, e.g. "$0.003150".
    """
    return f"${price}"
