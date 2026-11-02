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

    Keys are (model_id, inputs_hash) tuples. The caller puts the provider
    name into ``model_id``: two providers can share a path such as
    ``chat/completions`` with very different prices.

    Unpaid requests fill the cache too, so it is bounded. Once it holds
    ``max_entries`` keys, expired entries are dropped, then the oldest.

    An entry can carry ``extra``: values pricing left in the request body
    for submit, which a cache hit would otherwise lose.
    """

    def __init__(self, ttl: int = 60, max_entries: int = 10_000) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._store: dict[str, tuple[Decimal, float, dict[str, Any]]] = {}

    @staticmethod
    def _make_key(model_id: str, inputs: dict[str, Any], raw: bytes | None = None) -> str:
        """Create a deterministic cache key from model_id and inputs.

        File bytes are hashed on their own. Putting the raw upload into
        ``json.dumps`` would copy the whole file into the key.

        ``raw`` is the JSON body exactly as received. Hashing it skips
        re-serializing the parsed body (about 10 ms for 10 MB of base64).
        Equal bytes parse to equal inputs, so they share a price; bodies
        that only differ in key order or spacing get separate entries.

        The full digest is kept: the client chooses every byte, and a
        truncated hash would let it find a cheap body that shares the key
        of an expensive one.
        """
        if raw is not None:
            return f"{model_id}:raw:{hashlib.sha256(raw).hexdigest()}"
        safe: dict[str, Any] = {key: value for key, value in inputs.items() if key != "_file_bytes"}
        file_bytes = inputs.get("_file_bytes")
        if isinstance(file_bytes, (bytes, bytearray)):
            safe["_file_sha256"] = hashlib.sha256(file_bytes).hexdigest()
        inputs_json = json.dumps(safe, sort_keys=True, default=str)
        inputs_hash = hashlib.sha256(inputs_json.encode()).hexdigest()
        return f"{model_id}:{inputs_hash}"

    def key(self, model_id: str, inputs: dict[str, Any], raw: bytes | None = None) -> str | None:
        """Return the cache key for a request, or None when caching is off.

        A request computes it once and passes it to :meth:`get_by_key` and
        :meth:`set_by_key`: hashing a 24 MB upload takes tens of ms.
        """
        if self._ttl <= 0:
            return None
        return self._make_key(model_id, inputs, raw)

    def get(self, model_id: str, inputs: dict[str, Any]) -> Decimal | None:
        """Return cached price if it exists and hasn't expired."""
        return self.get_by_key(self.key(model_id, inputs))

    def set(self, model_id: str, inputs: dict[str, Any], price: Decimal) -> None:
        """Store a price in the cache with the current timestamp."""
        self.set_by_key(self.key(model_id, inputs), price)

    def get_by_key(self, key: str | None) -> Decimal | None:
        """Return the price stored under a :meth:`key` if it hasn't expired."""
        entry = self.entry_by_key(key)
        return None if entry is None else entry[0]

    def entry_by_key(self, key: str | None) -> tuple[Decimal, dict[str, Any]] | None:
        """Return the price and extra stored under a :meth:`key`, unless expired."""
        if key is None or self._ttl <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        price, timestamp, extra = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return price, dict(extra)

    def set_by_key(
        self, key: str | None, price: Decimal, extra: dict[str, Any] | None = None
    ) -> None:
        """Store a price, and optional extra, under a :meth:`key` with the current timestamp."""
        if key is None or self._ttl <= 0:
            return
        now = time.monotonic()
        # Re-inserting moves the key to the end, so dict order stays oldest first.
        self._store.pop(key, None)
        if len(self._store) >= self._max_entries:
            self._evict(now)
        self._store[key] = (price, now, dict(extra or {}))

    def _evict(self, now: float) -> None:
        """Drop expired entries, then the oldest ones, to make room."""
        expired = [key for key, (_, stored, _) in self._store.items() if now - stored > self._ttl]
        for key in expired:
            del self._store[key]
        while len(self._store) >= self._max_entries:
            del self._store[next(iter(self._store))]

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()


def apply_commission(
    base_price: Decimal, commission_rate: float, gas_surcharge: float = 0.0
) -> Decimal:
    """Apply commission markup to a base price.

    Args:
        base_price: The provider's base cost in USD.
        commission_rate: Commission as a decimal (e.g., 0.04 for 4%).
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
