"""Read upstream account balances for the dashboard.

Only providers that expose a balance on the same API key the gateway
uses are queried. Results are cached so the dashboard refresh does not
call every provider on each poll.

A balance is marked low when it is time to top up:

- dollars: under $20
- CloudConvert credits: under 300
- ElevenLabs characters: under 10,000 left
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 60.0
_REQUEST_TIMEOUT_S = 8.0
_LOW_USD = Decimal("20")
_LOW_CREDITS = Decimal("300")
_LOW_CHARACTERS = Decimal("10000")

_SUPPORTED = {
    "openrouter",
    "wavespeed",
    "spicyapi",
    "cloudconvert",
    "fishaudio",
    "elevenlabs",
}
_FETCHERS: dict[str, str] = {
    "wavespeed": "/balance",
    "spicyapi": "/api/v1/chat/credit",
    "cloudconvert": "/users/me",
    "fishaudio": "/wallet/self/api-credit",
}


@dataclass
class _CachedBalance:
    """One provider's last balance check."""

    fetched_at: float
    payload: dict[str, Any] | None


_cache: dict[str, _CachedBalance] = {}
_lock = asyncio.Lock()


def clear_balance_cache() -> None:
    """Drop cached balances. Used by tests."""
    _cache.clear()


def _decimal(value: Any) -> Decimal:
    """Parse a JSON number or numeric string into Decimal."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("balance is not a number") from exc


def _format_usd(amount: Decimal) -> str:
    """Format a dollar amount, keeping the minus sign in front of the symbol."""
    rounded = amount.quantize(Decimal("0.01"))
    sign = "-" if rounded < 0 else ""
    return f"{sign}${abs(rounded):,.2f}"


def _format_count(amount: Decimal, suffix: str) -> str:
    """Format an integer quantity with thousands separators."""
    number = int(amount.to_integral_value())
    return f"{number:,} {suffix}"


def _payload(amount: Decimal, unit: str, *, low: bool) -> dict[str, Any]:
    """Build the dashboard balance object."""
    if unit == "usd":
        label = _format_usd(amount)
    elif unit == "credits":
        label = _format_count(amount, "credits")
    else:
        label = _format_count(amount, "chars")
    return {
        "amount": f"{amount:.2f}" if unit == "usd" else str(int(amount.to_integral_value())),
        "unit": unit,
        "label": label,
        "low": low,
    }


def _usd(amount: Decimal) -> dict[str, Any]:
    """Dollar balance. Low when less than $20 is left."""
    return _payload(amount, "usd", low=amount < _LOW_USD)


def _bearer(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _join(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def parse_openrouter(body: dict[str, Any]) -> dict[str, Any]:
    """Remaining OpenRouter account credits: purchased minus used.

    This endpoint accepts a management key, not an inference key.
    """
    data = body.get("data", body)
    remaining = _decimal(data["total_credits"]) - _decimal(data["total_usage"])
    return _usd(remaining)


def parse_openrouter_key(body: dict[str, Any]) -> dict[str, Any] | None:
    """Remaining spend on the current OpenRouter key.

    An inference key can read its own credit limit and budget. When neither
    is set, the account wallet is not visible and this returns None.
    """
    data = body.get("data", body)
    amounts: list[Decimal] = []
    if data.get("limit_remaining") is not None:
        amounts.append(_decimal(data["limit_remaining"]))
    budget = data.get("effective_budget")
    if isinstance(budget, dict) and budget.get("remaining_usd") is not None:
        amounts.append(_decimal(budget["remaining_usd"]))
    if not amounts:
        return None
    return _usd(min(amounts))


def parse_wavespeed(body: dict[str, Any]) -> dict[str, Any]:
    """WaveSpeed account balance in USD."""
    if body.get("code") not in (None, 200):
        raise ValueError("wavespeed balance was not successful")
    return _usd(_decimal(body["data"]["balance"]))


def parse_spicyapi(body: dict[str, Any]) -> dict[str, Any]:
    """SpicyAPI spendable balance. Held funds are not included."""
    if body.get("code") not in (None, 200):
        raise ValueError("spicyapi balance was not successful")
    return _usd(_decimal(body["data"]["available"]))


def parse_cloudconvert(body: dict[str, Any]) -> dict[str, Any]:
    """CloudConvert conversion credits."""
    credits = _decimal(body["data"]["credits"])
    return _payload(credits, "credits", low=credits < _LOW_CREDITS)


def parse_fishaudio(body: dict[str, Any]) -> dict[str, Any]:
    """Fish Audio API credit, charged in dollars."""
    return _usd(_decimal(body["credit"]))


def parse_elevenlabs(body: dict[str, Any]) -> dict[str, Any]:
    """ElevenLabs characters still left in the current period."""
    subscription = body.get("subscription", body)
    limit = _decimal(subscription["character_limit"])
    remaining = limit - _decimal(subscription["character_count"])
    return _payload(remaining, "characters", low=remaining < _LOW_CHARACTERS)


def _failure_hint(response: httpx.Response) -> str:
    """Short provider error text that does not include credentials."""
    try:
        body = response.json()
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail")
    message = ""
    if isinstance(detail, dict):
        message = str(detail.get("status") or "")
    elif isinstance(detail, str):
        message = detail
    else:
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(body.get("message"), str):
            message = body["message"]
    message = " ".join(message.split())[:80]
    lowered = message.lower()
    if "sk-" in lowered or "bearer " in lowered:
        return ""
    return message


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    """GET a JSON object or raise for an HTTP error."""
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("balance response was not an object")
    return body


async def _read_openrouter(
    client: httpx.AsyncClient,
    config: ProviderConfig,
) -> dict[str, Any]:
    """Read the account wallet with a management key, otherwise the key limit.

    A management key cannot call models, so it is stored separately and used
    only for the credit balance.
    """
    management_key = config.balance_api_key.strip()
    if management_key:
        credits = await _get_json(
            client,
            _join(config.base_url, "/credits"),
            _bearer(management_key),
        )
        return parse_openrouter(credits)
    headers = _bearer(config.api_key)
    key_body = await _get_json(client, _join(config.base_url, "/key"), headers)
    parsed = parse_openrouter_key(key_body)
    if parsed is not None:
        return parsed
    credits = await _get_json(client, _join(config.base_url, "/credits"), headers)
    return parse_openrouter(credits)


async def _read_elevenlabs(
    client: httpx.AsyncClient,
    config: ProviderConfig,
) -> dict[str, Any]:
    """Read characters left. Some keys can only see /v1/user."""
    headers = {"xi-api-key": config.api_key}
    last_error: Exception | None = None
    for path in ("/v1/user/subscription", "/v1/user"):
        try:
            body = await _get_json(client, _join(config.base_url, path), headers)
            return parse_elevenlabs(body)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


async def _read(
    client: httpx.AsyncClient,
    name: str,
    config: ProviderConfig,
) -> dict[str, Any]:
    """Fetch and parse one provider balance."""
    if name == "openrouter":
        return await _read_openrouter(client, config)
    if name == "elevenlabs":
        return await _read_elevenlabs(client, config)
    headers = _bearer(config.api_key)
    if name == "cloudconvert":
        headers["Accept"] = "application/json"
    body = await _get_json(client, _join(config.base_url, _FETCHERS[name]), headers)
    parsers = {
        "wavespeed": parse_wavespeed,
        "spicyapi": parse_spicyapi,
        "cloudconvert": parse_cloudconvert,
        "fishaudio": parse_fishaudio,
    }
    return parsers[name](body)


async def _read_safe(
    client: httpx.AsyncClient,
    name: str,
    config: ProviderConfig,
) -> dict[str, Any] | None:
    """Return a balance, or None when the provider call fails."""
    try:
        return await _read(client, name, config)
    except httpx.HTTPStatusError as exc:
        hint = _failure_hint(exc.response)
        path = exc.request.url.path
        if hint:
            logger.warning(
                "Balance check for %s failed: HTTP %s %s (%s)",
                name,
                exc.response.status_code,
                path,
                hint,
            )
        else:
            logger.warning(
                "Balance check for %s failed: HTTP %s %s",
                name,
                exc.response.status_code,
                path,
            )
    except Exception:
        logger.warning("Balance check for %s failed", name)
    return None


async def fetch_balances(configs: dict[str, ProviderConfig]) -> dict[str, dict[str, Any]]:
    """Return cached balances for providers that publish one.

    A failed refresh keeps the previous balance. Providers without a
    balance API, or without an API key, are omitted.
    """
    now = time.monotonic()
    fresh: dict[str, dict[str, Any]] = {}
    due: list[tuple[str, ProviderConfig]] = []
    for name, config in configs.items():
        if name not in _SUPPORTED or not config.enabled or not config.api_key:
            continue
        cached = _cache.get(name)
        if cached and now - cached.fetched_at < _CACHE_TTL_S:
            if cached.payload is not None:
                fresh[name] = cached.payload
        else:
            due.append((name, config))
    if not due:
        return fresh

    async with _lock:
        now = time.monotonic()
        still_due: list[tuple[str, ProviderConfig]] = []
        for name, config in due:
            cached = _cache.get(name)
            if cached and now - cached.fetched_at < _CACHE_TTL_S:
                if cached.payload is not None:
                    fresh[name] = cached.payload
            else:
                still_due.append((name, config))
        if not still_due:
            return fresh

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
            loaded = await asyncio.gather(
                *(_read_safe(client, name, config) for name, config in still_due)
            )
        checked_at = time.monotonic()
        for (name, _config), payload in zip(still_due, loaded, strict=True):
            previous = _cache.get(name)
            if payload is None and previous and previous.payload is not None:
                fresh[name] = previous.payload
                _cache[name] = _CachedBalance(checked_at, previous.payload)
            else:
                _cache[name] = _CachedBalance(checked_at, payload)
                if payload is not None:
                    fresh[name] = payload
    return fresh
