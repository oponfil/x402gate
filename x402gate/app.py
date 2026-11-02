"""FastAPI application for x402gate.

Defines the API routes and wires up providers, pricing, and payment handling.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import http.cookiejar
import json
import logging
import os
import pathlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.formparsers import MultiPartException, MultiPartParser

from x402gate.core import stats
from x402gate.core.balances import fetch_balances
from x402gate.core.config import AppConfig, load_config
from x402gate.core.http import KEEPALIVE_EXPIRY_S, close_shared_client
from x402gate.core.http import timeout as upstream_timeout
from x402gate.core.payment import (
    InFlightAmounts,
    PaymentClaims,
    PaymentHandler,
    payment_key,
    settlement_state,
)
from x402gate.core.prepaid import (
    build_signing_message,
    deposit,
    format_usd,
    get_balance,
    release,
    reserve,
    validate_timestamp,
    verify_wallet_signature,
)
from x402gate.core.prepaid import (
    init as init_prepaid,
)
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError, outcome_unknown
from x402gate.providers.cloudconvert import CloudConvertProvider
from x402gate.providers.elevenlabs import ElevenLabsProvider
from x402gate.providers.fishaudio import FishAudioProvider
from x402gate.providers.minimax import MinimaxProvider
from x402gate.providers.nanogpt import NanoGptProvider
from x402gate.providers.openrouter import OpenRouterProvider
from x402gate.providers.socialdownload import SocialDownloadProvider
from x402gate.providers.spicyapi import SpicyApiProvider
from x402gate.providers.tungsten import TungstenProvider
from x402gate.providers.wavespeed import WaveSpeedProvider

logger = logging.getLogger(__name__)

# Railway closes an HTTP response after 5 minutes with no bytes, and after
# 15 minutes even when bytes are flowing. A pad of newlines is valid JSON
# whitespace, so clients still parse the object that follows.
_HEARTBEAT_INTERVAL_S = 20.0
_HEARTBEAT_PAD = b"\n" * 8192
# A payment stays claimed this long after it settles. An RPC node that has
# not seen the transaction yet could otherwise pass a replay through verify.
# A Solana blockhash lives about a minute, so its claim outlives the signature.
_BASE_CLAIM_HOLD_S = 60.0
_SOLANA_CLAIM_HOLD_S = 180.0
# A Base settlement with an unknown outcome can still land until the
# authorization's validBefore. Its claim lasts that long plus this margin
# for clock skew, capped so a far-off validBefore cannot pin it forever.
_UNKNOWN_CLAIM_MARGIN_S = 60.0
_MAX_CLAIM_HOLD_S = 86400.0
# A Base settlement with an unknown outcome is followed by its receipt. The
# poll starts at a Base block and backs off, since a long validBefore can keep
# it going for hours.
_RECONCILE_FIRST_POLL_S = 2.0
_RECONCILE_MAX_POLL_S = 60.0
# A reconcile stopped by shutdown looks the receipt up once more, for at most
# this long, so the log says whether the payer is owed the lost credit.
_RECONCILE_LAST_LOOKUP_S = 5.0
# Multipart framing (boundaries, small text fields) on top of the file limit.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
# CloudConvert takes one file and a few text fields.
_MULTIPART_MAX_FIELDS = 20
# Only these providers read an uploaded file; every other one takes JSON. A
# multipart body for another provider gets a 400 before it is read, so an
# unpaid client cannot make the gateway hold an upload nobody would use.
_MULTIPART_PROVIDERS: tuple[type[BaseProvider], ...] = (CloudConvertProvider,)
# Uploads are read into memory before the request is paid. At most this many
# maximum-size uploads are read unpaid at once; more get a 503, since an
# out-of-memory restart wipes every prepaid balance.
_UNPAID_UPLOAD_SLOTS = 2
# A multipart body must arrive within this time. Its bytes hold the unpaid
# budget while it is read, so a client trickling bytes cannot hold it forever.
# 600 s is a maximum-size upload at 0.5 MB/s.
_UPLOAD_READ_TIMEOUT_S = 600.0
_TOPUP_BODY_LIMIT_BYTES = 64 * 1024
# Parsed JSON takes about 80 bytes per empty object, so a body made of tiny
# containers grows 20x in memory. Real requests (base64 media, chat) have few.
_MAX_JSON_CONTAINERS = 200_000
# A longer JSON integer is refused while the body is parsed. Python parses
# and prints a long integer in quadratic time, so a body of them cost seconds
# of unpaid CPU, parsed and again re-encoded for pricing. 2**128 has 39 digits.
_MAX_JSON_INT_CHARS = 40
# Keys the gateway adds to the body itself; a client must not supply them.
_INTERNAL_BODY_KEYS = frozenset({"_caller", "_file_bytes", "_file_name", "_priced_rates"})
# Keys get_price leaves in the body for submit (OpenRouter and NanoGPT cap
# steered routing at the rates they priced). The price cache keeps them with
# the price, so a cached price puts them back.
_PRICED_BODY_KEYS = ("_priced_rates",)
# A 429 from pricing (all audio parse slots busy) clears in moments.
_PRICING_RETRY_AFTER_S = "2"
# A body above this is hashed for the price cache in a worker thread. sha256
# releases the GIL, so a 24 MB upload (~30 ms) does not stall other requests.
_HASH_IN_THREAD_BYTES = 1024 * 1024
# The passthrough has its own pool, so long passthrough LLM calls never take
# connections from managed provider calls. No cap on open connections, as
# before pooling; only idle ones are limited.
_PASSTHROUGH_LIMITS = httpx.Limits(
    max_connections=None,
    max_keepalive_connections=50,
    keepalive_expiry=KEEPALIVE_EXPIRY_S,
)
# One passthrough client per event loop, like core/http.py's shared client.
_passthrough: httpx.AsyncClient | None = None
_passthrough_loop: asyncio.AbstractEventLoop | None = None


def _passthrough_client() -> httpx.AsyncClient:
    """Return the pooled passthrough client for the running event loop.

    It stores no cookies. The client is shared by every end user, so a
    Set-Cookie the upstream sends one user must not reach the next.
    """
    global _passthrough, _passthrough_loop
    loop = asyncio.get_running_loop()
    if _passthrough is None or _passthrough.is_closed or _passthrough_loop is not loop:
        no_cookies = http.cookiejar.CookieJar(
            policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[])
        )
        _passthrough = httpx.AsyncClient(
            timeout=upstream_timeout(120.0), limits=_PASSTHROUGH_LIMITS, cookies=no_cookies
        )
        _passthrough_loop = loop
    return _passthrough


async def _close_passthrough_client() -> None:
    """Close the passthrough pool (on shutdown)."""
    global _passthrough, _passthrough_loop
    if _passthrough is not None and not _passthrough.is_closed:
        await _passthrough.aclose()
    _passthrough = None
    _passthrough_loop = None


def _authorization_ttl_s(payment_sig: str) -> float:
    """Seconds until a Base authorization's validBefore, capped.

    A payload without one falls back to the validity the gateway asks for.
    """
    try:
        payload = json.loads(base64.b64decode(payment_sig))
        valid_before = int(payload["payload"]["authorization"]["validBefore"])
    except Exception:
        return float(config.payment.max_timeout)
    now = time.time()
    return min(valid_before, now + _MAX_CLAIM_HOLD_S) - now


def _claim_hold_s(payment_sig: str, network: str, state: str) -> float:
    """How long a payment stays claimed once its settle has returned.

    A failed settle frees it at once. A Base settlement whose outcome is
    unknown may still land while the authorization is valid, and until
    then a replay passes verify, so the claim lasts that long.
    """
    if state == "failed":
        return 0.0
    if not network.startswith("eip155:"):
        return _SOLANA_CLAIM_HOLD_S
    if state == "ok":
        return _BASE_CLAIM_HOLD_S
    hold = _authorization_ttl_s(payment_sig) + _UNKNOWN_CLAIM_MARGIN_S
    return max(hold, _BASE_CLAIM_HOLD_S)


class _JsonIntTooLongError(ValueError):
    """A client JSON body holds an integer longer than _MAX_JSON_INT_CHARS."""


def _parse_json_int(text: str) -> int:
    """Parse one JSON integer, refusing an overly long one."""
    if len(text) > _MAX_JSON_INT_CHARS:
        raise _JsonIntTooLongError(f"JSON integer longer than {_MAX_JSON_INT_CHARS} characters")
    return int(text)


def _loads_client_json(raw: bytes) -> Any:
    """Parse a client's JSON body; raises _JsonIntTooLongError on a too-long integer."""
    return json.loads(raw, parse_int=_parse_json_int)


def _error_response(
    status_code: int,
    error: str,
    *,
    provider: str | None = None,
    **extra: Any,
) -> JSONResponse:
    """Build a uniform JSON error response."""
    body: dict[str, Any] = {"error": error, "status": status_code}
    if provider:
        body["provider"] = provider
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


def _prepaid_report(remaining: Decimal, charged: Decimal) -> dict[str, str]:
    """Cost and balance of a prepaid request, as sent in the body and headers.

    The body copy matters for heartbeat responses, whose headers go out
    before the cost is known.
    """
    return {"cost": format_usd(charged), "balance": format_usd(remaining)}


def _prepaid_headers(report: dict[str, str]) -> dict[str, str]:
    """Response headers carrying a :func:`_prepaid_report`."""
    return {"X-Prepaid-Balance": report["balance"], "X-Prepaid-Cost": report["cost"]}


def _record_onchain_settlement(
    provider_name: str,
    network: str,
    settlement: dict[str, Any],
    settle_latency: float,
    revenue_usd: Decimal | None = None,
) -> None:
    """Record gas and per-network stats for a successful on-chain settlement.

    ``revenue_usd`` overrides the amount taken from the settlement dict.
    Top-up passes the exact credited payment so the network card matches
    the Decimal amount the client signed.
    """
    gas_cost = Decimal(str(settlement.get("gas_cost_usd", 0.0)))
    stats.record_gas(provider_name, gas_cost)
    if revenue_usd is None:
        revenue_usd = Decimal(str(settlement.get("amount_usdc", 0)))
    stats.record_settlement(
        network,
        settle_latency,
        settlement.get("gas_cost_usd", 0.0),
        settlement.get("gas_cost_native", 0.0),
        settlement.get("gas_label", ""),
        revenue_usd=revenue_usd,
    )


async def _refund_settled_payment(payer: str, amount: Decimal) -> dict[str, str]:
    """Credit a settled x402 payment back to the payer's prepaid balance.

    Used when the provider fails after the payment settled, which happens
    before the provider call. The balance is in-memory and is lost on restart.
    """
    new_balance = await deposit(payer, amount)
    stats.record_refund_credit(amount)
    logger.warning(
        "Refund: $%s credited to prepaid %s (balance: $%s)",
        amount,
        (payer[:12] + "…") if payer else "",
        new_balance,
    )
    return {
        "refunded_to_prepaid": str(amount),
        "prepaid_balance": str(new_balance),
    }


# One map for the landing page and the dashboard. Anything missing is System.
PROVIDER_CATEGORIES: dict[str, str] = {
    "openrouter": "LLM",
    "nanogpt": "LLM",
    "blockrun": "LLM",
    "wavespeed": "Image & Video",
    "tungsten": "Image & Video",
    "spicyapi": "Image & Video",
    "elevenlabs": "Text-to-Speech",
    "minimax": "Text-to-Speech",
    "fishaudio": "Text-to-Speech",
    "cloudconvert": "Utility",
    "socialdownload": "Utility",
}


def category_order() -> list[str]:
    """Category headings in map order, with System last for topup and unknowns."""
    order: list[str] = []
    for category in PROVIDER_CATEGORIES.values():
        if category not in order:
            order.append(category)
    if "System" not in order:
        order.append("System")
    return order


# Provider registry: config name → provider class.
# A passthrough provider needs no class. A managed provider needs one line here.
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "wavespeed": WaveSpeedProvider,
    "openrouter": OpenRouterProvider,
    "nanogpt": NanoGptProvider,
    "tungsten": TungstenProvider,
    "cloudconvert": CloudConvertProvider,
    "socialdownload": SocialDownloadProvider,
    "elevenlabs": ElevenLabsProvider,
    "minimax": MinimaxProvider,
    "fishaudio": FishAudioProvider,
    "spicyapi": SpicyApiProvider,
}

# Global state (initialized in lifespan)
config: AppConfig
providers: dict[str, BaseProvider | None] = {}  # None = passthrough (no provider object)
payment_handler: PaymentHandler
price_cache: PriceCache
_pending_jobs: set[asyncio.Task] = set()
# Followers of Base settlements whose outcome was unknown (_reconcile_base).
_pending_reconciles: set[asyncio.Task] = set()
payment_claims = PaymentClaims()
base_in_flight = InFlightAmounts()
# Upload bytes in memory for requests that are not paid yet.
_unpaid_upload_bytes = 0


async def _settlement_receipt(
    network: str, tx_hash: str, payer: str, amount: Decimal
) -> dict[str, Any] | None:
    """The receipt lookup for a reconcile; None (not resolved) if it crashes."""
    try:
        return await payment_handler.settlement_receipt(network, tx_hash, payer, amount)
    except Exception:
        logger.exception("Receipt lookup for Base tx %s failed", tx_hash)
        return None


async def _reconcile_base(
    network: str,
    payer: str,
    payment_sig: str,
    settlement: dict[str, Any] | None,
    amount: Decimal,
    on_landed: Callable[[dict[str, Any]], Awaitable[None]],
    owed: Callable[[], bool] | None = None,
    on_unresolved: Callable[[str], None] | None = None,
) -> None:
    """Follow a Base settlement whose outcome was unknown until it resolves.

    The receipt is polled by hash rather than by settling again: the SDK's
    pending entry is single-use and expires. ``on_landed`` runs only once
    the receipt shows the transfer; a reverted duplicate never reaches it.
    Polling stops once the authorization has expired, since the transfer
    can no longer land. :func:`_start_base_reconcile` holds the payer's
    in-flight amount until this returns.

    ``on_unresolved`` hears "failed" (the transfer reverted) or "expired"
    when the reconcile ends without a landing, so a caller stops promising
    a refund that nothing will pay. ``owed`` tells a shutdown log whether
    a landed transfer is owed back (see :func:`_log_unreconciled`).
    """
    tx_hash = (settlement or {}).get("transaction") or ""
    deadline = time.monotonic() + _claim_hold_s(payment_sig, network, "unknown")
    delay = _RECONCILE_FIRST_POLL_S
    outcome: dict[str, Any] | None = None
    try:
        while True:
            expired = time.monotonic() >= deadline
            if tx_hash:
                outcome = await _settlement_receipt(network, tx_hash, payer, amount)
            if outcome is not None or expired:
                break
            await asyncio.sleep(min(delay, max(deadline - time.monotonic(), 0.0)))
            delay = min(delay * 2, _RECONCILE_MAX_POLL_S)
    except asyncio.CancelledError:
        await _log_unreconciled(network, tx_hash, payer, amount, owed)
        raise
    state = settlement_state(outcome)
    if state == "ok":
        try:
            await on_landed(outcome)
        except Exception:
            logger.exception("Recording landed Base tx %s failed", tx_hash)
        return
    if on_unresolved is not None:
        try:
            on_unresolved("failed" if state == "failed" else "expired")
        except Exception:
            logger.exception("Recording unresolved Base tx %s failed", tx_hash)
    if state == "failed":
        logger.warning(
            "Base tx %s from %s did not move $%s; nothing credited",
            tx_hash,
            payer[:12] + "…",
            amount,
        )
    else:
        logger.error(
            "Base tx %s from %s never resolved before its authorization expired",
            tx_hash or "?",
            payer[:12] + "…",
        )


async def _log_unreconciled(
    network: str,
    tx_hash: str,
    payer: str,
    amount: Decimal,
    owed: Callable[[], bool] | None = None,
) -> None:
    """Log a reconcile that shutdown stopped, after one last receipt lookup.

    The prepaid balance lives in memory, so a credit made now would die
    with the process anyway. The log tells whether the payer is owed it.
    A landed transfer is owed for a top-up or a job that failed with a
    known error. ``owed`` returns False for a job that was delivered or
    whose outcome is unknown: that payment is kept. Without ``owed``, or
    if it raises, the transfer is logged as owed.
    """
    outcome = None
    if tx_hash:
        try:
            outcome = await asyncio.wait_for(
                _settlement_receipt(network, tx_hash, payer, amount), _RECONCILE_LAST_LOOKUP_S
            )
        except TimeoutError:
            outcome = None
    state = settlement_state(outcome)
    is_owed = True
    if state == "ok" and owed is not None:
        try:
            is_owed = bool(owed())
        except Exception:
            logger.exception("Checking whether Base tx %s is owed failed", tx_hash)
    if state == "ok" and is_owed:
        logger.error(
            "Shutdown: Base tx %s from %s landed; $%s is owed and was not credited",
            tx_hash,
            payer[:12] + "…",
            amount,
        )
    elif state == "ok":
        logger.warning(
            "Shutdown: Base tx %s from %s landed; $%s kept (job delivered or outcome "
            "unknown), nothing owed",
            tx_hash,
            payer[:12] + "…",
            amount,
        )
    elif state == "failed":
        logger.warning(
            "Shutdown: Base tx %s from %s did not move $%s; nothing owed",
            tx_hash,
            payer[:12] + "…",
            amount,
        )
    else:
        logger.error(
            "Shutdown before Base tx %s from %s resolved; $%s not reconciled",
            tx_hash or "?",
            payer[:12] + "…",
            amount,
        )


def _start_base_reconcile(
    network: str,
    payer: str,
    payment_sig: str,
    settlement: dict[str, Any] | None,
    amount: Decimal,
    on_landed: Callable[[dict[str, Any]], Awaitable[None]],
    *,
    owed: Callable[[], bool] | None = None,
    on_unresolved: Callable[[str], None] | None = None,
) -> None:
    """Run :func:`_reconcile_base` in the background, taking over the in-flight amount.

    The amount stays in base_in_flight until the task ends, so another
    authorization over the same USDC fails the balance check meanwhile. It
    is released when the task is done, however it ends: a task cancelled
    before its first step never runs the coroutine's own cleanup.
    """
    task = asyncio.create_task(
        _reconcile_base(
            network, payer, payment_sig, settlement, amount, on_landed, owed, on_unresolved
        )
    )
    _pending_reconciles.add(task)
    task.add_done_callback(functools.partial(_reconcile_done, payer, amount))


def _reconcile_done(payer: str, amount: Decimal, task: asyncio.Task) -> None:
    """Forget a finished reconcile and release its amount, logging a crash."""
    _pending_reconciles.discard(task)
    base_in_flight.remove(payer, amount)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Base reconcile crashed", exc_info=task.exception())


async def _read_body_capped(request: Request, limit: int) -> bytes | None:
    """Read the request body, or return None once it passes ``limit`` bytes.

    The body is read before pricing and payment, so an unpaid client could
    otherwise fill the memory. An out-of-memory restart wipes every prepaid
    balance.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


class _UploadTooLargeError(Exception):
    """A multipart body passed its size limit while it was being read."""


class _UploadBudgetError(Exception):
    """Unpaid uploads in memory would pass their budget."""


def _unpaid_upload_budget(max_upload_bytes: int) -> int:
    """Bytes all unpaid uploads may hold in memory together.

    While a body is parsed each file byte exists twice: in the spool and in
    the copy read from it, until the form is closed.
    """
    return _UNPAID_UPLOAD_SLOTS * 2 * (max_upload_bytes + _MULTIPART_OVERHEAD_BYTES)


class _UnpaidUpload:
    """Upload bytes one request holds in memory before it is paid."""

    def __init__(self) -> None:
        self.size = 0

    def take(self, size: int, limit: int) -> bool:
        """Hold ``size`` bytes, or return False if all requests would pass ``limit``."""
        global _unpaid_upload_bytes
        if _unpaid_upload_bytes + size > limit:
            return False
        _unpaid_upload_bytes += size
        self.size += size
        return True

    def keep(self, size: int) -> None:
        """Give back all but ``size`` bytes, once the spool is closed."""
        global _unpaid_upload_bytes
        size = min(size, self.size)
        _unpaid_upload_bytes -= self.size - size
        self.size = size

    def release(self) -> None:
        """Give the bytes back once the request is paid or answered."""
        self.keep(0)


async def _capped_stream(
    request: Request, limit: int, upload: _UnpaidUpload, budget: int
) -> AsyncGenerator[bytes, None]:
    """Yield the request body, raising _UploadTooLargeError past ``limit`` bytes.

    A chunked upload has no Content-Length, so the bytes are counted as they
    arrive. Otherwise an unpaid client could stream any size to the spool.
    Each chunk takes its share of the unpaid budget before it is spooled
    (twice: spool and read copy), so parallel uploads cannot fill the memory
    before any check; past ``budget`` it raises _UploadBudgetError.
    """
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _UploadTooLargeError
        if not upload.take(2 * len(chunk), budget):
            raise _UploadBudgetError
        yield chunk


def _track_job(task: asyncio.Task) -> None:
    """Keep a provider job alive until it is charged or refunded."""
    _pending_jobs.add(task)
    task.add_done_callback(_job_done)


def _job_done(task: asyncio.Task) -> None:
    """Forget a finished job, logging a crash nobody else may see."""
    _pending_jobs.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.error("Provider job crashed", exc_info=task.exception())


async def _warm_up(name: str, provider: BaseProvider) -> None:
    """Run a provider's warm-up; a failure leaves it to load lazily."""
    try:
        await provider.warm_up()
    except Exception:
        logger.warning("Provider '%s' warm-up failed; it will load on demand", name, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global config, providers, payment_handler, price_cache

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")
    init_prepaid(timestamp_window=config.gateway.prepaid_timestamp_window)

    # Initialize payment handler
    payment_handler = PaymentHandler(
        networks=config.payment.networks,
        max_timeout=config.payment.max_timeout,
    )

    # Initialize price cache
    price_cache = PriceCache(ttl=config.gateway.price_cache_ttl)

    # Register providers from config
    for name, provider_config in config.providers.items():
        if not provider_config.enabled:
            logger.info("Provider '%s' is disabled, skipping", name)
            continue

        if provider_config.type == "passthrough":
            providers[name] = None  # No provider object needed
            logger.info("Provider '%s' registered (passthrough)", name)
        elif name in PROVIDER_REGISTRY:
            kwargs = {}
            if name == "openrouter":
                gw = config.gateway
                kwargs["default_max_tokens"] = gw.default_max_tokens
                kwargs["web_search_tokens_per_result"] = gw.web_search_tokens_per_result
                kwargs["default_web_search_max_results"] = gw.default_web_search_max_results
                kwargs["web_search_cost_per_result"] = gw.web_search_cost_per_result
            elif name == "wavespeed":
                kwargs["price_cache_ttl"] = config.gateway.price_cache_ttl
            elif name == "nanogpt":
                kwargs["default_max_tokens"] = config.gateway.default_max_tokens
            elif name == "spicyapi":
                kwargs["default_max_tokens"] = config.gateway.default_max_tokens
                kwargs["price_cache_ttl"] = config.gateway.price_cache_ttl
            elif name == "tungsten":
                kwargs["jwt_token"] = provider_config.jwt_token
                kwargs["cf_clearance"] = provider_config.cf_clearance
            provider = PROVIDER_REGISTRY[name](config=provider_config, **kwargs)
            providers[name] = provider
            logger.info("Provider '%s' registered (managed)", name)
        else:
            logger.warning("Unknown provider '%s', skipping", name)

    # --- Dynamic route registration from config ---
    for name in providers:
        pcfg = config.providers[name]
        if pcfg.type == "passthrough":
            # Passthrough: transparent proxy, no payment handling
            def _make_passthrough(n: str):
                async def _route(path: str, request: Request) -> Any:
                    return await _passthrough_proxy(n, path, request)

                _route.__doc__ = f"Transparent passthrough proxy for {n}."
                return _route

            app.add_api_route(
                f"/v1/{name}/{{path:path}}",
                _make_passthrough(name),
                methods=["POST"],
                name=f"{name}_proxy",
            )
            logger.info("Route registered: POST /v1/%s/{{path}} (passthrough)", name)
        else:
            # Managed: full x402 payment flow
            def _make_managed(n: str):
                async def _route(path: str, request: Request) -> Any:
                    return await _handle_managed_request(n, path, request)

                _route.__doc__ = f"Managed proxy for {n}."
                return _route

            app.add_api_route(
                f"/v1/{name}/{{path:path}}",
                _make_managed(name),
                methods=["POST"],
                name=f"{name}_proxy",
            )
            logger.info("Route registered: POST /v1/%s/{{path}} (managed)", name)

    # Initialize dashboard stats and log capture
    stats.init(
        list(providers.keys()),
        [nc.network for nc in config.payment.networks.values()],
    )
    stats.install_log_handler()

    # Load catalogs now, so the first paid request does not wait for them.
    warm_ups = [
        asyncio.create_task(_warm_up(name, provider))
        for name, provider in providers.items()
        if provider is not None
    ]

    logger.info(
        "x402gate started on %s:%d with %d provider(s)",
        config.gateway.host,
        config.gateway.port,
        len(providers),
    )

    yield  # Application is running

    # Let cancelled warm-ups unwind before the pools close, so none reopens one.
    for task in warm_ups:
        task.cancel()
    await asyncio.gather(*warm_ups, return_exceptions=True)
    # Shutdown: finish provider jobs, which settle, then refund or record
    # their payments
    if _pending_jobs:
        logger.info("Waiting for %d provider job(s)...", len(_pending_jobs))
        await asyncio.gather(*_pending_jobs, return_exceptions=True)
    # Unknown Base settlements can take until validBefore, up to hours. What
    # they would credit lives in memory and dies with the process, so they
    # stop here. Each looks its receipt up once more (at most
    # _RECONCILE_LAST_LOOKUP_S, all at once) and logs whether the payer is owed.
    reconciles = list(_pending_reconciles)
    for task in reconciles:
        task.cancel()
    await asyncio.gather(*reconciles, return_exceptions=True)
    for provider in providers.values():
        if provider is not None:
            await provider.close()
    await close_shared_client()
    await _close_passthrough_client()
    logger.info("x402gate shut down")


app = FastAPI(
    title="x402gate",
    description="Transparent x402 payment proxy for AI services",
    version="unversioned",
    lifespan=lifespan,
)


# --- Service Discovery ---


@app.get("/", include_in_schema=False)
async def service_info(request: Request) -> Response:
    """Service manifest: HTML for browsers, JSON for AI agents.

    Builds a single data dict used by both responses — no duplication.
    """
    base_url = os.environ.get("BASE_URL", "")
    networks = list(config.payment.networks.keys())
    commission_pct = int(config.gateway.commission * 100)
    gas_fee = config.gateway.gas_surcharge

    # -- Build provider docs and examples (used by both JSON and HTML) --
    provider_docs: dict[str, Any] = {}
    provider_info: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for name, pcfg in config.providers.items():
        if not pcfg.enabled:
            continue
        ptype = getattr(pcfg, "type", "managed") or "managed"
        provider_info.append(
            {
                "name": name,
                "type": ptype,
                "category": PROVIDER_CATEGORIES.get(name, "System"),
                "description": pcfg.description or "",
                "docs_url": pcfg.docs_url or "",
            }
        )

        doc: dict[str, Any] = {"endpoint_pattern": f"/v1/{name}/{{model_path}}"}
        if pcfg.docs_url:
            doc["api_reference"] = pcfg.docs_url

        ex_list = []
        for ex_field in sorted(f for f in pcfg.model_fields if f.startswith("example_request")):
            ex = getattr(pcfg, ex_field, None)
            if not ex:
                continue
            model = ex.get("model", "MODEL_PATH")
            body = ex.get("body", {})
            entry = {"method": "POST", "path": f"/v1/{name}/{model}", "body": body}
            ex_list.append(entry)
            examples.append(
                {
                    "provider": name,
                    "model": model,
                    "path": f"/v1/{name}/{model}",
                    "body_json": json.dumps(body, ensure_ascii=False),
                    "content_type": ex.get("content_type", "json"),
                    "body_fields": list(body.items()),
                }
            )
        if len(ex_list) == 1:
            doc["example_request"] = ex_list[0]
        elif ex_list:
            doc["example_requests"] = ex_list
        provider_docs[name] = doc

    prepaid = {
        "description": (
            "Top-up a prepaid balance to skip per-request blockchain transactions. "
            "Commission and gas are charged once at top-up. Subsequent requests "
            "deduct only the provider's actual cost from the balance."
        ),
        "wallets": ["Solana (Ed25519)", "Base/EVM (EIP-191)"],
        "topup_limits": {
            "min": f"${config.gateway.min_prepaid_topup}",
            "max": f"${config.gateway.max_prepaid_topup}",
        },
        "endpoints": {
            "topup": f"{base_url}/v1/topup",
            "balance": f"{base_url}/v1/balance/{{wallet_address}}",
        },
        "documentation": "https://github.com/oponfil/x402gate/blob/main/docs/prepaid.md",
    }

    service_data = {
        "base_url": base_url,
        "networks": networks,
        "commission": f"{commission_pct}% + ${gas_fee} gas",
        "provider_info": provider_info,
        "category_order": category_order(),
        "provider_docs": provider_docs,
        "prepaid": prepaid,
        "examples": examples,
        "max_upload_mb": config.gateway.max_upload_mb,
        "source": "https://github.com/oponfil/x402gate",
        "documentation": "https://github.com/oponfil/x402gate#readme",
        "endpoints": {
            "openapi": f"{base_url}/openapi.json",
            "docs": f"{base_url}/docs",
            "health": f"{base_url}/health",
            "providers": f"{base_url}/v1/providers",
            "topup": f"{base_url}/v1/topup",
            "balance": f"{base_url}/v1/balance/{{wallet_address}}",
            "ai_plugin": f"{base_url}/.well-known/ai-plugin.json",
            "skill": f"{base_url}/skill",
        },
    }

    # HTML for browsers — render Jinja2 template with the same data
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        template_path = pathlib.Path(__file__).parent / "templates" / "index.html"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_path.parent),
            autoescape=True,
        )
        template = env.get_template(template_path.name)
        html = template.render(**service_data)
        return HTMLResponse(content=html)

    # JSON for AI agents
    json_data = {
        "name": "x402gate",
        "description": (
            "Transparent pay-per-request proxy for AI services via the x402 protocol. "
            "Send a POST request to any provider endpoint — if no payment header is "
            "included, a 402 response is returned with USDC payment options. "
            "The request body format is defined by the upstream provider. "
            "x402gate forwards it unchanged, except that chat requests may get caps "
            "that keep the cost within the quote (max_tokens, provider.max_price, "
            "web plugin limits), WaveSpeed's enable_sync_mode is dropped (the gateway "
            "polls), and options it cannot price before payment, such as "
            "stream: true, are refused with HTTP 400 before any payment. "
            "Calls longer than 20 seconds are kept open with newline heartbeats "
            "(status 200); read the body for the result or error. "
            "See provider_docs for API references."
        ),
        "payment_protocol": "x402",
        "payment_asset": "USDC",
        "networks": networks,
        "providers": list(providers.keys()),
        "commission": service_data["commission"] + " per request",
        "provider_docs": provider_docs,
        "prepaid": prepaid,
        "endpoints": service_data["endpoints"],
        "source": service_data["source"],
        "documentation": service_data["documentation"],
    }
    return JSONResponse(content=json_data)


@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def ai_plugin_manifest() -> dict[str, Any]:
    """Standard AI plugin manifest for agent discoverability."""
    base_url = os.environ.get("BASE_URL", "")
    # Build provider usage hints from config
    provider_hints = []
    for name, pcfg in config.providers.items():
        if not pcfg.enabled:
            continue
        hint = f"Provider '{name}': POST /v1/{name}/{{model_path}}"
        if pcfg.docs_url:
            hint += f" (docs: {pcfg.docs_url})"
        for ex_field in sorted(f for f in pcfg.model_fields if f.startswith("example_request")):
            ex = getattr(pcfg, ex_field, None)
            if ex:
                model = ex.get("model", "MODEL_PATH")
                body = ex.get("body", {})
                hint += f". Example: POST /v1/{name}/{model} with body {json.dumps(body)}"
        provider_hints.append(hint)
    providers_text = ". ".join(provider_hints)
    networks_text = ", ".join(
        f"{n} ({c.type.upper()}, USDC)" for n, c in config.payment.networks.items()
    )
    return {
        "schema_version": "v1",
        "name_for_human": "x402gate",
        "name_for_model": "x402gate",
        "description_for_human": "Pay-per-request proxy for AI services via x402/USDC",
        "description_for_model": (
            "x402gate is a transparent payment proxy for AI services using the x402 protocol. "
            "The request body format is defined by the upstream provider; "
            "x402gate forwards it unchanged, except for chat cost caps (max_tokens, "
            "provider.max_price), WaveSpeed's enable_sync_mode (dropped; the gateway "
            "polls) and HTTP 400 before payment for options it cannot "
            "price, such as stream: true. See the provider docs for body schema. "
            f"{providers_text}. "
            "If no payment is attached, the server returns HTTP 402 with accepted payment "
            "options (scheme, network, asset, amount, pay_to address). "
            "Attach a valid x402 PAYMENT-SIGNATURE header and resubmit to get the AI response. "
            f"Supported networks: {networks_text}. "
            "Prices are dynamic and fetched from the upstream provider at request time."
        ),
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url": f"{base_url}/openapi.json",
        },
        "contact_email": "oponfil@github.com",
        "legal_info_url": "https://github.com/oponfil/x402gate/blob/main/LICENSE",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok"}


@app.get("/skill", include_in_schema=False)
async def agent_skill() -> Response:
    """Serve the SKILL.md file for autonomous AI agents."""
    skill_path = (
        pathlib.Path(__file__).parent.parent / ".agents" / "skills" / "x402gate" / "SKILL.md"
    )
    if not skill_path.exists():
        return JSONResponse(status_code=404, content={"error": "SKILL.md not found"})
    return Response(content=skill_path.read_text(encoding="utf-8"), media_type="text/markdown")


# --- Dashboard ---


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> Response:
    """Public dashboard with provider status, statistics, and live logs."""
    # Build provider type map for the template
    provider_types: dict[str, str] = {}
    for name in providers:
        pcfg = config.providers.get(name)
        provider_types[name] = getattr(pcfg, "type", "managed") or "managed"

    template_path = pathlib.Path(__file__).parent / "templates" / "dashboard.html"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent),
        autoescape=False,  # We control the template; JS needs raw JSON
    )
    template = env.get_template(template_path.name)
    html = template.render(
        provider_types_json=json.dumps(provider_types),
        provider_categories_json=json.dumps(PROVIDER_CATEGORIES),
    )
    return HTMLResponse(content=html)


@app.get("/v1/stats")
async def get_stats() -> dict[str, Any]:
    """Return current gateway statistics, including upstream balances."""
    snapshot = stats.get_stats()
    try:
        balances = await fetch_balances(
            {
                name: config.providers[name]
                for name in snapshot["providers"]
                if name in config.providers
            }
        )
    except Exception:
        logger.warning("Provider balance snapshot failed")
        balances = {}
    for card_name, card in snapshot["providers"].items():
        card["balance"] = balances.get(card_name)
    return snapshot


@app.get("/v1/logs")
async def get_logs(limit: int = stats.MAX_LOG_ENTRIES) -> list[dict[str, Any]]:
    """Return recent log entries (newest first)."""
    return stats.get_logs(limit=limit)


@app.post("/v1/topup")
async def topup(request: Request) -> Response:
    """Top up prepaid balance via x402 payment.

    The client sends a standard x402 PAYMENT-SIGNATURE for the desired
    top-up amount. Gateway verifies, settles, deducts commission + gas,
    and credits the remainder to the sender's prepaid balance.
    """
    # Parse desired top-up amount from request body (default: min_prepaid_topup)
    t_start = time.monotonic()
    min_topup = Decimal(str(config.gateway.min_prepaid_topup))
    max_topup = Decimal(str(config.gateway.max_prepaid_topup))

    raw_body = await _read_body_capped(request, _TOPUP_BODY_LIMIT_BYTES)
    if raw_body is None:
        return _error_response(413, "Request body too large")
    try:
        body = _loads_client_json(raw_body) if raw_body else {}
    except _JsonIntTooLongError:
        return _error_response(400, "Invalid JSON body")
    except Exception:
        body = {}

    body_amount = body.get("amount") if isinstance(body, dict) else None
    if body_amount is not None:
        try:
            requested_amount = Decimal(str(body_amount))
        except Exception:
            return _error_response(400, "Invalid amount value")
        if not requested_amount.is_finite():
            return _error_response(400, "Invalid amount value")
        if requested_amount < min_topup:
            return _error_response(400, f"Top-up amount below minimum (${min_topup})")
        if requested_amount > max_topup:
            return _error_response(400, f"Top-up amount exceeds maximum (${max_topup})")
    else:
        requested_amount = min_topup

    payment_sig = payment_handler.extract_payment_signature(request)
    if not payment_sig:
        return payment_handler.create_payment_required(requested_amount)

    # Parse the requested top-up amount from the payment payload
    try:
        payload_dict = json.loads(base64.b64decode(payment_sig))
        accepted = payload_dict.get("accepted", {})
        raw_amount = int(accepted.get("amount", 0))
        topup_amount = Decimal(str(raw_amount)) / Decimal("1000000")  # USDC 6 decimals
    except Exception:
        return _error_response(400, "Invalid payment payload")

    if topup_amount <= 0:
        return _error_response(400, "Top-up amount must be positive")
    if topup_amount < min_topup or topup_amount > max_topup:
        return _error_response(
            400,
            f"Top-up amount must be between ${min_topup} and ${max_topup}",
        )
    if body_amount is not None and topup_amount != requested_amount:
        return _error_response(400, "Signed amount does not match the requested top-up")

    # Verify payment
    is_valid, payment_network, payer, reason = await payment_handler.verify(
        payment_sig, topup_amount
    )
    if not is_valid or not payer:
        msg = f"Payment verification failed: {reason}" if reason else "Payment verification failed"
        return _error_response(402, msg)

    # Calculate net credit: topup_amount minus commission and gas
    commission = topup_amount * Decimal(str(config.gateway.commission))
    gas_fee = Decimal(str(config.gateway.gas_surcharge))
    net_credit = topup_amount - commission - gas_fee
    if net_credit <= 0:
        return _error_response(400, "Top-up amount too small to cover fees")

    # One payment credits one top-up, and parallel copies must not each
    # broadcast it. The claim is shared with paid jobs.
    payment_id = payment_key(payment_sig)
    if not payment_claims.claim(payment_id):
        return _error_response(
            409,
            "This payment is already used by another request. "
            "Check your balance before signing a new one.",
        )

    # Verify does not count this payer's other Base authorizations that have
    # not landed, each with its own nonce, so the balance must cover them too,
    # as on the job path. An unknown settlement keeps its amount counted.
    is_base = payment_network.startswith("eip155:")
    if is_base:
        in_flight = base_in_flight.add(payer, topup_amount)
        has_funds = await payment_handler.check_evm_balance(payment_network, payer, in_flight)
        if not has_funds:
            base_in_flight.remove(payer, topup_amount)
            payment_claims.release(payment_id)
            if has_funds is None:
                return _error_response(503, "Payment network unavailable, retry")
            if in_flight > topup_amount:
                return _error_response(
                    402,
                    "Insufficient USDC balance: other payments from this wallet are still settling",
                )
            return _error_response(402, "Insufficient USDC balance")

    async def _credit(settlement: dict[str, Any], settle_latency: float) -> Decimal:
        """Credit a settled top-up and record it."""
        new_balance = await deposit(payer, net_credit)
        _record_onchain_settlement(
            "topup", payment_network, settlement, settle_latency, revenue_usd=topup_amount
        )
        stats.record_topup(topup_amount)
        # Revenue = full top-up amount, cost = net credit given to user.
        # Card shows profit as if all balance was spent (commission + gas_surcharge - gas).
        stats.record_revenue("topup", topup_amount, net_credit)
        return new_balance

    async def _credit_landed(receipt_settlement: dict[str, Any]) -> None:
        """Credit a top-up whose unknown settlement later landed."""
        new_balance = await _credit(receipt_settlement, time.monotonic() - t_settle_start)
        logger.info(
            "Top-up tx %s landed; credited $%s to %s (balance: $%s)",
            receipt_settlement.get("transaction"),
            net_credit,
            payer[:12] + "…",
            new_balance,
        )

    # Settle before crediting. A failed or expired transaction must not
    # leave a prepaid balance that was never paid for.
    t_settle_start = time.monotonic()
    settlement: dict[str, Any] | None = None
    state = "unknown"
    try:
        settlement = await payment_handler.settle(
            payment_sig,
            topup_amount,
            payment_network,
            extra_context={"provider_name": "topup", "provider_cost": 0},
        )
        state = settlement_state(settlement)
    finally:
        payment_claims.release(payment_id, _claim_hold_s(payment_sig, payment_network, state))
        if is_base and state == "unknown":
            # A top-up that lands is always owed to the payer.
            _start_base_reconcile(
                payment_network,
                payer,
                payment_sig,
                settlement,
                topup_amount,
                _credit_landed,
                owed=lambda: True,
            )
        elif is_base:
            base_in_flight.remove(payer, topup_amount)
    settle_latency = time.monotonic() - t_settle_start
    if settlement is None:
        # settle crashed, maybe after sending. The claim stays held, so the
        # same payment cannot be credited twice; a new one could pay twice.
        logger.error("Top-up settlement status unknown for %s; nothing credited", payer[:12] + "…")
        return _error_response(
            502,
            "Payment status unknown. Do not sign a new payment: check your wallet "
            "and balance before trying again.",
            outcome_unknown=True,
        )
    if state == "failed":
        reason = settlement.get("error_reason") or "unknown"
        return _error_response(402, f"Payment settlement failed: {reason}")

    warning = "Balance is stored in memory only. It will be lost on server restart."
    if state == "unknown" and is_base:
        # Sent, but its confirmation wait failed. Another authorization over
        # the same USDC may have taken the funds, and then this one reverts,
        # so nothing is credited until the receipt shows the transfer.
        tx_hash = settlement.get("transaction") or ""
        logger.error(
            "Top-up settlement status unknown for %s, tx %s; credit waits for the receipt",
            payer[:12] + "…",
            tx_hash or "?",
        )
        stats.record_request("topup", time.monotonic() - t_start, True)
        stats.record_overhead(payment_network, time.monotonic() - t_start)
        content: dict[str, Any] = {
            "pubkey": payer,
            "credited": "0",
            "pending_credit": str(net_credit),
            "balance": str(get_balance(payer)),
            "warning": warning,
            "settlement_pending": True,
            "transaction": tx_hash,
            "message": "Do not sign a new payment; your balance is credited "
            "once the transaction confirms.",
        }
        return JSONResponse(status_code=202, content=content)

    if state == "unknown":
        # Solana: sent, but its confirmation wait failed. It most likely
        # lands; refusing the credit would lose the payer's money.
        new_balance = await deposit(payer, net_credit)
        stats.record_topup(topup_amount)
        stats.record_revenue("topup", topup_amount, net_credit)
        logger.error(
            "Top-up settlement status unknown; credited $%s to %s, tx %s",
            net_credit,
            payer[:12] + "…",
            settlement.get("transaction") or "?",
        )
    else:
        new_balance = await _credit(settlement, settle_latency)

    stats.record_request("topup", time.monotonic() - t_start, True)
    stats.record_overhead(payment_network, time.monotonic() - t_start)
    content = {
        "pubkey": payer,
        "credited": str(net_credit),
        "balance": str(new_balance),
        "warning": warning,
    }
    if state == "unknown":
        content["settlement_pending"] = True
    return JSONResponse(content=content)


@app.get("/v1/balance/{pubkey}")
async def check_balance(pubkey: str) -> dict[str, Any]:
    """Check prepaid balance for a wallet."""
    balance = get_balance(pubkey)
    return {
        "pubkey": pubkey,
        "balance": str(balance),
    }


# --- Provider Info ---


@app.get("/v1/providers")
async def list_providers() -> dict[str, Any]:
    """List all registered providers and their status."""
    result: dict[str, Any] = {}
    for name in providers:
        pcfg = config.providers[name]
        info: dict[str, Any] = {
            "enabled": True,
            "base_url": pcfg.base_url,
        }
        if pcfg.type == "passthrough":
            info["type"] = "passthrough"
        result[name] = info
    return {"providers": result}


# --- Passthrough Proxy (for x402-native providers) ---


async def _passthrough_proxy(provider_name: str, path: str, request: Request) -> Response:
    """Transparent HTTP proxy for x402-native providers like BlockRun.

    Forwards everything as-is, including 402 responses and Payment-Signature
    headers.  x402gate does NOT handle payments — the client pays the
    upstream provider directly.
    """
    provider_cfg = config.providers.get(provider_name)
    if provider_cfg is None or provider_name not in providers:
        return JSONResponse(
            status_code=503,
            content={"error": f"{provider_name} provider is not configured"},
        )

    url = f"{provider_cfg.base_url.rstrip('/')}/{path}"
    body = await request.body()

    # Forward relevant headers (especially Payment-Signature for x402)
    forward_headers: dict[str, str] = {}
    for key in ("content-type", "payment-signature", "accept", "authorization"):
        if val := request.headers.get(key):
            forward_headers[key] = val

    t_start = time.monotonic()
    try:
        # The pooled client keeps the upstream connection open between requests.
        resp = await _passthrough_client().post(
            url, content=body, headers=forward_headers, timeout=upstream_timeout(120.0)
        )
    except httpx.TimeoutException:
        stats.record_request(
            provider_name,
            time.monotonic() - t_start,
            False,
            error_msg="Upstream request timed out",
        )
        return _error_response(504, "Upstream request timed out", provider=provider_name)
    except httpx.RequestError:
        stats.record_request(
            provider_name,
            time.monotonic() - t_start,
            False,
            error_msg="Upstream request failed",
        )
        return _error_response(502, "Upstream request failed", provider=provider_name)

    latency = time.monotonic() - t_start
    success = resp.status_code < 400
    stats.record_request(
        provider_name,
        latency,
        success,
        error_msg=resp.text[:200] if not success else None,
    )

    # Filter response headers (avoid hop-by-hop headers)
    safe_headers: dict[str, str] = {}
    skip = {"transfer-encoding", "connection", "keep-alive", "content-encoding", "content-length"}
    for k, v in resp.headers.items():
        if k.lower() not in skip:
            safe_headers[k] = v

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=safe_headers,
    )


# --- Managed Provider Proxy (unified flow for all managed providers) ---


async def _handle_managed_request(provider_name: str, path: str, request: Request) -> Response:
    """Run a managed request, freeing its unpaid upload however it ends."""
    upload = _UnpaidUpload()
    try:
        return await _managed_flow(provider_name, path, request, upload)
    finally:
        upload.release()


async def _managed_flow(
    provider_name: str, path: str, request: Request, upload: _UnpaidUpload
) -> Response:
    """Unified x402 payment flow for managed providers.

    Flow:
    1. Parse request body
    2. Fetch dynamic price from provider (with cache)
    3. Apply commission (4% + gas fee)
    4. If no PAYMENT-SIGNATURE header → return 402
    5. Verify payment via facilitator
    5c. Settle before the provider call, inside the job task so heartbeats
        cover it. A failed settle does not call the provider. If the
        provider fails afterwards, a settled amount is credited to the
        payer's prepaid balance; an unknown Base one once it lands.
    6. Forward request to provider as-is
    7. If async (task_id present) → poll for result; else use response directly
    8. Calculate the actual cost
    9. Record revenue for the settled payment
    10. Return result to client
    """
    provider = providers.get(provider_name)
    if provider is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"{provider_name} provider is not configured"},
        )

    t_start = time.monotonic()  # Total client wait time starts here

    # 1. Parse request body (JSON or multipart/form-data for file uploads)
    max_upload_bytes = config.gateway.max_upload_mb * 1024 * 1024
    content_type = request.headers.get("content-type", "")
    # The JSON body as received: the price cache hashes these bytes instead of
    # serializing the parsed body again. Multipart bodies have none.
    raw_body: bytes | None = None
    hashed_size = 0
    if "multipart/form-data" in content_type:
        if not isinstance(provider, _MULTIPART_PROVIDERS):
            return _error_response(
                400,
                f"{provider_name} takes a JSON body. Multipart uploads are not accepted.",
                provider=provider.name,
            )
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_upload_bytes + _MULTIPART_OVERHEAD_BYTES:
            return _error_response(
                413,
                f"Request too large. Maximum file size: {config.gateway.max_upload_mb} MB",
            )
        form = None
        try:
            # Parsed from a counted stream: a chunked upload declares no size.
            stream = _capped_stream(
                request,
                max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
                upload,
                _unpaid_upload_budget(max_upload_bytes),
            )
            form = await asyncio.wait_for(
                MultiPartParser(
                    request.headers,
                    stream,
                    max_files=1,
                    max_fields=_MULTIPART_MAX_FIELDS,
                ).parse(),
                _UPLOAD_READ_TIMEOUT_S,
            )
            body: dict[str, Any] = {}
            for key in form:
                value = form[key]
                if hasattr(value, "read"):  # UploadFile
                    # Starlette spools uploads to disk; check the size before
                    # pulling the file into memory.
                    if (getattr(value, "size", None) or 0) > max_upload_bytes:
                        return _error_response(
                            413,
                            f"File too large ({value.size / 1024 / 1024:.1f} MB). "
                            f"Maximum: {config.gateway.max_upload_mb} MB",
                        )
                    file_bytes = await value.read()
                    if len(file_bytes) > max_upload_bytes:
                        return _error_response(
                            413,
                            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
                            f"Maximum: {config.gateway.max_upload_mb} MB",
                        )
                    body["_file_bytes"] = file_bytes
                    body["_file_name"] = getattr(value, "filename", "input_file")
                    hashed_size = len(file_bytes)
                elif key not in _INTERNAL_BODY_KEYS:
                    body[key] = value
        except _UploadTooLargeError:
            return _error_response(
                413,
                f"Request too large. Maximum file size: {config.gateway.max_upload_mb} MB",
            )
        except _UploadBudgetError:
            response = _error_response(503, "Too many uploads in progress. Retry in a few seconds.")
            response.headers["Retry-After"] = "5"
            return response
        except TimeoutError:
            return _error_response(408, "Upload took too long. Send the file faster.")
        except MultiPartException as e:
            return _error_response(400, f"Invalid multipart form data: {e.message}")
        except Exception:
            return _error_response(400, "Invalid multipart form data")
        finally:
            # Uploads are spooled to temp files; free them once read. Only the
            # read copy stays in memory until the request is paid.
            if form is not None:
                await form.close()
            upload.keep(hashed_size)
    else:
        raw_body = await _read_body_capped(request, config.gateway.max_json_mb * 1024 * 1024)
        if raw_body is None:
            return _error_response(
                413, f"Request body too large. Maximum: {config.gateway.max_json_mb} MB"
            )
        if raw_body.count(b"{") + raw_body.count(b"[") > _MAX_JSON_CONTAINERS:
            return _error_response(413, "Request body has too many JSON objects and arrays")
        try:
            body = _loads_client_json(raw_body)
        except Exception:
            return _error_response(400, "Invalid JSON body")
        if not isinstance(body, dict):
            return _error_response(400, "JSON body must be an object")
        for key in _INTERNAL_BODY_KEYS:
            body.pop(key, None)
        hashed_size = len(raw_body)

    # 2. Fetch price (with cache)
    try:
        # Providers share paths such as chat/completions at different prices.
        price_key = f"{provider_name}/{path}"
        # One hash serves both the lookup and the store.
        if hashed_size > _HASH_IN_THREAD_BYTES:
            cache_key = await asyncio.to_thread(price_cache.key, price_key, body, raw_body)
        else:
            cache_key = price_cache.key(price_key, body, raw_body)
        cached = price_cache.entry_by_key(cache_key)
        if cached is not None:
            base_price, priced_extra = cached
            body.update(priced_extra)
        else:
            base_price = await provider.get_price(path, body)
            priced_extra = {key: body[key] for key in _PRICED_BODY_KEYS if key in body}
            price_cache.set_by_key(cache_key, base_price, priced_extra)
    except ProviderError as e:
        # An unpaid client can cause any 4xx (an unknown model, say), so only
        # provider-side failures count against health. The detail can echo
        # client text, and the dashboard is public, so it gets a fixed one.
        if e.status_code >= 500:
            stats.record_request(
                provider_name, 0, False, error_msg=f"Pricing failed: HTTP {e.status_code}"
            )
        response = _error_response(e.status_code, e.detail, provider=e.provider)
        if e.status_code == 429:
            response.headers["Retry-After"] = _PRICING_RETRY_AFTER_S
        return response
    except Exception:
        # A gateway bug, not the provider's failure: nothing is charged and
        # provider health is untouched.
        logger.exception("%s pricing crashed", provider_name)
        return _error_response(400, "Could not price this request", provider=provider.name)

    # 3. Apply commission
    final_price = apply_commission(
        base_price, config.gateway.commission, config.gateway.gas_surcharge
    )

    # 4. Check for payment (x402 or prepaid)
    payment_sig = payment_handler.extract_payment_signature(request)
    prepaid_pubkey = request.headers.get("x-prepaid-pubkey")
    prepaid_mode = False
    verify_s = 0.0
    payment_network = ""
    payer = ""
    upfront_settlement: dict[str, Any] | None = None
    payment_id = ""
    prepaid_reserved: Decimal | None = None

    if payment_sig:
        # 5a. Standard x402 payment flow
        t_verify_start = time.monotonic()
        is_valid, payment_network, payer, reason = await payment_handler.verify(
            payment_sig, final_price
        )
        if not is_valid:
            verify_s = time.monotonic() - t_verify_start
            msg = (
                f"Payment verification failed: {reason}"
                if reason
                else "Payment verification failed"
            )
            return _error_response(402, msg)
        # One payment buys one generation. Parallel requests with the same
        # payment all pass verify, since nothing has landed on-chain yet.
        payment_id = payment_key(payment_sig)
        if not payment_claims.claim(payment_id):
            return _error_response(
                409,
                "This payment is already used by another request. Sign a new payment.",
            )
        # The balance must also cover this payer's other Base requests that
        # have not settled yet, each signed with its own nonce. Without it,
        # parallel authorizations over one balance each cost us a reverted
        # transaction.
        if payment_network.startswith("eip155:"):
            in_flight = base_in_flight.add(payer, final_price)
            has_funds = await payment_handler.check_evm_balance(payment_network, payer, in_flight)
            if not has_funds:
                base_in_flight.remove(payer, final_price)
                payment_claims.release(payment_id)
                verify_s = time.monotonic() - t_verify_start
                if has_funds is None:
                    return _error_response(503, "Payment network unavailable, retry")
                if in_flight > final_price:
                    return _error_response(
                        402,
                        "Insufficient USDC balance: other payments from this wallet "
                        "are still settling",
                    )
                return _error_response(402, "Insufficient USDC balance")
        verify_s = time.monotonic() - t_verify_start
    elif prepaid_pubkey:
        # 5b. Prepaid balance flow
        prepaid_sig = request.headers.get("x-prepaid-signature", "")
        prepaid_ts = request.headers.get("x-prepaid-timestamp", "")

        if not prepaid_sig or not prepaid_ts:
            return _error_response(401, "Missing X-PREPAID-SIGNATURE or X-PREPAID-TIMESTAMP header")

        try:
            ts_int = int(prepaid_ts)
        except ValueError:
            return _error_response(401, "Invalid X-PREPAID-TIMESTAMP")

        if not validate_timestamp(ts_int):
            return _error_response(401, "X-PREPAID-TIMESTAMP expired or too far in the future")

        msg = build_signing_message(f"{provider_name}/{path}", ts_int)
        if not verify_wallet_signature(prepaid_pubkey, prepaid_sig, msg):
            return _error_response(401, "Invalid prepaid signature")

        # Hold the estimate before the provider runs. The hold is atomic, so
        # parallel requests cannot all pass a balance check and then go free.
        if not await reserve(prepaid_pubkey, base_price):
            current_balance = get_balance(prepaid_pubkey)
            return _error_response(
                402,
                f"Insufficient prepaid balance: ${current_balance} < ${base_price}",
                balance=str(current_balance),
            )
        prepaid_reserved = base_price
        prepaid_mode = True
    else:
        return payment_handler.create_payment_required(final_price)
    # Paid or held: the upload may stay in memory for the job.
    upload.release()

    is_base = payment_network.startswith("eip155:")

    # A Base settlement with an unknown outcome is followed by a reconcile,
    # which may see the transfer land before or after the job ends. Whichever
    # comes second settles it once: a known failure is credited to prepaid,
    # a kept payment is counted as revenue. "refund" is None until the job
    # ends; "cost" is the provider cost of a kept payment. "resolved" is
    # "failed" or "expired" once the reconcile ended without a landing.
    base_follow: dict[str, Any] = {
        "landed": None,
        "refund": None,
        "cost": None,
        "done": False,
        "resolved": None,
    }

    async def _follow_up() -> dict[str, str]:
        """Settle a landed Base payment against the job's outcome, at most once."""
        landed = base_follow["landed"]
        if base_follow["done"] or landed is None or base_follow["refund"] is None:
            return {}
        # Set before any await, so the two callers cannot both get here.
        base_follow["done"] = True
        if base_follow["refund"]:
            return await _refund_settled_payment(payer, final_price)
        stats.record_revenue(
            provider_name, Decimal(str(landed.get("amount_usdc", 0))), base_follow["cost"]
        )
        return {}

    def _job_ended(*, refund: bool, cost: Decimal = Decimal("0")) -> None:
        """Note how the job ended, for a Base payment that lands later."""
        base_follow["refund"] = refund
        base_follow["cost"] = cost

    def _unresolved(state: str) -> None:
        """Note that the reconcile ended without the transfer landing."""
        base_follow["resolved"] = state

    def _owed() -> bool:
        """Whether a landed Base payment is owed back: only a known failure.

        A job that has not ended (refund None) counts as an unknown outcome,
        so its payment is kept.
        """
        return base_follow["refund"] is True

    async def _settle() -> JSONResponse | None:
        """5c. Settle the x402 payment before the provider call.

        Returns a 402 if the settlement failed; the provider is then not
        called. A Solana transaction carries a blockhash that expires in
        about a minute, before a long job finishes, and a Base payer could
        move the USDC while the job runs. This runs inside the job task, so
        heartbeats cover a slow settle and a client that leaves does not
        stop it.
        """
        nonlocal upfront_settlement
        t_settle_start = time.monotonic()

        async def _landed(receipt_settlement: dict[str, Any]) -> None:
            """Record a Base settlement whose unknown outcome later landed."""
            _record_onchain_settlement(
                provider_name,
                payment_network,
                receipt_settlement,
                time.monotonic() - t_settle_start,
            )
            base_follow["landed"] = receipt_settlement
            await _follow_up()

        state = "unknown"
        try:
            upfront_settlement = await payment_handler.settle(
                payment_sig,
                final_price,
                payment_network,
                extra_context={
                    "provider_name": provider_name,
                    "estimated_cost": float(base_price),
                    "provider_cost": float(base_price),
                    "generation_s": 0.0,
                    "t_client": time.monotonic() - t_start,
                    "commission_rate": config.gateway.commission,
                    "gas_surcharge": config.gateway.gas_surcharge,
                },
            )
            state = settlement_state(upfront_settlement)
        finally:
            # Only a clear failure frees the payment at once. A crash or a
            # cancel may still have broadcast the transaction, and an unknown
            # Base outcome keeps the claim until the authorization expires.
            payment_claims.release(payment_id, _claim_hold_s(payment_sig, payment_network, state))
            if is_base and state == "unknown":
                # The amount stays counted until the receipt resolves, so a
                # new nonce cannot spend the same USDC in the meantime.
                _start_base_reconcile(
                    payment_network,
                    payer,
                    payment_sig,
                    upfront_settlement,
                    final_price,
                    _landed,
                    owed=_owed,
                    on_unresolved=_unresolved,
                )
            elif is_base:
                base_in_flight.remove(payer, final_price)
        if state == "failed" or upfront_settlement is None:
            # None means settle crashed, maybe before sending anything. The
            # claim stays held in case the transaction went out.
            reason = (upfront_settlement or {}).get("error_reason") or "unknown"
            return _error_response(402, f"Payment settlement failed: {reason}")
        if state == "unknown":
            # Sent, but its confirmation wait failed: it most likely lands. The
            # client gets the job. On Base the reconcile credits a known
            # failure once the transfer lands; on Solana nothing follows the
            # receipt, so a failure is not credited.
            logger.error(
                "%s settlement status unknown for %s; running the job anyway",
                "Base" if is_base else "Solana",
                provider_name,
            )
        else:
            _record_onchain_settlement(
                provider_name,
                payment_network,
                upfront_settlement,
                time.monotonic() - t_settle_start,
            )
        return None

    async def _provider_error(
        status_code: int,
        detail: str,
        *,
        provider_label: str | None = None,
        keep_payment: bool = False,
        **extra: Any,
    ) -> JSONResponse:
        """Return a provider error and settle the funds it held.

        Funds go back only when the provider reported the failure. With
        ``keep_payment`` the outcome is unknown: the job may have run and
        billed us, so the prepaid hold is charged and a settled x402
        payment is kept. Otherwise a settled payment goes to the payer's
        prepaid balance, and an unknown Base settlement once it lands.
        """
        if keep_payment:
            extra["outcome_unknown"] = True
            logger.warning(
                "%s outcome unknown, keeping the payment: %s", provider_name, detail[:200]
            )
        report = None
        if prepaid_reserved is not None:
            kept = prepaid_reserved if keep_payment else Decimal("0")
            remaining, charged = await release(prepaid_pubkey, prepaid_reserved, kept)
            if charged:
                stats.record_prepaid_usage(provider_name, charged)
            report = _prepaid_report(remaining, charged)
            extra["prepaid"] = report
        else:
            settlement = upfront_settlement
            tx_hash = (settlement or {}).get("transaction") or ""
            if settlement_state(settlement) == "unknown" and is_base and tx_hash:
                # The reconcile follows this transaction. A known failure is
                # credited once the receipt shows the transfer, which may
                # already have happened; a revert or an expiry credits nothing.
                _job_ended(refund=not keep_payment, cost=base_price)
                credited = await _follow_up()
                resolved = base_follow["resolved"]
                if credited:
                    extra.update(credited)
                elif not keep_payment and resolved == "failed":
                    # The transfer reverted: the payer was charged nothing.
                    logger.warning(
                        "%s failed and Base tx %s reverted; nothing charged or credited",
                        provider_name,
                        tx_hash,
                    )
                elif not keep_payment and resolved is not None:
                    logger.error(
                        "%s failed and Base tx %s never resolved before its authorization "
                        "expired; nothing credited",
                        provider_name,
                        tx_hash,
                    )
                elif not keep_payment:
                    extra["settlement_pending"] = True
                    extra["pending_refund"] = str(final_price)
                    extra["transaction"] = tx_hash
                    logger.error(
                        "%s failed and the payment status is unknown; the refund waits "
                        "for the receipt of tx %s",
                        provider_name,
                        tx_hash,
                    )
            elif settlement_state(settlement) == "unknown":
                logger.error(
                    "%s failed and the payment status is unknown; nothing credited",
                    provider_name,
                )
            if settlement_state(settlement) == "ok":
                if keep_payment:
                    stats.record_revenue(
                        provider_name,
                        Decimal(str(settlement.get("amount_usdc", 0))),
                        base_price,
                    )
                else:
                    extra.update(await _refund_settled_payment(payer, final_price))
        response = _error_response(status_code, detail, provider=provider_label, **extra)
        if report is not None:
            response.headers.update(_prepaid_headers(report))
        return response

    async def _job_failed(exc: TaskTimeoutError | ProviderError) -> JSONResponse:
        """Record a failed provider call and build its error response."""
        t_err = time.monotonic() - t_gen_start
        if isinstance(exc, TaskTimeoutError):
            stats.record_request(provider_name, t_err, False, error_msg=f"Timeout {exc.timeout}s")
            return await _provider_error(
                504,
                f"Task timed out after {exc.timeout}s",
                provider_label=provider.name,
                keep_payment=True,
                task_id=exc.task_id,
            )
        stats.record_request(provider_name, t_err, False, error_msg=exc.detail)
        return await _provider_error(
            exc.status_code,
            exc.detail,
            provider_label=exc.provider,
            keep_payment=outcome_unknown(exc),
        )

    async def _job_crashed(exc: Exception) -> JSONResponse:
        """Handle an error that is not a ProviderError.

        It is a gateway bug, and the provider may have run and billed the
        job, so the outcome is unknown and the payment is kept.
        """
        logger.error("%s call crashed after payment", provider_name, exc_info=exc)
        stats.record_request(
            provider_name,
            time.monotonic() - t_gen_start,
            False,
            error_msg=f"Gateway error: {type(exc).__name__}",
        )
        return await _provider_error(
            502,
            "Gateway error while handling the provider response",
            provider_label=provider.name,
            keep_payment=True,
        )

    body["_caller"] = (prepaid_pubkey or payer or "x402")[:12]
    # Reset by _run_job once the payment has settled.
    t_gen_start = time.monotonic()
    # Set by _run_job once the provider answers; _respond reads them.
    result: dict[str, Any] = {}
    is_async = False
    task_id = ""

    async def _respond(output: Any) -> JSONResponse:
        """Settle a finished provider call and build the client response."""
        generation_s = time.monotonic() - t_gen_start

        # 8. Calculate actual cost (if provider supports it)
        #    x402 exact scheme requires settling the signed amount (final_price).
        #    Actual cost is used for dashboard revenue stats and prepaid deduction.
        #    Pass the full `result` (not `output`) so the provider can access
        #    top-level fields like "usage" -- for embeddings, output is a list
        #    and would not have .get().
        try:
            actual_base_price = await provider.calculate_actual_cost(body, result)
        except Exception:
            logger.exception("%s actual cost failed; using the estimate", provider_name)
            actual_base_price = None
        if actual_base_price is None:
            actual_base_price = base_price  # fallback to estimate

        if prepaid_mode:
            # The reservation already took base_price. Keep the actual cost, capped
            # at that hold, and return whatever is left.
            if prepaid_reserved is not None and actual_base_price > prepaid_reserved:
                logger.warning(
                    "Prepaid actual $%s exceeds reserved $%s for %s; charging the reservation",
                    actual_base_price,
                    prepaid_reserved,
                    prepaid_pubkey[:12] + "…",
                )
            remaining, charged = await release(
                prepaid_pubkey, prepaid_reserved or Decimal("0"), actual_base_price
            )
            t_client = time.monotonic() - t_start
            logger.info(
                "Prepaid %s: -$%s from %s (remaining: $%s, %.1fs)",
                provider_name,
                charged,
                prepaid_pubkey[:12] + "…",
                remaining,
                t_client,
            )
            stats.record_request(provider_name, generation_s, True)
            # Prepaid: record usage on the provider card (revenue=cost, profit=0).
            # Bottom-line totals will exclude the revenue part to avoid
            # double-counting with the topup that funded this balance.
            stats.record_prepaid_usage(provider_name, charged)
            # For sync providers where data was extracted from result (e.g.,
            # embeddings), preserve top-level metadata (usage, model).
            if not is_async and output is not result:
                content = {k: v for k, v in result.items() if k != "data"}
                content["data"] = output
            else:
                content = {"data": output}
            report = _prepaid_report(remaining, charged)
            content["prepaid"] = report
            return JSONResponse(content=content, headers=_prepaid_headers(report))

        # 9. The payment settled before the provider call. An unknown Base
        #    settlement is counted as revenue if the reconcile sees it land.
        if is_base and settlement_state(upfront_settlement) == "unknown":
            _job_ended(refund=False, cost=actual_base_price)
            await _follow_up()
        elif upfront_settlement is not None:
            stats.record_revenue(
                provider_name,
                Decimal(str(upfront_settlement.get("amount_usdc", 0))),
                actual_base_price,
            )
        if upfront_settlement is not None:
            logger.info(
                "%s completed after upfront settle: cost=$%s (%.1fs)",
                provider_name,
                actual_base_price,
                generation_s,
            )
        t_client = time.monotonic() - t_start

        # 10. Return result to client
        stats.record_request(provider_name, generation_s, True)
        stats.record_overhead(payment_network, t_client - generation_s)
        # For sync providers where data was extracted from result (e.g.,
        # embeddings), preserve top-level metadata (usage, model).
        if not is_async and output is not result:
            content = {k: v for k, v in result.items() if k != "data"}
            content["data"] = output
        else:
            content = {"data": output}
        return JSONResponse(
            content=content,
            headers={
                "X-Timing-Verify": f"{verify_s:.2f}",
                "X-Timing-Generation": f"{generation_s:.2f}",
            },
        )

    async def _run_job() -> JSONResponse:
        """Settle the payment, call the provider, then charge or refund it."""
        nonlocal result, is_async, task_id, t_gen_start
        if payment_sig:
            refused = await _settle()
            if refused is not None:
                return refused
        # 6. Forward request to provider
        t_gen_start = time.monotonic()
        try:
            if isinstance(provider, SpicyApiProvider):
                # SpicyAPI refuses a fresh quote above the price charged here.
                result = await provider.submit(
                    path, body, prepaid=prepaid_mode, paid_price=base_price
                )
            else:
                result = await provider.submit(path, body, prepaid=prepaid_mode)

            # 7. Determine if response is async (needs polling) or sync (ready)
            #    - Sync providers (OpenRouter, TTS, etc.) return results directly,
            #      often in OpenAI format with "choices" or "data" (list) key
            #    - Async providers (WaveSpeed) return {"data": {"id": ..., "status": ...}}
            output = result.get("data", result)
            is_async = (
                isinstance(output, dict)
                and "id" in output
                and "status" in output
                and "choices" not in result
            )
            if is_async:
                task_id = output["id"]
                output = await provider.get_result(task_id)
        except (TaskTimeoutError, ProviderError) as e:
            return await _job_failed(e)
        except Exception as e:
            return await _job_crashed(e)
        return await _respond(output)

    # The job runs in its own task, from the settle on, so a client that
    # disconnects does not stop it. It still finishes and is charged or
    # refunded on its outcome.
    # Synchronous calls (chat, TTS) run the same way: a long one sends no
    # bytes until it ends, and Railway would cut the idle connection.
    finish = asyncio.create_task(_run_job())
    _track_job(finish)
    try:
        return await asyncio.wait_for(asyncio.shield(finish), _HEARTBEAT_INTERVAL_S)
    except TimeoutError:
        job = f"task {task_id}" if task_id else "call"
        logger.info("Holding %s %s open with heartbeats", provider_name, job)
        return StreamingResponse(
            _stream_finished_job(finish, provider_name, job),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


async def _stream_finished_job(
    finish: asyncio.Task[JSONResponse],
    provider_name: str,
    job: str,
) -> Any:
    """Send JSON whitespace until a provider job finishes, then its body.

    The HTTP status is already 200 once the first pad goes out. A later
    failure is the usual error JSON in that body. If the client leaves,
    only the stream stops: ``finish`` keeps running.
    """
    try:
        while not finish.done():
            yield _HEARTBEAT_PAD
            try:
                await asyncio.wait_for(asyncio.shield(finish), _HEARTBEAT_INTERVAL_S)
            except TimeoutError:
                continue
            except Exception:
                break
        yield finish.result().body
    except (GeneratorExit, asyncio.CancelledError):
        logger.warning(
            "Client left %s %s before the result; it is billed on its outcome",
            provider_name,
            job,
        )
        raise
