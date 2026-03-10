"""FastAPI application for x402gate.

Defines the API routes and wires up providers, pricing, and payment handling.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any

import httpx
import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from x402gate import __version__
from x402gate.core import stats
from x402gate.core.config import AppConfig, load_config
from x402gate.core.payment import PaymentHandler
from x402gate.core.prepaid import (
    build_signing_message,
    deduct,
    deposit,
    get_balance,
    validate_timestamp,
    verify_wallet_signature,
)
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError
from x402gate.providers.cloudconvert import CloudConvertProvider
from x402gate.providers.openrouter import OpenRouterProvider
from x402gate.providers.socialdownload import SocialDownloadProvider
from x402gate.providers.tungsten import TungstenProvider
from x402gate.providers.wavespeed import WaveSpeedProvider

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Provider registry: maps config name → provider class.
# Passthrough providers don't need a class — they're handled generically.
# To add a new managed provider, add one line here and create the class.
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "wavespeed": WaveSpeedProvider,
    "openrouter": OpenRouterProvider,
    "tungsten": TungstenProvider,
    "cloudconvert": CloudConvertProvider,
    "socialdownload": SocialDownloadProvider,
}

# Global state (initialized in lifespan)
config: AppConfig
providers: dict[str, BaseProvider | None] = {}  # None = passthrough (no provider object)
payment_handler: PaymentHandler
price_cache: PriceCache
_pending_settlements: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global config, providers, payment_handler, price_cache

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded successfully")

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
    stats.init(list(providers.keys()))
    stats.install_log_handler()

    logger.info(
        "x402gate started on %s:%d with %d provider(s)",
        config.gateway.host,
        config.gateway.port,
        len(providers),
    )

    yield  # Application is running

    # Shutdown: wait for pending settlements
    if _pending_settlements:
        logger.info("Waiting for %d pending settlement(s)...", len(_pending_settlements))
        await asyncio.gather(*_pending_settlements, return_exceptions=True)
    for provider in providers.values():
        if provider is not None:
            await provider.close()
    logger.info("x402gate shut down")


app = FastAPI(
    title="x402gate",
    description="Transparent x402 payment proxy for AI services",
    version=__version__,
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
        ptype = getattr(pcfg, "type", "managed") or "managed"
        provider_info.append(
            {
                "name": name,
                "type": ptype,
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
        "version": __version__,
        "base_url": base_url,
        "networks": networks,
        "commission": f"{commission_pct}% + ${gas_fee} gas",
        "provider_info": provider_info,
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
        "version": __version__,
        "description": (
            "Transparent pay-per-request proxy for AI services via the x402 protocol. "
            "Send a POST request to any provider endpoint — if no payment header is "
            "included, a 402 response is returned with USDC payment options. "
            "The request body format is defined by the upstream provider — "
            "x402gate forwards it as-is. See provider_docs for API references."
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
            "The request body format is defined by the upstream provider — "
            "x402gate forwards it as-is. See the provider docs for body schema. "
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
    )
    return HTMLResponse(content=html)


@app.get("/v1/stats")
async def get_stats() -> dict[str, Any]:
    """Return current gateway statistics."""
    return stats.get_stats()


@app.get("/v1/logs")
async def get_logs(limit: int = 200) -> list[dict[str, Any]]:
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
    min_topup = Decimal(str(config.gateway.min_prepaid_topup))
    max_topup = Decimal(str(config.gateway.max_prepaid_topup))

    try:
        body = await request.json()
    except Exception:
        body = {}

    raw_amount = body.get("amount") if isinstance(body, dict) else None
    if raw_amount is not None:
        try:
            requested_amount = Decimal(str(raw_amount))
        except Exception:
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

    # Verify payment
    is_valid, payment_network, payer = await payment_handler.verify(payment_sig, topup_amount)
    if not is_valid or not payer:
        return _error_response(402, "Payment verification failed")

    # Calculate net credit: topup_amount minus commission and gas
    commission = topup_amount * Decimal(str(config.gateway.commission))
    gas_fee = Decimal(str(config.gateway.gas_surcharge))
    net_credit = topup_amount - commission - gas_fee
    if net_credit <= 0:
        return _error_response(400, "Top-up amount too small to cover fees")

    # Credit the prepaid balance
    new_balance = await deposit(payer, net_credit)

    # Settle payment in background
    async def _settle_topup() -> None:
        try:
            await payment_handler.settle(
                payment_sig,
                topup_amount,
                payment_network,
                extra_context={"provider_name": "topup", "provider_cost": 0},
            )
        except Exception:
            logger.exception("Top-up settlement failed for %s", payer)

    task = asyncio.create_task(_settle_topup())
    _pending_settlements.add(task)
    task.add_done_callback(_pending_settlements.discard)

    stats.record_topup(topup_amount)

    return JSONResponse(
        content={
            "pubkey": payer,
            "credited": str(net_credit),
            "balance": str(new_balance),
            "warning": "Balance is stored in memory only. It will be lost on server restart.",
        }
    )


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
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, content=body, headers=forward_headers)

    latency = time.monotonic() - t_start
    success = resp.status_code < 400
    stats.record_request(
        provider_name, latency, success,
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
    """Unified x402 payment flow for managed providers.

    Flow:
    1. Parse request body
    2. Fetch dynamic price from provider (with cache)
    3. Apply commission (4% + gas fee)
    4. If no PAYMENT-SIGNATURE header → return 402
    5. Verify payment via facilitator
    6. Forward request to provider as-is
    7. If async (task_id present) → poll for result; else use response directly
    8. Settle payment on-chain (background)
    9. Return result to client
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
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            body: dict[str, Any] = {}
            for key in form:
                value = form[key]
                if hasattr(value, "read"):  # UploadFile
                    file_bytes = await value.read()
                    if len(file_bytes) > max_upload_bytes:
                        return _error_response(
                            413,
                            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
                            f"Maximum: {config.gateway.max_upload_mb} MB",
                        )
                    body["_file_bytes"] = file_bytes
                    body["_file_name"] = getattr(value, "filename", "input_file")
                else:
                    body[key] = value
        except Exception:
            return _error_response(400, "Invalid multipart form data")
    else:
        try:
            body = await request.json()
        except Exception:
            return _error_response(400, "Invalid JSON body")

    # 2. Fetch price (with cache)
    try:
        cached_price = price_cache.get(path, body)
        if cached_price is not None:
            base_price = cached_price
        else:
            base_price = await provider.get_price(path, body)
            price_cache.set(path, body, base_price)
    except ProviderError as e:
        return _error_response(e.status_code, e.detail, provider=e.provider)

    # 3. Apply commission
    final_price = apply_commission(
        base_price, config.gateway.commission, config.gateway.gas_surcharge
    )

    # 4. Check for payment (x402 or prepaid)
    payment_sig = payment_handler.extract_payment_signature(request)
    prepaid_pubkey = request.headers.get("x-prepaid-pubkey")
    prepaid_mode = False

    if payment_sig:
        # 5a. Standard x402 payment flow
        is_valid, payment_network, _ = await payment_handler.verify(payment_sig, final_price)
        if not is_valid:
            return _error_response(402, "Payment verification failed")
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

        # Check balance against base_price (no commission — already paid at top-up)
        current_balance = get_balance(prepaid_pubkey)
        if current_balance < base_price:
            return _error_response(
                402,
                f"Insufficient prepaid balance: ${current_balance} < ${base_price}",
                balance=str(current_balance),
            )
        prepaid_mode = True
    else:
        return payment_handler.create_payment_required(final_price)

    # 6. Forward request to provider
    t_gen_start = time.monotonic()
    try:
        result = await provider.submit(path, body, prepaid=prepaid_mode)
    except ProviderError as e:
        t_err = time.monotonic() - t_start
        stats.record_request(provider_name, t_err, False, error_msg=e.detail)
        return _error_response(e.status_code, e.detail, provider=e.provider)

    # 7. Determine if response is async (needs polling) or sync (ready)
    #    - Sync providers (OpenRouter, etc.) return results directly,
    #      often in OpenAI format with "choices" key
    #    - Async providers (WaveSpeed) return {"data": {"id": ..., "status": ...}}
    task_data = result.get("data", result)
    is_async = (
        isinstance(task_data, dict)
        and "id" in task_data
        and "status" in task_data
        and "choices" not in result
    )

    if is_async:
        task_id = task_data["id"]
        # Async provider — poll for completion
        try:
            output = await provider.get_result(task_id)
        except TaskTimeoutError as e:
            # Don't settle — client keeps their money
            t_err = time.monotonic() - t_start
            stats.record_request(provider_name, t_err, False, error_msg=f"Timeout {e.timeout}s")
            return _error_response(
                504,
                f"Task timed out after {e.timeout}s",
                provider=provider.name,
                task_id=e.task_id,
            )
        except ProviderError as e:
            # Don't settle — task failed
            t_err = time.monotonic() - t_start
            stats.record_request(provider_name, t_err, False, error_msg=e.detail)
            return _error_response(e.status_code, e.detail, provider=e.provider)
    else:
        # Synchronous provider — use result directly
        output = task_data

    generation_s = time.monotonic() - t_gen_start  # Provider processing time

    # 8. Calculate actual cost (if provider supports it)
    #    x402 exact scheme requires settling the signed amount (final_price).
    #    Actual cost is used only for Transaction Summary reporting.
    actual_base_price = await provider.calculate_actual_cost(body, output)
    if actual_base_price is None:
        actual_base_price = base_price  # fallback to estimate

    if prepaid_mode:
        # Prepaid: deduct actual cost from balance (no settlement needed)
        deducted = await deduct(prepaid_pubkey, actual_base_price)
        remaining = get_balance(prepaid_pubkey)
        if not deducted:
            logger.error(
                "Prepaid deduction failed for %s (tried $%s, had $%s)",
                prepaid_pubkey[:12] + "…",
                actual_base_price,
                remaining,
            )
        else:
            t_client = time.monotonic() - t_start
            logger.info(
                "Prepaid %s: -$%s from %s (remaining: $%s, %.1fs)",
                provider_name,
                actual_base_price,
                prepaid_pubkey[:12] + "…",
                remaining,
                t_client,
            )
        t_prepaid = time.monotonic() - t_start
        stats.record_request(provider_name, t_prepaid, True)
        stats.record_revenue(provider_name, final_price, actual_base_price)
        return JSONResponse(
            content={"data": output},
            headers={"X-Prepaid-Balance": str(remaining)},
        )

    # 9. Settle payment in background (don't block client)
    t_client = time.monotonic() - t_start

    async def background_settle() -> None:
        """Settle payment and log financial summary in background."""
        try:
            ctx = {
                "provider_name": provider_name,
                "estimated_cost": float(base_price),
                "provider_cost": float(actual_base_price),
                "generation_s": generation_s,
                "t_client": t_client,
                "commission_rate": config.gateway.commission,
                "gas_surcharge": config.gateway.gas_surcharge,
            }
            t_settle_start = time.monotonic()
            settlement = await payment_handler.settle(
                payment_sig,
                final_price,
                payment_network,
                extra_context=ctx,
            )
            settle_latency = time.monotonic() - t_settle_start

            if not settlement:
                logger.error("Settlement returned None for %s — check logs above", payment_network)
            elif not settlement.get("success"):
                logger.error(
                    "Settlement FAILED for %s: %s",
                    payment_network,
                    settlement.get("error_reason", "unknown"),
                )
            else:
                stats.record_revenue(
                    provider_name,
                    Decimal(str(settlement.get("amount_usdc", 0))),
                    actual_base_price,
                )
                stats.record_settlement(
                    payment_network,
                    settle_latency,
                    settlement.get("gas_cost_usd", 0.0),
                    settlement.get("gas_cost_native", 0.0),
                    settlement.get("gas_label", ""),
                )
        except Exception:
            logger.exception("Background settlement crashed for %s", payment_network)

    task = asyncio.create_task(background_settle())
    _pending_settlements.add(task)
    task.add_done_callback(_pending_settlements.discard)

    # 10. Return result to client
    stats.record_request(provider_name, t_client, True)
    return JSONResponse(content={"data": output})
