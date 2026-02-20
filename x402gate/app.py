"""FastAPI application for x402gate.

Defines the API routes and wires up providers, pricing, and payment handling.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from x402gate.core.config import AppConfig, load_config
from x402gate.core.payment import PaymentHandler
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError
from x402gate.providers.openrouter import OpenRouterProvider
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
                kwargs["default_max_tokens"] = config.gateway.default_max_tokens
            provider = PROVIDER_REGISTRY[name](config=provider_config, **kwargs)
            providers[name] = provider
            logger.info("Provider '%s' registered (managed)", name)
        else:
            logger.warning("Unknown provider '%s', skipping", name)

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


_VERSION = importlib.metadata.version("x402gate")

app = FastAPI(
    title="x402gate",
    description="Transparent x402 payment proxy for AI services",
    version=_VERSION,
    lifespan=lifespan,
)



# --- Service Discovery ---


@app.get("/", include_in_schema=False)
async def service_info(request: Request) -> Response:
    """Service manifest: HTML for browsers, JSON for AI agents."""
    base_url = os.environ.get("BASE_URL", "")
    networks = list(config.payment.networks.keys())
    commission_pct = int(config.gateway.commission * 100)
    gas_fee = config.gateway.gas_surcharge
    provider_docs = {}
    for name, pcfg in config.providers.items():
        doc: dict[str, Any] = {
            "endpoint_pattern": f"/v1/{name}/{{model_path}}",
        }
        if pcfg.docs_url:
            doc["api_reference"] = pcfg.docs_url
        examples = []
        for ex_field in sorted(f for f in pcfg.model_fields if f.startswith("example_request")):
            ex = getattr(pcfg, ex_field, None)
            if ex:
                model = ex.get("model", "MODEL_PATH")
                examples.append({
                    "method": "POST",
                    "path": f"/v1/{name}/{model}",
                    "body": ex.get("body", {}),
                })
        if len(examples) == 1:
            doc["example_request"] = examples[0]
        elif examples:
            doc["example_requests"] = examples
        provider_docs[name] = doc

    # If browser requests HTML, return a human-readable landing page
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        provider_list = ""
        for name, pcfg in config.providers.items():
            ptype = getattr(pcfg, "type", "managed") or "managed"
            docs_link = f' &middot; <a href="{pcfg.docs_url}">API docs</a>' if pcfg.docs_url else ""
            provider_list += (
                f'<li><strong>{name}</strong> ({ptype})'
                f" &mdash; <code>/v1/{name}/...</code>{docs_link}</li>\n"
            )
        # Build curl examples from provider configs
        import json as _json
        examples_html = ""
        for name, pcfg in config.providers.items():
            for ex_field in sorted(f for f in pcfg.model_fields if f.startswith("example_request")):
                ex = getattr(pcfg, ex_field, None)
                if not ex:
                    continue
                model = ex.get("model", "MODEL")
                body = ex.get("body", {})
                body_json = _json.dumps(body, ensure_ascii=False)
                path = f"/v1/{name}/{model}"
                examples_html += f"""
    <h4>{name} &mdash; <code>{model}</code></h4>
    <pre><code>curl -X POST {base_url}{path} \\
  -H "Content-Type: application/json" \\
  -d '{body_json}'

# Response: 402 Payment Required
# Sign payment, then retry with PAYMENT-SIGNATURE header</code></pre>
"""
        examples_html += """
    <div class="note">Payment is settled <strong>only</strong> on success (HTTP 200). On any error, no USDC is
        transferred.</div>"""

        template_path = pathlib.Path(__file__).parent / "templates" / "index.html"
        html = template_path.read_text(encoding="utf-8")
        html = (
            html.replace("{{ version }}", _VERSION)
            .replace("{{ networks }}", ", ".join(networks))
            .replace("{{ provider_list }}", provider_list)
            .replace("{{ commission }}", f"{commission_pct}% + ${gas_fee} gas")
            .replace("{{ base_url }}", base_url)
            .replace("{{ examples }}", examples_html)
        )
        return HTMLResponse(content=html)

    # Otherwise return JSON for AI agents
    return JSONResponse(content={
        "name": "x402gate",
        "version": _VERSION,
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
        "commission": f"{commission_pct}% + ${gas_fee} gas per request",
        "provider_docs": provider_docs,
        "endpoints": {
            "openapi": f"{base_url}/openapi.json",
            "docs": f"{base_url}/docs",
            "health": f"{base_url}/health",
            "providers": f"{base_url}/v1/providers",
            "ai_plugin": f"{base_url}/.well-known/ai-plugin.json",
        },
        "source": "https://github.com/oponfil/x402gate",
        "documentation": "https://github.com/oponfil/x402gate#readme",
    })



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
                import json
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


async def _passthrough_proxy(
    provider_name: str, path: str, request: Request
) -> Response:
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, content=body, headers=forward_headers)

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


@app.post("/v1/blockrun/{path:path}")
async def blockrun_proxy(path: str, request: Request) -> Any:
    """Transparent passthrough proxy for BlockRun (x402-native)."""
    return await _passthrough_proxy("blockrun", path, request)


# --- Managed Provider Proxy (unified flow for all managed providers) ---


async def _handle_managed_request(
    provider_name: str, path: str, request: Request
) -> Response:
    """Unified x402 payment flow for managed providers.

    Flow:
    1. Parse request body
    2. Fetch dynamic price from provider (with cache)
    3. Apply commission (5% + gas fee)
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

    # 1. Parse request body
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

    # 4. Check for payment
    payment_sig = payment_handler.extract_payment_signature(request)
    if not payment_sig:
        return payment_handler.create_payment_required(final_price)

    # 5. Verify payment (auto-detects network from payload)
    is_valid, payment_network = await payment_handler.verify(payment_sig, final_price)
    if not is_valid:
        return _error_response(402, "Payment verification failed")

    t_start = time.monotonic()  # Start timing after verification

    # 6. Forward request to provider
    t_gen_start = time.monotonic()
    try:
        result = await provider.submit(path, body)
    except ProviderError as e:
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
            return _error_response(
                504, f"Task timed out after {e.timeout}s",
                provider=provider.name, task_id=e.task_id,
            )
        except ProviderError as e:
            # Don't settle — task failed
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

    # 9. Settle payment in background (don't block client)
    t_client = time.monotonic() - t_start

    async def background_settle() -> None:
        """Settle payment and log financial summary in background."""
        try:
            ctx = {
                "estimated_cost": float(base_price),
                "provider_cost": float(actual_base_price),
                "generation_s": generation_s,
                "t_client": t_client,
                "commission_rate": config.gateway.commission,
                "gas_surcharge": config.gateway.gas_surcharge,
            }
            settlement = await payment_handler.settle(
                payment_sig,
                final_price,
                payment_network,
                extra_context=ctx,
            )

            if not settlement:
                logger.error("Settlement returned None for %s — check logs above", payment_network)
            elif not settlement.get("success"):
                logger.error(
                    "Settlement FAILED for %s: %s",
                    payment_network,
                    settlement.get("error_reason", "unknown"),
                )
        except Exception:
            logger.exception("Background settlement crashed for %s", payment_network)

    task = asyncio.create_task(background_settle())
    _pending_settlements.add(task)
    task.add_done_callback(_pending_settlements.discard)

    # 9. Return result to client
    return JSONResponse(content={"data": output})


# --- Per-Provider Routes ---


@app.post("/v1/wavespeed/{path:path}")
async def wavespeed_proxy(path: str, request: Request) -> Any:
    """Managed proxy for WaveSpeed AI (image/video generation)."""
    return await _handle_managed_request("wavespeed", path, request)


@app.post("/v1/openrouter/{path:path}")
async def openrouter_proxy(path: str, request: Request) -> Any:
    """Managed proxy for OpenRouter (300+ LLM models)."""
    return await _handle_managed_request("openrouter", path, request)
