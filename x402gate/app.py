"""FastAPI application for x402gate.

Defines the API routes and wires up providers, pricing, and payment handling.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from x402gate.core.config import AppConfig, load_config
from x402gate.core.payment import PaymentHandler
from x402gate.core.pricing import PriceCache, apply_commission
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import BaseProvider, ProviderError
from x402gate.providers.wavespeed import WaveSpeedProvider

logger = logging.getLogger(__name__)

# Global state (initialized in lifespan)
config: AppConfig
providers: dict[str, BaseProvider] = {}
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

    # Register providers
    for name, provider_config in config.providers.items():
        if not provider_config.enabled:
            logger.info("Provider '%s' is disabled, skipping", name)
            continue

        if name == "wavespeed":
            provider = WaveSpeedProvider(config=provider_config)
            providers[name] = provider
            logger.info("Provider '%s' registered", name)
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
async def service_info() -> dict[str, Any]:
    """Machine-readable service manifest for AI agents."""
    base_url = os.environ.get("BASE_URL", "")
    networks = list(config.payment.networks.keys())
    commission_pct = int(config.gateway.commission * 100)
    gas_fee = config.gateway.min_commission
    provider_docs = {}
    for name, pcfg in config.providers.items():
        doc: dict[str, Any] = {
            "endpoint_pattern": f"/v1/{name}/{{model_path}}",
        }
        if pcfg.docs_url:
            doc["api_reference"] = pcfg.docs_url
        if pcfg.example_request:
            model = pcfg.example_request.get("model", "MODEL_PATH")
            doc["example_request"] = {
                "method": "POST",
                "path": f"/v1/{name}/{model}",
                "body": pcfg.example_request.get("body", {}),
            }
        provider_docs[name] = doc
    return {
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
    }



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
        if pcfg.example_request:
            import json
            model = pcfg.example_request.get("model", "MODEL_PATH")
            body = pcfg.example_request.get("body", {})
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
    return {
        "providers": {
            name: {
                "enabled": True,
                "base_url": provider.config.base_url,
            }
            for name, provider in providers.items()
        }
    }


# --- WaveSpeed Proxy ---


@app.post("/v1/wavespeed/{path:path}")
async def wavespeed_proxy(path: str, request: Request) -> Any:
    """Transparent proxy endpoint for WaveSpeed AI.

    Flow:
    1. Parse request body
    2. Fetch dynamic price from WaveSpeed Pricing API (with cache)
    3. Apply 5% commission
    4. If no PAYMENT-SIGNATURE header -> return 402
    5. Verify payment via facilitator
    6. Forward request to WaveSpeed as-is
    7. Poll for async task result
    8. Settle payment on-chain
    9. Return result to client
    """
    provider = providers.get("wavespeed")
    if provider is None:
        return JSONResponse(
            status_code=503,
            content={"error": "WaveSpeed provider is not configured"},
        )

    # 1. Parse request body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body"},
        )

    # 2. Fetch price (with cache)
    try:
        cached_price = price_cache.get(path, body)
        if cached_price is not None:
            base_price = cached_price
        else:
            base_price = await provider.get_price(path, body)
            price_cache.set(path, body, base_price)
    except ProviderError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": f"Pricing unavailable: {e.detail}"},
        )

    # 3. Apply commission
    final_price = apply_commission(
        base_price, config.gateway.commission, config.gateway.min_commission
    )

    # 4. Check for payment
    payment_sig = payment_handler.extract_payment_signature(request)
    if not payment_sig:
        return payment_handler.create_payment_required(final_price)

    # 5. Verify payment (auto-detects network from payload)
    is_valid, payment_network = await payment_handler.verify(payment_sig, final_price)
    if not is_valid:
        return JSONResponse(
            status_code=402,
            content={"error": "Payment verification failed"},
        )

    t_start = time.monotonic()  # Start timing after verification

    # 6. Forward request to WaveSpeed
    try:
        result = await provider.submit(path, body)
    except ProviderError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": f"Provider error: {e.detail}"},
        )

    # 7. Poll for result
    task_data = result.get("data", result)
    task_id = task_data.get("id")
    if not task_id:
        # Synchronous response (no polling needed)
        await payment_handler.settle(payment_sig, final_price, payment_network)
        return result

    try:
        output = await provider.get_result(task_id)
    except TaskTimeoutError as e:
        # Don't settle — client keeps their money
        return JSONResponse(
            status_code=504,
            content={
                "error": f"Task timed out after {e.timeout}s",
                "task_id": e.task_id,
            },
        )
    except ProviderError as e:
        # Don't settle — task failed
        return JSONResponse(
            status_code=e.status_code,
            content={"error": f"Task failed: {e.detail}"},
        )

    # 8. Settle payment in background (don't block client)
    t_client = time.monotonic() - t_start

    async def background_settle() -> None:
        """Settle payment and log financial summary in background."""
        try:
            inference_ms = output.get("timings", {}).get("inference", 0)
            ctx = {
                "provider_cost": float(base_price),
                "inference_ms": inference_ms,
                "t_client": t_client,
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

    # 9. Return result immediately to client
    return JSONResponse(content={"data": output})
