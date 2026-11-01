"""x402 payment handling for x402gate.

Supports both EVM (Base) and SVM (Solana) networks via the x402 SDK.
Verifies and settles payments directly on-chain using local facilitator
signers — no external facilitator service required.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from web3 import Web3
from x402 import PaymentPayload, PaymentRequirements
from x402.mechanisms.evm.exact.facilitator import ExactEvmScheme
from x402.mechanisms.evm.signers import FacilitatorWeb3Signer
from x402.mechanisms.svm.exact.facilitator import ExactSvmScheme

from x402gate.core import http
from x402gate.core.config import NetworkConfig
from x402gate.core.pricing import format_price_for_x402
from x402gate.core.svm_signer import SolanaSigner

logger = logging.getLogger(__name__)

# Cached token prices for gas cost estimation
_price_cache: dict[str, float] = {"ethereum": 0.0, "solana": 0.0, "ts": 0.0, "failed_at": 0.0}
_PRICE_TTL = 300  # 5 minutes
# A failed fetch waits this long before the next try.
_PRICE_RETRY_AFTER_FAILURE = 60
_PRICE_TIMEOUT_S = 5.0
_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
# Two signatures, the fee of a payment without a priority fee.
_SVM_FALLBACK_FEE_LAMPORTS = 10_000

# The price only converts gas to USD for the dashboard, so settlement never
# waits for it: a stale price is refreshed in the background, and a running
# refresh is held here so it is not garbage-collected.
_background_tasks: set[asyncio.Task] = set()
_price_refresh: asyncio.Task | None = None


async def _fetch_prices() -> dict:
    """Fetch ETH and SOL prices from CoinGecko."""
    r = await http.shared_client().get(
        _COINGECKO_URL,
        params={"ids": "ethereum,solana", "vs_currencies": "usd"},
        timeout=http.timeout(_PRICE_TIMEOUT_S),
    )
    return r.json()


def _prices_fresh(now: float) -> bool:
    """True when the cache needs no fetch: fresh, or a fetch failed recently."""
    if now - _price_cache["ts"] < _PRICE_TTL and (
        _price_cache["ethereum"] > 0 or _price_cache["solana"] > 0
    ):
        return True
    return now - _price_cache["failed_at"] < _PRICE_RETRY_AFTER_FAILURE


async def _refresh_prices() -> None:
    """Refresh cached ETH and SOL prices from CoinGecko. Never raises."""
    now = time.time()
    if _prices_fresh(now):
        return
    try:
        data = await _fetch_prices()
        if not isinstance(data, dict) or not (
            data.get("ethereum", {}).get("usd") or data.get("solana", {}).get("usd")
        ):
            raise ValueError(f"no prices in response: {str(data)[:200]}")
        _price_cache["ethereum"] = data.get("ethereum", {}).get("usd", _price_cache["ethereum"])
        _price_cache["solana"] = data.get("solana", {}).get("usd", _price_cache["solana"])
        _price_cache["ts"] = now
        logger.info(
            "Prices updated: ETH=$%.2f, SOL=$%.2f", _price_cache["ethereum"], _price_cache["solana"]
        )
    except Exception as e:
        _price_cache["failed_at"] = now
        logger.warning("Failed to fetch prices from CoinGecko: %s", e)


def _spawn(coro: Any) -> asyncio.Task:
    """Run ``coro`` in the background, keeping a reference until it ends."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_done)
    return task


def _background_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background payment task failed: %s", task.exception())


def _schedule_price_refresh() -> None:
    """Start a background price refresh when the cache is stale."""
    global _price_refresh
    if _prices_fresh(time.time()):
        return
    running = _price_refresh
    if (
        running is not None
        and not running.done()
        and running.get_loop() is asyncio.get_running_loop()
    ):
        return
    _price_refresh = _spawn(_refresh_prices())


def _cached_price(coin: str) -> float:
    """Return the cached USD price of ``coin`` (0 if never fetched) without waiting."""
    _schedule_price_refresh()
    return _price_cache[coin]


def _settlement_failure_reason(result: Any) -> str:
    """Build a readable reason from a failed settle result.

    The SDK often sets ``error_message`` to None, which used to log as the
    text "None". Prefer the first real detail attached to the result.
    """
    error_reason = getattr(result, "error_reason", None) or "unknown"
    detail = getattr(result, "error_message", None) or getattr(result, "invalid_message", None)
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or str(detail)
    if detail:
        return f"{error_reason}: {detail}"
    return str(error_reason)


def _canonical_nonce(nonce: str) -> str:
    """Return the nonce as plain lowercase hex.

    The SDK reads it with ``bytes.fromhex``, which accepts an optional 0x,
    either case and spaces, so all of those spellings are one nonce.
    """
    try:
        return bytes.fromhex(nonce.strip().removeprefix("0x").removeprefix("0X")).hex()
    except ValueError:
        return nonce.lower()


def payment_key(payment_signature: str) -> str:
    """Return a stable id for one payment.

    A Base authorization is identified by its payer and nonce, so re-encoding
    the same signed authorization does not produce a new id. A Solana
    transaction is identified by a hash of its bytes, anything else by a hash
    of its inner payload, and an unreadable header by a hash of the header.
    """
    try:
        payload_dict = json.loads(base64.b64decode(payment_signature))
        network = str(payload_dict.get("accepted", {}).get("network", ""))
        inner = payload_dict.get("payload", {})
        authorization = inner.get("authorization") if isinstance(inner, dict) else None
        if isinstance(authorization, dict) and authorization.get("nonce"):
            payer = str(authorization.get("from", "")).lower()
            return f"{network}:{payer}:{_canonical_nonce(str(authorization['nonce']))}"
        transaction = inner.get("transaction") if isinstance(inner, dict) else None
        if isinstance(transaction, str):
            digest = hashlib.sha256(base64.b64decode(transaction)).hexdigest()
        else:
            digest = hashlib.sha256(json.dumps(inner, sort_keys=True).encode()).hexdigest()
        return f"{network}:{digest}"
    except Exception:
        return "raw:" + hashlib.sha256(payment_signature.encode()).hexdigest()


class PaymentClaims:
    """Payments that a request is using right now, or used a moment ago.

    One signed payment must buy one generation. Without this, the same
    payment sent in parallel requests passes verify in all of them, because
    nothing has landed on-chain yet. A claim lasts until the request's
    settlement finishes. After a successful settle the claim is kept for
    ``hold_s`` more, so an RPC node that has not seen the transaction yet
    cannot let a replay through.
    """

    def __init__(self) -> None:
        self._claims: dict[str, float] = {}

    def claim(self, key: str) -> bool:
        """Take the payment for this request. False if another request has it."""
        now = time.monotonic()
        expired = [k for k, until in self._claims.items() if until <= now]
        for k in expired:
            del self._claims[k]
        if key in self._claims:
            return False
        self._claims[key] = float("inf")
        return True

    def release(self, key: str, hold_s: float = 0.0) -> None:
        """Free the payment, keeping it blocked for ``hold_s`` seconds."""
        if hold_s > 0:
            self._claims[key] = time.monotonic() + hold_s
        else:
            self._claims.pop(key, None)

    def clear(self) -> None:
        """Forget every claim."""
        self._claims.clear()


class InFlightAmounts:
    """USDC that each Base payer has promised to requests still settling.

    Each authorization has its own nonce, so the payment claim cannot see
    two of them drawing on one balance. Without this, a wallet holding the
    price of one job could start many paid provider calls at once; only
    one settlement would land.
    """

    def __init__(self) -> None:
        self._amounts: dict[str, Decimal] = {}

    def add(self, payer: str, amount: Decimal) -> Decimal:
        """Promise ``amount`` and return the payer's total in flight."""
        key = payer.lower()
        total = self._amounts.get(key, Decimal("0")) + amount
        self._amounts[key] = total
        return total

    def remove(self, payer: str, amount: Decimal) -> None:
        """Drop a promise once its settlement has finished."""
        key = payer.lower()
        left = self._amounts.get(key, Decimal("0")) - amount
        if left > 0:
            self._amounts[key] = left
        else:
            self._amounts.pop(key, None)

    def clear(self) -> None:
        """Forget every promise."""
        self._amounts.clear()


def settlement_state(settlement: dict[str, Any] | None) -> str:
    """Classify a settle result as "ok", "failed" or "unknown".

    Unknown means the transaction may still land: settle crashed (None), or
    it was broadcast and its confirmation wait failed (settlement_pending).
    """
    if settlement is None or settlement.get("pending"):
        return "unknown"
    return "ok" if settlement.get("success") else "failed"


class _NetworkScheme:
    """Holds a network's scheme, signer, and config for verify/settle."""

    def __init__(
        self, net_cfg: NetworkConfig, scheme: Any, w3: Any = None, svm_signer: Any = None
    ) -> None:
        self.config = net_cfg
        self.scheme = scheme
        self.w3 = w3  # Web3 instance (EVM only, for gas queries)
        self.svm_signer = svm_signer  # SolanaSigner instance (SVM only, for tx fee queries)


class PaymentHandler:
    """Handles x402 payment verification and settlement.

    Supports multiple networks (EVM + SVM). Each network is configured
    independently with its own scheme, signer, and RPC endpoint.
    """

    def __init__(
        self,
        networks: dict[str, NetworkConfig],
        max_timeout: int = 3600,
    ) -> None:
        self._max_timeout = max_timeout
        self._schemes: dict[str, _NetworkScheme] = {}

        for name, net_cfg in networks.items():
            try:
                if net_cfg.type == "evm":
                    self._init_evm(name, net_cfg)
                elif net_cfg.type == "svm":
                    self._init_svm(name, net_cfg)
                else:
                    logger.warning(
                        "Unknown network type '%s' for '%s', skipping", net_cfg.type, name
                    )
            except Exception as e:
                logger.warning(
                    "Failed to initialize network '%s' (%s): %s — skipping",
                    name,
                    net_cfg.type,
                    e,
                )

    def _init_evm(self, name: str, cfg: NetworkConfig) -> None:
        """Initialize an EVM network (Base, Ethereum, etc.)."""

        signer = FacilitatorWeb3Signer(
            private_key=cfg.facilitator_key,
            rpc_url=cfg.rpc_url,
        )
        w3 = Web3(Web3.HTTPProvider(cfg.rpc_url))
        scheme = ExactEvmScheme(signer)

        self._schemes[cfg.network] = _NetworkScheme(cfg, scheme, w3=w3)
        logger.info(
            "EVM network '%s' initialized (address=%s, network=%s)",
            name,
            signer.address,
            cfg.network,
        )

    def _init_svm(self, name: str, cfg: NetworkConfig) -> None:
        """Initialize an SVM network (Solana)."""

        signer = SolanaSigner(
            private_key=cfg.facilitator_key,
            rpc_url=cfg.rpc_url,
        )
        scheme = ExactSvmScheme(signer)

        self._schemes[cfg.network] = _NetworkScheme(cfg, scheme, svm_signer=signer)
        logger.info(
            "SVM network '%s' initialized (address=%s, network=%s)",
            name,
            signer.address,
            cfg.network,
        )

    def get_all_payment_requirements(self, price: Decimal) -> list[dict[str, Any]]:
        """Build payment requirements for ALL configured networks.

        The client chooses which network to pay on.

        Args:
            price: Final price in USD (after commission).

        Returns:
            List of payment requirements dicts — one per network.
        """
        requirements = []
        for _network_id, ns in self._schemes.items():
            req = self._build_requirements(price, ns)
            requirements.append(req)
        return requirements

    def _build_requirements(self, price: Decimal, ns: _NetworkScheme) -> dict[str, Any]:
        """Build payment requirements for a single network."""
        cfg = ns.config
        price_str = format_price_for_x402(price)

        # USDC has 6 decimals on both EVM and Solana
        amount = str(int(price * 1_000_000))

        req: dict[str, Any] = {
            "scheme": "exact",
            "network": cfg.network,
            "payTo": cfg.pay_to,
            "price": price_str,
            "asset": cfg.token_address,
            "amount": amount,
            "maxTimeoutSeconds": self._max_timeout,
        }

        if cfg.type == "evm":
            req["extra"] = {
                "name": cfg.token_name,
                "version": cfg.token_version,
            }
        elif cfg.type == "svm":
            # SVM requires feePayer in extra
            req["extra"] = ns.scheme.get_extra(cfg.network)

        return req

    def create_payment_required(self, price: Decimal) -> JSONResponse:
        """Build an HTTP 402 response with x402 payment requirements.

        Returns requirements for all configured networks so the client
        can choose which chain to pay on.
        """
        accepts = self.get_all_payment_requirements(price)
        body = {
            "error": "Payment Required",
            "accepts": accepts,
        }
        return JSONResponse(
            status_code=402,
            content=body,
            headers={"Content-Type": "application/json"},
        )

    def _detect_network(self, payment_signature: str) -> tuple[str, dict]:
        """Detect which network a payment signature is for.

        Args:
            payment_signature: Base64-encoded payment payload.

        Returns:
            Tuple of (network_id, payload_dict).
        """
        payload_json = base64.b64decode(payment_signature)
        payload_dict = json.loads(payload_json)

        # x402 v2 payload has 'accepted.network'
        network = payload_dict.get("accepted", {}).get("network", "")
        return network, payload_dict

    def get_payment_requirements_for_network(self, price: Decimal, network: str) -> dict[str, Any]:
        """Get requirements for a specific network."""
        ns = self._schemes.get(network)
        if not ns:
            raise ValueError(f"Unknown network: {network}")
        return self._build_requirements(price, ns)

    @staticmethod
    def extract_payment_signature(request: Request) -> str | None:
        """Extract the PAYMENT-SIGNATURE header from a request."""
        return request.headers.get("payment-signature")

    async def verify(self, payment_signature: str, price: Decimal) -> tuple[bool, str, str, str]:
        """Verify a payment signature on the correct network.

        Args:
            payment_signature: Base64-encoded payment payload.
            price: Expected payment amount.

        Returns:
            Tuple of (is_valid, network_id, payer_address, invalid_reason).
        """
        # Settle follows a passed verify, so the gas price is fetched now and
        # is in the cache by then.
        _schedule_price_refresh()
        try:
            network, payload_dict = self._detect_network(payment_signature)
            ns = self._schemes.get(network)
            if not ns:
                logger.warning("Payment on unsupported network: %s", network)
                return False, network, "", "unsupported_network"

            payload = PaymentPayload.model_validate(payload_dict)
            requirements_dict = self._build_requirements(price, ns)
            requirements = PaymentRequirements.model_validate(requirements_dict)

            result = await asyncio.to_thread(ns.scheme.verify, payload, requirements)
            amount_usdc = int(requirements.amount) / 1_000_000
            payer = getattr(result, "payer", "") or ""
            invalid_reason = getattr(result, "invalid_reason", "") or ""
            invalid_message = getattr(result, "invalid_message", "") or ""
            if result.is_valid:
                logger.info(
                    "Payment verified: $%g USDC from %s on %s (valid=True)",
                    amount_usdc,
                    payer[:10] + "…" if payer else "",
                    network,
                )
            else:
                logger.warning(
                    "Payment rejected: $%g USDC from %s on %s — reason=%s: %s",
                    amount_usdc,
                    payer[:10] + "…" if payer else "",
                    network,
                    invalid_reason,
                    invalid_message,
                )
            reason = f"{invalid_reason}: {invalid_message}" if invalid_message else invalid_reason
            return result.is_valid, network, payer, reason

        except Exception:
            logger.exception("Payment verification failed")
            return False, "", "", "internal_error"

    async def check_evm_balance(self, network: str, payer: str, amount: Decimal) -> bool | None:
        """Check if payer has sufficient USDC balance on EVM.

        EVM verify() only checks the ECDSA signature — not the on-chain balance.
        This method prevents free generation from clients with insufficient funds.
        An RPC failure is not treated as a success: the caller must stop the
        request, because Base settlement happens after the result is returned.

        Args:
            network: Network ID (e.g. "eip155:8453").
            payer: Payer EVM address (0x...).
            amount: Required amount in USD.

        Returns:
            True if balance >= required amount, False if the balance is too
            small, None if the RPC check could not be completed.
        """
        ns = self._schemes.get(network)
        if not ns or not ns.w3:
            return True  # non-EVM or no w3 — skip check

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                bal_sig = ns.w3.keccak(text="balanceOf(address)")[:4]
                data = bal_sig + bytes(12) + bytes.fromhex(payer[2:])
                raw = await asyncio.to_thread(
                    ns.w3.eth.call, {"to": ns.config.token_address, "data": data}
                )
                balance = int.from_bytes(raw, "big")
                required = int(amount * 1_000_000)
                if balance < required:
                    logger.warning(
                        "Insufficient USDC: %s has %d, need %d",
                        payer[:10] + "…",
                        balance,
                        required,
                    )
                return balance >= required
            except Exception as e:
                last_error = e
                logger.warning(
                    "EVM balance check failed for %s (attempt %d/2): %s",
                    payer[:10] + "…",
                    attempt + 1,
                    e,
                )
        logger.error(
            "EVM balance check unavailable for %s: %s",
            payer[:10] + "…",
            last_error,
        )
        return None

    async def settle(
        self,
        payment_signature: str,
        price: Decimal,
        network: str,
        extra_context: dict | None = None,
    ) -> dict[str, Any] | None:
        """Settle a verified payment on-chain.

        Args:
            payment_signature: Base64-encoded payment payload.
            price: Payment amount.
            network: Network ID (from verify result).

        Returns:
            Settlement response dict on success, None on failure.
        """
        try:
            _, payload_dict = self._detect_network(payment_signature)
            ns = self._schemes.get(network)
            if not ns:
                logger.error("Cannot settle — unknown network: %s", network)
                return None

            payload = PaymentPayload.model_validate(payload_dict)
            requirements_dict = self._build_requirements(price, ns)
            requirements = PaymentRequirements.model_validate(requirements_dict)

            result = await asyncio.to_thread(ns.scheme.settle, payload, requirements)
            if not result.success and getattr(result, "error_reason", "") == "settlement_pending":
                # Broadcast, but the confirmation wait failed. The SDK keeps the
                # transaction, and settling the same payload again waits for it
                # instead of sending a new one.
                logger.warning("Settlement pending on %s; waiting for it once more", network)
                result = await asyncio.to_thread(ns.scheme.settle, payload, requirements)
            amount_usdc = int(requirements.amount) / 1_000_000
            settle_data = result.__dict__.copy()

            if result.success:
                # Gas cost reporting
                gas_cost_native = 0.0
                gas_cost_usd = 0.0
                gas_label = ""

                if ns.config.type == "evm" and ns.w3:
                    try:
                        receipt = await asyncio.to_thread(
                            ns.w3.eth.wait_for_transaction_receipt,
                            result.transaction,
                            timeout=30,
                        )
                        gas_cost_wei = receipt.gasUsed * receipt.effectiveGasPrice
                        gas_cost_native = gas_cost_wei / 1e18
                        eth_price = _cached_price("ethereum")
                        gas_cost_usd = gas_cost_native * eth_price if eth_price else 0
                        gas_label = "ETH"
                    except Exception as e:
                        logger.warning("EVM gas cost lookup failed: %s", e)
                        gas_label = "ETH"
                elif ns.config.type == "svm" and ns.svm_signer:
                    # The fee follows from the signed message, so no RPC call:
                    # a getTransaction here delayed every Solana job.
                    try:
                        fee_lamports = ns.svm_signer.fee_lamports(payload.payload["transaction"])
                    except Exception as e:
                        logger.warning("SVM gas cost lookup failed: %s", e)
                        fee_lamports = _SVM_FALLBACK_FEE_LAMPORTS
                    gas_cost_native = fee_lamports / 1e9
                    sol_price = _cached_price("solana")
                    gas_cost_usd = gas_cost_native * sol_price if sol_price else 0
                    gas_label = "SOL"

                settle_data["gas_cost_native"] = gas_cost_native
                settle_data["gas_cost_usd"] = gas_cost_usd
                settle_data["gas_label"] = gas_label
                settle_data["amount_usdc"] = amount_usdc
                settle_data["network_type"] = ns.config.type

                # Compact single-line transaction summary
                provider_cost = 0.0
                generation_s = 0.0
                t_client = 0.0
                provider_label = "?"
                profit = amount_usdc - gas_cost_usd

                if extra_context:
                    provider_cost = extra_context.get("provider_cost", 0)
                    generation_s = extra_context.get("generation_s", 0)
                    provider_label = extra_context.get("provider_name", "?")
                    t_client = extra_context.get("t_client", generation_s)
                    profit = amount_usdc - provider_cost - gas_cost_usd

                def _fmt(v: float) -> str:
                    """Format USD with meaningful precision, no trailing zeros."""
                    return f"{v:.6f}".rstrip("0").rstrip(".")

                logger.info(
                    "TX %s settled: $%s -> cost=$%s gas=$%s %s profit=$%s (%.1fs -> %.1fs)",
                    provider_label,
                    _fmt(amount_usdc),
                    _fmt(provider_cost),
                    _fmt(gas_cost_usd),
                    gas_label,
                    _fmt(profit),
                    generation_s,
                    t_client,
                )
            else:
                reason = _settlement_failure_reason(result)
                logger.warning(
                    "Settlement failed: $%.6f USDC from %s on %s -- %s",
                    amount_usdc,
                    getattr(result, "payer", "") or "",
                    network,
                    reason,
                )
                settle_data["error_reason"] = reason
                settle_data["pending"] = getattr(result, "error_reason", "") == "settlement_pending"
            return settle_data

        except Exception:
            logger.exception("Payment settlement failed")
            return None
