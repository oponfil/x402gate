"""x402 payment handling for x402gate.

Supports both EVM (Base) and SVM (Solana) networks via the x402 SDK.
Verifies and settles payments directly on-chain using local facilitator
signers — no external facilitator service required.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from x402 import PaymentPayload, PaymentRequirements

from x402gate.core.config import NetworkConfig
from x402gate.core.pricing import format_price_for_x402

logger = logging.getLogger(__name__)

# Cached token prices for gas cost estimation
_price_cache: dict[str, float] = {"ethereum": 0.0, "solana": 0.0, "ts": 0.0}
_PRICE_TTL = 300  # 5 minutes


def _fetch_prices_sync() -> dict:
    """Fetch ETH and SOL prices from CoinGecko (blocking)."""
    import httpx

    r = httpx.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": "ethereum,solana", "vs_currencies": "usd"},
        timeout=5.0,
    )
    return r.json()


async def _refresh_prices() -> None:
    """Refresh cached ETH and SOL prices from CoinGecko."""
    now = time.time()
    if now - _price_cache["ts"] < _PRICE_TTL and (
        _price_cache["ethereum"] > 0 or _price_cache["solana"] > 0
    ):
        return
    try:
        data = await asyncio.to_thread(_fetch_prices_sync)
        _price_cache["ethereum"] = data.get("ethereum", {}).get("usd", _price_cache["ethereum"])
        _price_cache["solana"] = data.get("solana", {}).get("usd", _price_cache["solana"])
        _price_cache["ts"] = now
        logger.info(
            "Prices updated: ETH=$%.2f, SOL=$%.2f", _price_cache["ethereum"], _price_cache["solana"]
        )
    except Exception as e:
        logger.warning("Failed to fetch prices from CoinGecko: %s", e)


async def _get_eth_price() -> float:
    await _refresh_prices()
    return _price_cache["ethereum"]


async def _get_sol_price() -> float:
    await _refresh_prices()
    return _price_cache["solana"]


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
        from web3 import Web3
        from x402.mechanisms.evm.exact.facilitator import ExactEvmScheme
        from x402.mechanisms.evm.signers import FacilitatorWeb3Signer

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
        from x402.mechanisms.svm.exact.facilitator import ExactSvmScheme

        from x402gate.core.svm_signer import SolanaSigner

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

    async def verify(self, payment_signature: str, price: Decimal) -> tuple[bool, str]:
        """Verify a payment signature on the correct network.

        Args:
            payment_signature: Base64-encoded payment payload.
            price: Expected payment amount.

        Returns:
            Tuple of (is_valid, network_id).
        """
        try:
            network, payload_dict = self._detect_network(payment_signature)
            ns = self._schemes.get(network)
            if not ns:
                logger.warning("Payment on unsupported network: %s", network)
                return False, network

            payload = PaymentPayload.model_validate(payload_dict)
            requirements_dict = self._build_requirements(price, ns)
            requirements = PaymentRequirements.model_validate(requirements_dict)

            result = await asyncio.to_thread(ns.scheme.verify, payload, requirements)
            amount_usdc = int(requirements.amount) / 1_000_000
            logger.info(
                "Payment verified: $%g USDC from %s on %s (valid=%s)",
                amount_usdc,
                result.payer,
                network,
                result.is_valid,
            )
            return result.is_valid, network

        except Exception:
            logger.exception("Payment verification failed")
            return False, ""

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
                        eth_price = await _get_eth_price()
                        gas_cost_usd = gas_cost_native * eth_price if eth_price else 0
                        gas_label = "ETH"
                    except Exception as e:
                        logger.warning("EVM gas cost lookup failed: %s", e)
                        gas_label = "ETH"
                elif ns.config.type == "svm" and ns.svm_signer:
                    try:
                        from solders.signature import Signature as SolSignature

                        tx_sig = SolSignature.from_string(result.transaction)
                        tx_resp = await ns.svm_signer.get_transaction_async(
                            tx_sig,
                            commitment="confirmed",
                            max_supported_transaction_version=0,
                        )
                        fee_lamports = tx_resp.value.transaction.meta.fee
                        gas_cost_native = fee_lamports / 1e9
                        sol_price = await _get_sol_price()
                        gas_cost_usd = gas_cost_native * sol_price if sol_price else 0
                        gas_label = "SOL"
                    except Exception as e:
                        logger.warning("SVM gas cost lookup failed: %s", e)
                        gas_cost_native = 0.000005
                        gas_cost_usd = 0
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
                logger.warning(
                    "Settlement failed: $%.6f USDC from %s on %s -- %s: %s",
                    amount_usdc,
                    result.payer,
                    network,
                    result.error_reason,
                    getattr(result, "error_message", "no details"),
                )
            return settle_data

        except Exception:
            logger.exception("Payment settlement failed")
            return None
