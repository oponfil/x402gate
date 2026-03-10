"""Configuration loader for x402gate.

Loads settings from config.yaml and interpolates environment variables.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator


def _interpolate_env(value: Any) -> Any:
    """Replace ${VAR} placeholders with environment variable values."""
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{(\w+)\}")
    match = pattern.search(value)
    if not match:
        return value
    result = value
    for m in pattern.finditer(value):
        env_var = m.group(1)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ValueError(
                f"Environment variable '{env_var}' is required but not set. "
                f"Add it to your .env file or export it."
            )
        result = result.replace(m.group(0), env_value)
    return result


def _interpolate_dict(data: dict) -> dict:
    """Recursively interpolate env vars in a dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _interpolate_dict(value)
        elif isinstance(value, list):
            result[key] = [_interpolate_env(item) for item in value]
        else:
            result[key] = _interpolate_env(value)
    return result


class GatewayConfig(BaseModel):
    """Gateway server settings."""

    host: str = "0.0.0.0"
    port: int = 4021
    commission: float = 0.04
    gas_surcharge: float = 0.001  # fixed $0.001 gas surcharge per request
    default_max_tokens: int = 1024  # default max_tokens when client omits it
    web_search_tokens_per_result: int = 2048  # estimated extra input tokens per web search result
    default_web_search_max_results: int = 3  # default max_results when client omits it
    web_search_cost_per_result: float = 0.004  # $0.004 per search result (OpenRouter Exa: $4/1000)
    price_cache_ttl: int = 60
    max_upload_mb: int = 200  # max file upload size in MB (all in RAM, no disk)
    max_prepaid_topup: float = 10.0  # max single top-up amount in USD
    min_prepaid_topup: float = 0.10  # min single top-up amount in USD


class NetworkConfig(BaseModel):
    """Configuration for a single payment network (EVM or SVM)."""

    type: Literal["evm", "svm"]
    network: str  # CAIP-2 network id, e.g. "eip155:8453" or "solana:5eykt..."
    pay_to: str  # Recipient address (EVM hex or Solana base58)
    token_address: str  # Token contract/mint address
    token_name: str = "USD Coin"  # EIP-712 domain name (EVM only)
    token_version: str = "2"  # EIP-712 domain version (EVM only)
    rpc_url: str
    facilitator_key: str  # Private key (EVM hex or Solana base58 keypair)


class PaymentConfig(BaseModel):
    """x402 payment settings — supports multiple networks."""

    networks: dict[str, NetworkConfig]
    max_timeout: int = 3600

    @field_validator("networks")
    @classmethod
    def validate_has_networks(cls, v: dict) -> dict:
        if not v:
            raise ValueError("At least one payment network must be configured")
        return v


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    type: str = "managed"  # "managed" = full x402 flow, "passthrough" = transparent proxy
    enabled: bool = True
    base_url: str
    api_key: str = ""
    poll_interval: int = 2
    poll_timeout: int = 300
    docs_url: str = ""  # Link to provider API documentation
    description: str = ""  # Short description shown on the landing page
    example_request: dict[str, Any] = {}  # Example request: {model, body}
    example_request_2: dict[str, Any] = {}  # Second example: {model, body}
    # Tungsten-specific: cookie-based auth (no API keys)
    jwt_token: str = ""
    cf_clearance: str = ""
    fixed_price_usd: float = 0.0  # Fixed price per request (when provider has no pricing API)


class AppConfig(BaseModel):
    """Root application configuration."""

    gateway: GatewayConfig = GatewayConfig()
    payment: PaymentConfig
    providers: dict[str, ProviderConfig] = {}


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Environment variables in the format ${VAR} are interpolated before
    parsing. The .env file (if present) should be loaded beforehand.

    Args:
        path: Path to the config.yaml file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required env vars are missing or config is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    interpolated = _interpolate_dict(raw)
    return AppConfig(**interpolated)
