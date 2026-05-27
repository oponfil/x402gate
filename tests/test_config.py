"""Unit tests for configuration loading."""

import os
from pathlib import Path

import pytest

from x402gate.core.config import load_config


@pytest.fixture
def config_yaml(tmp_path: Path) -> Path:
    """Create a temporary config.yaml for testing."""
    config_content = """
gateway:
  host: "127.0.0.1"
  port: 8080
  commission: 0.10
  price_cache_ttl: 30

payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "0x1234567890abcdef1234567890abcdef12345678"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"
  max_timeout: 3600

providers:
  wavespeed:
    enabled: true
    base_url: "https://api.wavespeed.ai/api/v3"
    api_key: "test-key"
    poll_interval: 5
    poll_timeout: 120
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_with_env(tmp_path: Path) -> Path:
    """Create a config.yaml that uses env interpolation."""
    config_content = """
gateway:
  port: 4021

payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "${TEST_PAY_TO}"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"
  max_timeout: 3600

providers:
  wavespeed:
    enabled: true
    base_url: "https://api.wavespeed.ai/api/v3"
    api_key: "${TEST_API_KEY}"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return config_file


class TestLoadConfig:
    """Tests for config loading and validation."""

    def test_load_valid_config(self, config_yaml: Path):
        """A valid config file is parsed correctly."""
        config = load_config(config_yaml)
        assert config.gateway.host == "127.0.0.1"
        assert config.gateway.port == 8080
        assert config.gateway.commission == 0.10
        assert config.gateway.price_cache_ttl == 30

    def test_payment_config(self, config_yaml: Path):
        """Payment settings are loaded correctly."""
        config = load_config(config_yaml)
        base = config.payment.networks["base"]
        assert base.pay_to == "0x1234567890abcdef1234567890abcdef12345678"
        assert base.network == "eip155:8453"
        assert base.type == "evm"
        assert base.token_address == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
        assert config.payment.max_timeout == 3600

    def test_provider_config(self, config_yaml: Path):
        """Provider settings are loaded correctly."""
        config = load_config(config_yaml)
        ws = config.providers["wavespeed"]
        assert ws.enabled is True
        assert ws.base_url == "https://api.wavespeed.ai/api/v3"
        assert ws.api_key == "test-key"
        assert ws.poll_interval == 5
        assert ws.poll_timeout == 120

    def test_example_request_4_preserved(self, tmp_path: Path):
        """Fourth example (e.g. STT) must survive load_config for discovery endpoints."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "0x1234567890abcdef1234567890abcdef12345678"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"

providers:
  openrouter:
    enabled: true
    base_url: "https://openrouter.ai/api/v1"
    api_key: "test-key"
    example_request_4:
      model: "audio/transcriptions"
      body:
        model: "openai/whisper-1"
        input_audio:
          data: "UklGRg=="
          format: "wav"
""")
        config = load_config(config_file)
        ex4 = config.providers["openrouter"].example_request_4
        assert ex4["model"] == "audio/transcriptions"
        assert ex4["body"]["model"] == "openai/whisper-1"

    def test_missing_file_raises(self):
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_invalid_config_raises(self, tmp_path: Path):
        """A config with no networks raises ValueError."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("""
payment:
  networks: {}
""")
        with pytest.raises((ValueError, Exception)):  # noqa: B017
            load_config(config_file)


class TestEnvInterpolation:
    """Tests for ${VAR} environment variable interpolation."""

    def test_env_vars_interpolated(self, config_with_env: Path):
        """${VAR} placeholders are replaced with env values."""
        os.environ["TEST_PAY_TO"] = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        os.environ["TEST_API_KEY"] = "my-secret-key"
        try:
            config = load_config(config_with_env)
            base = config.payment.networks["base"]
            assert base.pay_to == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
            assert config.providers["wavespeed"].api_key == "my-secret-key"
        finally:
            del os.environ["TEST_PAY_TO"]
            del os.environ["TEST_API_KEY"]

    def test_missing_env_raises(self, config_with_env: Path):
        """Missing required env vars (payment) raise ValueError."""
        # Make sure the variables are NOT set
        os.environ.pop("TEST_PAY_TO", None)
        os.environ.pop("TEST_API_KEY", None)
        # TEST_PAY_TO is in payment section → strict → ValueError
        with pytest.raises(ValueError, match="required but not set"):
            load_config(config_with_env)

    def test_missing_provider_key_raises(self, config_with_env: Path):
        """Missing provider API key raises ValueError."""
        # Set payment vars but leave provider key unset
        os.environ["TEST_PAY_TO"] = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        os.environ.pop("TEST_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="required but not set"):
                load_config(config_with_env)
        finally:
            del os.environ["TEST_PAY_TO"]

    def test_missing_tungsten_cookie_raises(self, tmp_path: Path):
        """Missing Tungsten auth cookies should still fail fast."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "${TEST_PAY_TO}"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"

providers:
  tungsten:
    enabled: true
    base_url: "https://api.tungsten.run/v1"
    jwt_token: "${TEST_TUNGSTEN_JWT}"
    cf_clearance: "${TEST_TUNGSTEN_CF}"
    fixed_price_usd: 0.01
""")

        os.environ["TEST_PAY_TO"] = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        os.environ.pop("TEST_TUNGSTEN_JWT", None)
        os.environ.pop("TEST_TUNGSTEN_CF", None)
        try:
            with pytest.raises(ValueError, match="required but not set"):
                load_config(config_file)
        finally:
            del os.environ["TEST_PAY_TO"]


class TestDisabledProvider:
    """Tests for provider enable/disable."""

    def test_disabled_provider_in_config(self, tmp_path: Path):
        """Disabled providers are parsed but flagged."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "0x1234567890abcdef1234567890abcdef12345678"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"

providers:
  wavespeed:
    enabled: false
    base_url: "https://api.wavespeed.ai/api/v3"
""")
        config = load_config(config_file)
        assert config.providers["wavespeed"].enabled is False

    def test_disabled_provider_does_not_require_missing_env(self, tmp_path: Path):
        """Disabled providers should not fail on unresolved env placeholders."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
payment:
  networks:
    base:
      type: "evm"
      network: "eip155:8453"
      pay_to: "0x1234567890abcdef1234567890abcdef12345678"
      token_address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
      rpc_url: "https://mainnet.base.org"
      facilitator_key: "0x0000000000000000000000000000000000000000000000000000000000000001"

providers:
  openrouter:
    enabled: false
    base_url: "https://openrouter.ai/api/v1"
    api_key: "${MISSING_OPENROUTER_API_KEY}"
""")

        os.environ.pop("MISSING_OPENROUTER_API_KEY", None)

        config = load_config(config_file)

        assert config.providers["openrouter"].enabled is False
        assert config.providers["openrouter"].api_key == "${MISSING_OPENROUTER_API_KEY}"
