# Adding a New Provider

x402gate supports multiple AI service providers through a modular plugin system. This guide walks you through adding a new one.

## Overview

Each provider implements 3 methods:

| Method | Purpose |
|---|---|
| `get_price(model_path, inputs)` | Fetch dynamic pricing from provider's API |
| `submit(path, body)` | Forward the client's request as-is |
| `get_result(task_id)` | Poll for async task completion |

## Step-by-Step

### 1. Create the Provider Module

Create `x402gate/providers/your_provider.py`:

```python
from __future__ import annotations

from decimal import Decimal
from typing import Any

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import poll_result, proxy_request
from x402gate.providers.base import BaseProvider, ProviderError


class YourProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="your_provider", config=config)

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        # Call your provider's pricing API
        result = await proxy_request(
            base_url=self._config.base_url,
            path="pricing",  # your pricing endpoint
            body={"model": model_path, "params": inputs},
            api_key=self._config.api_key,
        )
        return Decimal(str(result["price"]))

    async def submit(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await proxy_request(
            base_url=self._config.base_url,
            path=path,
            body=body,
            api_key=self._config.api_key,
        )

    async def get_result(self, task_id: str) -> dict[str, Any]:
        return await poll_result(
            base_url=self._config.base_url,
            task_id=task_id,
            api_key=self._config.api_key,
            poll_interval=self._config.poll_interval,
            poll_timeout=self._config.poll_timeout,
        )
```

### 2. Register in the App

Add your provider to `x402gate/app.py` in the `lifespan` function:

```python
from x402gate.providers.your_provider import YourProvider

# In lifespan():
if name == "your_provider":
    provider = YourProvider(config=provider_config)
    providers[name] = provider
```

### 3. Add a Route

Add a catch-all route in `x402gate/app.py`:

```python
@app.post("/v1/your_provider/{path:path}")
async def your_provider_proxy(path: str, request: Request) -> Any:
    return await _handle_proxy_request("your_provider", path, request)
```

### 4. Add Configuration

Add the provider to `config.yaml`:

```yaml
providers:
  your_provider:
    enabled: true
    base_url: "https://api.yourprovider.com/v1"
    api_key: "${YOUR_PROVIDER_API_KEY}"
    poll_interval: 2
    poll_timeout: 300
```

### 5. Write Tests

Create `tests/test_your_provider.py` with mocked API responses.

## Tips

- **Synchronous providers**: If your provider returns results immediately (no polling), return the result from `submit()` and have `get_result()` raise `NotImplementedError`
- **Custom pricing**: If your provider doesn't have a pricing API, you can hardcode prices in `get_price()` or read them from config
- **Error mapping**: Map provider-specific error codes to `ProviderError` with appropriate HTTP status codes
