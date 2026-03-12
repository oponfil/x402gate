# Adding a New Provider

x402gate supports multiple AI service providers through a modular plugin system. Routes are registered automatically from `config.yaml` at startup — no manual route definitions needed.

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

### 2. Register in the Provider Registry

Add your provider class to `PROVIDER_REGISTRY` in `x402gate/app.py`:

```python
from x402gate.providers.your_provider import YourProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "wavespeed": WaveSpeedProvider,
    "openrouter": OpenRouterProvider,
    "tungsten": TungstenProvider,
    "cloudconvert": CloudConvertProvider,
    "socialdownload": SocialDownloadProvider,
    "your_provider": YourProvider,  # ← add this line
}
```

The route `POST /v1/your_provider/{path}` is registered automatically from config at startup.

### 3. Add Configuration

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

### 4. Write Tests

Create `tests/test_your_provider.py` with mocked API responses.

## Tips

- **Synchronous providers**: If your provider returns results immediately (no polling), return the result from `submit()` and have `get_result()` raise `NotImplementedError`
- **Custom pricing**: If your provider doesn't have a pricing API, you can hardcode prices in `get_price()` or read them from config
- **Error mapping**: Map provider-specific error codes to `ProviderError` with appropriate HTTP status codes

## Multipart/File Upload Providers

Some providers (e.g. CloudConvert) accept files instead of JSON. For these:

1. **In `submit()`**, read file bytes from `body["_file_bytes"]` and filename from `body["_file_name"]` — these are injected by `app.py` when the client sends `multipart/form-data`

2. **In `config.yaml`**, set `content_type: "multipart"` in `example_request` so the landing page renders a curl with `-F` flags:

```yaml
  example_request:
    model: "convert"
    content_type: "multipart"
    body:
      output_format: "pdf"
```

3. **Client sends** `multipart/form-data` instead of JSON:

```bash
curl -X POST https://x402gate.io/v1/cloudconvert/convert \
  -F "file=@document.docx" \
  -F "output_format=pdf"
```

See `x402gate/providers/cloudconvert.py` for a complete example.
