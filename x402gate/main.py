"""Entry point for x402gate server.

Run with: python -m x402gate.main
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn


def main() -> None:
    """Start the x402gate server."""
    # Load .env file if present
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Read host/port from env or defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "4021"))

    uvicorn.run(
        "x402gate.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
