# Contributing to x402gate

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Getting Started

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

## Style Guide

### Language

All code, comments, docstrings, commit messages, and documentation must be written in **English**.

### Formatting & Linting

We use [Ruff](https://docs.astral.sh/ruff/) for both formatting and linting:

```bash
# Format code
ruff format .

# Check for lint errors
ruff check .

# Auto-fix lint errors
ruff check --fix .
```

### Code Conventions

- **Type hints**: Required on all function signatures
- **Async**: Use `async/await` for all I/O operations (HTTP calls, file reads)
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes
- **Docstrings**: Google style on all public functions and classes
- **Imports**: Sorted by `ruff` (stdlib → third-party → local)

### Example

```python
async def fetch_price(model_id: str, inputs: dict[str, Any]) -> Decimal:
    """Fetch dynamic pricing from the provider's API.

    Args:
        model_id: Model identifier (e.g. 'wavespeed-ai/flux-dev').
        inputs: Request parameters to price.

    Returns:
        Base price in USD as a Decimal.

    Raises:
        ProviderError: If the pricing API call fails.
    """
    ...
```

## Adding a New Provider

See [docs/add-provider.md](docs/add-provider.md) for a step-by-step guide on implementing a new AI service provider.

## Running Tests

```bash
# Unit + integration tests
pytest tests/ -v --ignore=tests/e2e

# E2E tests (requires .env.test with real credentials)
pytest tests/e2e/ -v -s
```

## Pull Request Checklist

Before submitting a PR, please ensure:

- [ ] All tests pass (`pytest tests/ -v --ignore=tests/e2e`)
- [ ] Code is formatted (`ruff format .`)
- [ ] No lint errors (`ruff check .`)
- [ ] New code has type hints and docstrings
- [ ] PR description explains the change and motivation
