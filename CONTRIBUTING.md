# Coding Rules for x402gate

## Imports

- **All imports must be at the top of the file.** No inline/local imports inside functions, methods, or conditionals.
- Order: stdlib → third-party → local (`x402gate.*`, `tests.*`).
- Separate each group with a blank line.

## Code Style

- **Python 3.11+** — use modern syntax (type hints, `match`, `|` union).
- Keep functions focused and small.
- Use `async/await` for all I/O-bound operations.
- Use `Decimal` for monetary values, never `float`.
- Thread safety: use `asyncio.Lock` for shared mutable state.

## DRY — Don't Repeat Yourself

- **Never duplicate logic.** If the same code appears in more than one place, extract it into a shared helper function.
- Common E2E utilities go in `tests/e2e/helpers.py`.
- Common core logic goes in `x402gate/core/`.

## Testing

- Every new feature MUST have unit tests (`tests/test_*.py`) and integration tests (`tests/test_integration.py`).
- E2E tests go in `tests/e2e/`.
- Shared E2E logic goes in `tests/e2e/helpers.py` — don't duplicate.
- Tests should be self-contained and not depend on external state.

## Configuration

- All configurable values go in `GatewayConfig` (or related Pydantic models) in `x402gate/core/config.py`.
- Corresponding entries must be added to `config.yaml` with comments.
- Use `${ENV_VAR}` syntax for secrets.

## Documentation

- Update `README.md` when adding features.
- Detailed docs go in `docs/`.
- Update `x402gate/templates/index.html` (landing page) when features are user-facing.

## Logging

- Use `logger.info()` for important events.
- Keep production logs compact — no debug dumps.
- Monetary values in logs: `$X.XXXXXXX` format.

## Security

- Ed25519 signatures for prepaid authentication.
- Timestamp validation (60s window) to prevent replay attacks.
- Never log private keys or full signatures.
