"""Shared HTTP client for upstream calls.

Opening a fresh ``httpx.AsyncClient`` per call costs a TCP and TLS handshake
(50-200 ms) every time. One pooled client keeps connections to each upstream
host open between requests. Callers pass their own ``timeout=`` per request;
use :func:`timeout` so a host that does not answer fails within
``CONNECT_TIMEOUT_S`` instead of the whole read timeout.
"""

from __future__ import annotations

import asyncio
from http.cookiejar import CookieJar, DefaultCookiePolicy

import httpx

# A host that does not accept the connection within this is down. Failing
# fast here is safe: a connect error means nothing was sent.
CONNECT_TIMEOUT_S = 10.0
# Below the idle timeout of common load balancers (60 s), so the pool drops a
# connection before the server does. A request on a connection the server
# already closed would fail after sending, which reads as an unknown outcome.
KEEPALIVE_EXPIRY_S = 30.0

_LIMITS = httpx.Limits(
    max_connections=200,
    max_keepalive_connections=50,
    keepalive_expiry=KEEPALIVE_EXPIRY_S,
)

# One client per event loop: a pooled connection belongs to the loop that
# opened it (tests run each case in a new loop).
_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def timeout(seconds: float) -> httpx.Timeout:
    """Return a request timeout of ``seconds`` with short connect and pool limits.

    A full pool fails with PoolTimeout, which like a connect error means
    nothing was sent, instead of waiting out the whole read timeout.
    """
    short = min(seconds, CONNECT_TIMEOUT_S)
    return httpx.Timeout(seconds, connect=short, pool=short)


def shared_client() -> httpx.AsyncClient:
    """Return the pooled client for the running event loop."""
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        # Calls share one client, so a cookie set by one provider response
        # (a load-balancer pin, say) must not ride along on every later call.
        _client = httpx.AsyncClient(
            timeout=timeout(60.0),
            limits=_LIMITS,
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
        )
        _client_loop = loop
    return _client


async def close_shared_client() -> None:
    """Close the pooled client (on shutdown)."""
    global _client, _client_loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None
