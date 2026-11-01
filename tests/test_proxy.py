"""Unit tests for request forwarding in core/proxy.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core.proxy import ProxyError, proxy_request

BASE_URL = "https://api.wavespeed.ai/api/v3"


async def _send(post: AsyncMock) -> dict:
    with (
        patch("httpx.AsyncClient.post", post),
        patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
    ):
        return await proxy_request(BASE_URL, "wavespeed-ai/flux-dev", {"prompt": "x"}, "key")


class TestProxyRequestRetries:
    """Only requests that never reached the provider are sent again."""

    async def test_read_timeout_is_not_resent(self):
        """A submit that timed out after sending may have created a task already."""
        post = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProxyError) as caught:
            await _send(post)
        assert post.call_count == 1
        assert caught.value.status_code == 504

    async def test_connect_timeout_is_retried(self):
        post = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("no route"),
                httpx.Response(200, json={"data": {"id": "t1", "status": "created"}}),
            ]
        )
        result = await _send(post)
        assert post.call_count == 2
        assert result["data"]["id"] == "t1"

    async def test_connect_timeout_gives_up_after_retries(self):
        post = AsyncMock(side_effect=httpx.ConnectTimeout("no route"))
        with pytest.raises(ProxyError):
            await _send(post)
        assert post.call_count == 3

    async def test_server_error_is_still_retried(self):
        post = AsyncMock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"data": {"id": "t1", "status": "created"}}),
            ]
        )
        result = await _send(post)
        assert post.call_count == 2
        assert result["data"]["id"] == "t1"
