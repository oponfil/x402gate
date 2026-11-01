"""Unit tests for the WaveSpeed provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.wavespeed import WaveSpeedProvider


def _make_provider() -> WaveSpeedProvider:
    return WaveSpeedProvider(
        ProviderConfig(
            base_url="https://api.wavespeed.ai/api/v3",
            api_key="test_key",
            poll_interval=2,
            poll_timeout=60,
        )
    )


def _patched(method: str, mock: AsyncMock):
    """Patch one httpx.AsyncClient method and skip proxy sleeps."""
    return (
        patch(f"httpx.AsyncClient.{method}", mock),
        patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
    )


class TestGetPrice:
    async def test_server_error_is_retried(self):
        """A price quote starts no job, so a 5xx is retried."""
        post = AsyncMock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"data": {"unit_price": 0.003}}),
            ]
        )
        p1, p2 = _patched("post", post)
        with p1, p2:
            price = await _make_provider().get_price("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert price == Decimal("0.003")
        assert post.call_count == 2

    async def test_rejected_key_is_502_without_upstream_text(self):
        post = AsyncMock(return_value=httpx.Response(401, text="invalid api key abc123"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL


class TestSubmit:
    async def test_server_error_is_sent_once_and_refunded(self):
        """A resend after a 5xx could start a second paid job."""
        post = AsyncMock(return_value=httpx.Response(500, text="upstream down"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert outcome_unknown(caught.value) is False

    async def test_gateway_timeout_keeps_the_payment(self):
        post = AsyncMock(return_value=httpx.Response(504, text="gateway timeout"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_is_502(self, status):
        post = AsyncMock(return_value=httpx.Response(status, text="no credit"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @pytest.mark.parametrize(
        "body",
        [{"data": None}, {"data": {"status": "created"}}, {"data": {"id": "t1"}}, {"x": 1}],
    )
    async def test_success_without_task_id_is_an_error(self, body):
        """No task id means no output can be delivered, but a task may exist."""
        post = AsyncMock(return_value=httpx.Response(200, json=body))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert "returned no task id" in caught.value.detail
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize(
        "body",
        [
            {"data": {"id": "t1", "status": "created"}},
            {"data": {"output": {"images": ["base64img"]}}},
            {"data": {"outputs": ["https://out.png"]}},
            {"outputs": ["ok"]},
        ],
    )
    async def test_task_or_sync_result_is_returned(self, body):
        post = AsyncMock(return_value=httpx.Response(200, json=body))
        p1, p2 = _patched("post", post)
        with p1, p2:
            result = await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert result == body

    async def test_redirect_is_refunded(self):
        """httpx does not follow redirects, so WaveSpeed never ran the call."""
        post = AsyncMock(
            return_value=httpx.Response(307, headers={"location": "https://elsewhere/"})
        )
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False


class TestGetResult:
    async def test_lost_task_keeps_the_payment(self):
        get = AsyncMock(return_value=httpx.Response(404, text="not found"))
        p1, p2 = _patched("get", get)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_result("t1")
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is True

    async def test_rejected_key_while_polling_is_502(self):
        """The task was accepted and may still run, so the payment is kept."""
        get = AsyncMock(return_value=httpx.Response(402, text="insufficient balance"))
        p1, p2 = _patched("get", get)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_result("t1")
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is True
