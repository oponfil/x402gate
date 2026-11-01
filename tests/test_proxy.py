"""Unit tests for request forwarding and task polling in core/proxy.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core import http
from x402gate.core.proxy import (
    UPSTREAM_CREDENTIALS_DETAIL,
    ProxyError,
    TaskTimeoutError,
    poll_result,
    proxy_request,
)
from x402gate.providers.base import outcome_unknown

BASE_URL = "https://api.wavespeed.ai/api/v3"
CREATED = {"data": {"id": "t1", "status": "created"}}


async def _send(post: AsyncMock, *, idempotent: bool = False) -> dict:
    with (
        patch("httpx.AsyncClient.post", post),
        patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
    ):
        return await proxy_request(
            BASE_URL, "wavespeed-ai/flux-dev", {"prompt": "x"}, "key", idempotent=idempotent
        )


async def _poll(get: AsyncMock, *, poll_interval: int = 2, poll_timeout: int = 60):
    """Run poll_result against mocked responses; return (result or error, sleeps).

    The sleeps leave out the short pause before the first poll.
    """
    sleep = AsyncMock()
    with (
        patch("httpx.AsyncClient.get", get),
        patch("x402gate.core.proxy.asyncio.sleep", sleep),
    ):
        try:
            outcome = await poll_result(
                BASE_URL, "t1", "key", poll_interval=poll_interval, poll_timeout=poll_timeout
            )
        except (ProxyError, TaskTimeoutError) as exc:
            outcome = exc
    sleeps = [call.args[0] for call in sleep.await_args_list]
    assert sleeps[0] == pytest.approx(0.3)
    return outcome, sleeps[1:]


def _status(status: str, **extra) -> httpx.Response:
    return httpx.Response(200, json={"data": {"id": "t1", "status": status, **extra}})


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

    async def test_connect_error_is_retried(self):
        """A connection that never opened sent nothing, so it is safe to resend."""
        post = AsyncMock(
            side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=CREATED)]
        )
        result = await _send(post)
        assert post.call_count == 2
        assert result["data"]["id"] == "t1"

    async def test_idempotent_server_error_is_retried(self):
        """A price quote starts no job, so a 5xx is sent again."""
        post = AsyncMock(
            side_effect=[httpx.Response(503, text="busy"), httpx.Response(200, json=CREATED)]
        )
        result = await _send(post, idempotent=True)
        assert post.call_count == 2
        assert result["data"]["id"] == "t1"

    async def test_submit_server_error_is_not_resent(self):
        """A submit that got a 5xx may have created a task; a resend pays twice."""
        post = AsyncMock(return_value=httpx.Response(503, text="busy"))
        with pytest.raises(ProxyError) as caught:
            await _send(post)
        assert post.call_count == 1
        assert caught.value.status_code == 503
        # The provider reported the error itself, so the client is refunded.
        assert outcome_unknown(caught.value) is False

    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_submit_gateway_timeout_is_unknown(self, status):
        """A gateway timeout means the request was forwarded and may have landed."""
        post = AsyncMock(return_value=httpx.Response(status, text="gateway timeout"))
        with pytest.raises(ProxyError) as caught:
            await _send(post)
        assert post.call_count == 1
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize("status", [301, 307, 308])
    async def test_submit_redirect_is_a_known_failure(self, status):
        """httpx does not follow redirects, so a 3xx never ran the call."""
        post = AsyncMock(
            return_value=httpx.Response(status, headers={"location": "https://elsewhere/"})
        )
        with pytest.raises(ProxyError) as caught:
            await _send(post)
        assert post.call_count == 1
        assert caught.value.status_code == status
        assert outcome_unknown(caught.value) is False

    async def test_unreadable_success_body_is_unknown(self):
        """A 200 we cannot parse still means the provider accepted the call."""
        post = AsyncMock(return_value=httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(ProxyError) as caught:
            await _send(post)
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is True


class TestPollResult:
    """Polling outcomes and which of them refund the client."""

    async def test_rate_limit_and_server_errors_keep_polling(self):
        """429 and 5xx are transient; the job keeps running upstream."""
        get = AsyncMock(
            side_effect=[
                httpx.Response(429, text="slow down"),
                httpx.Response(503, text="busy"),
                _status("completed", outputs=["https://out.png"]),
            ]
        )
        result, sleeps = await _poll(get)
        assert result["outputs"] == ["https://out.png"]
        # Waits double while the failures last.
        assert sleeps == [4, 8]

    async def test_backoff_resets_and_is_capped(self):
        get = AsyncMock(
            side_effect=[httpx.Response(503, text="busy")] * 6
            + [_status("processing"), _status("completed")]
        )
        result, sleeps = await _poll(get, poll_timeout=600)
        assert result["status"] == "completed"
        assert sleeps == [4, 8, 16, 30, 30, 30, 2]

    async def test_persistent_server_errors_time_out(self):
        """The poll timeout still ends it, and a timeout keeps the payment."""
        get = AsyncMock(return_value=httpx.Response(500, text="down"))
        outcome, sleeps = await _poll(get, poll_timeout=20)
        assert isinstance(outcome, TaskTimeoutError)
        assert outcome_unknown(outcome) is True
        # The last wait is cut short at the deadline (0.3 s went before the first poll).
        assert sum(sleeps) == pytest.approx(20 - 0.3)
        assert get.call_count == 3

    async def test_not_found_right_after_creation_is_tolerated(self):
        get = AsyncMock(
            side_effect=[
                httpx.Response(404, text="not found"),
                httpx.Response(404, text="not found"),
                _status("completed"),
            ]
        )
        result, _ = await _poll(get)
        assert result["status"] == "completed"

    async def test_not_found_grace_is_time_based(self):
        """At a 1s poll interval, a few quick 404s are still within the grace."""
        get = AsyncMock(
            side_effect=[httpx.Response(404, text="not found")] * 6 + [_status("completed")]
        )
        result, _ = await _poll(get, poll_interval=1)
        assert result["status"] == "completed"

    async def test_persistent_not_found_is_unknown(self):
        """The task was accepted, so losing it is not a known failure."""
        get = AsyncMock(return_value=httpx.Response(404, text="not found"))
        outcome, sleeps = await _poll(get)
        assert isinstance(outcome, ProxyError)
        assert outcome.status_code == 502
        # Given up at the first poll 20s after the first 404, not before.
        assert sum(sleeps) >= 20
        assert sum(sleeps[:-1]) < 20
        assert get.call_count == len(sleeps) + 1
        assert outcome_unknown(outcome) is True

    async def test_failed_task_is_a_known_failure(self):
        get = AsyncMock(return_value=_status("failed", error="boom"))
        outcome, _ = await _poll(get)
        assert isinstance(outcome, ProxyError)
        assert outcome.detail == "boom"
        assert outcome_unknown(outcome) is False

    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_credentials_become_502(self, status):
        """An upstream 402 must not reach x402 clients as a payment request.

        The task was accepted with this key at submission and may still run,
        so the payment is kept.
        """
        get = AsyncMock(return_value=httpx.Response(status, text="invalid api key"))
        outcome, _ = await _poll(get)
        assert isinstance(outcome, ProxyError)
        assert outcome.status_code == 502
        assert outcome.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(outcome) is True

    async def test_redirect_is_unknown(self):
        """A redirected poll cannot read the task, but the task was accepted."""
        get = AsyncMock(
            return_value=httpx.Response(307, headers={"location": "https://elsewhere/"})
        )
        outcome, _ = await _poll(get)
        assert isinstance(outcome, ProxyError)
        assert outcome.status_code == 502
        assert get.call_count == 1
        assert outcome_unknown(outcome) is True

    async def test_unreadable_body_keeps_polling(self):
        get = AsyncMock(
            side_effect=[
                httpx.Response(200, text="<html>oops</html>"),
                httpx.Response(200, json=["not", "a", "dict"]),
                _status("completed"),
            ]
        )
        result, _ = await _poll(get)
        assert result["status"] == "completed"


class TestPollSpeed:
    """Short jobs are seen finished soon after they finish."""

    async def test_first_poll_is_soon_and_gaps_grow_to_poll_interval(self):
        get = AsyncMock(side_effect=[_status("processing")] * 6 + [_status("completed")])
        result, sleeps = await _poll(get, poll_interval=2)
        assert result["status"] == "completed"
        assert sleeps == pytest.approx([0.4, 0.6, 0.9, 1.35, 2, 2])

    async def test_early_gaps_never_exceed_a_short_poll_interval(self):
        get = AsyncMock(side_effect=[_status("processing")] * 4 + [_status("completed")])
        _, sleeps = await _poll(get, poll_interval=1)
        assert sleeps == pytest.approx([0.4, 0.6, 0.9, 1])

    async def test_rate_limit_backs_off_from_the_full_interval(self):
        """Early fast polls do not shorten the 429 backoff."""
        get = AsyncMock(
            side_effect=[_status("processing"), httpx.Response(429, text="slow down")]
            + [_status("completed")]
        )
        _, sleeps = await _poll(get, poll_interval=2)
        assert sleeps == pytest.approx([0.4, 4])

    async def test_short_timeout_still_ends_on_time(self):
        get = AsyncMock(return_value=_status("processing"))
        outcome, sleeps = await _poll(get, poll_timeout=1)
        assert isinstance(outcome, TaskTimeoutError)
        assert 0.3 + sum(sleeps) == pytest.approx(1)


class TestPooledClient:
    """Calls reuse the shared pooled client instead of opening one each."""

    async def test_no_client_is_opened_per_call(self):
        http.shared_client()  # the pool exists before the calls
        post = AsyncMock(return_value=httpx.Response(200, json=CREATED))
        get = AsyncMock(return_value=_status("completed"))
        with (
            patch.object(httpx.AsyncClient, "__init__", side_effect=AssertionError("new client")),
            patch("httpx.AsyncClient.post", post),
            patch("httpx.AsyncClient.get", get),
            patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
        ):
            await proxy_request(BASE_URL, "wavespeed-ai/flux-dev", {"prompt": "x"}, "key")
            await proxy_request(BASE_URL, "wavespeed-ai/flux-dev", {"prompt": "y"}, "key")
            await poll_result(BASE_URL, "t1", "key")
        assert not http.shared_client().is_closed

    async def test_request_timeout_keeps_the_short_connect_limit(self):
        post = AsyncMock(return_value=httpx.Response(200, json=CREATED))
        with patch("httpx.AsyncClient.post", post):
            await proxy_request(
                BASE_URL, "wavespeed-ai/flux-dev", {"prompt": "x"}, "key", request_timeout=120
            )
        timeout = post.await_args.kwargs["timeout"]
        assert timeout.read == 120
        assert timeout.connect == http.CONNECT_TIMEOUT_S

    async def test_pool_timeout_is_resent(self):
        """Waiting for a free pooled connection sends nothing."""
        post = AsyncMock(
            side_effect=[httpx.PoolTimeout("pool full"), httpx.Response(200, json=CREATED)]
        )
        result = await _send(post)
        assert post.call_count == 2
        assert result["data"]["id"] == "t1"
