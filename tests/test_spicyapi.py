"""Unit tests for the SpicyAPI provider."""

from __future__ import annotations

import json
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL, TaskTimeoutError
from x402gate.providers import spicyapi
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.spicyapi import SpicyApiProvider

_BODY = {
    "model": "alibaba/z-image-spicy/text-to-image",
    "input": {"prompt": "A lighthouse at dusk"},
    "_caller": "abcd1234ef01",
}


def _config(**overrides) -> ProviderConfig:
    defaults = {
        "base_url": "https://api.spicyapi.ai",
        "api_key": "sk-spicy-test",
        "poll_interval": 0,
        "poll_timeout": 5,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _provider(**overrides) -> SpicyApiProvider:
    provider = SpicyApiProvider(config=_config(**overrides))
    client = AsyncMock()
    client.is_closed = False
    # The provider calls through the shared pool; tests hand it this mock.
    provider._client = client
    provider._get_client = lambda: client
    return provider


def _response(status: int, payload: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def _quote(cost: str = "0.010", quote_id: str = "quote_1", unit: str = "per_image") -> MagicMock:
    return _response(
        200,
        {
            "code": 200,
            "msg": "success",
            "data": {
                "quoteId": quote_id,
                "estimatedCost": cost,
                "maxCharge": cost,
                "currency": "USD",
                "unit": unit,
                "expiresAt": "2099-01-01T00:00:00Z",
            },
        },
    )


def _task(state: str, task_id: str = "job_1", **extra) -> MagicMock:
    data = {"taskId": task_id, "state": state, "estimatedCost": "0.010"}
    data.update(extra)
    return _response(
        202 if state in {"queued", "running"} else 200,
        {
            "code": 200,
            "msg": "success",
            "data": data,
        },
    )


class TestGetPrice:
    """Tests for SpicyApiProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_returns_quoted_cost(self):
        """The quote's estimatedCost is the price, and the quote is kept."""
        provider = _provider()
        provider._client.request.return_value = _quote()

        price = await provider.get_price("jobs", _BODY)

        assert price == Decimal("0.010")
        call = provider._client.request.await_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/api/v1/jobs/quote")
        assert call.kwargs["json"] == {
            "model": _BODY["model"],
            "input": _BODY["input"],
        }
        assert call.kwargs["headers"]["Authorization"] == "Bearer sk-spicy-test"
        assert len(provider._quotes) == 1

    @pytest.mark.asyncio
    async def test_rejects_unknown_path(self):
        """Jobs and chat are the only SpicyAPI routes."""
        provider = _provider()
        with pytest.raises(ProviderError, match="chat/completions"):
            await provider.get_price("images", _BODY)

    @pytest.mark.asyncio
    async def test_rejects_missing_input(self):
        """A body without an input object is a client error."""
        provider = _provider()
        with pytest.raises(ProviderError, match="input") as exc:
            await provider.get_price("jobs", {"model": "some/model"})
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_client_quote_fields(self):
        """The gateway owns quoting, so a client quoteId is not forwarded."""
        provider = _provider()
        with pytest.raises(ProviderError, match="quoteId") as exc:
            await provider.get_price("jobs", {**_BODY, "quoteId": "q", "expectedCost": "0.01"})
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_schema_error_is_returned_to_client(self):
        """A 400 from the quote is the model's schema error, not a gateway failure."""
        provider = _provider()
        provider._client.request.return_value = _response(
            400, {"code": 400, "msg": "prompt is required"}
        )
        with pytest.raises(ProviderError, match="prompt is required") as exc:
            await provider.get_price("jobs", _BODY)
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_low_balance_is_unavailable(self):
        """A quote the gateway account cannot hold must not become a client 402."""
        provider = _provider()
        provider._client.request.return_value = _response(
            402, {"code": 40201, "msg": "insufficient balance"}
        )
        with pytest.raises(ProviderError, match="cannot accept this charge") as exc:
            await provider.get_price("jobs", _BODY)
        assert exc.value.status_code == 503


class TestPriceCache:
    """Unpaid quotes must not grow the provider's memory without bound."""

    def test_prices_are_capped_oldest_first(self, monkeypatch):
        """Past the cap, the oldest prices are dropped to make room."""
        monkeypatch.setattr(spicyapi, "_CACHE_MAX_ENTRIES", 3)
        provider = _provider()

        for index in range(5):
            provider._remember_price(f"key{index}", Decimal(index))

        assert list(provider._priced) == ["key2", "key3", "key4"]

    def test_stale_prices_are_pruned(self):
        """A price older than the TTL is dropped on the next insert and not served."""
        provider = _provider()
        old = time.monotonic() - spicyapi._PRICED_TTL_SECONDS - 1
        provider._priced["old"] = (Decimal("0.01"), old)

        assert provider._price_for("old") is None
        provider._remember_price("new", Decimal("0.02"))

        assert list(provider._priced) == ["new"]

    @pytest.mark.asyncio
    async def test_quotes_are_capped(self, monkeypatch):
        """Unexpired quotes are capped too."""
        monkeypatch.setattr(spicyapi, "_CACHE_MAX_ENTRIES", 2)
        provider = _provider()
        provider._client.request.return_value = _quote()

        for index in range(3):
            await provider.get_price("jobs", {**_BODY, "input": {"prompt": f"p{index}"}})

        assert len(provider._quotes) == 2
        assert len(provider._priced) == 2

    def test_cache_key_is_a_hash(self):
        """A large chat body is not kept whole as a cache key."""
        key = SpicyApiProvider._cache_key({"model": "m", "input": {"prompt": "x" * 100_000}})
        assert len(key) == 64

    @pytest.mark.asyncio
    async def test_submit_without_a_price_on_record_is_refused(self):
        """An evicted price never lets the job run at an unchecked price."""
        provider = _provider()
        provider._client.request.return_value = _quote()
        await provider.get_price("jobs", _BODY)
        provider._priced.clear()
        provider._client.request.reset_mock()

        with pytest.raises(ProviderError, match="no longer on record") as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 409
        assert not outcome_unknown(exc.value)
        provider._client.request.assert_not_awaited()


class TestSubmit:
    """Tests for SpicyApiProvider.submit()."""

    @pytest.mark.asyncio
    async def test_image_job_reuses_quote_and_waits(self):
        """Image jobs attach the stored quote and hold the create call."""
        provider = _provider()
        provider._client.request.side_effect = [_quote(cost="0.010"), _task("queued")]

        await provider.get_price("jobs", _BODY)
        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_1"
        assert result["data"]["status"] == "processing"
        create = provider._client.request.await_args_list[1]
        assert create.args[1].endswith("/api/v1/jobs/createTask")
        assert create.kwargs["json"]["quoteId"] == "quote_1"
        assert create.kwargs["json"]["expectedCost"] == "0.010"
        assert create.kwargs["json"]["input"] == _BODY["input"]
        assert "_caller" not in create.kwargs["json"]
        assert create.kwargs["params"] == {"wait": 45}
        assert create.kwargs["headers"]["Idempotency-Key"]

    @pytest.mark.asyncio
    async def test_video_job_does_not_wait(self):
        """Video jobs return as soon as the task is accepted."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(unit="per_second"),
            _task("queued", task_id="job_video"),
        ]

        await provider.get_price("jobs", _BODY)
        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_video"
        assert provider._client.request.await_args_list[1].kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_finished_image_is_synchronous(self):
        """A task that finishes inside wait is returned without polling."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(),
            _task(
                "succeeded",
                cost="0.009",
                output={"assets": [{"url": "https://cdn.example/out.png"}]},
            ),
        ]

        await provider.get_price("jobs", _BODY)
        result = await provider.submit("jobs", _BODY)

        assert "status" not in result["data"]
        assert result["data"]["state"] == "succeeded"
        assert result["data"]["output"]["assets"][0]["url"].endswith("out.png")
        assert await provider.calculate_actual_cost(_BODY, result) == Decimal("0.009")

    @pytest.mark.asyncio
    async def test_failed_task_is_not_a_success(self):
        """A terminal failure from createTask does not look like a finished job."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(),
            _task("failed", errorCode="content_rejected", errorMessage="refused"),
        ]
        await provider.get_price("jobs", _BODY)
        with pytest.raises(ProviderError, match="refused") as exc:
            await provider.submit("jobs", _BODY)
        assert exc.value.status_code == 502

    @pytest.mark.asyncio
    async def test_rejects_a_higher_requote(self):
        """A paid retry must not run when the new quote is above the offered price."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(cost="0.01"),
            _quote(cost="0.05", quote_id="q2"),
        ]

        await provider.get_price("jobs", _BODY)
        for quote in provider._quotes.values():
            quote.expires_at = 0

        with pytest.raises(ProviderError, match="increased") as exc:
            await provider.submit("jobs", _BODY)
        assert exc.value.status_code == 409
        assert provider._client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_create_when_quote_is_stale(self):
        """40901 quotes again and submits once more at the same price."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(quote_id="q1"),
            _response(409, {"code": 40901, "msg": "quote no longer matches"}),
            _quote(quote_id="q2"),
            _task("queued", task_id="job_retried"),
        ]

        await provider.get_price("jobs", _BODY)
        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_retried"
        create_calls = [
            call
            for call in provider._client.request.await_args_list
            if call.args[1].endswith("/api/v1/jobs/createTask")
        ]
        assert [call.kwargs["json"]["quoteId"] for call in create_calls] == ["q1", "q2"]
        assert (
            create_calls[0].kwargs["headers"]["Idempotency-Key"]
            != (create_calls[1].kwargs["headers"]["Idempotency-Key"])
        )


class TestGetResult:
    """Tests for SpicyApiProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_polls_until_succeeded(self):
        """Queued then succeeded returns the task record and its cost."""
        provider = _provider()
        provider._client.request.side_effect = [
            _task("running"),
            _task(
                "succeeded",
                cost="0.008",
                output={"assets": [{"url": "https://cdn.example/clip.mp4"}]},
            ),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"
        assert result["output"]["assets"][0]["url"].endswith("clip.mp4")
        poll = provider._client.request.await_args_list[0]
        assert poll.args[0] == "GET"
        assert poll.args[1].endswith("/api/v1/jobs/recordInfo")
        assert poll.kwargs["params"] == {"taskId": "job_1"}
        assert await provider.calculate_actual_cost({}, {"data": {"id": "job_1"}}) == Decimal(
            "0.008"
        )

    @pytest.mark.asyncio
    async def test_first_poll_is_sent_without_a_delay(self):
        """A job that finished while queued is read at once, not after a poll interval."""
        provider = _provider(poll_interval=2)
        provider._sleep = AsyncMock()
        provider._client.request.return_value = _task("succeeded", cost="0.008")

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"
        provider._sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_poll_raises(self):
        """A failed task becomes a provider error and has no settled cost."""
        provider = _provider()
        provider._client.request.return_value = _task(
            "expired", errorCode="timeout", errorMessage="deadline passed"
        )
        with pytest.raises(ProviderError, match="deadline passed") as exc:
            await provider.get_result("job_1")
        assert not outcome_unknown(exc.value)
        assert await provider.calculate_actual_cost({}, {"data": {"id": "job_1"}}) is None

    @pytest.mark.asyncio
    async def test_timeout(self):
        """A zero poll budget times out before calling the provider."""
        provider = _provider(poll_timeout=0)
        with pytest.raises(TaskTimeoutError):
            await provider.get_result("job_1")
        provider._client.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rate_limit_and_5xx_keep_polling_with_backoff(self):
        """A 429 or 5xx while polling is not a job failure; polling backs off."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _response(429, {"code": 429, "msg": "rate limited"}),
            _response(503, {"code": 503, "msg": "busy"}),
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"
        delays = [call.args[0] for call in provider._sleep.await_args_list]
        assert delays == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_rate_limit_until_deadline_is_an_unknown_outcome(self, monkeypatch):
        """Being rate limited until the poll timeout keeps the payment."""
        clock = {"now": 1000.0}

        async def fake_sleep(seconds: float) -> None:
            clock["now"] += seconds

        monkeypatch.setattr(
            spicyapi, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time)
        )
        provider = _provider(poll_timeout=60)
        provider._sleep = fake_sleep
        provider._client.request.return_value = _response(429, {"code": 429, "msg": "slow down"})

        with pytest.raises(TaskTimeoutError) as exc:
            await provider.get_result("job_1")

        assert outcome_unknown(exc.value)
        assert clock["now"] == 1060.0

    @pytest.mark.asyncio
    async def test_envelope_rate_limit_and_non_json_keep_polling(self):
        """A 200 carrying code 429/5xx or a non-JSON page is a transient hiccup."""
        provider = _provider()
        provider._sleep = AsyncMock()
        not_json = _response(200, {})
        not_json.json.side_effect = ValueError("not json")
        provider._client.request.side_effect = [
            _response(200, {"code": 429, "msg": "rate limited"}),
            _response(200, {"code": "500", "msg": "internal"}),
            not_json,
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"
        delays = [call.args[0] for call in provider._sleep.await_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_envelope_rate_limit_until_deadline_is_an_unknown_outcome(self, monkeypatch):
        """A 200/429 envelope until the poll timeout keeps the payment."""
        clock = {"now": 1000.0}

        async def fake_sleep(seconds: float) -> None:
            clock["now"] += seconds

        monkeypatch.setattr(
            spicyapi, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time)
        )
        provider = _provider(poll_timeout=60)
        provider._sleep = fake_sleep
        provider._client.request.return_value = _response(200, {"code": 429, "msg": "slow down"})

        with pytest.raises(TaskTimeoutError) as exc:
            await provider.get_result("job_1")

        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_other_envelope_error_still_fails(self):
        """A 200 with a non-transient envelope code stays a known failure."""
        provider = _provider()
        provider._client.request.return_value = _response(
            200, {"code": 400, "msg": "task id invalid"}
        )

        with pytest.raises(ProviderError, match="task id invalid") as exc:
            await provider.get_result("job_1")

        assert not outcome_unknown(exc.value)


def _fake_clock(monkeypatch) -> dict[str, float]:
    """Make spicyapi's monotonic clock advance only when the provider sleeps."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        spicyapi, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=time.time)
    )
    return clock


def _non_json(status: int = 200) -> MagicMock:
    response = _response(status, {})
    response.json.side_effect = ValueError("not json")
    response.text = "<html>bad gateway</html>"
    return response


async def _priced_provider() -> SpicyApiProvider:
    """A provider whose next request is the createTask after a stored quote."""
    provider = _provider()
    provider._sleep = AsyncMock()
    provider._client.request.return_value = _quote(unit="per_second")
    await provider.get_price("jobs", _BODY)
    provider._client.request.reset_mock(return_value=True)
    return provider


def _create_calls(provider: SpicyApiProvider) -> list:
    return [
        call
        for call in provider._client.request.await_args_list
        if call.args[1].endswith("/api/v1/jobs/createTask")
    ]


class TestEnvelopeTransient:
    """Envelope codes are HTTP statuses or five digits that start with one."""

    @pytest.mark.parametrize("code", [429, "503", 500, 599, 42901, 50001, "50301"])
    def test_rate_limit_and_server_codes_are_transient(self, code):
        assert spicyapi._envelope_transient(_response(200, {"code": code}))

    @pytest.mark.parametrize("code", [200, None, 400, 404, 40001, 40401, 40201, 40901, 60001, "x"])
    def test_client_and_success_codes_are_not_transient(self, code):
        assert not spicyapi._envelope_transient(_response(200, {"code": code}))

    def test_unreadable_body_is_transient(self):
        assert spicyapi._envelope_transient(_non_json())

    @pytest.mark.asyncio
    async def test_five_digit_client_code_while_polling_is_not_retried(self):
        """A 200 with 40001 fails at once instead of polling until the deadline."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.return_value = _response(
            200, {"code": 40001, "msg": "taskId is malformed"}
        )

        with pytest.raises(ProviderError, match="malformed") as exc:
            await provider.get_result("job_1")

        assert exc.value.status_code == 400
        assert not outcome_unknown(exc.value)
        assert provider._client.request.await_count == 1


class TestCreateTaskOutcome:
    """A createTask that may have made the task keeps the payment."""

    @pytest.mark.asyncio
    async def test_read_timeout_then_connect_errors_is_unknown(self):
        """The first attempt may have created the task; the last one says nothing."""
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            httpx.ReadTimeout("read timed out"),
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
        ]

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert outcome_unknown(exc.value)
        keys = {call.kwargs["headers"]["Idempotency-Key"] for call in _create_calls(provider)}
        assert len(_create_calls(provider)) == 3
        assert len(keys) == 1

    @pytest.mark.asyncio
    async def test_read_error_then_5xx_is_unknown(self):
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            httpx.ReadError("connection reset"),
            _response(503, {"code": 503, "msg": "busy"}),
            _response(500, {"code": 500, "msg": "internal"}),
        ]

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_write_timeout_then_stale_quote_is_not_sent_again(self):
        """After a create that may have landed, 40901 does not start a second job."""
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            httpx.WriteTimeout("write timed out"),
            _response(409, {"code": 40901, "msg": "quote no longer matches"}),
        ]

        with pytest.raises(ProviderError, match="40901") as exc:
            await provider.submit("jobs", _BODY)

        assert outcome_unknown(exc.value)
        assert provider._client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_gateway_timeout_is_unknown(self):
        provider = await _priced_provider()
        provider._client.request.return_value = _response(524, {"code": 524, "msg": "timeout"})

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_connect_errors_and_5xx_alone_are_known_failures(self):
        """Connections that never opened and plain 5xx answers refund the client."""
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("connect timed out"),
            _response(503, {"code": 503, "msg": "busy"}),
        ]

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 503
        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_read_timeout_then_success_returns_the_task(self):
        """A retry with the same Idempotency-Key that returns the task is a success."""
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            httpx.ReadTimeout("read timed out"),
            _task("queued", task_id="job_replayed"),
        ]

        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_replayed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _non_json(),
            _response(200, ["not", "an", "object"]),
            _response(200, {"code": 500, "msg": "internal"}),
            _response(200, {"code": 42901, "msg": "slow down"}),
            _response(200, {"code": 200, "msg": "success"}),
            _response(200, {"code": 200, "data": {"state": "queued"}}),
            _response(202, {"code": 200, "data": {"taskId": "", "state": "queued"}}),
        ],
        ids=[
            "non-json",
            "not-an-object",
            "code-500",
            "code-42901",
            "no-data",
            "no-task-id",
            "empty-task-id",
        ],
    )
    async def test_unreadable_2xx_is_unknown(self, answer):
        """SpicyAPI accepted the call, so the task may exist and bill."""
        provider = await _priced_provider()
        provider._client.request.return_value = answer

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 502
        assert outcome_unknown(exc.value)
        assert len(_create_calls(provider)) == 1

    @pytest.mark.asyncio
    async def test_2xx_with_client_error_code_is_a_known_failure(self):
        """A 4xxxx code is SpicyAPI refusing the job before it made one."""
        provider = await _priced_provider()
        provider._client.request.return_value = _response(
            200, {"code": 40001, "msg": "input.prompt is required"}
        )

        with pytest.raises(ProviderError, match="prompt is required") as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 400
        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_2xx_with_low_balance_code_is_unavailable(self):
        provider = await _priced_provider()
        provider._client.request.return_value = _response(
            200, {"code": 40201, "msg": "insufficient balance"}
        )

        with pytest.raises(ProviderError, match="cannot accept this charge") as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 503
        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _response(200, {"code": 40301, "msg": "forbidden"}),
            _response(403, {"code": 403, "msg": "forbidden"}),
        ],
        ids=["envelope-40301", "http-403"],
    )
    async def test_forbidden_create_is_a_gateway_credentials_error(self, answer):
        """A client would read a 403 as its own credentials being bad."""
        provider = await _priced_provider()
        provider._client.request.return_value = answer

        with pytest.raises(ProviderError) as exc:
            await provider.submit("jobs", _BODY)

        assert exc.value.status_code == 502
        assert exc.value.detail == UPSTREAM_CREDENTIALS_DETAIL

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["pending", "processing", ""])
    async def test_unlisted_create_state_is_pending(self, state):
        """Only a state SpicyAPI documents as failed refunds an accepted task."""
        provider = await _priced_provider()
        provider._client.request.return_value = _task(state, task_id="job_new")

        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_new"
        assert result["data"]["status"] == "processing"

    @pytest.mark.asyncio
    async def test_2xx_with_stale_quote_code_quotes_again(self):
        provider = await _priced_provider()
        provider._client.request.side_effect = [
            _response(200, {"code": 40901, "msg": "quote expired"}),
            _quote(quote_id="q2", unit="per_second"),
            _task("queued", task_id="job_retried"),
        ]

        result = await provider.submit("jobs", _BODY)

        assert result["data"]["id"] == "job_retried"
        assert [call.kwargs["json"]["quoteId"] for call in _create_calls(provider)] == [
            "quote_1",
            "q2",
        ]


class TestPollAfterAcceptance:
    """The task was accepted, so losing sight of it keeps the payment."""

    @pytest.mark.asyncio
    async def test_brief_404_is_tolerated(self):
        """A new task may not be readable yet."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _response(404, {"code": 404, "msg": "task not found"}),
            _response(404, {"code": 404, "msg": "task not found"}),
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"

    @pytest.mark.asyncio
    async def test_many_quick_404s_within_grace_keep_polling(self, monkeypatch):
        """A poll count alone is a few seconds; the grace is also time based."""
        clock = _fake_clock(monkeypatch)

        async def fake_sleep(seconds: float) -> None:
            clock["now"] += 1

        provider = _provider(poll_timeout=60)
        provider._sleep = fake_sleep
        provider._client.request.side_effect = [
            *[_response(404, {"code": 404, "msg": "task not found"}) for _ in range(10)],
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _response(404, {"code": 404, "msg": "task not found"}),
            _response(200, {"code": 40401, "msg": "task not found"}),
        ],
        ids=["http-404", "envelope-40401"],
    )
    async def test_404_past_grace_is_unknown(self, monkeypatch, answer):
        clock = _fake_clock(monkeypatch)

        async def fake_sleep(seconds: float) -> None:
            clock["now"] += 5

        provider = _provider(poll_timeout=300)
        provider._sleep = fake_sleep
        provider._client.request.return_value = answer

        with pytest.raises(ProviderError, match="no longer finds task job_1") as exc:
            await provider.get_result("job_1")

        assert exc.value.status_code == 502
        assert outcome_unknown(exc.value)
        assert clock["now"] - 1000.0 >= spicyapi._NOT_FOUND_GRACE_S

    @pytest.mark.asyncio
    async def test_404_count_resets_when_the_task_is_seen(self, monkeypatch):
        clock = _fake_clock(monkeypatch)

        async def fake_sleep(seconds: float) -> None:
            clock["now"] += 5

        missing = _response(404, {"code": 404, "msg": "task not found"})
        provider = _provider(poll_timeout=300)
        provider._sleep = fake_sleep
        provider._client.request.side_effect = [
            *[missing] * 4,
            _task("running"),
            *[missing] * 4,
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _response(401, {"code": 401, "msg": "invalid api key"}),
            _response(403, {"code": 403, "msg": "forbidden"}),
            _response(402, {"code": 40201, "msg": "insufficient balance"}),
            _response(200, {"code": 40101, "msg": "invalid api key"}),
        ],
        ids=["401", "403", "402", "envelope-40101"],
    )
    async def test_rejected_key_while_polling_is_unknown(self, answer):
        provider = _provider()
        provider._client.request.return_value = answer

        with pytest.raises(ProviderError) as exc:
            await provider.get_result("job_1")

        assert exc.value.status_code == 502
        assert exc.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_redirect_while_polling_is_unknown(self):
        provider = _provider()
        provider._client.request.return_value = _response(302, {})

        with pytest.raises(ProviderError, match="redirected") as exc:
            await provider.get_result("job_1")

        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_network_errors_keep_polling(self):
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            httpx.ReadTimeout("read timed out"),
            httpx.ConnectError("refused"),
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "record",
        [{"taskId": "job_1"}, {"taskId": "job_1", "state": "pending"}],
        ids=["missing-state", "pending"],
    )
    async def test_unlisted_state_keeps_polling(self, record):
        """A record without a terminal state may be a job still running."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _response(200, {"code": 200, "data": record}),
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"

    @pytest.mark.asyncio
    async def test_unlisted_state_until_the_deadline_keeps_the_payment(self, monkeypatch):
        provider = _provider(poll_timeout=10)
        provider._sleep = AsyncMock()
        clock = iter(float(second) for second in range(100))
        monkeypatch.setattr(spicyapi.time, "monotonic", lambda: next(clock))
        provider._client.request.return_value = _response(
            200, {"code": 200, "data": {"taskId": "job_1", "state": "processing"}}
        )

        with pytest.raises(TaskTimeoutError):
            await provider.get_result("job_1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["failed", "canceled", "cancelled", "expired"])
    async def test_failed_states_are_known_failures(self, state):
        provider = _provider()
        provider._client.request.return_value = _task(state, errorMessage="gave up")

        with pytest.raises(ProviderError, match="gave up") as exc:
            await provider.get_result("job_1")

        assert exc.value.status_code == 502
        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_success_envelope_without_data_keeps_polling(self):
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _response(200, {"code": 200, "msg": "success"}),
            _task("succeeded", cost="0.008"),
        ]

        result = await provider.get_result("job_1")

        assert result["state"] == "succeeded"


_CHAT = {
    "model": "deepseek/v4.1-flash/chat",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "reasoning_effort": "low",
    "_caller": "abcd1234ef01",
}


class _StreamResponse:
    """Minimal httpx stream response for chat tests."""

    def __init__(self, status: int, lines: list[str] | None = None, payload: dict | None = None):
        self.status_code = status
        self.headers = {"x-spicy-task-id": "job_chat"} if status < 400 else {}
        self._lines = lines or []
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    async def aread(self) -> bytes:
        return self.text.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamContext:
    def __init__(self, response: _StreamResponse):
        self._response = response

    async def __aenter__(self) -> _StreamResponse:
        return self._response

    async def __aexit__(self, *args: object) -> bool:
        return False


def _chat_lines() -> list[str]:
    first = (
        'data: {"id":"chatcmpl_1","model":"deepseek/v4.1-flash/chat",'
        '"choices":[{"index":0,"delta":{"content":"Hello"}}]}'
    )
    second = 'data: {"choices":[{"index":0,"delta":{"content":" there."},"finish_reason":"stop"}]}'
    usage = 'data: {"usage":{"prompt_tokens":12,"completion_tokens":4}}'
    return [first, second, usage, "data: [DONE]"]


def _attach_stream(provider: SpicyApiProvider, responses: list[_StreamResponse]) -> None:
    queue = list(responses)

    def open_stream(*args: object, **kwargs: object) -> _StreamContext:
        return _StreamContext(queue.pop(0))

    provider._client.stream = open_stream


class TestChat:
    """Tests for POST /v1/spicyapi/chat/completions."""

    @pytest.mark.asyncio
    async def test_quotes_openai_body_as_text_input(self):
        """Chat price is the SpicyAPI quote, with the output cap inside input."""
        provider = _provider()
        provider._client.request.return_value = _quote(cost="0.0098895", unit="per_1k_tokens")

        price = await provider.get_price("chat/completions", _CHAT)

        assert price == Decimal("0.0098895")
        sent = provider._client.request.await_args.kwargs["json"]
        assert sent == {
            "model": _CHAT["model"],
            "input": {
                "messages": _CHAT["messages"],
                "max_tokens": 64,
                "reasoning_effort": "low",
            },
        }

    @pytest.mark.asyncio
    async def test_omitted_max_tokens_uses_gateway_default(self):
        """A missing cap must not fall through to the model's large default."""
        provider = _provider()
        provider._default_max_tokens = 2048
        provider._client.request.return_value = _quote(unit="per_1k_tokens")
        body = {key: value for key, value in _CHAT.items() if key != "max_tokens"}

        await provider.get_price("chat/completions", body)

        sent = provider._client.request.await_args.kwargs["json"]
        assert sent["input"]["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_stream_is_rejected(self):
        """The gateway returns one JSON body, so SSE is not offered to the client."""
        provider = _provider()
        with pytest.raises(ProviderError, match="Streaming is not supported") as exc:
            await provider.get_price("chat/completions", {**_CHAT, "stream": True})
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_folds_the_stream_and_reads_settled_cost(self):
        """A quoted chat is streamed, then settled from the task record."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello there."
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["task_id"] == "job_chat"
        assert "status" not in result
        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.001")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "record",
        [
            _response(200, {"code": 40401, "msg": "task not found"}),
            _response(200, {"code": 50001, "msg": "internal"}),
            _non_json(),
        ],
        ids=["envelope-40401", "envelope-50001", "non-json"],
    )
    async def test_unreadable_cost_record_keeps_the_completion(self, record):
        """The chat already ran and billed; a bad record falls back to the quote."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            *[record] * (1 + len(spicyapi._COST_LOOKUP_DELAYS)),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert result["choices"][0]["message"]["content"] == "Hello there."
        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.0098895")

    @pytest.mark.asyncio
    async def test_cost_record_readable_on_a_later_try_is_used(self):
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            _response(200, {"code": 40401, "msg": "task not found"}),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_submit_sends_the_quoted_price(self):
        """The stream call carries the quote the client was shown."""
        provider = _provider()
        captured: dict = {}

        def open_stream(*args: object, **kwargs: object) -> _StreamContext:
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            captured["url"] = args[1] if len(args) > 1 else ""
            return _StreamContext(_StreamResponse(200, _chat_lines()))

        provider._client.stream = open_stream
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]

        await provider.get_price("chat/completions", _CHAT)
        await provider.submit("chat/completions", _CHAT)

        assert captured["json"]["quoteId"] == "qchat"
        assert captured["json"]["expectedCost"] == "0.0098895"
        assert captured["json"]["input"]["messages"] == _CHAT["messages"]
        assert "_caller" not in captured["json"]
        assert captured["headers"]["Idempotency-Key"]
        assert str(captured["url"]).endswith("/api/v1/jobs/stream")

    @pytest.mark.asyncio
    async def test_content_rejection_is_a_client_error(self):
        """A model refusal before any text does not look like a finished chat."""
        provider = _provider()
        provider._client.request.return_value = _quote(unit="per_1k_tokens")
        _attach_stream(
            provider,
            [
                _StreamResponse(
                    200,
                    [
                        "event: error",
                        'data: {"error":{"code":"content_rejected","message":"refused"}}',
                    ],
                )
            ],
        )

        await provider.get_price("chat/completions", _CHAT)
        with pytest.raises(ProviderError, match="refused") as exc:
            await provider.submit("chat/completions", _CHAT)
        assert exc.value.status_code == 400
        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_error_after_output_keeps_payment(self):
        """A mid-stream error after text was billed is an unknown outcome, not re-sent."""
        provider = _provider()
        provider._client.request.return_value = _quote(unit="per_1k_tokens")
        opened: list[int] = []
        lines = [
            'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}',
            "event: error",
            'data: {"error":{"code":"upstream","message":"model no longer available"}}',
        ]

        def open_stream(*args: object, **kwargs: object) -> _StreamContext:
            opened.append(1)
            return _StreamContext(_StreamResponse(200, lines))

        provider._client.stream = open_stream

        await provider.get_price("chat/completions", _CHAT)
        with pytest.raises(ProviderError, match="after output") as exc:
            await provider.submit("chat/completions", _CHAT)

        assert outcome_unknown(exc.value)
        assert len(opened) == 1

    @pytest.mark.asyncio
    async def test_stream_cut_off_after_output_keeps_payment(self):
        """A stream that closes without finish_reason after text is an unknown outcome."""
        provider = _provider()
        provider._client.request.return_value = _quote(unit="per_1k_tokens")
        lines = ['data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}']
        _attach_stream(provider, [_StreamResponse(200, lines)])

        await provider.get_price("chat/completions", _CHAT)
        with pytest.raises(ProviderError, match="before the answer finished") as exc:
            await provider.submit("chat/completions", _CHAT)

        assert outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_empty_stream_is_a_known_failure(self):
        """A stream that closes before any output refunds the client."""
        provider = _provider()
        provider._client.request.return_value = _quote(unit="per_1k_tokens")
        _attach_stream(provider, [_StreamResponse(200, [": keepalive"])])

        await provider.get_price("chat/completions", _CHAT)
        with pytest.raises(ProviderError, match="before the answer finished") as exc:
            await provider.submit("chat/completions", _CHAT)

        assert not outcome_unknown(exc.value)

    @pytest.mark.asyncio
    async def test_retries_stream_when_quote_is_stale(self):
        """40901 quotes again and streams once more at the same price."""
        provider = _provider()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="q1", unit="per_1k_tokens"),
            _quote(cost="0.0098895", quote_id="q2", unit="per_1k_tokens"),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(
            provider,
            [
                _StreamResponse(409, payload={"code": 40901, "msg": "quote no longer matches"}),
                _StreamResponse(200, _chat_lines()),
            ],
        )

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert result["task_id"] == "job_chat"
        assert provider._client.stream is not None


class TestSettledCostLookup:
    """The settled chat cost is read right after the stream, then with short pauses."""

    @pytest.mark.asyncio
    async def test_settled_cost_on_the_first_read_needs_no_pause(self):
        provider = _provider(poll_interval=2)
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.001")
        assert all(call.args[0] == 0 for call in provider._sleep.await_args_list)

    @pytest.mark.asyncio
    async def test_unsettled_cost_is_read_again_after_short_pauses(self):
        """Pauses grow from a quarter second instead of a fixed poll interval each."""
        provider = _provider(poll_interval=2)
        provider._sleep = AsyncMock()
        pending = _response(200, {"code": 200, "data": {"taskId": "job_chat", "state": "running"}})
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            pending,
            pending,
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.001")
        delays = [call.args[0] for call in provider._sleep.await_args_list]
        assert [delay for delay in delays if delay] == [0.25, 0.5]

    @pytest.mark.asyncio
    async def test_never_settled_falls_back_to_the_quote_within_the_old_budget(self):
        """Giving up no sooner than before keeps the exact charge as likely as it was."""
        provider = _provider(poll_interval=2)
        provider._sleep = AsyncMock()
        pending = _response(200, {"code": 200, "data": {"taskId": "job_chat", "state": "running"}})
        reads = 1 + len(spicyapi._COST_LOOKUP_DELAYS)
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            *[pending] * reads,
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.0098895")
        assert provider._client.request.await_count == 1 + reads
        delays = [call.args[0] for call in provider._sleep.await_args_list]
        # The old lookup waited two poll intervals (2 s each) in all.
        assert 3.5 <= sum(delays) <= 4.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            _response(429, {"code": 429, "msg": "rate limited"}),
            _response(503, {"code": 503, "msg": "unavailable"}),
            _response(404, {"code": 404, "msg": "task not found"}),
            httpx.ConnectError("refused"),
            httpx.RemoteProtocolError("dropped"),
        ],
        ids=["429", "503", "404", "connect-error", "dropped"],
    )
    async def test_failed_read_is_retried_and_the_exact_cost_still_used(self, failure):
        """One failed read must not bill the quote while later reads can settle."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            failure,
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.001")
        assert provider._client.request.await_count == 3

    @pytest.mark.asyncio
    async def test_failing_reads_fall_back_to_the_quote_after_the_last_read(self):
        provider = _provider()
        provider._sleep = AsyncMock()
        reads = 1 + len(spicyapi._COST_LOOKUP_DELAYS)
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            *[_response(503, {"code": 503, "msg": "unavailable"})] * reads,
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert result["choices"][0]["message"]["content"] == "Hello there."
        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.0098895")
        assert provider._client.request.await_count == 1 + reads

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            _response(401, {"code": 401, "msg": "bad key"}),
            httpx.ReadTimeout("slow"),
        ],
        ids=["rejected-key", "read-timeout"],
    )
    async def test_lasting_or_slow_failure_stops_the_lookup(self, failure):
        """A rejected key will not change; a timeout already made the client wait."""
        provider = _provider()
        provider._sleep = AsyncMock()
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            failure,
        ]
        _attach_stream(provider, [_StreamResponse(200, _chat_lines())])

        await provider.get_price("chat/completions", _CHAT)
        result = await provider.submit("chat/completions", _CHAT)

        assert await provider.calculate_actual_cost(_CHAT, result) == Decimal("0.0098895")
        assert provider._client.request.await_count == 2


class TestSharedPool:
    """SpicyAPI calls go through the gateway's pooled client."""

    @pytest.mark.asyncio
    async def test_uses_the_shared_pool_and_close_leaves_it_open(self):
        provider = SpicyApiProvider(config=_config())
        client = provider._get_client()

        assert client is http.shared_client()
        await provider.close()
        assert not client.is_closed
        await http.close_shared_client()

    @pytest.mark.asyncio
    async def test_requests_fail_fast_when_the_host_does_not_connect(self):
        """Each call keeps its read timeout but gets the short connect limit."""
        provider = _provider()
        provider._client.request.return_value = _quote()

        await provider.get_price("jobs", _BODY)

        timeout = provider._client.request.await_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == spicyapi._DEFAULT_TIMEOUT
        assert timeout.connect == http.CONNECT_TIMEOUT_S

    @pytest.mark.asyncio
    async def test_chat_stream_has_the_short_connect_limit(self):
        provider = _provider(poll_timeout=900)
        captured: dict = {}

        def open_stream(*args: object, **kwargs: object) -> _StreamContext:
            captured["timeout"] = kwargs.get("timeout")
            return _StreamContext(_StreamResponse(200, _chat_lines()))

        provider._client.stream = open_stream
        provider._client.request.side_effect = [
            _quote(cost="0.0098895", quote_id="qchat", unit="per_1k_tokens"),
            _task("succeeded", task_id="job_chat", cost="0.001", settled=True),
        ]

        await provider.get_price("chat/completions", _CHAT)
        await provider.submit("chat/completions", _CHAT)

        assert captured["timeout"].connect == http.CONNECT_TIMEOUT_S
        assert captured["timeout"].read == 900.0
