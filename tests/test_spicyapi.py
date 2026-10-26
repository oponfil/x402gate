"""Unit tests for the SpicyAPI provider."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import TaskTimeoutError
from x402gate.providers.base import ProviderError
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
    provider._client = client
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
        """Only the jobs endpoint is part of the gateway contract."""
        provider = _provider()
        with pytest.raises(ProviderError, match="/v1/spicyapi/jobs"):
            await provider.get_price("chat/completions", _BODY)

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
        provider._priced["ignored"] = Decimal("0.01")

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
    async def test_failed_poll_raises(self):
        """A failed task becomes a provider error and has no settled cost."""
        provider = _provider()
        provider._client.request.return_value = _task(
            "expired", errorCode="timeout", errorMessage="deadline passed"
        )
        with pytest.raises(ProviderError, match="deadline passed"):
            await provider.get_result("job_1")
        assert await provider.calculate_actual_cost({}, {"data": {"id": "job_1"}}) is None

    @pytest.mark.asyncio
    async def test_timeout(self):
        """A zero poll budget times out before calling the provider."""
        provider = _provider(poll_timeout=0)
        with pytest.raises(TaskTimeoutError):
            await provider.get_result("job_1")
        provider._client.request.assert_not_awaited()
