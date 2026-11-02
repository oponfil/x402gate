"""Unit tests for shared provider helpers."""

from __future__ import annotations

import httpx
import pytest

from x402gate.core.proxy import ProxyError, TaskTimeoutError
from x402gate.providers.base import (
    ProviderError,
    optional_str,
    outcome_unknown,
    raise_for_gateway_timeout,
    require_text,
)


def _raised(make_error, cause: BaseException | None = None, *, suppress: bool = False):
    """Raise make_error() while handling cause, and return what was raised."""
    try:
        try:
            if cause is not None:
                raise cause
            raise make_error()
        except BaseException as inner:
            if inner is cause:
                if suppress:
                    raise make_error() from None
                raise make_error() from inner
            raise
    except BaseException as outer:
        return outer


def _provider_error() -> ProviderError:
    return ProviderError(provider="test", detail="boom")


class TestOutcomeUnknown:
    """Which failures may still have run, and billed us, at the provider."""

    def test_poll_timeout_is_unknown(self):
        assert outcome_unknown(TaskTimeoutError(task_id="t", timeout=10)) is True

    @pytest.mark.parametrize(
        "cause",
        [
            httpx.ReadTimeout("slow"),
            httpx.WriteTimeout("slow"),
            httpx.ReadError("dropped"),
            httpx.WriteError("dropped"),
            httpx.RemoteProtocolError("dropped"),
        ],
    )
    def test_request_that_may_have_landed_is_unknown(self, cause):
        assert outcome_unknown(_raised(_provider_error, cause)) is True

    def test_suppressed_context_still_counts(self):
        """``raise ... from None`` hides the timeout from tracebacks, not from us."""
        assert outcome_unknown(_raised(_provider_error, httpx.ReadTimeout("x"), suppress=True))

    def test_nested_wrapping_is_followed(self):
        """WaveSpeed wraps a ProxyError that was raised from a read timeout."""
        proxy_error = _raised(
            lambda: ProxyError(504, "timed out"), httpx.ReadTimeout("x"), suppress=True
        )
        assert outcome_unknown(_raised(_provider_error, proxy_error)) is True

    @pytest.mark.parametrize(
        "cause",
        [
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("no route"),
            httpx.PoolTimeout("busy"),
        ],
    )
    def test_request_that_never_left_is_known(self, cause):
        assert outcome_unknown(_raised(_provider_error, cause)) is False

    def test_reported_failure_is_known(self):
        assert outcome_unknown(_provider_error()) is False


class TestRequireText:
    """Only a non-blank string is text that can be priced."""

    def test_returns_the_text(self):
        assert require_text({"text": " hi "}, "test") == " hi "

    @pytest.mark.parametrize("text", [None, "", "   ", ["x" * 5000], 5, {"a": "b"}, True])
    def test_rejects_anything_else(self, text):
        with pytest.raises(ProviderError) as caught:
            require_text({"text": text}, "test")
        assert caught.value.status_code == 400

    def test_rejects_missing_text(self):
        with pytest.raises(ProviderError):
            require_text({}, "test")


class TestOptionalStr:
    """Optional string fields: missing or null is empty, other types are a 400."""

    @pytest.mark.parametrize(
        ("inputs", "expected"), [({}, ""), ({"k": None}, ""), ({"k": "v"}, "v")]
    )
    def test_accepts_missing_null_and_strings(self, inputs, expected):
        assert optional_str(inputs, "k", "test") == expected

    @pytest.mark.parametrize("value", [5, 0, False, ["v"], {"a": 1}])
    def test_rejects_other_types(self, value):
        with pytest.raises(ProviderError) as caught:
            optional_str({"k": value}, "k", "test")
        assert caught.value.status_code == 400
        assert "'k'" in caught.value.detail


class TestRaiseForGatewayTimeout:
    """504/520/524 from a proxy in front of the provider keeps the payment."""

    @pytest.mark.parametrize("status", [504, 520, 524])
    def test_gateway_timeout_is_unknown(self, status):
        with pytest.raises(ProviderError) as caught:
            raise_for_gateway_timeout(status, "test", "Test")
        assert caught.value.status_code == 504
        assert str(status) in caught.value.detail
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize("status", [200, 400, 500, 502, 503, 521, 522, 523])
    def test_other_statuses_pass(self, status):
        raise_for_gateway_timeout(status, "test", "Test")
