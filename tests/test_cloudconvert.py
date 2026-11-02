"""Unit tests for the CloudConvert provider."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import struct
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL, TaskTimeoutError
from x402gate.providers import cloudconvert
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.cloudconvert import CloudConvertProvider, _sync_wait_url


def _make_config(**overrides) -> ProviderConfig:
    """Create a ProviderConfig for CloudConvert tests."""
    defaults = {
        "base_url": "https://api.cloudconvert.com/v2",
        "api_key": "test_api_key",
        "fixed_price_usd": 0.03,
        "poll_interval": 0,
        "poll_timeout": 10,
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _make_provider(**overrides) -> CloudConvertProvider:
    """Create a CloudConvertProvider for tests."""
    config = _make_config(**overrides)
    return CloudConvertProvider(config=config)


# A base URL with no known sync host: get_result() polls, as it always did.
_POLL_ONLY = "https://cloudconvert-proxy.example/v2"

_VALID_BODY = {"output_format": "pdf", "_file_bytes": b"data", "_file_name": "a.docx"}


class TestGetPrice:
    """Tests for CloudConvertProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_returns_fixed_price(self):
        """Returns the configured fixed price."""
        provider = _make_provider(fixed_price_usd=0.03)
        price = await provider.get_price("convert", dict(_VALID_BODY))
        assert price == Decimal("0.03")

    @pytest.mark.asyncio
    async def test_returns_custom_price(self):
        """Returns a custom fixed price."""
        provider = _make_provider(fixed_price_usd=0.05)
        price = await provider.get_price("convert", dict(_VALID_BODY))
        assert price == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_fallback_on_zero_price(self):
        """Falls back to $0.03 if fixed_price_usd is 0."""
        provider = _make_provider(fixed_price_usd=0.0)
        price = await provider.get_price("convert", dict(_VALID_BODY))
        assert price == Decimal("0.03")

    @pytest.mark.asyncio
    async def test_optimize_needs_no_output_format(self):
        """Optimize is priced without output_format, as submit() accepts it."""
        provider = _make_provider()
        body = {"operation": "optimize", "_file_bytes": b"%PDF-1.7"}
        assert await provider.get_price("optimize", body) == Decimal("0.03")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({"output_format": "pdf"}, "No file provided"),
            ({"_file_bytes": b"data"}, "output_format"),
            ({"operation": "merge", **_VALID_BODY}, "Unsupported operation"),
            ({"output_format": "pdf", "_file_bytes": 1}, "No file provided"),
            ({"output_format": "pdf", "_file_bytes": {"a": 1}}, "No file provided"),
            ({"output_format": "pdf", "_file_bytes": ["x"]}, "No file provided"),
            ({"output_format": "pdf", "_file_bytes": "text"}, "No file provided"),
            ({**_VALID_BODY, "output_format": 1}, "output_format"),
            ({**_VALID_BODY, "output_format": ["pdf"]}, "output_format"),
            ({**_VALID_BODY, "input_format": 1}, "input_format"),
        ],
    )
    async def test_rejects_invalid_body_before_payment(self, body, match):
        """A body submit() would refuse is refused by get_price(), before payment."""
        provider = _make_provider()
        with pytest.raises(ProviderError, match=match) as exc_info:
            await provider.get_price("convert", body)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_bytearray_and_empty_input_format_accepted(self):
        """Valid requests keep working: bytearray uploads and a null input_format."""
        provider = _make_provider()
        body = {"output_format": "pdf", "_file_bytes": bytearray(b"data"), "input_format": None}
        assert await provider.get_price("convert", body) == Decimal("0.03")

    @pytest.mark.asyncio
    async def test_non_string_file_name_does_not_crash(self):
        """A non-string _file_name falls back to the default name."""
        provider = _make_provider()
        body = {**_VALID_BODY, "_file_name": 123}
        assert await provider.get_price("convert", body) == Decimal("0.03")


class TestSubmit:
    """Tests for CloudConvertProvider.submit()."""

    @pytest.mark.asyncio
    async def test_submit_missing_output_format(self):
        """Raises ProviderError when output_format is missing."""
        provider = _make_provider()
        with pytest.raises(Exception, match="output_format"):
            await provider.submit("convert", {"_file_bytes": b"data"})

    @pytest.mark.asyncio
    async def test_submit_missing_file(self):
        """Raises ProviderError when no file is provided."""
        provider = _make_provider()
        with pytest.raises(Exception, match="No file provided"):
            await provider.submit("convert", {"output_format": "pdf"})

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Successful submission creates job and uploads file."""
        provider = _make_provider()

        # Mock job creation response
        job_response = MagicMock()
        job_response.status_code = 200
        job_response.json.return_value = {
            "data": {
                "id": "job_123",
                "status": "processing",
                "tasks": [
                    {
                        "name": "upload-file",
                        "status": "waiting",
                        "result": {
                            "form": {
                                "url": "https://storage.cloudconvert.com/upload/123",
                                "parameters": {"signature": "abc"},
                            }
                        },
                    },
                    {"name": "process-file", "status": "waiting"},
                    {"name": "export-file", "status": "waiting"},
                ],
            }
        }

        # Mock upload response
        upload_response = MagicMock()
        upload_response.status_code = 200

        with patch("x402gate.providers.cloudconvert.shared_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [job_response, upload_response]
            mock_client_cls.return_value = mock_client

            result = await provider.submit(
                "convert",
                {
                    "output_format": "pdf",
                    "input_format": "docx",
                    "_file_bytes": b"fake_docx_data",
                    "_file_name": "test.docx",
                },
            )

        assert result["id"] == "job_123"
        assert result["status"] == "processing"

    @pytest.mark.asyncio
    async def test_submit_job_creation_fails(self):
        """Raises ProviderError when job creation fails."""
        provider = _make_provider()

        error_response = MagicMock()
        error_response.status_code = 422
        error_response.text = "Invalid output format"

        with patch("x402gate.providers.cloudconvert.shared_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="Failed to create job"):
                await provider.submit(
                    "convert",
                    {
                        "output_format": "xyz",
                        "_file_bytes": b"data",
                        "_file_name": "test.txt",
                    },
                )


class TestGetResult:
    """Tests for CloudConvertProvider.get_result()."""

    @pytest.mark.asyncio
    async def test_poll_until_finished(self):
        """Polls until job status is 'finished'."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=0, poll_timeout=10)

        # First poll: processing; second poll: finished
        processing_response = MagicMock()
        processing_response.status_code = 200
        processing_response.json.return_value = {
            "data": {"id": "job_123", "status": "processing", "tasks": []}
        }

        finished_response = MagicMock()
        finished_response.status_code = 200
        finished_response.json.return_value = {
            "data": {
                "id": "job_123",
                "status": "finished",
                "tasks": [
                    {
                        "name": "export-file",
                        "status": "finished",
                        "result": {
                            "files": [
                                {
                                    "url": "https://storage.cloudconvert.com/result.pdf",
                                    "filename": "output.pdf",
                                }
                            ]
                        },
                    }
                ],
            }
        }

        with patch("x402gate.providers.cloudconvert.shared_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [processing_response, finished_response]
            mock_client_cls.return_value = mock_client

            result = await provider.get_result("job_123")

        assert result["status"] == "completed"
        assert result["url"] == "https://storage.cloudconvert.com/result.pdf"
        assert result["filename"] == "output.pdf"

    @pytest.mark.asyncio
    async def test_poll_job_error(self):
        """Raises ProviderError when job fails."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=0, poll_timeout=10)

        error_response = MagicMock()
        error_response.status_code = 200
        error_response.json.return_value = {
            "data": {
                "id": "job_fail",
                "status": "error",
                "tasks": [
                    {
                        "name": "process-file",
                        "status": "error",
                        "code": "INPUT_TASK_FAILED",
                        "message": "The input file could not be converted",
                    }
                ],
            }
        }

        with patch("x402gate.providers.cloudconvert.shared_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="Conversion failed"):
                await provider.get_result("job_fail")

    @pytest.mark.asyncio
    async def test_poll_timeout(self):
        """Raises TaskTimeoutError when polling exceeds timeout."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=0, poll_timeout=0)

        processing_response = MagicMock()
        processing_response.status_code = 200
        processing_response.json.return_value = {
            "data": {"id": "job_slow", "status": "processing", "tasks": []}
        }

        with patch("x402gate.providers.cloudconvert.shared_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = processing_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="did not complete"):
                await provider.get_result("job_slow")


def _response(status_code: int, payload=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if isinstance(payload, Exception):
        resp.json.side_effect = payload
    else:
        resp.json.return_value = payload
    return resp


def _mock_client(method: str, responses: list) -> MagicMock:
    mock_client = AsyncMock()
    getattr(mock_client, method).side_effect = responses
    return mock_client


_FINISHED = {
    "data": {
        "id": "job_1",
        "status": "finished",
        "tasks": [
            {
                "name": "export-file",
                "status": "finished",
                "result": {"files": [{"url": "https://out.pdf", "filename": "out.pdf"}]},
            }
        ],
    }
}


class TestUpstreamStatuses:
    """Upstream statuses the client must not see as they are."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_job_creation_rejected_key_is_502(self, status):
        """An upstream 402 must not reach x402 clients as a payment request."""
        provider = _make_provider()
        client = _mock_client("post", [_response(status, text="Invalid API key")])
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("convert", {"output_format": "pdf", "_file_bytes": b"data"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @pytest.mark.asyncio
    async def test_poll_rejected_key_is_502(self):
        """The job was created with this key and may still run: keep the payment."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=1, poll_timeout=10)
        client = _mock_client("get", [_response(403, text="Forbidden")])
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_result("job_1")
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is True

    @pytest.mark.asyncio
    async def test_poll_rate_limit_keeps_polling(self):
        """A 429 while polling is transient; the job keeps running."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=1, poll_timeout=10)
        client = _mock_client("get", [_response(429, text="Too Many"), _response(200, _FINISHED)])
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"

    @pytest.mark.asyncio
    async def test_poll_unreadable_body_keeps_polling(self):
        """A garbled body must not escape get_result() as a raw exception."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=1, poll_timeout=10)
        client = _mock_client(
            "get",
            [
                _response(200, ValueError("bad json"), text="<html>"),
                _response(200, ["not", "a", "job"]),
                _response(200, _FINISHED),
            ],
        )
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"


class TestExtractResult:
    """Tests for CloudConvertProvider._extract_result()."""

    def test_extracts_url_from_export_task(self):
        """Extracts file URL from finished export task."""
        job = {
            "tasks": [
                {
                    "name": "export-file",
                    "status": "finished",
                    "result": {
                        "files": [
                            {"url": "https://cdn.cloudconvert.com/file.pdf", "filename": "file.pdf"}
                        ]
                    },
                }
            ]
        }
        result = CloudConvertProvider._extract_result(job)
        assert result["url"] == "https://cdn.cloudconvert.com/file.pdf"
        assert result["filename"] == "file.pdf"

    def test_raises_if_no_export_files(self):
        """Raises ProviderError if no export files found."""
        job = {"tasks": [{"name": "export-file", "status": "finished", "result": {"files": []}}]}
        with pytest.raises(Exception, match="No export files"):
            CloudConvertProvider._extract_result(job)


class TestExtractError:
    """Tests for CloudConvertProvider._extract_error()."""

    def test_extracts_error_message(self):
        """Extracts error message from failed task."""
        job = {
            "tasks": [
                {
                    "name": "process-file",
                    "status": "error",
                    "code": "SANDBOX_FILE_NOT_ALLOWED",
                    "message": "File type not allowed in sandbox",
                }
            ]
        }
        error = CloudConvertProvider._extract_error(job)
        assert "SANDBOX_FILE_NOT_ALLOWED" in error
        assert "File type not allowed" in error

    def test_returns_unknown_if_no_error(self):
        """Returns 'Unknown error' when no error task found."""
        job = {"tasks": [{"name": "upload-file", "status": "finished"}]}
        error = CloudConvertProvider._extract_error(job)
        assert error == "Unknown error"


# ---------------------------------------------------------------------------
# shared client and the sync wait API
# ---------------------------------------------------------------------------

_POLL_URL = "https://api.cloudconvert.com/v2/jobs/job_1"
_WAIT_URL = "https://sync.api.cloudconvert.com/v2/jobs/job_1"
_PROCESSING = {"data": {"id": "job_1", "status": "processing", "tasks": []}}
_FAILED = {
    "data": {
        "id": "job_1",
        "status": "error",
        "tasks": [{"name": "process-file", "status": "error", "code": "X", "message": "bad"}],
    }
}


def _routed_client(wait: list, poll: list) -> AsyncMock:
    """Return a client whose GET answers the sync and the poll URL from their own lists.

    An item is a response or an exception to raise.
    """
    answers = {_WAIT_URL: list(wait), _POLL_URL: list(poll)}

    async def get(url, **kwargs):
        answer = answers[url].pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    client = AsyncMock()
    client.get.side_effect = get
    return client


def _urls(client: AsyncMock) -> list[str]:
    return [call.args[0] for call in client.get.call_args_list]


class TestSubmitUsesSharedClient:
    """Job creation and upload go through the pooled client with the old timeouts."""

    @pytest.mark.asyncio
    async def test_job_and_upload_timeouts(self):
        provider = _make_provider()
        job = _response(
            200,
            {
                "data": {
                    "id": "job_1",
                    "tasks": [
                        {
                            "name": "upload-file",
                            "result": {"form": {"url": "https://up.example", "parameters": {}}},
                        }
                    ],
                }
            },
        )
        client = _mock_client("post", [job, _response(201)])
        with patch("x402gate.providers.cloudconvert.shared_client", return_value=client):
            await provider.submit("convert", dict(_VALID_BODY))
        create, upload = client.post.call_args_list
        assert create.kwargs["timeout"] == http.timeout(30.0)
        assert upload.kwargs["timeout"] == http.timeout(120.0)


class TestSyncWaitUrl:
    """Which base URLs have a sync host."""

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://api.cloudconvert.com/v2", _WAIT_URL),
            ("https://api.cloudconvert.com/v2/", _WAIT_URL),
            (
                "https://api.sandbox.cloudconvert.com/v2",
                "https://sync.api.sandbox.cloudconvert.com/v2/jobs/job_1",
            ),
            (_POLL_ONLY, None),
            ("http://api.cloudconvert.com/v2", None),
            ("https://api.cloudconvert.com.evil.example/v2", None),
        ],
    )
    def test_maps_base_url(self, base_url, expected):
        assert _sync_wait_url(base_url, "job_1") == expected


class TestSyncWait:
    """get_result() learns the job ended from the sync API, without a poll delay."""

    @pytest.mark.asyncio
    async def test_finished_job_returns_at_once(self):
        """One sync request, no sleep and no poll."""
        provider = _make_provider(poll_interval=2, poll_timeout=600)
        client = _routed_client(wait=[_response(200, _FINISHED)], poll=[])
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL]
        sleep.assert_not_called()
        # Each wait is capped, so a proxy never sees an endless silent request.
        assert client.get.call_args.kwargs["timeout"] == http.timeout(60.0)

    @pytest.mark.asyncio
    async def test_wait_is_capped_by_poll_timeout(self):
        """A wait never runs past the budget: poll_timeout plus the old first second."""
        provider = _make_provider(poll_timeout=5)
        client = _routed_client(wait=[_response(200, _FINISHED)], poll=[])
        with patch("x402gate.providers.cloudconvert.shared_client", return_value=client):
            await provider.get_result("job_1")
        assert client.get.call_args.kwargs["timeout"].read == 6.0

    @pytest.mark.asyncio
    async def test_failed_job_is_refunded(self):
        """A job CloudConvert reports failed is a known failure, as when polled."""
        provider = _make_provider()
        client = _routed_client(wait=[_response(200, _FAILED)], poll=[])
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            pytest.raises(ProviderError, match="Conversion failed") as caught,
        ):
            await provider.get_result("job_1")
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False

    @pytest.mark.asyncio
    async def test_wait_that_runs_out_is_checked_and_reopened(self):
        """After a capped wait runs out, one poll checks the job, then the wait reopens."""
        provider = _make_provider(poll_interval=2)
        client = _routed_client(
            wait=[httpx.ReadTimeout("cap"), _response(200, _FINISHED)],
            poll=[_response(200, _PROCESSING)],
        )
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL, _POLL_URL, _WAIT_URL]
        sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_after_a_hung_wait_finds_the_finished_job(self):
        """A sync API that never answers cannot hide a finished job."""
        provider = _make_provider()
        client = _routed_client(
            wait=[httpx.ReadTimeout("cap")],
            poll=[_response(200, _FINISHED)],
        )
        with patch("x402gate.providers.cloudconvert.shared_client", return_value=client):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _response(404, text="no route"),
            _response(403, text="forbidden"),
            _response(302, text=""),
            _response(200, ValueError("bad json"), text="<html>"),
            _response(200, _PROCESSING),
        ],
    )
    async def test_sync_api_that_does_not_work_falls_back_to_polling(self, answer):
        """The sync API is dropped for this job; polling (with its pause) takes over."""
        provider = _make_provider(poll_interval=2)
        client = _routed_client(
            wait=[answer],
            poll=[_response(200, _PROCESSING), _response(200, _FINISHED)],
        )
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL, _POLL_URL, _POLL_URL]
        sleep.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "answer",
        [
            _response(500, text="oops"),
            _response(429, text="slow down"),
            httpx.ConnectError("down"),
            httpx.RemoteProtocolError("dropped"),
            httpx.PoolTimeout("pool full"),
        ],
    )
    async def test_passing_sync_failure_is_retried_after_a_poll(self, answer):
        """A dropped pooled connection or a 5xx/429 keeps the sync API for the next round."""
        provider = _make_provider(poll_interval=2)
        client = _routed_client(
            wait=[answer, _response(200, _FINISHED)],
            poll=[_response(200, _PROCESSING)],
        )
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL, _POLL_URL, _WAIT_URL]
        # A pause before retrying, so a failing sync host is never hit in a tight loop.
        sleep.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    async def test_sync_api_dropped_after_three_failures_in_a_row(self):
        """Three passing failures in a row mean the sync host is down: poll the rest."""
        provider = _make_provider(poll_interval=2)
        oops = _response(500, text="oops")
        client = _routed_client(
            wait=[oops, oops, oops],
            poll=[_response(200, _PROCESSING)] * 3 + [_response(200, _FINISHED)],
        )
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL, _POLL_URL] * 3 + [_POLL_URL]
        assert sleep.await_count == 3

    @pytest.mark.asyncio
    async def test_wait_that_runs_out_resets_the_failure_count(self):
        """Only failures in a row count: a working wait in between starts over."""
        provider = _make_provider(poll_interval=2)
        oops = _response(500, text="oops")
        client = _routed_client(
            wait=[oops, oops, httpx.ReadTimeout("cap"), oops, oops, _response(200, _FINISHED)],
            poll=[_response(200, _PROCESSING)] * 5,
        )
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_WAIT_URL, _POLL_URL] * 5 + [_WAIT_URL]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("status", "unknown"), [(403, True), (404, False)])
    async def test_sync_errors_are_classified_by_the_poll(self, status, unknown):
        """A sync 403/404 decides nothing; the poll's answer keeps today's rules."""
        provider = _make_provider()
        client = _routed_client(
            wait=[_response(status, text="nope")],
            poll=[_response(status, text="nope")],
        )
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await provider.get_result("job_1")
        assert caught.value.status_code == (502 if status == 403 else 404)
        assert outcome_unknown(caught.value) is unknown

    @pytest.mark.asyncio
    async def test_times_out_after_poll_timeout(self):
        """Time spent inside sync waits counts: they end in a timeout, payment kept."""
        provider = _make_provider(poll_timeout=100)
        clock = _FakeClock()

        async def get(url, **kwargs):
            if url == _WAIT_URL:
                clock.now += kwargs["timeout"].read
                raise httpx.ReadTimeout("cap")
            return _response(200, _PROCESSING)

        client = AsyncMock()
        client.get.side_effect = get
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.get_running_loop", return_value=clock),
            pytest.raises(TaskTimeoutError) as caught,
        ):
            await provider.get_result("job_1")
        assert outcome_unknown(caught.value) is True
        # poll_timeout plus the old uncounted first second, in capped waits.
        waits = [
            c.kwargs["timeout"].read for c in client.get.call_args_list if c.args[0] == _WAIT_URL
        ]
        assert waits == [60.0, 41.0]
        # The job is still checked once after the last wait.
        assert _urls(client)[-1] == _POLL_URL


class _FakeClock:
    """Stands in for the event loop's clock; tests move ``now`` by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


class TestTimeBudget:
    """poll_timeout gives every job at least the time the old polling loop gave it."""

    @pytest.mark.asyncio
    async def test_poll_latency_is_not_counted(self):
        """Polls that take time do not eat into poll_timeout, as before.

        poll_interval=2, poll_timeout=10: the old loop polled at 0, 2, 4, 6 and
        8 s of sleep. With each poll taking 1 s, the fifth poll runs 12 s in,
        past poll_timeout of wall-clock time, and must still see the result.
        """
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=2, poll_timeout=10)
        clock = _FakeClock()
        answers = [_response(200, _PROCESSING)] * 4 + [_response(200, _FINISHED)]

        async def get(url, **kwargs):
            clock.now += 1.0
            return answers.pop(0)

        async def sleep(seconds):
            clock.now += seconds

        client = AsyncMock()
        client.get.side_effect = get
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.get_running_loop", return_value=clock),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", side_effect=sleep),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert clock.now > 10

    @pytest.mark.asyncio
    async def test_polling_stops_after_the_old_number_of_polls(self):
        """The budget is not open-ended: the same polls as the old loop, then a timeout."""
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=2, poll_timeout=10)
        client = AsyncMock()
        client.get.return_value = _response(200, _PROCESSING)
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(TaskTimeoutError),
        ):
            await provider.get_result("job_1")
        # Old loop: polls at 0, 2, 4, 6, 8 s; the first second of grace adds one at 10 s.
        assert client.get.await_count == 6

    @pytest.mark.asyncio
    async def test_sync_waits_are_charged_at_the_old_polling_rate(self):
        """With polls taking 0.5 s and poll_interval=2, 1 s of waiting counted 0.8 s before.

        The old loop's last poll came about 1 + 49 x 2.5 = 123.5 s in for
        poll_timeout=100; sync waits must reach past that, not stop at 100 s.
        """
        provider = _make_provider(poll_interval=2, poll_timeout=100)
        clock = _FakeClock()

        async def get(url, **kwargs):
            if url == _WAIT_URL:
                clock.now += kwargs["timeout"].read
                raise httpx.ReadTimeout("cap")
            clock.now += 0.5
            return _response(200, _PROCESSING)

        client = AsyncMock()
        client.get.side_effect = get
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.get_running_loop", return_value=clock),
            pytest.raises(TaskTimeoutError),
        ):
            await provider.get_result("job_1")
        waits = [
            c.kwargs["timeout"].read for c in client.get.call_args_list if c.args[0] == _WAIT_URL
        ]
        # 101 s of budget at 0.8 per second is 126.25 s of waiting.
        assert sum(waits) == pytest.approx(126.25)
        assert clock.now > 123.5


class TestSyncWaitSlots:
    """Sync waits hold pooled connections, so only so many run at once."""

    @pytest.mark.asyncio
    async def test_polls_when_every_slot_is_busy(self):
        provider = _make_provider(poll_interval=2)
        client = _routed_client(
            wait=[], poll=[_response(200, _PROCESSING), _response(200, _FINISHED)]
        )
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
            patch.object(cloudconvert, "_SYNC_WAIT_SLOTS", asyncio.Semaphore(0)),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"
        assert _urls(client) == [_POLL_URL, _POLL_URL]
        sleep.assert_awaited_once_with(2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "wait",
        [
            [_response(200, _FINISHED)],
            [_response(200, _FAILED)],
            [_response(404), httpx.ReadTimeout("cap")],
        ],
    )
    async def test_slot_is_given_back(self, wait):
        provider = _make_provider()
        client = _routed_client(wait=wait, poll=[_response(200, _FINISHED)])
        slots = asyncio.Semaphore(1)
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch.object(cloudconvert, "_SYNC_WAIT_SLOTS", slots),
            contextlib.suppress(ProviderError),
        ):
            await provider.get_result("job_1")
        assert not slots.locked()


class TestPollWithoutDelay:
    """Polling (no sync host) starts at once instead of after a fixed second."""

    @pytest.mark.asyncio
    async def test_first_poll_is_immediate(self):
        provider = _make_provider(base_url=_POLL_ONLY, poll_interval=2)
        client = _mock_client("get", [_response(200, _FINISHED)])
        sleep = AsyncMock()
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", sleep),
        ):
            await provider.get_result("job_1")
        sleep.assert_not_called()
        assert client.get.call_args.kwargs["timeout"] == http.timeout(30.0)


_CREATED = {
    "data": {
        "id": "job_1",
        "tasks": [
            {
                "name": "upload-file",
                "result": {"form": {"url": "https://up.example", "parameters": {}}},
            }
        ],
    }
}


class TestSubmitTransportFailures:
    """A job only runs once its whole file arrived; before that a failure is refunded."""

    async def _submit(self, responses: list) -> ProviderError:
        client = _mock_client("post", responses)
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await _make_provider().submit("convert", dict(_VALID_BODY))
        return caught.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ReadTimeout("slow"),
            httpx.ReadError("dropped"),
            httpx.RemoteProtocolError("dropped"),
            httpx.ConnectError("refused"),
        ],
    )
    async def test_job_creation_failure_is_refunded(self, error):
        """No file was uploaded, so the job cannot run and nothing is billed."""
        caught = await self._submit([error])
        assert caught.status_code == 502
        assert "Failed to create job" in caught.detail
        assert outcome_unknown(caught) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.WriteTimeout("slow"),
            httpx.WriteError("dropped"),
            httpx.ConnectError("refused"),
            httpx.PoolTimeout("busy"),
        ],
    )
    async def test_upload_that_never_arrived_is_refunded(self, error):
        """The file never fully arrived, so the conversion cannot start."""
        caught = await self._submit([_response(201, _CREATED), error])
        assert "File upload failed" in caught.detail
        assert outcome_unknown(caught) is False

    async def _upload(self, answer, *, sent: bool) -> ProviderError:
        """Submit with an upload that raises or returns ``answer``.

        With ``sent``, httpx reads the whole file first, as when every byte
        was written to the socket.
        """

        async def post(url, **kwargs):
            if "files" not in kwargs:
                return _response(201, _CREATED)
            if sent:
                _name, upload = kwargs["files"]["file"]
                while upload.read(64 * 1024):
                    pass
            if isinstance(answer, BaseException):
                raise answer
            return answer

        client = AsyncMock()
        client.post.side_effect = post
        with (
            patch("x402gate.providers.cloudconvert.shared_client", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await _make_provider().submit("convert", dict(_VALID_BODY))
        return caught.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ReadTimeout("slow"),
            httpx.ReadError("dropped"),
            httpx.RemoteProtocolError("dropped"),
            httpx.WriteError("dropped"),
        ],
    )
    async def test_answer_lost_after_upload_keeps_the_payment(self, error):
        """The whole file was sent: the conversion may run and be billed."""
        caught = await self._upload(error, sent=True)
        assert "File upload failed" in caught.detail
        assert outcome_unknown(caught) is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            httpx.ReadTimeout("slow"),
            httpx.ReadError("reset"),
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
        ],
    )
    async def test_read_error_before_the_whole_file_was_sent_is_refunded(self, error):
        """httpcore hides a write that failed midway and reports the read after it."""
        caught = await self._upload(error, sent=False)
        assert "File upload failed" in caught.detail
        assert outcome_unknown(caught) is False

    @pytest.mark.asyncio
    async def test_upload_error_status_is_refunded(self):
        caught = await self._submit([_response(201, _CREATED), _response(500, text="down")])
        assert caught.status_code == 502
        assert outcome_unknown(caught) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503, 521])
    async def test_other_error_statuses_after_the_upload_are_refunded(self, status):
        caught = await self._upload(_response(status, text="down"), sent=True)
        assert caught.status_code == 502
        assert outcome_unknown(caught) is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [504, 520, 524])
    async def test_gateway_timeout_after_the_upload_keeps_the_payment(self, status):
        """A proxy gave up after the whole file was sent: it may have been stored."""
        caught = await self._upload(_response(status, text="timeout"), sent=True)
        assert caught.status_code == 504
        assert outcome_unknown(caught) is True

    @pytest.mark.asyncio
    async def test_gateway_timeout_before_the_whole_file_was_sent_is_refunded(self):
        caught = await self._upload(_response(524, text="timeout"), sent=False)
        assert caught.status_code == 502
        assert outcome_unknown(caught) is False


class TestUploadOverTheNetwork:
    """A real upload host that resets or stalls; httpcore turns a failed write into a read."""

    _SIZE = 20 * 1024 * 1024

    async def _submit(self, upload_host) -> tuple[ProviderError, int]:
        """Submit a 20 MB file to a local server whose upload is handled by ``upload_host``."""
        received = 0

        async def handle(reader, writer):
            nonlocal received
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode()
                length = int(head.lower().split("content-length:")[1].split("\r\n")[0])
                if not head.startswith("POST /v2/jobs"):
                    break
                await reader.readexactly(length)
                port = writer.get_extra_info("sockname")[1]
                form = {"url": f"http://127.0.0.1:{port}/upload", "parameters": {"k": "v"}}
                body = json.dumps(
                    {
                        "data": {
                            "id": "job_1",
                            "tasks": [{"name": "upload-file", "result": {"form": form}}],
                        }
                    }
                ).encode()
                writer.write(
                    b"HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            received = await upload_host(reader, writer, length)

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        provider = _make_provider(base_url=f"http://127.0.0.1:{port}/v2")
        body = {"output_format": "pdf", "_file_bytes": b"x" * self._SIZE, "_file_name": "a.docx"}
        try:
            with pytest.raises(ProviderError) as caught:
                await provider.submit("convert", body)
        finally:
            server.close()
            await http.close_shared_client()
        return caught.value, received

    @staticmethod
    def _reset(writer) -> None:
        sock = writer.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.transport.abort()

    @pytest.mark.asyncio
    async def test_reset_midway_is_refunded(self):
        async def upload_host(reader, writer, length):
            chunk = await reader.read(64 * 1024)
            self._reset(writer)
            return len(chunk)

        caught, received = await self._submit(upload_host)
        assert received < self._SIZE
        assert outcome_unknown(caught) is False

    @pytest.mark.asyncio
    async def test_reset_after_the_whole_file_keeps_the_payment(self):
        async def upload_host(reader, writer, length):
            await reader.readexactly(length)
            self._reset(writer)
            return length

        caught, received = await self._submit(upload_host)
        assert received > self._SIZE
        assert outcome_unknown(caught) is True

    @pytest.mark.asyncio
    async def test_stalled_reader_is_refunded(self):
        async def upload_host(reader, writer, length):
            await asyncio.sleep(5)
            return 0

        with patch.object(http, "timeout", return_value=httpx.Timeout(0.5)):
            caught, _ = await self._submit(upload_host)
        assert outcome_unknown(caught) is False
