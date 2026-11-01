"""Unit tests for the CloudConvert provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.cloudconvert import CloudConvertProvider


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

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [job_response, upload_response]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
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

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
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
        provider = _make_provider(poll_interval=0, poll_timeout=10)

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

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = [processing_response, finished_response]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await provider.get_result("job_123")

        assert result["status"] == "completed"
        assert result["url"] == "https://storage.cloudconvert.com/result.pdf"
        assert result["filename"] == "output.pdf"

    @pytest.mark.asyncio
    async def test_poll_job_error(self):
        """Raises ProviderError when job fails."""
        provider = _make_provider(poll_interval=0, poll_timeout=10)

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

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = error_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(Exception, match="Conversion failed"):
                await provider.get_result("job_fail")

    @pytest.mark.asyncio
    async def test_poll_timeout(self):
        """Raises TaskTimeoutError when polling exceeds timeout."""
        provider = _make_provider(poll_interval=0, poll_timeout=0)

        processing_response = MagicMock()
        processing_response.status_code = 200
        processing_response.json.return_value = {
            "data": {"id": "job_slow", "status": "processing", "tasks": []}
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = processing_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
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
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
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
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(ProviderError) as caught,
        ):
            await provider.submit("convert", {"output_format": "pdf", "_file_bytes": b"data"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @pytest.mark.asyncio
    async def test_poll_rejected_key_is_502(self):
        """The job was created with this key and may still run: keep the payment."""
        provider = _make_provider(poll_interval=1, poll_timeout=10)
        client = _mock_client("get", [_response(403, text="Forbidden")])
        with (
            patch("httpx.AsyncClient", return_value=client),
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
        provider = _make_provider(poll_interval=1, poll_timeout=10)
        client = _mock_client("get", [_response(429, text="Too Many"), _response(200, _FINISHED)])
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("x402gate.providers.cloudconvert.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.get_result("job_1")
        assert result["url"] == "https://out.pdf"

    @pytest.mark.asyncio
    async def test_poll_unreadable_body_keeps_polling(self):
        """A garbled body must not escape get_result() as a raw exception."""
        provider = _make_provider(poll_interval=1, poll_timeout=10)
        client = _mock_client(
            "get",
            [
                _response(200, ValueError("bad json"), text="<html>"),
                _response(200, ["not", "a", "job"]),
                _response(200, _FINISHED),
            ],
        )
        with (
            patch("httpx.AsyncClient", return_value=client),
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
