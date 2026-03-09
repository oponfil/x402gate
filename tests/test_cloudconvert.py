"""Unit tests for the CloudConvert provider."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from x402gate.core.config import ProviderConfig
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


class TestGetPrice:
    """Tests for CloudConvertProvider.get_price()."""

    @pytest.mark.asyncio
    async def test_returns_fixed_price(self):
        """Returns the configured fixed price."""
        provider = _make_provider(fixed_price_usd=0.03)
        price = await provider.get_price("convert", {})
        assert price == Decimal("0.03")

    @pytest.mark.asyncio
    async def test_returns_custom_price(self):
        """Returns a custom fixed price."""
        provider = _make_provider(fixed_price_usd=0.05)
        price = await provider.get_price("convert", {"output_format": "pdf"})
        assert price == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_fallback_on_zero_price(self):
        """Falls back to $0.03 if fixed_price_usd is 0."""
        provider = _make_provider(fixed_price_usd=0.0)
        price = await provider.get_price("convert", {})
        assert price == Decimal("0.03")


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
