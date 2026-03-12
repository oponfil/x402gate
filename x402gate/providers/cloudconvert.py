"""CloudConvert file conversion provider for x402gate.

Implements the BaseProvider interface for CloudConvert's REST API v2:
- Fixed pricing ($0.03 per operation)
- Supported operations: convert (format conversion), optimize (file compression)
- Job creation via POST /v2/jobs
- File upload via multipart POST to the upload URL
- Result polling via GET /v2/jobs/{id}
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import httpx

from x402gate.core.config import ProviderConfig
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)


class CloudConvertProvider(BaseProvider):
    """CloudConvert file conversion and optimization provider.

    Converts files between 200+ formats (documents, images, video, audio)
    and optimizes (compresses) PDF, PNG, JPG files.
    Uses a fixed price per operation since CloudConvert has no dynamic
    pricing API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(name="cloudconvert", config=config)

    async def get_price(self, model_path: str, inputs: dict[str, Any]) -> Decimal:
        """Return fixed price per conversion.

        CloudConvert uses credit-based pricing with no per-request pricing API,
        so we charge a flat rate configured via fixed_price_usd.
        """
        price = Decimal(str(self._config.fixed_price_usd))
        if price <= 0:
            price = Decimal("0.03")
        return price

    async def submit(
        self,
        path: str,
        body: dict[str, Any],
        *,
        prepaid: bool = False,
    ) -> dict[str, Any]:
        """Create a CloudConvert job, upload the file, and return job info.

        The body should contain:
            - operation (str, optional): "convert" (default) or "optimize"
            - output_format (str): Target format (required for convert, optional for optimize)
            - input_format (str, optional): Source format (auto-detected if omitted)
            - _file_bytes (bytes): Raw file content (injected by app.py from multipart)
            - _file_name (str): Original filename (injected by app.py from multipart)

        Returns:
            Dict with "id" and "status" keys for polling.
        """
        operation = body.get("operation", "convert")
        if operation not in ("convert", "optimize"):
            raise ProviderError(
                provider=self.name,
                detail=f"Unsupported operation '{operation}'. Use 'convert' or 'optimize'.",
                status_code=400,
            )

        output_format = body.get("output_format")
        if operation == "convert" and not output_format:
            raise ProviderError(
                provider=self.name,
                detail="'output_format' is required for convert (e.g. 'pdf', 'png', 'mp4')",
                status_code=400,
            )

        file_bytes = body.get("_file_bytes")
        file_name = body.get("_file_name", "input_file")
        if not file_bytes:
            raise ProviderError(
                provider=self.name,
                detail="No file provided. Send file as multipart/form-data.",
                status_code=400,
            )

        # Log file details for debugging OPEN_FAILED issues
        ext = file_name.rsplit(".", 1)[-1] if "." in file_name else "(none)"
        magic = file_bytes[:8] if isinstance(file_bytes, bytes) else b""
        caller = body.get("_caller", "?")
        logger.info(
            "CloudConvert %s: file=%s ext=%s size=%d bytes magic=%s caller=%s",
            operation,
            file_name,
            ext,
            len(file_bytes),
            magic[:8].hex(),
            caller,
        )

        # Build processing task options
        if operation == "optimize":
            process_task: dict[str, Any] = {
                "operation": "optimize",
                "input": "upload-file",
            }
            input_format = body.get("input_format")
            if input_format:
                process_task["input_format"] = input_format
        else:
            process_task = {
                "operation": "convert",
                "input": "upload-file",
                "output_format": output_format,
            }
            input_format = body.get("input_format")
            if input_format:
                process_task["input_format"] = input_format

        # 1. Create CloudConvert job
        job_payload = {
            "tasks": {
                "upload-file": {
                    "operation": "import/upload",
                },
                "process-file": process_task,
                "export-file": {
                    "operation": "export/url",
                    "input": "process-file",
                },
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._config.base_url.rstrip('/')}/jobs",
                    json=job_payload,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"Failed to create job: {resp.text}",
                    status_code=resp.status_code,
                )

            job = resp.json().get("data", resp.json())

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"Failed to create job: {e}",
                status_code=502,
            ) from e

        # 2. Find upload task and upload the file
        upload_task = None
        for task in job.get("tasks", []):
            if task.get("name") == "upload-file":
                upload_task = task
                break

        if not upload_task:
            raise ProviderError(
                provider=self.name,
                detail="Upload task not found in job response",
                status_code=502,
            )

        upload_url = upload_task.get("result", {}).get("form", {}).get("url")
        upload_params = upload_task.get("result", {}).get("form", {}).get("parameters", {})

        if not upload_url:
            raise ProviderError(
                provider=self.name,
                detail="Upload URL not found in task response",
                status_code=502,
            )

        try:
            # Upload: multipart POST with form parameters + file
            async with httpx.AsyncClient(timeout=120.0) as client:
                files = {"file": (file_name, file_bytes)}
                resp = await client.post(
                    upload_url,
                    data=upload_params,
                    files=files,
                )

            if resp.status_code >= 400:
                raise ProviderError(
                    provider=self.name,
                    detail=f"File upload failed: {resp.text}",
                    status_code=502,
                )

        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                provider=self.name,
                detail=f"File upload failed: {e}",
                status_code=502,
            ) from e

        job_id = job.get("id")
        logger.info(
            "CloudConvert job created: %s (%s, %s → %s)",
            job_id,
            operation,
            input_format or "auto",
            output_format or "optimized",
        )

        return {"id": job_id, "status": "processing"}

    async def get_result(self, task_id: str) -> dict[str, Any]:
        """Poll CloudConvert for job completion.

        Polls GET /v2/jobs/{id} until the job finishes, fails, or times out.

        Returns:
            Dict with job result including export URLs.
        """
        url = f"{self._config.base_url.rstrip('/')}/jobs/{task_id}"
        poll_interval = self._config.poll_interval
        poll_timeout = self._config.poll_timeout
        elapsed = 0
        poll_count = 0

        # Brief delay before first poll
        await asyncio.sleep(1)

        async with httpx.AsyncClient(timeout=30.0) as client:
            while elapsed < poll_timeout:
                poll_count += 1
                try:
                    resp = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {self._config.api_key}"},
                    )

                    if resp.status_code >= 500:
                        logger.warning(
                            "CloudConvert job %s poll #%d: HTTP %d after %ds",
                            task_id,
                            poll_count,
                            resp.status_code,
                            elapsed,
                        )
                    elif resp.status_code >= 400:
                        raise ProviderError(
                            provider=self.name,
                            detail=f"Job polling failed: {resp.text}",
                            status_code=resp.status_code,
                        )
                    else:
                        job = resp.json().get("data", resp.json())
                        status = job.get("status", "")

                        if status == "finished":
                            logger.info(
                                "CloudConvert job %s finished after %ds (%d polls)",
                                task_id,
                                elapsed,
                                poll_count,
                            )
                            # Extract result URLs from export task
                            return self._extract_result(job)

                        if status == "error":
                            error_msg = self._extract_error(job)
                            logger.error(
                                "CloudConvert job %s failed: %s",
                                task_id,
                                error_msg,
                            )
                            raise ProviderError(
                                provider=self.name,
                                detail=f"Conversion failed: {error_msg}",
                                status_code=502,
                            )

                        if poll_count <= 5 or poll_count % 10 == 0:
                            logger.info(
                                "CloudConvert job %s poll #%d: status=%s, elapsed=%ds/%ds",
                                task_id,
                                poll_count,
                                status,
                                elapsed,
                                poll_timeout,
                            )

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    logger.warning(
                        "CloudConvert job %s poll #%d: %s: %s",
                        task_id,
                        poll_count,
                        type(e).__name__,
                        e,
                    )

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        from x402gate.core.proxy import TaskTimeoutError

        raise TaskTimeoutError(task_id=task_id, timeout=poll_timeout)

    @staticmethod
    def _extract_result(job: dict[str, Any]) -> dict[str, Any]:
        """Extract export URLs from a finished job."""
        for task in job.get("tasks", []):
            if task.get("name") == "export-file" and task.get("status") == "finished":
                files = task.get("result", {}).get("files", [])
                if files:
                    return {
                        "status": "completed",
                        "files": files,
                        "url": files[0].get("url"),
                        "filename": files[0].get("filename"),
                    }

        raise ProviderError(
            provider="cloudconvert",
            detail="No export files found in finished job",
            status_code=502,
        )

    @staticmethod
    def _extract_error(job: dict[str, Any]) -> str:
        """Extract error message from a failed job."""
        errors: list[str] = []
        for task in job.get("tasks", []):
            if task.get("status") == "error":
                code = task.get("code", "")
                msg = task.get("message", "")
                task_name = task.get("name", task.get("operation", ""))

                # Build base error
                base = f"{code}: {msg}" if code else msg

                # Extract detailed errors from result.errors (if present)
                result_errors = task.get("result", {})
                if isinstance(result_errors, dict):
                    detail_list = result_errors.get("errors", [])
                    if detail_list:
                        details = "; ".join(
                            e.get("message", str(e)) if isinstance(e, dict) else str(e)
                            for e in detail_list
                        )
                        base = f"{base} ({details})" if base else details

                if base:
                    errors.append(f"[{task_name}] {base}" if task_name else base)

        return " | ".join(errors) if errors else "Unknown error"
