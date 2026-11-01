"""CloudConvert file conversion provider for x402gate.

Implements the BaseProvider interface for CloudConvert's REST API v2:
- Fixed pricing ($0.03 per operation)
- Supported operations: convert (format conversion), optimize (file compression)
- Job creation via POST /v2/jobs
- File upload via multipart POST to the upload URL
- Result via the sync API (GET sync.api.cloudconvert.com/v2/jobs/{id}),
  falling back to polling GET /v2/jobs/{id}
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from x402gate.core import http
from x402gate.core.config import ProviderConfig
from x402gate.core.http import shared_client
from x402gate.core.proxy import (
    UPSTREAM_CREDENTIALS_DETAIL,
    TaskTimeoutError,
    UpstreamMayHaveRun,
    rejected_credentials,
)
from x402gate.providers.base import BaseProvider, ProviderError

logger = logging.getLogger(__name__)

# The sync API holds a request open until the job ends. Each wait is capped,
# since proxies may drop a request that receives nothing for long; a wait
# that runs out is checked with a plain poll and then opened again.
_SYNC_WAIT_S = 60.0
_JOB_ENDED = ("finished", "error")
# A sync wait that fails for a passing reason (a dropped pooled connection,
# a 5xx or 429) is retried after a poll and a pause; this many in a row
# means the sync host is down, and the rest of the job is polled.
_SYNC_MAX_FAILURES = 3
# Each sync wait holds one connection of the shared pool for up to
# _SYNC_WAIT_S. Past this many at once a job polls instead, so a burst of
# conversions cannot take the slots that other providers' requests need.
_SYNC_WAIT_SLOTS = asyncio.Semaphore(20)
# The old loop slept this long before its first poll without counting it
# against poll_timeout. It stays in the budget, so a job that used to
# finish in time is never cut off sooner.
_FIRST_POLL_GRACE_S = 1.0


class _SyncWaitError(Exception):
    """The sync API did not answer with an ended job; poll instead.

    ``transient`` is True when the failure may pass (a transport error, a
    5xx or 429), so the sync API is tried again after a poll.
    """

    def __init__(self, reason: str, *, transient: bool) -> None:
        super().__init__(reason)
        self.transient = transient


def _sync_wait_url(base_url: str, job_id: str) -> str | None:
    """Return the sync API URL that waits for ``job_id``, or None if there is none.

    https://api.cloudconvert.com/v2 maps to https://sync.api.cloudconvert.com/v2
    (and the sandbox api.sandbox.cloudconvert.com the same way). Any other
    base URL (a proxy, a test server) has no known sync host and is polled.
    """
    parts = urlsplit(base_url.rstrip("/"))
    host = parts.hostname or ""
    if (
        parts.scheme != "https"
        or not host.startswith("api.")
        or not host.endswith(".cloudconvert.com")
    ):
        return None
    return urlunsplit(parts._replace(netloc=f"sync.{parts.netloc}")) + f"/jobs/{job_id}"


def _validate_body(body: dict[str, Any], provider_name: str) -> tuple[str, Any, Any, str]:
    """Check the request fields and return (operation, output_format, file_bytes, file_name).

    Shared by get_price() and submit(): get_price() runs before payment, so a
    request that could never be converted is turned away before it is paid for.

    Raises:
        ProviderError: 400 on an unsupported operation, a convert without a
            string ``output_format``, a non-string ``input_format``, or no
            uploaded file.
    """
    operation = body.get("operation", "convert")
    if operation not in ("convert", "optimize"):
        raise ProviderError(
            provider=provider_name,
            detail=f"Unsupported operation '{operation}'. Use 'convert' or 'optimize'.",
            status_code=400,
        )

    output_format = body.get("output_format")
    if operation == "convert" and (not output_format or not isinstance(output_format, str)):
        raise ProviderError(
            provider=provider_name,
            detail="'output_format' is required for convert (e.g. 'pdf', 'png', 'mp4')",
            status_code=400,
        )

    # submit() forwards input_format only when it is set; a non-string one
    # (JSON bodies only, multipart fields are always strings) cannot succeed.
    input_format = body.get("input_format")
    if input_format and not isinstance(input_format, str):
        raise ProviderError(
            provider=provider_name,
            detail="'input_format' must be a string (e.g. 'docx', 'jpg')",
            status_code=400,
        )

    # Only a multipart upload yields bytes. A JSON body can set _file_bytes to
    # anything, which would crash or fail in submit() after payment.
    file_bytes = body.get("_file_bytes")
    if not isinstance(file_bytes, (bytes, bytearray)) or not file_bytes:
        raise ProviderError(
            provider=provider_name,
            detail="No file provided. Send file as multipart/form-data.",
            status_code=400,
        )

    # A non-string name (only possible from a hand-written JSON body) would
    # crash the extension logging in submit() after payment.
    file_name = body.get("_file_name", "input_file")
    if not isinstance(file_name, str):
        file_name = "input_file"
    return operation, output_format, file_bytes, file_name


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

        Raises:
            ProviderError: 400 if the request could not be converted (see
                ``_validate_body``).
        """
        _validate_body(inputs, self.name)
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
        operation, output_format, file_bytes, file_name = _validate_body(body, self.name)

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
            resp = await shared_client().post(
                f"{self._config.base_url.rstrip('/')}/jobs",
                json=job_payload,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=http.timeout(30.0),
            )

            if rejected_credentials(resp.status_code):
                logger.error(
                    "CloudConvert rejected our key (%d): %s", resp.status_code, resp.text[:200]
                )
                raise ProviderError(provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL)

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
            resp = await shared_client().post(
                upload_url,
                data=upload_params,
                files={"file": (file_name, file_bytes)},
                timeout=http.timeout(120.0),
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
        """Wait for the CloudConvert job to finish, fail, or time out.

        Waits on the sync API (GET sync.api.cloudconvert.com/v2/jobs/{id}),
        which answers the moment the job ends instead of at the next poll.
        A wait that runs out is followed by one plain poll, so a sync API
        that hangs cannot hide a finished job. Any other sync answer than a
        readable finished or failed job is followed by a plain poll of
        GET /v2/jobs/{id}, whose handling of each status decides whether the
        client is charged; a sync API that does not work is then dropped
        for the rest of the job.

        poll_timeout is counted as the old polling loop counted it: the
        pauses between polls, not the time the polls themselves take. Time
        in a sync wait is charged at the rate the old loop charged wall-clock
        time (poll_interval out of every poll_interval plus one poll), so no
        job gets less time than polling alone would have given it.

        Returns:
            Dict with job result including export URLs.
        """
        url = f"{self._config.base_url.rstrip('/')}/jobs/{task_id}"
        wait_url = _sync_wait_url(self._config.base_url, task_id)
        poll_interval = self._config.poll_interval
        poll_timeout = self._config.poll_timeout
        budget = poll_timeout + _FIRST_POLL_GRACE_S if poll_timeout > 0 else 0.0
        loop = asyncio.get_running_loop()
        started = loop.time()
        poll_count = 0
        slept = 0.0  # pauses between polls, counted in full
        synced = 0.0  # wall-clock time inside sync waits
        polled = 0.0  # wall-clock time inside plain polls, never counted
        sync_failures = 0

        while True:
            if poll_count and poll_interval > 0:
                rate = poll_interval / (poll_interval + polled / poll_count)
            else:
                rate = 1.0
            waited = slept + synced * rate
            if waited >= budget:
                break

            pause = wait_url is None
            if wait_url is not None and _SYNC_WAIT_SLOTS.locked():
                # Every sync slot is busy: this round polls, as before.
                pause = True
            elif wait_url is not None:
                wait_started = loop.time()
                try:
                    async with _SYNC_WAIT_SLOTS:
                        job = await self._wait_for_job(wait_url, task_id, (budget - waited) / rate)
                except _SyncWaitError as e:
                    pause = True
                    sync_failures = sync_failures + 1 if e.transient else _SYNC_MAX_FAILURES
                    if sync_failures >= _SYNC_MAX_FAILURES:
                        logger.info(
                            "CloudConvert job %s: polling instead of waiting (%s)", task_id, e
                        )
                        wait_url = None
                    else:
                        logger.warning(
                            "CloudConvert job %s: sync wait failed (%s), retrying after a poll",
                            task_id,
                            e,
                        )
                else:
                    sync_failures = 0
                    if job is not None:
                        return self._job_outcome(task_id, job, loop.time() - started, poll_count)
                finally:
                    synced += loop.time() - wait_started

            elapsed = loop.time() - started
            poll_count += 1
            poll_started = loop.time()
            try:
                resp = await shared_client().get(
                    url,
                    headers={"Authorization": f"Bearer {self._config.api_key}"},
                    timeout=http.timeout(30.0),
                )

                if resp.status_code >= 500 or resp.status_code == 429:
                    # Transient: the job keeps running, so keep polling.
                    logger.warning(
                        "CloudConvert job %s poll #%d: HTTP %d after %ds",
                        task_id,
                        poll_count,
                        resp.status_code,
                        elapsed,
                    )
                elif rejected_credentials(resp.status_code):
                    logger.error(
                        "CloudConvert job %s poll rejected our key (%d): %s",
                        task_id,
                        resp.status_code,
                        resp.text[:200],
                    )
                    # The job was created and uploaded with this key and
                    # may still run upstream, so the payment is kept.
                    raise ProviderError(
                        provider=self.name, detail=UPSTREAM_CREDENTIALS_DETAIL
                    ) from UpstreamMayHaveRun(f"Poll of job {task_id} rejected our key")
                elif resp.status_code >= 400:
                    raise ProviderError(
                        provider=self.name,
                        detail=f"Job polling failed: {resp.text}",
                        status_code=resp.status_code,
                    )
                else:
                    job = self._parse_job(resp)
                    status = job.get("status", "")

                    if status in _JOB_ENDED:
                        return self._job_outcome(task_id, job, elapsed, poll_count)

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
            finally:
                polled += loop.time() - poll_started

            # Back to the sync API at once after a wait that ran out; a poll
            # that stands alone repeats after a pause.
            if pause:
                await asyncio.sleep(poll_interval)
                slept += poll_interval

        raise TaskTimeoutError(task_id=task_id, timeout=poll_timeout)

    async def _wait_for_job(
        self, wait_url: str, task_id: str, remaining: float
    ) -> dict[str, Any] | None:
        """Wait on the sync API until the job ends, for at most _SYNC_WAIT_S.

        Only a finished or failed job is returned. Failures are never
        classified here: a plain poll then sees the same state and decides.

        Returns:
            The ended job, or None when the wait ran out first.

        Raises:
            _SyncWaitError: The sync API gave any other answer; transient
                for a transport error, a 5xx or a 429.
        """
        try:
            resp = await shared_client().get(
                wait_url,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                timeout=http.timeout(min(_SYNC_WAIT_S, remaining)),
            )
        except httpx.ReadTimeout:
            logger.info("CloudConvert job %s still running after a sync wait", task_id)
            return None
        except httpx.HTTPError as e:
            raise _SyncWaitError(f"{type(e).__name__}: {e}", transient=True) from e
        if resp.status_code != 200:
            raise _SyncWaitError(
                f"HTTP {resp.status_code}",
                transient=resp.status_code >= 500 or resp.status_code == 429,
            )
        job = self._parse_job(resp)
        if job.get("status") not in _JOB_ENDED:
            # Answered before the job ended: this endpoint is not waiting.
            raise _SyncWaitError(f"status={job.get('status')!r}", transient=False)
        return job

    def _job_outcome(
        self, task_id: str, job: dict[str, Any], elapsed: float, poll_count: int
    ) -> dict[str, Any]:
        """Return the result of a finished job, or raise for a failed one."""
        if job.get("status") == "finished":
            logger.info(
                "CloudConvert job %s finished after %ds (%d polls)",
                task_id,
                elapsed,
                poll_count,
            )
            # Extract result URLs from export task
            return self._extract_result(job)

        error_msg = self._extract_error(job)
        fmt_info = self._extract_formats(job)
        logger.error(
            "CloudConvert job %s failed%s: %s",
            task_id,
            f" ({fmt_info})" if fmt_info else "",
            error_msg,
        )
        raise ProviderError(
            provider=self.name,
            detail=(
                f"Conversion failed ({fmt_info}): {error_msg}"
                if fmt_info
                else f"Conversion failed: {error_msg}"
            ),
            status_code=502,
        )

    @staticmethod
    def _parse_job(resp: httpx.Response) -> dict[str, Any]:
        """Return the job from a poll response, or {} when the body is unreadable.

        A garbled answer says nothing about the job, so polling goes on
        instead of an exception escaping get_result().
        """
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        job = payload.get("data", payload) if isinstance(payload, dict) else None
        if not isinstance(job, dict):
            logger.warning("CloudConvert poll returned an unreadable body: %s", resp.text[:200])
            return {}
        return job

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
    def _extract_formats(job: dict[str, Any]) -> str:
        """Extract 'input→output' format string from a job's process-file task."""
        for task in job.get("tasks", []):
            if task.get("name") == "process-file":
                inp = task.get("input_format") or task.get("result", {}).get("input", {}).get(
                    "ext", ""
                )
                out = task.get("output_format", "")
                if inp or out:
                    return f"{inp or '?'}→{out or '?'}"
        return ""

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
