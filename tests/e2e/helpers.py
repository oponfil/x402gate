"""Shared helpers for E2E test clients."""

import base64
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger("x402-e2e-helpers")

OUTPUT_DIR = Path(__file__).parent / "output"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config_yaml() -> dict[str, Any]:
    """Load project config.yaml with UTF-8 (Windows default cp1252 breaks on unicode)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Timings:
    """Collect step timings and output structured TIMINGS: line.

    Usage::

        timings = Timings()
        with timings.measure("pricing"):
            response = await http_client.post(...)
        with timings.measure("signing"):
            payload = await x402_client.create_payment_payload(...)
        timings.add_server_timings(response)  # parse X-Timing-* headers
        timings.output()  # prints TIMINGS:pricing=1.07,signing=0.01,...
    """

    def __init__(self):
        self._steps: list[tuple[str, float]] = []

    @contextmanager
    def measure(self, name: str):
        """Context manager that records elapsed time for a named step."""
        t0 = time.monotonic()
        yield
        self._steps.append((name, time.monotonic() - t0))

    def add_server_timings(self, response):
        """Parse X-Timing-Verify and X-Timing-Generation from response headers."""
        verify = response.headers.get("x-timing-verify")
        generation = response.headers.get("x-timing-generation")
        if verify is not None:
            self._steps.append(("server_verify", float(verify)))
        if generation is not None:
            self._steps.append(("server_generation", float(generation)))

    def output(self):
        """Print structured TIMINGS line for conftest to parse."""
        parts = ",".join(f"{k}={v:.2f}" for k, v in self._steps)
        print(f"TIMINGS:{parts}")


def _timestamp() -> str:
    """Return current timestamp string for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_images(images: list, label: str) -> list[Path]:
    """Decode and save base64 images from provider responses to tests/e2e/output/.

    Supports three formats:
    - Tungsten: {"base64_png": "..."} dicts
    - WaveSpeed: "data:image/...;base64,..." data-URLs
    - Raw base64 strings

    Returns list of saved file paths.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("#", "")
    ts = _timestamp()
    saved = []

    for i, img in enumerate(images):
        fname = f"{ts}_{safe_label}_{i + 1}.png"
        fpath = OUTPUT_DIR / fname
        img_bytes = None

        if isinstance(img, dict) and "base64_png" in img:
            img_bytes = base64.b64decode(img["base64_png"])
        elif isinstance(img, str) and img.startswith("data:"):
            b64_data = img.split(",", 1)[1] if "," in img else img
            img_bytes = base64.b64decode(b64_data)
        elif isinstance(img, str) and len(img) > 100:
            img_bytes = base64.b64decode(img)

        if img_bytes:
            fpath.write_bytes(img_bytes)
            saved.append(fpath)
            logger.info("Saved image: %s (%d bytes)", fpath, len(img_bytes))
        else:
            logger.warning("Skipped unsaved image %d: %s", i + 1, str(img)[:80])

    return saved


async def save_from_urls(urls: list[str], label: str, http_client: httpx.AsyncClient) -> list[Path]:
    """Download and save media files from URLs to tests/e2e/output/.

    Returns list of saved file paths.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("#", "")
    ts = _timestamp()
    saved = []

    for i, url in enumerate(urls):
        ext = Path(url.split("?")[0]).suffix or ".png"
        fpath = OUTPUT_DIR / f"{ts}_{safe_label}_{i + 1}{ext}"
        resp = await http_client.get(url, timeout=60.0)
        fpath.write_bytes(resp.content)
        saved.append(fpath)
        logger.info("Saved media: %s (%d bytes)", fpath, len(resp.content))

    return saved


def save_audio(audio_b64: str, label: str, content_type: str = "audio/mpeg") -> Path:
    """Decode and save base64 audio to tests/e2e/output/.

    Returns the saved file path.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("#", "")
    ts = _timestamp()

    # Determine extension from content-type
    ext_map = {"audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/ogg": ".ogg"}
    ext = ext_map.get(content_type, ".mp3")

    fpath = OUTPUT_DIR / f"{ts}_{safe_label}{ext}"
    audio_bytes = base64.b64decode(audio_b64)
    fpath.write_bytes(audio_bytes)
    logger.info("Saved audio: %s (%d bytes)", fpath, len(audio_bytes))
    return fpath


def print_timing_summary(
    label: str,
    timings: dict | None,
    generation_s: float | None,
    client_wait_s: float,
) -> None:
    """Print standardised timing summary for E2E tests.

    Args:
        label: Test label (e.g. "Base -> WaveSpeed").
        timings: Parsed TIMINGS dict or None.
        generation_s: Generation time from provider log, or None.
        client_wait_s: Total wall-clock time for client script.
    """
    print(f"\n=== [{label}] Timing ===")
    if timings:
        sv = timings.get("server_verify")
        sg = timings.get("server_generation")
        if sv is not None and sg is not None:
            network_overhead = timings["paid_request"] - sv - sg
            client_overhead = timings["pricing"] + timings["signing"]
            server_overhead = sv + network_overhead
            cp = timings["pricing"]
            cs = timings["signing"]
            print(
                f"Client overhead:         {client_overhead:.1f}s"
                f"  (pricing={cp:.1f} + signing={cs:.1f})"
            )
            print(
                f"Server overhead:         {server_overhead:.1f}s"
                f"  (verify={sv:.1f} + network={network_overhead:.1f})"
            )
            print(f"Generation time:         {sg:.1f}s")
        elif generation_s is not None:
            overhead = timings["paid_request"] - generation_s
            print(f"Client overhead:         {timings['pricing'] + timings['signing']:.1f}s")
            print(f"Server overhead:         {overhead:.1f}s")
            print(f"Generation time:         {generation_s:.1f}s")
        else:
            print(f"Paid request:            {timings['paid_request']:.1f}s")

        dl = timings.get("download", 0.0)
        if dl > 0:
            print(f"Download:                {dl:.1f}s")

        client_timings = {k: v for k, v in timings.items() if not k.startswith("server_")}
        total_timings = sum(client_timings.values())
        other = client_wait_s - total_timings
        print(f"Other (subprocess):      {other:.1f}s")
    elif generation_s is not None:
        print(f"Generation time:         {generation_s:.1f}s")
    print(f"Total client time:       {client_wait_s:.1f}s")
