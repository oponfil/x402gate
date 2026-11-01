"""Unit tests for the WaveSpeed provider."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from x402gate.core.config import ProviderConfig
from x402gate.core.proxy import UPSTREAM_CREDENTIALS_DETAIL
from x402gate.providers.base import ProviderError, outcome_unknown
from x402gate.providers.wavespeed import WaveSpeedProvider, _parse_catalog


def _make_provider() -> WaveSpeedProvider:
    return WaveSpeedProvider(
        ProviderConfig(
            base_url="https://api.wavespeed.ai/api/v3",
            api_key="test_key",
            poll_interval=2,
            poll_timeout=60,
        )
    )


def _patched(method: str, mock: AsyncMock):
    """Patch one httpx.AsyncClient method and skip proxy sleeps."""
    return (
        patch(f"httpx.AsyncClient.{method}", mock),
        patch("x402gate.core.proxy.asyncio.sleep", new_callable=AsyncMock),
    )


class TestGetPrice:
    async def test_server_error_is_retried(self):
        """A price quote starts no job, so a 5xx is retried."""
        post = AsyncMock(
            side_effect=[
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"data": {"unit_price": 0.003}}),
            ]
        )
        p1, p2 = _patched("post", post)
        with p1, p2:
            price = await _make_provider().get_price("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert price == Decimal("0.003")
        assert post.call_count == 2

    async def test_rejected_key_is_502_without_upstream_text(self):
        post = AsyncMock(return_value=httpx.Response(401, text="invalid api key abc123"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_price("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL


class TestSubmit:
    async def test_server_error_is_sent_once_and_refunded(self):
        """A resend after a 5xx could start a second paid job."""
        post = AsyncMock(return_value=httpx.Response(500, text="upstream down"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert outcome_unknown(caught.value) is False

    async def test_gateway_timeout_keeps_the_payment(self):
        post = AsyncMock(return_value=httpx.Response(504, text="gateway timeout"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize("status", [401, 402, 403])
    async def test_rejected_key_is_502(self, status):
        post = AsyncMock(return_value=httpx.Response(status, text="no credit"))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is False

    @pytest.mark.parametrize(
        "body",
        [{"data": None}, {"data": {"status": "created"}}, {"data": {"id": "t1"}}, {"x": 1}],
    )
    async def test_success_without_task_id_is_an_error(self, body):
        """No task id means no output can be delivered, but a task may exist."""
        post = AsyncMock(return_value=httpx.Response(200, json=body))
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert "returned no task id" in caught.value.detail
        assert outcome_unknown(caught.value) is True

    @pytest.mark.parametrize(
        "body",
        [
            {"data": {"id": "t1", "status": "created"}},
            {"data": {"output": {"images": ["base64img"]}}},
            {"data": {"outputs": ["https://out.png"]}},
            {"outputs": ["ok"]},
        ],
    )
    async def test_task_or_sync_result_is_returned(self, body):
        post = AsyncMock(return_value=httpx.Response(200, json=body))
        p1, p2 = _patched("post", post)
        with p1, p2:
            result = await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert result == body

    async def test_redirect_is_refunded(self):
        """httpx does not follow redirects, so WaveSpeed never ran the call."""
        post = AsyncMock(
            return_value=httpx.Response(307, headers={"location": "https://elsewhere/"})
        )
        p1, p2 = _patched("post", post)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().submit("wavespeed-ai/flux-dev", {"prompt": "x"})
        assert post.call_count == 1
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is False


class TestGetResult:
    async def test_lost_task_keeps_the_payment(self):
        get = AsyncMock(return_value=httpx.Response(404, text="not found"))
        p1, p2 = _patched("get", get)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_result("t1")
        assert caught.value.status_code == 502
        assert outcome_unknown(caught.value) is True

    async def test_rejected_key_while_polling_is_502(self):
        """The task was accepted and may still run, so the payment is kept."""
        get = AsyncMock(return_value=httpx.Response(402, text="insufficient balance"))
        p1, p2 = _patched("get", get)
        with p1, p2, pytest.raises(ProviderError) as caught:
            await _make_provider().get_result("t1")
        assert caught.value.status_code == 502
        assert caught.value.detail == UPSTREAM_CREDENTIALS_DETAIL
        assert outcome_unknown(caught.value) is True


PRICING_URL = "https://api.wavespeed.ai/api/v3/model/pricing"
MODELS_URL = "https://api.wavespeed.ai/api/v3/models"
FLUX = "wavespeed-ai/flux-dev"


def _quote(price: float) -> httpx.Response:
    return httpx.Response(200, json={"data": {"unit_price": price}})


def _pricing_bodies(post: AsyncMock) -> list[dict]:
    return [call.kwargs["json"] for call in post.await_args_list if call.args[0] == PRICING_URL]


def _provider_with_types(types: dict[str, str], **kwargs) -> WaveSpeedProvider:
    provider = WaveSpeedProvider(_make_provider()._config, **kwargs)
    provider._model_types = dict(types)
    provider._catalog_loaded_at = time.monotonic()
    return provider


async def _probes_done(provider: WaveSpeedProvider) -> None:
    """Wait for the background probes and the checks they record."""
    await asyncio.gather(*provider._probes.values(), return_exceptions=True)


class TestPriceCache:
    """New prompts and seeds reuse a quote when they cannot change the price."""

    async def test_new_seed_reuses_the_quote(self):
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({})
        p1, p2 = _patched("post", post)
        with p1, p2:
            first = await provider.get_price(FLUX, {"prompt": "cat", "seed": 1})
            second = await provider.get_price(FLUX, {"prompt": "cat", "seed": 2})
        assert first == second == Decimal("0.01")
        assert post.call_count == 1

    async def test_price_relevant_field_is_quoted_again(self):
        post = AsyncMock(side_effect=[_quote(0.01), _quote(0.04)])
        provider = _provider_with_types({})
        p1, p2 = _patched("post", post)
        with p1, p2:
            small = await provider.get_price(FLUX, {"prompt": "cat", "size": "512*512"})
            large = await provider.get_price(FLUX, {"prompt": "cat", "size": "2048*2048"})
        assert (small, large) == (Decimal("0.01"), Decimal("0.04"))
        assert post.call_count == 2

    async def test_prompt_stays_in_the_key_for_a_model_not_in_the_catalog(self):
        """Without a type there is no telling it is not priced on its text."""
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({})
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat"})
            await provider.get_price(FLUX, {"prompt": "dog"})
        # One quote each, and no probe
        assert post.call_count == 2

    async def test_prompt_stays_in_the_key_for_a_speech_model(self):
        post = AsyncMock(return_value=_quote(0.03))
        provider = _provider_with_types({"minimax/speech": "text-to-audio"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price("minimax/speech", {"prompt": "short"})
            await provider.get_price("minimax/speech", {"prompt": "other"})
        assert post.call_count == 2

    async def test_image_model_that_ignores_the_prompt_reuses_the_quote(self):
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            first = await provider.get_price(FLUX, {"prompt": "cat", "size": "512*512"})
            await _probes_done(provider)
            second = await provider.get_price(FLUX, {"prompt": "a dog", "size": "512*512"})
        assert first == second == Decimal("0.01")
        # The real quote and the long-prompt probe, then nothing
        bodies = _pricing_bodies(post)
        assert len(bodies) == 2
        assert {len(body["inputs"]["prompt"]) > 1000 for body in bodies} == {False, True}
        assert all(body["inputs"]["size"] == "512*512" for body in bodies)

    async def test_ignored_prompt_still_quotes_other_fields(self):
        async def quote(url, json, **kwargs):
            return _quote(0.04 if json["inputs"]["size"] == "2048*2048" else 0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat", "size": "512*512"})
            await _probes_done(provider)
            large = await provider.get_price(FLUX, {"prompt": "dog", "size": "2048*2048"})
            await _probes_done(provider)
        assert large == Decimal("0.04")
        # Each size quoted and probed on its own
        assert post.call_count == 4

    async def test_image_model_priced_on_its_prompt_keeps_it_in_the_key(self):
        """A different price for a longer prompt means the prompt counts."""

        async def quote(url, json, **kwargs):
            return _quote(0.02 if len(json["inputs"]["prompt"]) > 1000 else 0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            first = await provider.get_price(FLUX, {"prompt": "cat"})
            await _probes_done(provider)
            second = await provider.get_price(FLUX, {"prompt": "dog"})
            again = await provider.get_price(FLUX, {"prompt": "dog", "seed": 7})
        assert first == second == again == Decimal("0.01")
        # Probe on the first miss only; "dog" quoted on its own; the seed reused it
        assert post.call_count == 3

    async def test_failed_probe_keeps_the_prompt_in_the_key(self):
        async def quote(url, json, **kwargs):
            if len(json["inputs"]["prompt"]) > 1000:
                return httpx.Response(400, text="prompt too long")
            return _quote(0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            assert await provider.get_price(FLUX, {"prompt": "cat"}) == Decimal("0.01")
            await _probes_done(provider)
            await provider.get_price(FLUX, {"prompt": "dog"})
        # Not probed again right away
        assert post.call_count == 3

    async def test_probe_is_sent_once(self):
        """A probe 5xx is not retried."""

        async def quote(url, json, **kwargs):
            if len(json["inputs"]["prompt"]) > 1000:
                return httpx.Response(503, text="busy")
            return _quote(0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat"})
            await _probes_done(provider)
        assert post.call_count == 2

    async def test_quote_expires(self):
        post = AsyncMock(side_effect=[_quote(0.01), _quote(0.02)])
        provider = _provider_with_types({})
        clock = [1000.0]
        p1, p2 = _patched("post", post)
        with p1, p2, patch("x402gate.providers.wavespeed.time.monotonic", lambda: clock[0]):
            assert await provider.get_price(FLUX, {"prompt": "cat"}) == Decimal("0.01")
            clock[0] += 61
            assert await provider.get_price(FLUX, {"prompt": "cat"}) == Decimal("0.02")

    async def test_failed_quote_is_not_cached(self):
        post = AsyncMock(side_effect=[httpx.Response(400, text="bad size"), _quote(0.01)])
        provider = _provider_with_types({})
        p1, p2 = _patched("post", post)
        with p1, p2:
            with pytest.raises(ProviderError):
                await provider.get_price(FLUX, {"prompt": "cat"})
            assert await provider.get_price(FLUX, {"prompt": "cat"}) == Decimal("0.01")

    async def test_cache_is_bounded(self):
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({})
        p1, p2 = _patched("post", post)
        with p1, p2, patch("x402gate.providers.wavespeed._PRICE_CACHE_MAX_ENTRIES", 3):
            for size in range(5):
                await provider.get_price(FLUX, {"size": str(size)})
        assert len(provider._prices) == 3

    async def test_failed_probe_is_checked_again_later(self):
        """One transient probe error does not keep the prompt in the key for an hour."""
        probe_answers = [httpx.Response(503, text="busy"), _quote(0.01)]

        async def quote(url, json, **kwargs):
            if len(json["inputs"]["prompt"]) > 1000:
                return probe_answers.pop(0)
            return _quote(0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        clock = [1000.0]
        p1, p2 = _patched("post", post)
        with p1, p2, patch("x402gate.providers.wavespeed.time.monotonic", lambda: clock[0]):
            await provider.get_price(FLUX, {"prompt": "cat"})
            await _probes_done(provider)
            clock[0] += 61
            await provider.get_price(FLUX, {"prompt": "dog"})
            await _probes_done(provider)
            await provider.get_price(FLUX, {"prompt": "bird"})
        # cat + failed probe, dog + probe, then bird from the cache
        assert post.call_count == 4
        assert probe_answers == []

    async def test_check_holds_only_for_the_other_fields_it_ran_with(self):
        """A prompt may count only while another option is on, so that combination is probed."""

        async def quote(url, json, **kwargs):
            inputs = json["inputs"]
            if inputs.get("enable_prompt_expansion") and len(inputs["prompt"]) > 1000:
                return _quote(0.05)
            return _quote(0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        expand = {"size": "512*512", "enable_prompt_expansion": True}
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat", "size": "512*512"})
            await _probes_done(provider)
            await provider.get_price(FLUX, {"prompt": "cat", **expand})
            await _probes_done(provider)
            await provider.get_price(FLUX, {"prompt": "dog", **expand})
        bodies = _pricing_bodies(post)
        # The new combination was quoted and probed, and its prompt then counted
        assert len(bodies) == 5
        assert bodies[-1]["inputs"]["prompt"] == "dog"

    async def test_quote_does_not_wait_for_the_probe(self):
        hang = asyncio.Event()

        async def quote(url, json, **kwargs):
            if len(json["inputs"]["prompt"]) > 1000:
                await hang.wait()
            return _quote(0.01)

        post = AsyncMock(side_effect=quote)
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2:
            price = await asyncio.wait_for(provider.get_price(FLUX, {"prompt": "cat"}), 0.5)
            assert price == Decimal("0.01")
            assert len(provider._probes) == 1
            # A second miss while it runs starts no second probe
            await provider.get_price(FLUX, {"prompt": "dog"})
            assert len(provider._probes) == 1
            hang.set()
            await _probes_done(provider)
            await provider.get_price(FLUX, {"prompt": "bird"})
        # cat, probe, dog, then bird from the cache once the probe agreed
        assert post.call_count == 3
        assert provider._probes == {}

    async def test_close_stops_a_running_probe(self):
        async def quote(url, json, **kwargs):
            if len(json["inputs"]["prompt"]) > 1000:
                await asyncio.Event().wait()
            return _quote(0.01)

        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", AsyncMock(side_effect=quote))
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat"})
            (probe,) = provider._probes.values()
            await provider.close()
        assert probe.cancelled()
        assert provider._text_checks == {}

    async def test_zero_ttl_turns_the_cache_off(self):
        """price_cache_ttl 0 means fresh quotes every time, and no probes."""
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({FLUX: "text-to-image"}, price_cache_ttl=0)
        p1, p2 = _patched("post", post)
        with p1, p2:
            await provider.get_price(FLUX, {"prompt": "cat"})
            await provider.get_price(FLUX, {"prompt": "cat"})
        assert post.call_count == 2
        assert provider._prices == {}

    async def test_text_checks_are_bounded(self):
        post = AsyncMock(return_value=_quote(0.01))
        provider = _provider_with_types({FLUX: "text-to-image"})
        p1, p2 = _patched("post", post)
        with p1, p2, patch("x402gate.providers.wavespeed._TEXT_CHECK_MAX_ENTRIES", 3):
            for size in range(5):
                await provider.get_price(FLUX, {"prompt": "cat", "size": str(size)})
                await _probes_done(provider)
        assert len(provider._text_checks) == 3


class TestCatalog:
    """The catalog gives model types; it loads at startup and on demand."""

    async def test_warm_up_loads_the_catalog(self):
        get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": [
                        {"model_id": FLUX, "type": "text-to-image"},
                        {"model_id": "minimax/speech", "type": "text-to-audio"},
                        {"model_id": "broken"},
                    ],
                },
            )
        )
        provider = _make_provider()
        to_thread = AsyncMock(side_effect=asyncio.to_thread)
        with (
            patch("httpx.AsyncClient.get", get),
            patch("x402gate.providers.wavespeed.asyncio.to_thread", to_thread),
        ):
            await provider.warm_up()
        assert get.await_args.args[0] == MODELS_URL
        assert to_thread.await_args.args[0] is _parse_catalog
        assert provider._model_types == {FLUX: "text-to-image", "minimax/speech": "text-to-audio"}

    async def test_failed_warm_up_raises_for_the_app_to_log(self):
        get = AsyncMock(return_value=httpx.Response(500, text="down"))
        provider = _make_provider()
        with patch("httpx.AsyncClient.get", get), pytest.raises(ProviderError):
            await provider.warm_up()
        assert provider._model_types == {}

    async def test_missing_catalog_loads_in_the_background(self):
        get = AsyncMock(
            return_value=httpx.Response(
                200, json={"data": [{"model_id": FLUX, "type": "text-to-image"}]}
            )
        )
        post = AsyncMock(return_value=_quote(0.01))
        provider = _make_provider()
        with patch("httpx.AsyncClient.get", get), patch("httpx.AsyncClient.post", post):
            await provider.get_price(FLUX, {"prompt": "cat"})
            await provider._catalog_task
            await provider.get_price(FLUX, {"prompt": "dog"})
            await _probes_done(provider)
        assert get.call_count == 1
        assert provider._model_types == {FLUX: "text-to-image"}
        # The typed model is probed now
        assert post.call_count == 3

    async def test_failed_catalog_load_is_not_retried_at_once(self):
        get = AsyncMock(return_value=httpx.Response(500, text="down"))
        post = AsyncMock(return_value=_quote(0.01))
        provider = _make_provider()
        with patch("httpx.AsyncClient.get", get), patch("httpx.AsyncClient.post", post):
            await provider.get_price(FLUX, {"prompt": "cat"})
            await provider._catalog_task
            await provider.get_price(FLUX, {"prompt": "dog"})
        assert get.call_count == 1

    async def test_close_waits_for_a_catalog_refresh(self):
        async def slow_get(*args, **kwargs):
            await asyncio.Event().wait()

        provider = _make_provider()
        with (
            patch("httpx.AsyncClient.get", AsyncMock(side_effect=slow_get)),
            patch("httpx.AsyncClient.post", AsyncMock(return_value=_quote(0.01))),
        ):
            await provider.get_price(FLUX, {"prompt": "cat"})
            task = provider._catalog_task
            await asyncio.sleep(0)
            await provider.close()
        assert task.done()


class TestFinishedAtSubmit:
    """A result the submit already returned is not polled for again."""

    async def test_completed_submit_is_returned_without_polling(self):
        done = {"id": "t1", "status": "completed", "outputs": ["https://out.png"]}
        post = AsyncMock(return_value=httpx.Response(200, json={"code": 200, "data": done}))
        get = AsyncMock(side_effect=AssertionError("polled"))
        provider = _make_provider()
        with patch("httpx.AsyncClient.post", post), patch("httpx.AsyncClient.get", get):
            await provider.submit(FLUX, {"prompt": "x", "enable_sync_mode": True})
            result = await provider.get_result("t1")
        assert result == done
        assert get.call_count == 0

    async def test_kept_result_is_returned_once(self):
        done = {"id": "t1", "status": "completed", "outputs": ["https://out.png"]}
        post = AsyncMock(return_value=httpx.Response(200, json={"data": done}))
        get = AsyncMock(return_value=httpx.Response(200, json={"data": done}))
        provider = _make_provider()
        p1, p2 = _patched("get", get)
        with patch("httpx.AsyncClient.post", post), p1, p2:
            await provider.submit(FLUX, {"prompt": "x"})
            await provider.get_result("t1")
            await provider.get_result("t1")
        assert get.call_count == 1

    @pytest.mark.parametrize(
        "data",
        [
            # The sync wait ran out; the task is still running
            {
                "id": "t1",
                "status": "processing",
                "code": 5004,
                "outputs": [],
                "error": "Sync mode timed out while the prediction is still processing.",
            },
            {"id": "t1", "status": "created"},
            {"id": "t1", "status": "completed", "outputs": []},
        ],
    )
    async def test_unfinished_submit_is_polled(self, data):
        post = AsyncMock(return_value=httpx.Response(200, json={"data": data}))
        polled = {"id": "t1", "status": "completed", "outputs": ["https://out.png"]}
        get = AsyncMock(return_value=httpx.Response(200, json={"data": polled}))
        provider = _make_provider()
        p1, p2 = _patched("get", get)
        with patch("httpx.AsyncClient.post", post), p1, p2:
            await provider.submit(FLUX, {"prompt": "x"})
            result = await provider.get_result("t1")
        assert result == polled
        assert get.call_count == 1

    async def test_kept_results_are_bounded(self):
        provider = _make_provider()
        with patch("x402gate.providers.wavespeed._FINISHED_MAX_ENTRIES", 2):
            for n in range(4):
                provider._remember_if_finished(
                    {"id": f"t{n}", "status": "completed", "outputs": ["x"]}
                )
        assert list(provider._finished) == ["t2", "t3"]
