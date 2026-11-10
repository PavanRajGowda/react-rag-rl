"""Correlated timing coverage and isolation from retrieval/reward contracts."""

import asyncio
from types import SimpleNamespace

import pytest

from retrieval_service.async_client import AsyncSearchClient
from retrieval_service.dynamic_batching import dispatch_envelopes
from retrieval_service.embedded import EmbeddedSearch
from retrieval_service.timing import (
    SearchTrace,
    finish_timing,
    search_context,
    validate_timing,
)


def test_client_segments_are_additive_and_clock_skew_is_not_hidden():
    """Only correlated durations are subtracted; negative UTC estimates stay visible."""
    row = finish_timing(
        {
            "created_ns": 0,
            "enqueued_ns": 1_000_000_000,
            "sdk_handoff_ns": 3_000_000_000,
            "result_observed_ns": 10_000_000_000,
            "resumed_ns": 11_000_000_000,
            "sdk_handoff_utc_ns": 5_000_000_000,
            "result_observed_utc_ns": 12_000_000_000,
            "server": {
                "batch_function_seconds": 4,
                "batch_entered_utc_ns": 4_000_000_000,
                "batch_returning_utc_ns": 8_000_000_000,
            },
        }
    )
    assert (
        sum(
            row[k]
            for k in (
                "client_prepare_seconds",
                "client_queue_seconds",
                "sdk_handoff_to_result_seconds",
                "resume_delay_seconds",
            )
        )
        == row["client_total_seconds"]
        == 11
    )
    assert row["outside_server_function_seconds"] == 3
    assert row["approx_submission_to_batch_entry_seconds"] == -1
    with pytest.raises(RuntimeError, match="readiness"):
        validate_timing(row)


def test_envelope_dedup_keeps_telemetry_outside_evidence():
    """Different request IDs still share one query and retain independent response objects."""
    calls = []

    async def execute(payloads, trace_id=None):
        calls.append(payloads)
        assert payloads == [{"query": "a"}, {"query": "b"}]
        return {
            "responses": [{"status": "ok", "results": [q]} for q in payloads],
            "timing": {
                "batch_id": trace_id,
                "query_took_ms": [11, 22],
                "service_total_seconds": 0.01,
            },
        }

    inputs = [
        {"request_id": "1", "request": {"query": "a"}, "trace": True},
        {"request_id": "2", "request": {"query": "a"}, "trace": True},
        {"query": "b"},
    ]
    outputs, metrics = asyncio.run(dispatch_envelopes(inputs, execute))
    assert len(calls) == 1 and metrics["unique_queries"] == 2
    assert outputs[0]["response"] == outputs[1]["response"]
    assert outputs[0]["timing"]["batch_id"] == outputs[1]["timing"]["batch_id"]
    assert outputs[0]["timing"]["elasticsearch"]["query_took_ms"] == 11
    assert "timing" not in outputs[2]
    outputs[0]["response"]["results"].clear()
    assert outputs[1]["response"]["results"]


def test_engine_telemetry_preserves_results_and_failed_query_slots(tmp_path):
    """Validation and query timings align with original slots; transport failures remain data."""
    engine = EmbeddedSearch(tmp_path)
    engine.manifest = {"corpus_version": "test"}
    engine.es = SimpleNamespace(
        msearch=lambda **kw: {
            "responses": [
                {"took": 7, "hits": {"hits": []}},
                {"error": {"reason": "bad"}},
            ]
        }
    )
    payloads = [{"query": "a"}, {"query": ""}, {"query": "b"}]
    plain = engine.search_batch(payloads)
    timing = {}
    assert engine.search_batch(payloads, timing=timing) == plain
    assert timing["query_took_ms"] == [7, None, None]
    assert all(
        timing[k] >= 0
        for k in (
            "msearch_seconds",
            "format_seconds",
            "validation_prepare_seconds",
            "service_total_seconds",
        )
    )

    def fail(**kw):
        raise TimeoutError("injected")

    engine.es.msearch = fail
    timing = {}
    results = engine.search_batch(payloads, timing=timing)
    assert results[0]["code"] == "search_unavailable"
    assert timing["msearch_seconds"] >= 0 and timing["service_total_seconds"] >= 0
    assert timing["query_took_ms"] == [None] * 3


@pytest.mark.parametrize("cancel", [False, True])
def test_trace_correlates_turns_heartbeat_and_cancelled_late_outputs(tmp_path, cancel):
    """An abandoned traced request cannot break peers; evidence never includes timing fields."""
    trace = SearchTrace(tmp_path)

    async def scenario():
        queue = asyncio.Queue()
        arrived, release = asyncio.Event(), asyncio.Event()
        tasks = []

        async def handle(item):
            if item["request"]["query"] == "slow":
                arrived.set()
                await release.wait()
            outputs, _ = await dispatch_envelopes([item], execute)
            await queue.put(outputs[0])

        async def execute(payloads, trace_id):
            return {
                "responses": [{"status": "ok", "results": []}],
                "timing": {
                    "batch_id": trace_id,
                    "service_total_seconds": 0.001,
                    "msearch_seconds": 0.0005,
                    "format_seconds": 0.0001,
                    "query_took_ms": [0],
                },
            }

        async def mapped(inputs, **options):
            async def produce():
                async for item in inputs:
                    tasks.append(asyncio.create_task(handle(item)))

            producer = asyncio.create_task(produce())
            try:
                while True:
                    yield await queue.get()
            finally:
                producer.cancel()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(producer, *tasks, return_exceptions=True)

        endpoint = SimpleNamespace(map=SimpleNamespace(aio=mapped))
        async with AsyncSearchClient(endpoint, trace_sink=trace) as client:
            with search_context("episode", 2):
                slow = asyncio.create_task(client({"query": "slow"}))
            await arrived.wait()
            if cancel:
                slow.cancel()
                await asyncio.gather(slow, return_exceptions=True)
            release.set()
            assert await client({"query": "fast"}) == {"status": "ok", "results": []}
            if not cancel:
                await slow
            await asyncio.sleep(0.06)
        assert client._heartbeat_task.done()

    asyncio.run(scenario())
    assert len(trace.requests) == 2 and trace.heartbeats
    rows = {r["episode_id"]: r for r in trace.requests if "episode_id" in r}
    assert rows["episode"]["turn"] == 2
    for row in trace.requests:
        if "error" not in row:
            validate_timing(row)
    trace.save()
    assert (tmp_path / "search-timings.json").exists()


def test_traced_downstream_failure_keeps_failure_slots_and_partial_timings():
    """Failed RPCs preserve available boundaries instead of inventing ES measurements."""

    async def fail(payloads, trace_id):
        raise TimeoutError("injected")

    inputs = [{"request_id": "a", "request": {"query": "x"}, "trace": True}]
    outputs, _ = asyncio.run(dispatch_envelopes(inputs, fail))
    output = outputs[0]
    assert output["response"]["code"] == "search_unavailable"
    assert output["timing"]["downstream_rpc_seconds"] >= 0
    assert output["timing"]["elasticsearch"]["query_took_ms"] is None
    assert "service_total_seconds" not in output["timing"]["elasticsearch"]
