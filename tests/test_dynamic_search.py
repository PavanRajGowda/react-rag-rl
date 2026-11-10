"""Native batching adapter contracts; no Modal deployment or Elasticsearch calls."""

import asyncio
from types import SimpleNamespace

import pytest

from retrieval_service.async_client import AsyncSearchClient
from retrieval_service.dynamic_batching import BatchSettings, dispatch


def test_settings_are_configurable_and_bounded():
    """Only valid batches fitting the existing 512-query RPC may be deployed."""
    assert BatchSettings.from_env(
        {"SEARCH_DYNAMIC_BATCH_SIZE": "64", "SEARCH_DYNAMIC_WAIT_MS": "25"}
    ) == BatchSettings(64, 25)
    for size, wait in [(0, 10), (513, 10), (1, -1), (True, 10)]:
        with pytest.raises(ValueError):
            BatchSettings(size, wait)


def test_dispatch_deduplicates_full_payloads_and_preserves_errors_and_order():
    """Equal queries with different top_k remain distinct; results are independent."""
    calls = []

    async def execute(requests):
        calls.append(requests)
        return [
            {"status": "error", "code": "invalid_request"}
            if not r["query"]
            else {"status": "ok", "results": [{"k": r["top_k"]}]}
            for r in requests
        ]

    payloads = [
        {"query": "a", "top_k": 2},
        {"top_k": 2, "query": "a"},
        {"query": "", "top_k": 2},
        {"query": "a", "top_k": 3},
    ]
    results, metrics = asyncio.run(dispatch(payloads, execute))
    assert len(calls) == 1 and len(calls[0]) == 3
    assert results[2]["code"] == "invalid_request"
    assert results[3]["results"][0]["k"] == 3
    results[0]["results"][0]["k"] = 99
    assert results[1]["results"][0]["k"] == 2
    assert metrics["requests"] == 4 and metrics["unique_queries"] == 3


@pytest.mark.parametrize("wrong_count", [False, True])
def test_dispatch_shared_failure_resolves_every_slot_without_retry(wrong_count):
    """Remote failures and membership violations become serializable error responses."""
    calls = []

    async def execute(requests):
        calls.append(1)
        if wrong_count:
            return []
        raise TimeoutError("injected")

    result, _ = asyncio.run(dispatch([{"query": "a"}, {"query": "b"}], execute))
    assert calls == [1]
    assert all(r["code"] == "search_unavailable" and r["retryable"] for r in result)


def test_map_stream_routes_out_of_order_and_accepts_later_searches():
    """A fast chain advances before a slow peer; one map invocation owns both."""

    async def scenario():
        calls = []
        release = asyncio.Event()

        async def mapped(inputs, **options):
            assert options == {"order_outputs": False, "return_exceptions": False}
            calls.append(1)
            outputs = asyncio.Queue()

            async def handle(item):
                query = item["request"]["query"]
                if query == "slow":
                    await release.wait()
                await outputs.put(
                    {
                        "request_id": item["request_id"],
                        "response": {"status": "ok", "query": query},
                    }
                )

            async def produce():
                async for item in inputs:
                    workers.append(asyncio.create_task(handle(item)))

            workers = []
            producer = asyncio.create_task(produce())
            try:
                while True:
                    yield await outputs.get()
            finally:
                producer.cancel()
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(producer, *workers, return_exceptions=True)

        async with AsyncSearchClient(
            SimpleNamespace(map=SimpleNamespace(aio=mapped))
        ) as client:

            async def fast():
                assert (await client({"query": "fast"}))["query"] == "fast"
                assert (await client({"query": "next"}))["query"] == "next"
                release.set()

            slow = asyncio.create_task(client({"query": "slow"}))
            await asyncio.wait_for(fast(), 2)
            assert (await slow)["query"] == "slow"
        assert calls == [1]
        assert client._worker.done()

    asyncio.run(scenario())


def test_map_transport_failure_fails_pending_and_future_calls_without_retry():
    """An untagged map failure cannot be attributed to one search safely."""

    async def scenario():
        async def mapped(inputs, **options):
            async for item in inputs:
                raise ConnectionError("injected stream failure")
                yield item

        async with AsyncSearchClient(
            SimpleNamespace(map=SimpleNamespace(aio=mapped))
        ) as client:
            results = await asyncio.gather(
                client({"query": "a"}), client({"query": "b"}), return_exceptions=True
            )
            assert all(isinstance(r, ConnectionError) for r in results)
            with pytest.raises(ConnectionError):
                await client({"query": "c"})

    asyncio.run(scenario())
