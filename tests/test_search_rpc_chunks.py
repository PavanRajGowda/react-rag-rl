"""Large logical rollout pools retain membership through streaming map transport."""

import asyncio
from types import SimpleNamespace

import pytest

from tinker_training.__main__ import search_client


def endpoint(calls, fail_middle=False):
    """Stream tagged local results; no RPCs or retries are performed."""

    async def mapped(inputs, **settings):
        assert settings == {"order_outputs": False, "return_exceptions": False}
        async for item in inputs:
            payload = item["request"]
            calls.append(payload)
            failed = fail_middle and 1024 <= int(payload["query"]) < 1536
            yield {
                "request_id": item["request_id"],
                "response": TimeoutError("injected") if failed else payload,
            }

    return SimpleNamespace(map=SimpleNamespace(aio=mapped))


@pytest.mark.parametrize("fail_middle", [False, True])
def test_search_map_preserves_large_pool_membership(fail_middle):
    """All 2,560 inputs retain their slots; selected failures cannot shift peers."""

    async def scenario():
        calls = []
        template = search_client()
        template._endpoint = endpoint(calls, fail_middle)
        async with template.open_session() as search:
            results = await search.batch([{"query": str(i)} for i in range(2560)])
        assert len(calls) == len(results) == 2560
        for i, result in enumerate(results):
            if fail_middle and 1024 <= i < 1536:
                assert isinstance(result, TimeoutError)
            else:
                assert result["query"] == str(i)
        assert len({p["query"] for p in calls}) == 2560

    asyncio.run(scenario())


def test_map_deduplication_retains_top_k_and_independent_results():
    """Convenience batch deduplication still keys on the entire payload."""

    async def scenario():
        calls = []
        template = search_client()
        template._endpoint = endpoint(calls)
        async with template.open_session() as search:
            results = await search.batch(
                [
                    {"query": "1", "top_k": 2},
                    {"query": "1", "top_k": 3},
                    {"query": "1", "top_k": 2},
                ]
            )
        assert len(calls) == 2
        assert [r["top_k"] for r in results] == [2, 3, 2]
        results[0]["top_k"] = 99
        assert results[2]["top_k"] == 2

    asyncio.run(scenario())
