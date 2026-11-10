"""Async trajectory scheduling and hosted sampling contracts; no remote calls."""

import asyncio
from types import SimpleNamespace as NS

from baseline_agent.agent import AgentConfig
from batched_inference.baseline import async_agents
from tests.test_token_conversation import Tokenizer
from tinker_training.backend import Batch, Failure, Generation
from tinker_training.records import PolicyVersion


def setup():
    """Return an isolated toy tokenizer, policy and three distinct questions."""
    return (
        Tokenizer(),
        PolicyVersion("a" * 40, "b" * 40, "base", 0, "c" * 64),
        [{"id": k, "question": "Q", "example_index": i} for i, k in enumerate("abc")],
    )


def completion(request, tok):
    """Provide a search followed by an accepted answer using exact fixture IDs."""
    text = (
        "Find.</reasoning><search>" + request.episode_id + "</search>"
        if request.turn == 1
        else "Done.</reasoning><answer>yes</answer>"
    )
    ids = tuple(tok.encode(text, add_special_tokens=False))
    gen = Generation(request, ids, (-1.0,) * len(ids), "action")
    return gen, Batch((gen,), 0.123)


def evidence():
    """Return a successful retrieved document to permit final answers."""
    return {
        "status": "ok",
        "results": [{"document_id": "d", "title": "T", "text": "Fact"}],
    }


def test_fast_trajectory_finishes_while_other_is_waiting_on_generation_and_search():
    """Neither slow sampling nor slow retrieval creates a wave barrier."""

    async def scenario():
        tok, policy, qs = setup()
        slow_sample, slow_search = asyncio.Event(), asyncio.Event()
        pending_search = asyncio.Event()

        class Backend:
            async def generate_async(self, request):
                if request.episode_id == "a" and request.turn == 1:
                    await slow_sample.wait()
                return completion(request, tok)

        async def search(payload):
            if payload["query"] == "b":
                pending_search.set()
                await slow_search.wait()
            return evidence()

        stream = async_agents(qs, Backend(), tok, policy, AgentConfig(), search, 3)
        try:
            result, measurements = await asyncio.wait_for(anext(stream), 2)
            assert result["question"]["id"] == "c"
            assert pending_search.is_set()
            assert result["status"] == "answered"
            assert len(measurements) == 2
            assert all(t["generation_seconds"] == 0.123 for t in result["turns"])
            slow_sample.set()
            slow_search.set()
            rest = [r async for r in stream]
            assert {r[0]["question"]["id"] for r in rest} == {"a", "b"}
        finally:
            await stream.aclose()

    asyncio.run(scenario())


def test_concurrency_slot_refills_and_early_close_cancels_waiters():
    """Completed trajectories release slots immediately; early consumer exit cleans up."""

    async def scenario():
        tok, policy, qs = setup()
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        started = []

        class Backend:
            async def generate_async(self, request):
                started.append(request.episode_id)
                if request.episode_id == "a":
                    waiting.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                return completion(request, tok)

        async def search(payload):
            return evidence()

        stream = async_agents(qs, Backend(), tok, policy, AgentConfig(), search, 2)
        r, _ = await asyncio.wait_for(anext(stream), 2)
        assert r["question"]["id"] == "b"
        r, _ = await asyncio.wait_for(anext(stream), 2)
        assert r["question"]["id"] == "c"
        assert waiting.is_set()
        await stream.aclose()
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_sampling_and_search_errors_are_isolated_and_do_not_retry():
    """A failed generation and a failed retrieval leave a healthy peer intact."""

    async def scenario():
        tok, policy, qs = setup()
        calls = []

        class Backend:
            async def generate_async(self, request):
                calls.append((request.episode_id, request.turn))
                if request.episode_id == "a":
                    item = Failure(request, TimeoutError("sample timeout"))
                    return item, Batch((item,), 0.2)
                return completion(request, tok)

        async def search(payload):
            if payload["query"] == "b":
                raise ConnectionError("search failure")
            return evidence()

        rows = {
            r["question"]["id"]: r
            async for r, _ in async_agents(
                qs, Backend(), tok, policy, AgentConfig(), search, 3
            )
        }
        assert rows["a"]["status"] == rows["b"]["status"] == "infrastructure_error"
        assert rows["c"]["status"] == "answered"
        assert calls.count(("a", 1)) == calls.count(("b", 1)) == 1

    asyncio.run(scenario())


def test_native_tinker_adapter_awaits_sample_async_and_preserves_tokens():
    """Concurrent sampler calls overlap and share the exact replay-token decoder."""
    from batched_inference.baseline import BatchGenerator
    from tests.agent_fixtures import ToyTokenizer
    from tests.test_tinker_migration import TYPES, settings
    from tinker_training.backend import TinkerBackend
    from tinker_training.records import SamplingParameters

    async def scenario():
        tok = ToyTokenizer()
        cfg = settings()
        started = []
        release = asyncio.Event()

        async def sample_async(**kwargs):
            started.append(kwargs)
            if len(started) == 2:
                release.set()
            await release.wait()
            ids = tok.encode(
                "Find.</reasoning><search>facts</search>", add_special_tokens=False
            )
            return NS(sequences=[NS(tokens=ids, logprobs=[-1.0] * len(ids))])

        backend = TinkerBackend(NS(sample_async=sample_async), tok, cfg, types=TYPES)
        requests = []
        for i in range(2):
            generator = BatchGenerator(
                tok,
                cfg,
                backend.policy,
                {"id": str(i), "example_index": i},
                SamplingParameters(1.0, 1.0),
            )
            prefix = generator.make_prefix("Q")
            requests.append(generator.request(prefix))
        results = await asyncio.wait_for(
            asyncio.gather(*(backend.generate_async(r) for r in requests)), 2
        )
        assert len(started) == 2
        assert all(
            p["sampling_params"].stop == ["</search>", "</answer>"] for p in started
        )
        for (gen, measurement), request in zip(results, requests):
            assert gen.request is request
            assert len(gen.completion_ids) == len(gen.sampling_logprobs)
            assert measurement.seconds > 0

    asyncio.run(scenario())
