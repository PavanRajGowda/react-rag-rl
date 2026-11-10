from types import SimpleNamespace as NS

import pytest

from baseline_agent.agent import AgentConfig
from batched_inference.baseline import batched_agents
from tests.test_token_conversation import Tokenizer
from tinker_training.records import PolicyVersion


def test_async_search_and_remove_finished_agents():
    """Preserve completion and append-only token checks with individual async IO."""
    tok = Tokenizer()
    policy = PolicyVersion("a" * 40, "b" * 40, "base", 0, "c" * 64)
    waves, searches = [], []

    def generate(requests):
        """Return scripted per-turn completions and record generation-wave membership for
        scheduler assertions.
        """
        waves.append([r.episode_id for r in requests])
        outputs = []
        for r in requests:
            search = r.turn == 1 or (r.episode_id == "b" and r.turn == 2)
            text = (
                f"Find.</reasoning><search>{r.episode_id}</search>"
                if search
                else "Done.</reasoning><answer>yes</answer>"
            )
            ids = tuple(ord(c) + 1 for c in text)
            outputs.append(
                NS(request=r, completion_ids=ids, sampling_logprobs=(-1.0,) * len(ids))
            )
        return NS(generations=outputs, seconds=0.1)

    async def search(payload):
        """One async search response per requesting trajectory."""
        searches.append(payload["query"])
        return {
            "status": "ok",
            "results": [{"document_id": "d", "title": "T", "text": "Fact"}],
        }

    rows = list(
        batched_agents(
            [
                {"id": k, "question": "Q", "example_index": i}
                for i, k in enumerate(["a", "b"])
            ],
            NS(generate=generate),
            tok,
            policy,
            AgentConfig(),
            search,
            2,
        )
    )
    assert waves == [["a"], ["b"], ["a"], ["b"], ["b"]]
    assert searches == ["a", "b", "b"]
    assert [r["answer"] for r, _ in rows] == ["yes", "yes"]
    assert [len(r["turns"]) for r, _ in rows] == [2, 3]
    for result, _ in rows:
        first, second = result["turns"][:2]
        prefix = first["prompt_ids"] + first["completion_ids"]
        assert second["prompt_ids"][: len(prefix)] == prefix


def test_msearch_keeps_invalid_and_failed_queries_in_place():
    """Verify msearch keeps invalid and failed queries in place."""
    from retrieval_service.embedded import EmbeddedSearch

    engine = EmbeddedSearch(".")
    engine.manifest = {"corpus_version": "test"}
    calls = []

    def msearch(**kwargs):
        """Return fake ordered Elasticsearch msearch responses while recording the request
        body.
        """
        assert all(
            header == {"request_cache": True} for header in kwargs["searches"][::2]
        )
        assert all(
            body["track_total_hits"] is False for body in kwargs["searches"][1::2]
        )
        calls.append(kwargs)
        return {
            "responses": [
                {"hits": {"hits": [], "total": {"value": 0}}},
                {"error": {"type": "failure"}},
            ]
        }

    engine.es = NS(msearch=msearch)
    responses = engine.search_batch([{"query": "ok"}, {"query": " "}, {"query": "bad"}])
    assert [r["status"] for r in responses] == ["ok", "error", "error"]
    assert responses[1]["code"] == "invalid_request"
    assert responses[2]["code"] == "search_unavailable"
    assert len(calls[0]["searches"]) == 4
    assert engine.search_batch([]) == []
    with pytest.raises(ValueError):
        engine.search_batch([{"query": "q"}] * 513)


def test_failed_search_only_rejects_its_trajectory():
    """Verify failed search only rejects its trajectory."""
    tok = Tokenizer()
    policy = PolicyVersion("a" * 40, "b" * 40, "base", 0, "c" * 64)
    waves = []

    def generate(requests):
        """Return scripted per-turn completions and record generation-wave membership for
        scheduler assertions.
        """
        waves.append([r.episode_id for r in requests])
        output = []
        for r in requests:
            text = (
                f"Find.</reasoning><search>{r.episode_id}</search>"
                if r.turn == 1
                else "Done.</reasoning><answer>yes</answer>"
            )
            ids = tuple(ord(c) + 1 for c in text)
            output.append(
                NS(request=r, completion_ids=ids, sampling_logprobs=(-1.0,) * len(ids))
            )
        return NS(generations=output, seconds=0.1)

    async def search(payload):
        """A failed request belongs only to its requesting trajectory."""
        if payload["query"] == "a":
            return {"status": "error", "code": "search_unavailable"}
        return {
            "status": "ok",
            "results": [{"document_id": "d", "title": "T", "text": "Fact"}],
        }

    rows = list(
        batched_agents(
            [
                {"id": k, "question": "Q", "example_index": i}
                for i, k in enumerate(["a", "b"])
            ],
            NS(generate=generate),
            tok,
            policy,
            AgentConfig(),
            search,
            2,
        )
    )
    results = {r["question"]["id"]: r for r, _ in rows}
    assert results["a"]["status"] == "infrastructure_error"
    assert results["b"]["answer"] == "yes"
    assert waves == [["a"], ["b"], ["b"]]
