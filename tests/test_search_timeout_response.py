"""Remote search failures remain plain data without client transport dependencies."""

from types import SimpleNamespace

from retrieval_service.embedded import EmbeddedSearch


def test_msearch_timeout_preserves_invalid_slots(tmp_path):
    """Return ordered structured errors after one failed transport call, without retries."""
    calls = []

    def fail(**kwargs):
        """Simulate an Elasticsearch timeout without starting a server."""
        calls.append(kwargs)
        raise TimeoutError("read deadline exceeded")

    engine = EmbeddedSearch(tmp_path)
    engine.manifest = {"corpus_version": "test"}
    engine.es = SimpleNamespace(msearch=fail)
    results = engine.search_batch([{"query": "one"}, {"query": ""}, {"query": "two"}])
    assert len(calls) == 1
    assert [r["code"] for r in results] == [
        "search_unavailable",
        "invalid_request",
        "search_unavailable",
    ]
    assert results[0]["batch_queries"] == 2
    assert "TimeoutError" in results[0]["message"]
    assert results[0]["seconds"] >= 0
