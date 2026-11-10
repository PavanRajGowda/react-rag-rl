import pytest

from baseline_agent.evidence import EvidenceStore, information_event, visible_evidence
from baseline_agent.protocol import select_history
from tests.helpers import passage


def response(*keys):
    """Build a successful retrieval response for testing deduplication, retention and
    evidence limits.
    """
    return {
        "status": "ok",
        "results": [passage() | {"context_id": k, "text": k} for k in keys],
    }


def test_fifo_dedup_keeps_distinct_passages_of_same_article():
    """Verify fifo dedup keeps distinct passages of same article."""
    store = EvidenceStore(2, 12000)
    first = information_event(response("one", "two"), store, "bounded")
    second = information_event(response("one", "three"), store, "bounded")
    assert [x["context_id"] for x in second["visible_evidence"]] == ["two", "three"]
    assert second["evicted_context_ids"] == ["one"]
    events = [first, {"kind": "model", "text": "search"}, second]
    assert select_history(events, "bounded") == events[1:]
    assert visible_evidence(events, "bounded") == second["visible_evidence"]
    assert all(len(x["content_sha256"]) == 64 for x in second["visible_evidence"])


def test_empty_and_error_search_preserve_evidence_and_explain():
    """Verify empty and error search preserve evidence and explain."""
    store = EvidenceStore(4, 12000)
    first = store.update(response("one"))
    empty = store.update(response())
    error = store.update({"status": "error", "message": "Query invalid"})
    assert (
        first["visible_evidence"]
        == empty["visible_evidence"]
        == error["visible_evidence"]
    )
    assert "no matching passages" in empty["text"]
    assert "Query invalid" in error["text"]


def test_whole_passage_eviction_and_content_integrity():
    """Verify whole passage eviction and content integrity."""
    store = EvidenceStore(4, 300)
    store.update(response("one"))
    event = store.update(
        {
            "status": "ok",
            "results": [passage() | {"context_id": "huge", "text": "x" * 1000}],
        }
    )
    assert event["visible_evidence"] == []
    assert event["evicted_context_ids"] == ["one", "huge"]
    with pytest.raises(ValueError, match="content changed"):
        store.update(
            {
                "status": "ok",
                "results": [passage() | {"context_id": "one", "text": "changed"}],
            }
        )


def test_append_history_deduplicates_without_eviction_or_rewriting():
    """Verify append history deduplicates without eviction or rewriting."""
    store = EvidenceStore(1, 1)  # Legacy limits do not constrain append mode.
    first = information_event(response("one", "two"), store, "append")
    saved_text = first["text"]
    second = information_event(response("one", "three"), store, "append")
    events = [first, {"kind": "model", "text": "next search"}, second]
    assert select_history(events, "append") == events
    assert first["text"] == saved_text
    assert [item["context_id"] for item in second["visible_evidence"]] == ["three"]
    assert [item["context_id"] for item in visible_evidence(events, "append")] == [
        "one",
        "two",
        "three",
    ]
    assert second["evicted_document_ids"] == []
    repeated = information_event(response("one"), store, "append")
    assert repeated["visible_evidence"] == []
    assert "no new passages" in repeated["text"]
    assert len(visible_evidence(events + [repeated], "append")) == 3


def test_append_detects_changed_passage_contents():
    """Verify append detects changed passage contents."""
    store = EvidenceStore(1, 1)
    information_event(response("one"), store, "append")
    changed = response("one")
    changed["results"][0]["text"] = "different content"
    with pytest.raises(ValueError, match="content changed"):
        information_event(changed, store, "append")
