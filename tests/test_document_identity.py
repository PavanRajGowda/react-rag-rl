from baseline_agent.evidence import EvidenceStore, information_event, visible_evidence
from baseline_agent.protocol import format_information
from retrieval_service.hotpotqa import paragraph
from tinker_training.process_events import events


def document(key):
    """Return a document-only HotpotQA retrieval fixture with no legacy passage/chunk
    identifier.
    """
    return {
        "document_id": key,
        "title": key,
        "text": "Supporting evidence.",
        "url": "https://en.wikipedia.org/" + key,
    }


def test_document_only_history_and_rendering():
    """Verify document only history and rendering."""
    docs = [document("wiki-1"), document("wiki-2")]
    store = EvidenceStore(1, 1)
    first = information_event({"status": "ok", "results": docs}, store, "append")
    second = information_event(
        {"status": "ok", "results": [docs[0], document("wiki-3")]}, store, "append"
    )
    visible = visible_evidence([first, second], "append")
    assert [item["document_id"] for item in visible] == ["wiki-1", "wiki-2", "wiki-3"]
    assert all("context_id" not in item for item in visible)
    assert first["evicted_document_ids"] == []
    assert "[wiki-1]" in format_information({"status": "ok", "results": docs})


def test_acquisition_is_one_binary_bonus_not_article_plus_passage():
    """Verify acquisition is one binary bonus not article plus passage."""
    generations = [
        {"turn": 1, "text": "<search>first</search>", "tokens": 3},
        {"turn": 2, "text": "<search>again</search>", "tokens": 3},
        {"turn": 3, "text": "<answer>yes</answer>", "tokens": 3},
    ]
    audit = {
        "searches": [
            {
                "turn": 1,
                "response": {"status": "ok", "results": [document("a"), document("b")]},
            },
            {"turn": 2, "response": {"status": "ok", "results": [document("c")]}},
        ]
    }
    turns = events(generations, audit, ["a", "b"])
    assert [turn["g"] for turn in turns] == [1, 0, 0]
    assert turns[0]["document_acquisition"] == 1
    assert "passage_acquisition" not in turns[0]
    assert turns[1]["after_coverage"] and turns[1]["Q"] == 1
    assert not turns[1]["no_new_documents"]


def test_prepared_abstract_has_no_chunk_identifiers():
    """Verify prepared abstract has no chunk identifiers."""
    row = paragraph(
        {
            "id": "1",
            "title": "Title",
            "url": "https://example.org",
            "text": ["Sentence."],
        }
    )
    assert row["document_id"] == "wiki-1"
    assert "context_id" not in row and "chunk_index" not in row


def test_bounded_compatibility_mode_logs_document_evictions():
    """Verify bounded compatibility mode logs document evictions."""
    store = EvidenceStore(1, 10000)
    event = store.update({"status": "ok", "results": [document("a"), document("b")]})
    assert event["evicted_document_ids"] == ["a"]
    assert "evicted_context_ids" not in event
    assert event["visible_evidence"][0]["document_id"] == "b"
