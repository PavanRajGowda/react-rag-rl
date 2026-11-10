from baseline_agent.retrieval_diagnostics import annotate_retrieval


def episode(mode="append", gold=("a", "b")):
    """Build a baseline trace with scripted search/evidence chronology for retrospective
    retrieval metrics.
    """
    return {
        "config": {"context_mode": mode},
        "turns": [{"turn": i} for i in (1, 2, 3)],
        "events": [],
        "searches": [
            {
                "turn": 1,
                "response": {"results": [{"context_id": "a0", "document_id": "a"}]},
            },
            {
                "turn": 2,
                "response": {"results": [{"context_id": "b0", "document_id": "b"}]},
            },
        ],
    }


def test_complete_evidence_accumulates_and_null_is_na():
    """Verify complete evidence accumulates and null is na."""
    r = annotate_retrieval(episode(), ("a", "b"))
    assert r["turns"][0]["retrieval_diagnostic"]["evidence_recall"] == 0.5
    assert r["turns"][2]["retrieval_diagnostic"]["gold_context_in_prompt"] is True
    assert r["retrieval_validation"]["gold_context_retrieved_ever"] is True
    assert (
        annotate_retrieval(episode(), ())["retrieval_validation"]["evidence_recall"]
        is None
    )


def test_replace_distinguishes_ever_and_active():
    """Verify replace distinguishes ever and active."""
    r = annotate_retrieval(episode("replace"), ("a", "b"))
    assert r["retrieval_validation"]["gold_context_retrieved_ever"] is True
    assert r["turns"][-1]["retrieval_diagnostic"]["gold_context_in_prompt"] is False


def test_bounded_uses_actual_visible_inventory_after_eviction():
    """Verify bounded uses actual visible inventory after eviction."""
    sample = episode("bounded")
    sample["turns"][0]["visible_evidence"] = []
    sample["turns"][1]["visible_evidence"] = [{"document_id": "a"}]
    sample["turns"][2]["visible_evidence"] = [{"document_id": "b"}]
    sample["events"] = [
        {"kind": "model"},
        {
            "kind": "information",
            "search_turn": 1,
            "visible_evidence": [{"document_id": "a"}],
        },
        {"kind": "model"},
        {
            "kind": "information",
            "search_turn": 2,
            "visible_evidence": [{"document_id": "b"}],
        },
        {"kind": "model"},
    ]
    result = annotate_retrieval(sample, ("a", "b"))
    assert result["retrieval_validation"]["gold_context_retrieved_ever"] is True
    assert result["turns"][2]["retrieval_diagnostic"]["gold_context_in_prompt"] is False
    assert (
        result["turns"][1]["retrieval_diagnostic"]["gold_context_active_after_turn"]
        is False
    )


def test_export_keeps_document_lists_and_f1(tmp_path):
    """Verify export keeps document lists and f1."""
    import json

    from baseline_agent.evaluation import export_rows, record_result

    result = episode() | {
        "question": {"id": "q", "question": "Who?", "example_index": 0},
        "answer": "Aster",
        "status": "answered",
        "inference_seconds": 1,
        "model_revision": "revision",
    }
    for search in result["searches"]:
        search.update(seconds=0.1, request={"query": "Aster"})
    row = record_result(result, "Aster", tmp_path, ["a", "b"])
    summary = export_rows([row], tmp_path)
    assert row["token_f1"] == 1 and summary["gold_context_retrieval_rate"] == 1
    assert "retrieved_document_ids" in (tmp_path / "turns.csv").read_text()
    assert "boolean_mapping" not in json.loads((tmp_path / "summary.json").read_text())
