import json

import pytest
from fastapi.testclient import TestClient

from baseline_agent.dashboard import create_app
from baseline_agent.finqa_reasoning import join_row, program_operations


def example():
    label = {
        "id": "finqa_test_0",
        "split": "test",
        "original_answer": "25%",
        "question": "Change?",
        "program_answer": "0.25",
    }
    document = {"source_file": "pdf/A/2010/page_5.pdf"}
    original = [
        {
            "id": "A/2010/page_5.pdf-1",
            "filename": "A/2010/page_5.pdf",
            "qa": {
                "answer": "25%",
                "question": "Original change?",
                "program": "subtract(5, 4), divide(#0, 4)",
                "steps": [{}, {}],
                "gold_inds": {"table_1": "first", "text_0": "second"},
                "exe_ans": 0.25,
            },
        }
    ]
    return label, document, original


def test_join_requires_independent_source_and_answer_agreement():
    label, document, original = example()
    result = join_row(label, document, original)
    assert result["status"] == "matched"
    assert result["operation_count"] == 2 and result["reasoning_bucket"] == "2"
    assert result["supporting_fact_count"] == 2
    assert result["supporting_sources"] == ["table", "text"]
    assert not result["question_reformulation_semantically_verified"]
    assert join_row(label, {"source_file": "wrong"}, original)["status"] == "unmatched"
    assert (
        join_row(label | {"original_answer": "24%"}, document, original)["status"]
        == "unmatched"
    )
    assert (
        join_row(label | {"id": "finqa_test_7"}, document, original)["status"]
        == "unmatched"
    )


def test_null_answers_and_program_validation():
    label, document, original = example()
    original[0]["qa"]["answer"] = ""
    assert (
        join_row(label | {"original_answer": None}, document, original)["status"]
        == "matched"
    )
    original[0]["qa"]["steps"] = [{}]
    assert (
        join_row(label | {"original_answer": None}, document, original)["reason"]
        == "program_and_steps_disagree"
    )
    assert program_operations("table_sum(revenue, none), divide(#0, const_100)") == [
        "table_sum",
        "divide",
    ]
    with pytest.raises(ValueError):
        program_operations("subtract(5, 4) garbage")


def test_dashboard_rejects_stale_or_misaligned_annotations(tmp_path):
    label, document, original = example()
    annotation = join_row(label, document, original)
    run = tmp_path / "runs" / "run1"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}")
    trace = {
        "question": {
            "id": label["id"],
            "question": label["question"],
            "dataset_revision": "rev1",
        },
        "gold_answer": "0.25",
        "answer": "0.25",
        "validation": {"score": 1},
        "turns": [],
        "searches": [],
    }
    (run / "finqa_test_0.json").write_text(json.dumps(trace))
    sidecar = tmp_path / "reasoning.json"

    def assess(revision, row):
        sidecar.write_text(
            json.dumps(
                {"provenance": {"t2_revision": revision}, "rows": {label["id"]: row}}
            )
        )
        client = TestClient(
            create_app(tmp_path / "runs", tmp_path / "missing", sidecar)
        )
        return client.get("/api/runs/run1").json()["rows"][0]

    assert assess("rev1", annotation)["reasoning_bucket"] == "2"
    assert assess("rev2", annotation)["reasoning_bucket"] == "unknown"
    assert (
        assess("rev1", annotation | {"t2_question": "Different"})["reasoning_bucket"]
        == "unknown"
    )
