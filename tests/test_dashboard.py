import json

from fastapi.testclient import TestClient

from baseline_agent.dashboard import create_app, diagnose


def trace():
    return {
        "question": {"id": "q1", "question": "Change?", "example_index": 0},
        "status": "answered",
        "answer": "-25",
        "gold_answer": "0.25",
        "validation": {"score": 1},
        "evaluation_mode": "rag",
        "retrieval_validation": {"gold_context_id": "gold"},
        "turns": [
            {
                "turn": 1,
                "model_output": "<search><company>A</company><year>2010</year><query>revenue</query></search>",
            },
            {"turn": 2, "model_output": "<search>None needed</search>"},
            {
                "turn": 3,
                "model_output": "<answer>-25</answer>",
                "retrieval_diagnostic": {"gold_context_in_prompt": False},
            },
        ],
        "searches": [
            {
                "turn": 1,
                "request": {"query": "revenue"},
                "response": {"results": [{"context_id": "gold"}]},
            },
            {
                "turn": 2,
                "request": {"query": "revenue"},
                "response": {"results": [{"context_id": "other"}]},
            },
            {
                "turn": 2,
                "request": {"query": "Revenue"},
                "response": {"results": [{"context_id": "other"}]},
            },
        ],
    }


def test_diagnostics_keep_benchmark_and_evidence_presence_distinct():
    result = diagnose(trace())
    assert result["nm"] == 1
    assert not result["strict"] and not result["exact"]
    assert result["sign_disagreement"]
    assert result["ever"] and result["active"] is False
    assert result["first_hit_search"] == result["first_hit_rank"] == 1
    assert result["repeated_requests"] == 2
    assert result["repeated_sets"] == result["no_new_context"] == 1
    assert result["searches_after_hit"] == 2
    assert result["format_attempts"]["search"] == 2
    assert result["format_failures"]["search"] == 1
    assert result["recovered_format"]
    assert result["output_tokens"] is None


def test_closed_book_has_no_retrieval_measurement():
    item = trace()
    item.update(
        evaluation_mode="pretrained_only",
        searches=[],
        turns=[{"turn": 1, "model_output": "<answer>Yes</answer>"}],
        answer="Yes",
        gold_answer=1,
    )
    result = diagnose(item)
    assert result["strict"]
    assert result["ever"] is None and result["active"] is None
    assert not result["format_errors"]


def test_filter_exclusions_are_not_inferred_from_broad_fallback():
    item = trace()
    item["searches"] = [
        {
            "request": {},
            "response": {"match_method": "broad_query", "applied_filters": {}},
        }
    ]
    metadata = {"company_id": "issuer", "report_year": 2012}
    assert "reference_excluding_year_filter" not in diagnose(item, metadata)["reasons"]
    item["searches"][0]["response"]["applied_filters"] = {
        "company_ids": ["other"],
        "report_year": 2010,
    }
    reasons = diagnose(item, metadata)["reasons"]
    assert (
        reasons["reference_excluding_year_filter"]
        == reasons["reference_excluding_company_filter"]
        == 1
    )


def test_api_handles_bad_files_exports_and_rejects_unknown_paths(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"status": "completed"}))
    (run / "q1.json").write_text(json.dumps(trace()))
    (run / "broken.json").write_text("{")
    client = TestClient(create_app(tmp_path, tmp_path / "missing.jsonl"))
    assert client.get("/").status_code == 200
    assert len(client.get("/api/runs").json()) == 1
    data = client.get("/api/runs/run1").json()
    assert len(data["rows"]) == len(data["read_errors"]) == 1
    assert client.get("/api/runs/run1/questions/q1").json()["trace"]["answer"] == "-25"
    assert "'-25" in client.get("/api/runs/run1/export.csv").text
    assert client.get("/api/runs/nonexistent").status_code == 404
