import csv
import json

import pytest

from baseline_agent.evaluation import export_rows, record_result
from baseline_agent.scoring import (
    extract_numeric_value,
    numeric_value_match,
    score_answer,
)


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("94", "94.0", 1),
        ("-94", "94", 1),
        ("94000", "94", 1),
        ("0.94", "94", 1),
        ("95", "94", 0),
        ("0", "0.009", 1),
        ("0", "0.01", 0),
        ("1e400", "1e-400", 1),
        ("Yes", "1", 1),
        ("NO!", "0", 1),
        ("no", "1", 0),
        ("Yes", "0", 0),
        ("NaN", "1", 0),
        (None, "94", 0),
        ("Revenue in 2015 was unclear", "2015", 0),
    ],
)
def test_number_match(prediction, gold, expected):
    assert numeric_value_match(prediction, gold) == expected


def test_extraction_and_sign_visibility():
    assert str(extract_numeric_value("($1,234.50)")) == "-1234.50"
    assert str(extract_numeric_value("14.5%")) == "0.145"
    assert extract_numeric_value("1,23") is None
    assert extract_numeric_value("1 or 2") is None
    result = score_answer("-94", "94")
    assert result["score"] == 1 and result["sign_agrees"] is False
    assert score_answer("Yes", "1")["normalized_prediction"] == "1"


def test_evaluation_export_adds_gold_only_after_inference(tmp_path):
    result = {
        "question": {"id": "test_0", "question": "Revenue?", "example_index": 0},
        "answer": "94",
        "status": "answered",
        "config": {"context_mode": "replace"},
        "inference_seconds": 3.5,
        "turns": [
            {
                "model_input": "Question only",
                "generation_seconds": 2.5,
                "input_tokens": 10,
                "generated_tokens": 5,
            }
        ],
        "searches": [{"seconds": 1.0}],
        "events": [],
        "model_revision": "revision",
    }
    row = record_result(result, "94.0", tmp_path)
    summary = export_rows([row], tmp_path)
    with (tmp_path / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        written = next(csv.DictReader(stream))
    assert written["correct_answer"] == "94.0" and written["our_answer"] == "94"
    assert written["inference_seconds"] == "3.5"
    assert summary["answer_match_accuracy"] == 1
    assert "gold_answer" not in result
    assert json.loads((tmp_path / "test_0.json").read_text())["gold_answer"] == "94.0"


def test_tensorboard_records_scalar_and_trace(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from baseline_agent.observability import write_dashboard

    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "turns": [{"turn": 1, "model_output": "<answer>Yes</answer>"}],
                "searches": [],
            }
        )
    )
    row = {
        "id": "test",
        "question": "Question?",
        "correct_answer": "1",
        "our_answer": "Yes",
        "status": "answered",
        "score": 1.0,
        "inference_seconds": 3.0,
        "generation_seconds": 2.0,
        "search_seconds": 1.0,
        "search_attempts": 1,
        "generated_tokens": 10,
        "trace_file": str(trace),
    }
    write_dashboard([row], tmp_path / "events", start_step=4)
    events = EventAccumulator(str(tmp_path / "events")).Reload()
    assert events.Scalars("evaluation/answer_match")[0].step == 4
    assert events.Scalars("latency/inference_seconds")[0].value == 3
    assert (
        b"<answer>Yes</answer>"
        in events.Tensors("examples/result")[0].tensor_proto.string_val[0]
    )
