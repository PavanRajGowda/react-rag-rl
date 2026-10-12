import csv
import json

import pytest

from baseline_agent.evaluation import export_rows, record_result
from baseline_agent.protocol import ProtocolError, action_complete, parse_action
from baseline_agent.retrieval_diagnostics import annotate_retrieval


def episode(mode="replace"):
    searches = [
        {"response": {"status": "ok", "results": [{"context_id": "gold"}]}},
        {"response": {"status": "ok", "results": [{"context_id": "other"}]}},
    ]
    return {
        "config": {"context_mode": mode},
        "turns": [{"turn": i, "model_input": "original prompt"} for i in range(1, 5)],
        "searches": searches,
        "events": [
            {"kind": "model", "text": "first search"},
            {"kind": "information", "text": "gold evidence"},
            {"kind": "model", "text": "malformed search"},
            {"kind": "feedback", "text": "fix format"},
            {"kind": "model", "text": "second search"},
            {"kind": "information", "text": "other evidence"},
            {"kind": "model", "text": "answer"},
        ],
    }


def test_rolling_gold_context_hit_and_prompt_presence_are_distinct():
    original = episode()
    annotated = annotate_retrieval(original, "gold")
    turns = [t["retrieval_diagnostic"] for t in annotated["turns"]]
    assert [t["gold_context_in_prompt"] for t in turns] == [False, True, True, False]
    assert [t["gold_context_retrieved"] for t in turns] == [True, None, False, None]
    assert turns[-1]["gold_context_retrieved_ever"]
    assert [t["gold_context_active_after_turn"] for t in turns] == [
        True,
        True,
        False,
        False,
    ]
    assert annotated["turns"][0]["model_input"] == "original prompt"
    assert "retrieval_validation" not in original
    assert "retrieval_diagnostic" not in original["turns"][0]


def test_append_retains_gold_context_and_api_error_replaces_in_replace_mode():
    appended = annotate_retrieval(episode("append"), "gold")
    assert appended["turns"][-1]["retrieval_diagnostic"]["gold_context_in_prompt"]
    failed = episode()
    failed["searches"][-1]["response"] = {"status": "error"}
    annotated = annotate_retrieval(failed, "gold")
    assert (
        annotated["turns"][2]["retrieval_diagnostic"]["gold_context_retrieved"] is False
    )
    assert not annotated["turns"][-1]["retrieval_diagnostic"]["gold_context_in_prompt"]


def test_no_gold_context_does_not_mean_alternative_evidence_is_invalid():
    annotated = annotate_retrieval(episode(), "different_gold")
    assert not annotated["retrieval_validation"]["gold_context_retrieved_ever"]
    assert (
        "alternative valid evidence" in annotated["retrieval_validation"]["definition"]
    )


def test_action_stopping_ignores_quoted_tags_in_reasoning():
    assert not action_complete("<reasoning>The earlier action ended with </search>")
    with pytest.raises(ProtocolError, match="Close the reasoning block"):
        parse_action(
            "<reasoning>The example is <search><company>Acme</company><year>2010</year><query>revenue</query></search>"
        )
    assert not action_complete(
        "<reasoning>Search first.</reasoning><search><company>Acme</company>"
    )
    assert action_complete("<reasoning>Done.</reasoning><answer>42</answer>")
    assert action_complete(
        "<reasoning>Enough evidence.</reasoning><search>None needed</search>"
    )
    with pytest.raises(ProtocolError, match="final value in answer tags"):
        parse_action(
            "<reasoning>Done.</reasoning><search>None needed; calculation can be performed directly from the provided table.</search>"
        )


def test_placeholder_search_values_are_not_sent_to_api():
    with pytest.raises(ProtocolError, match="documentation placeholders"):
        parse_action(
            "<search><company>Company name</company><year>2010</year><query>Financial evidence needed</query></search>"
        )


def test_local_export_and_dashboard_include_turn_hits_without_prompt_changes(tmp_path):
    result = episode() | {
        "question": {"id": "q1", "example_index": 0, "question": "Revenue?"},
        "answer": "42",
        "status": "answered",
        "inference_seconds": 1.0,
        "model_revision": "revision",
    }
    for search in result["searches"]:
        search["seconds"] = 0.1
        search["request"] = {}
    row = record_result(result, "42", tmp_path, "gold")
    summary = export_rows([row], tmp_path)
    assert summary["gold_context_retrieval_rate"] == 1
    trace = json.loads((tmp_path / "q1.json").read_text(encoding="utf-8"))
    assert all(turn["model_input"] == "original prompt" for turn in trace["turns"])
    with (tmp_path / "turns.csv").open(encoding="utf-8-sig", newline="") as stream:
        turns = list(csv.DictReader(stream))
    assert turns[0]["gold_context_retrieved"] == "True"
    assert turns[1]["search_performed"] == "False"
    assert turns[-1]["gold_context_in_prompt"] == "False"
    assert json.loads(turns[0]["retrieved_context_ids"]) == ["gold"]
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from baseline_agent.observability import write_dashboard

    write_dashboard([row], tmp_path / "events")
    events = EventAccumulator(str(tmp_path / "events")).Reload()
    text = events.Tensors("examples/result")[0].tensor_proto.string_val[0].decode()
    assert "Gold context in prompt: **True**" in text
    assert "Gold context retrieved this turn: **No search**" in text
    assert events.Scalars("retrieval/gold_context_hit")[0].value == 1
