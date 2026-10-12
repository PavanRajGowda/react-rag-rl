import csv
import json

import pytest

from baseline_agent.agent import AgentConfig
from baseline_agent.compare import compare_runs
from baseline_agent.question_only import (
    QuestionOnlyGenerator,
    messages,
    run_question_only,
)
from baseline_agent.runner import behavior_hash, run_evaluation
from tests.test_batch_runner import local_setup


class FakeGenerator:
    def __init__(self, segment):
        self.segment = segment
        self.calls = 0

    def make_prefix(self, question):
        return json.dumps(messages(question))

    def generate(self, prompt):
        self.calls += 1
        return self.segment, {"generated_tokens": 10, "input_tokens": 20}


def test_one_pass_answer_has_no_retriever_or_gold_context():
    question = {
        "id": "q0",
        "example_index": 0,
        "question": "What is revenue?",
        "gold_answer": "SECRET_GOLD",
        "context": "SECRET_CONTEXT",
    }
    generator = FakeGenerator("<reasoning>Known fact.</reasoning><answer>42</answer>")
    result = run_question_only(question, generator, AgentConfig())
    assert generator.calls == 1 and result["searches"] == []
    assert result["answer"] == "42" and result["evaluation_mode"] == "pretrained_only"
    assert "SECRET" not in result["turns"][0]["model_input"]
    assert all(event["kind"] == "model" for event in result["events"])


def test_attempted_search_is_rejected_without_repair_or_tool_call():
    generator = FakeGenerator(
        "<reasoning>Need facts.</reasoning><search><company>Apple</company><year>2002</year><query>sales</query></search>"
    )
    result = run_question_only({"question": "Revenue?"}, generator, AgentConfig())
    assert result["status"] == "invalid_protocol" and result["answer"] is None
    assert generator.calls == 1 and result["searches"] == []


def test_unknown_is_an_abstention_not_a_numeric_guess():
    result = run_question_only(
        {"question": "Revenue?"},
        FakeGenerator("<reasoning>Unknown.</reasoning><answer>unknown</answer>"),
        AgentConfig(),
    )
    assert result["status"] == "abstained" and result["answer"] == "unknown"


def test_native_template_receives_only_question_and_no_tool_instructions():
    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            self.conversation = conversation
            return "CHAT"

    generator = object.__new__(QuestionOnlyGenerator)
    generator.tokenizer = Tokenizer()
    assert generator.make_prefix("Revenue?") == "CHAT<reasoning>"
    assert [m["role"] for m in generator.tokenizer.conversation] == ["system", "user"]
    assert generator.tokenizer.conversation[-1]["content"] == "Question: Revenue?"


def test_question_only_runner_does_not_probe_search_or_read_corpus(
    monkeypatch, tmp_path
):
    local_setup(monkeypatch, tmp_path, count=1)
    (tmp_path / "data/finqa/manifest.json").unlink()

    def invoke(questions, config, revision):
        return iter(
            [
                {
                    **run_question_only(
                        questions[0],
                        FakeGenerator(
                            "<reasoning>Known.</reasoning><answer>0</answer>"
                        ),
                        AgentConfig(**config),
                    ),
                    "config": config,
                    "model_id": "model",
                    "model_revision": revision,
                    "gpu": "H100",
                    "inference_seconds": 1.0,
                }
            ]
        )

    directory = run_evaluation(
        invoke,
        lambda r: None,
        lambda: pytest.fail("retriever touched"),
        "model",
        "revision",
        "H100",
        count=1,
        evaluation_mode="pretrained_only",
    )
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["search_schema_version"] is None
    assert manifest["evaluation_mode"] == "pretrained_only"
    trace = json.loads((directory / "q0.json").read_text())
    assert "retrieval_validation" not in trace


def test_mode_hashes_differ_and_cannot_reuse_rag_run(monkeypatch, tmp_path):
    local_setup(monkeypatch, tmp_path, count=1)
    directory = run_evaluation(
        None, None, None, "model", "revision", "H100", count=1, dry_run=True
    )
    assert behavior_hash("rag") != behavior_hash("pretrained_only")
    with pytest.raises(ValueError, match="Resume mismatch"):
        run_evaluation(
            None,
            None,
            None,
            "model",
            "revision",
            "H100",
            count=1,
            dry_run=True,
            resume=str(directory),
            evaluation_mode="pretrained_only",
        )


def paired_fixture(tmp_path):
    directories = [tmp_path / "rag", tmp_path / "pretrained"]
    metadata = {
        "model_id": "model",
        "model_revision": "revision",
        "gpu": "H100",
        "dtype": "float16",
        "dataset_revision": "data",
        "base_config": {"seed": 42, "temperature": 0.2, "max_new_tokens": 1024},
    }
    for index, directory in enumerate(directories):
        directory.mkdir()
        (directory / "run.json").write_text(
            json.dumps(
                metadata
                | {"evaluation_mode": "rag" if index == 0 else "pretrained_only"}
            )
        )
        rows = [
            {
                "id": f"q{i}",
                "question": "Revenue?",
                "correct_answer": "42",
                "example_index": i,
                "our_answer": "42",
                "score": int(i == index),
                "status": "answered",
            }
            for i in range(2)
        ]
        with (directory / "results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return directories


def test_paired_comparison_reports_retrieval_gains_and_losses(tmp_path):
    rag, pretrained = paired_fixture(tmp_path)
    summary = compare_runs(rag, pretrained, tmp_path / "comparison")
    assert summary["paired_questions"] == 2
    assert summary["rag_only_correct"] == 1
    assert summary["pretrained_only_correct"] == 1
    assert summary["both_correct"] == 0


def test_paired_comparison_rejects_different_model_or_gpu(tmp_path):
    rag, pretrained = paired_fixture(tmp_path)
    path = pretrained / "run.json"
    manifest = json.loads(path.read_text()) | {"gpu": "T4"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Runs differ in gpu"):
        compare_runs(rag, pretrained, tmp_path / "comparison")


def test_paired_comparison_allows_only_an_explicit_matching_prefix(tmp_path):
    rag, pretrained = paired_fixture(tmp_path)
    summary = compare_runs(rag, pretrained, tmp_path / "comparison", num_examples=1)
    assert summary["paired_questions"] == 1
    assert summary["rag_only_correct"] == 1
    with pytest.raises(ValueError, match="requested comparison count"):
        compare_runs(rag, pretrained, tmp_path / "invalid", num_examples=3)
