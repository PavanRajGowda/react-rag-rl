from copy import deepcopy
from dataclasses import asdict

import pytest

from baseline_agent.agent import AgentConfig
from baseline_agent.sampling import (
    evaluation_seed,
    repeat_examples,
    summarize_repetitions,
)


def examples():
    """Return a minimal labeled evaluation set for seeded repetition tests."""
    return [
        {
            "question": {"id": "q", "question": "Who?", "example_index": 0},
            "gold_answer": "Aster",
            "gold_document_ids": [],
        }
    ]


def test_repeat_seeds_are_stable_distinct_and_independent_of_order():
    """Verify repeat seeds are stable distinct and independent of order."""
    source = examples()
    original = deepcopy(source)
    result = repeat_examples(source)
    assert source == original
    assert len({r["question"]["id"] for r in result}) == 3
    assert len({r["question"]["sampling_seed"] for r in result}) == 3
    assert [r["question"]["sampling_seed"] for r in result] == [
        evaluation_seed(42, "q", i) for i in range(3)
    ]
    assert repeat_examples(source) == result
    assert repeat_examples(source, master_seed=43) != result
    assert all("gold_answer" not in r["question"] for r in result)
    with pytest.raises(ValueError):
        repeat_examples(source, 0)


def test_phase_persists_all_repetitions_and_resumes_without_regenerating(tmp_path):
    """Verify phase persists all repetitions and resumes without regenerating."""
    import json

    from baseline_agent.phases import run_phase
    from tinker_training.config import TrainingConfig

    config = TrainingConfig(temperature=1)
    expanded = repeat_examples(examples())
    calls = []

    def invoke(questions, values, concurrency):
        """Yield scripted repeated-evaluation results and record requests to verify resume
        avoids resampling.
        """
        calls.append(questions)
        for q in questions:
            yield {
                "question": q,
                "status": "answered",
                "answer": "Aster" if q["repetition_index"] == 0 else "Wrong",
                "turns": [],
                "searches": [],
                "events": [],
                "inference_seconds": 1,
                "model_revision": config.model_revision,
                "config": asdict(AgentConfig()),
            }

    assert run_phase("test", expanded, tmp_path, config, 3, invoke)
    assert run_phase("test", expanded, tmp_path, config, 3, invoke)
    assert len(calls) == 1
    assert len(list((tmp_path / "test/raw").glob("*.json"))) == 3
    summary = json.loads((tmp_path / "test/summary.json").read_text())[
        "repeated_evaluation"
    ]
    assert summary["mean_exact_match"] == pytest.approx(1 / 3)
    assert summary["pass_at_n"] == 1
    assert summary["mean_inference_seconds"] == 1


def test_partial_or_failed_repetitions_do_not_report_final_pass_at_n():
    """Verify partial or failed repetitions do not report final pass at n."""
    rows = [
        {
            "source_question_id": "q",
            "repetition_index": i,
            "repetitions": 3,
            "status": "answered",
            "score": 1,
            "exact_match": 1,
            "token_f1": 1,
            "search_attempts": 1,
            "generated_tokens": 10,
            "generation_turns": 2,
            "inference_seconds": 1,
        }
        for i in range(3)
    ]
    assert summarize_repetitions(rows[:2])["pass_at_n"] is None
    rows[2]["status"] = "infrastructure_error"
    assert summarize_repetitions(rows)["mean_exact_match"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_repetitions([rows[0], rows[0]])


def test_batch_generator_passes_registered_repetition_seed():
    """Verify batch generator passes registered repetition seed."""
    from types import SimpleNamespace

    from batched_inference.baseline import BatchGenerator
    from tinker_training.records import PolicyVersion, generation_seed

    question = repeat_examples(examples())[1]["question"]
    policy = PolicyVersion("a" * 40, "b" * 40, "base", 0, "c" * 64)
    generator = BatchGenerator(
        SimpleNamespace(encode=lambda *a, **k: [1, 2]),
        AgentConfig(temperature=1),
        policy,
        question,
    )
    request = generator.request("prompt<reasoning>")
    assert request.seed == generation_seed(question["sampling_seed"], 1)
    assert request.sampling.temperature == 1
