"""Pure diagnostics/live reader tests; no GPU, TensorBoard or cloud operations."""

from dataclasses import replace

import pytest

from baseline_agent.diagnostics import diagnose
from baseline_agent.diagnostics import diagnose as baseline_diagnose
from tests.agent_fixtures import ANSWER, CONFIG, QUESTION, SEARCH, recorder, submit
from tinker_training.advantages import prepare_update
from tinker_training.data import RewardLabel
from tinker_training.metrics import (
    distribution,
    episode_diagnostic,
    merge_rates,
    pair,
    rollout_metrics,
)
from tinker_training.observability import atomic_json
from tinker_training.replay import QuestionGroup
from tinker_training.results import Reader


def observed_rollout():
    """Build a prepared update with controller/retrieval audit information for metric
    tests.
    """
    config = replace(CONFIG, questions_per_update=1)
    first, second = recorder(0, config=config), recorder(1, config=config)
    submit(first, SEARCH)
    submit(first, SEARCH.replace("revenue", "cost"))
    submit(first, ANSWER)
    submit(second, SEARCH)
    submit(second, ANSWER.replace("42", "9"))
    label = RewardLabel(QUESTION.id, "42", "OLD_EVIDENCE")
    prepared = prepare_update(
        [QuestionGroup((first.finish(), second.finish()))], [label]
    )
    return prepared, label


def test_training_uses_same_baseline_diagnostic_function():
    """Verify training uses same baseline diagnostic function."""
    assert baseline_diagnose is diagnose
    prepared, label = observed_rollout()
    metrics, traces = rollout_metrics(prepared, [label])
    assert metrics["rates"]["em"] == pair(1, 2)
    assert metrics["rates"]["ever"] == pair(2, 2)
    assert metrics["rates"]["active"] == pair(1, 2)
    assert metrics["rates"]["groups_with_signal"] == pair(1, 1)
    assert metrics["rates"]["format_search"] == pair(3, 3)
    assert metrics["evidence_outcome"]["retrieved_correct"] == 1
    assert metrics["evidence_outcome"]["retrieved_incorrect"] == 1
    assert len(traces) == 2
    assert metrics["episodes"][0]["generation_seconds"] == pytest.approx(0.3)
    assert metrics["episodes"][0]["generations"] == 3


def test_terminal_search_failure_is_not_an_accuracy_zero():
    """Verify terminal search failure is not an accuracy zero."""
    prepared, label = observed_rollout()
    failed = replace(
        prepared.groups[0].episodes[0], status="infrastructure_error", answer=None
    )
    diagnostic, _ = episode_diagnostic(failed, label)
    assert diagnostic["scored"] is False and diagnostic["em"] is None


def test_rate_merging_uses_counts_not_average_percentages():
    """Verify rate merging uses counts not average percentages."""
    assert merge_rates([pair(1, 1), pair(0, 9)]) == pair(1, 10)
    assert merge_rates([])["value"] is None
    assert distribution([1, 2, 3])["std"] == 1


def fixture_files(root):
    """Write synthetic run/observability artifacts used by dashboard readers and export
    endpoints.
    """
    prepared, label = observed_rollout()
    rollout, traces = rollout_metrics(prepared, [label])
    rollout.update(
        rollout_id="a" * 24, sampling={"raw_correction_above_cap": pair(0, 10)}
    )
    for row in rollout["episodes"]:
        row["trace_key"] = ("b" if row["sample_index"] == 0 else "c") * 24
        atomic_json(
            root / "traces" / f"{row['trace_key']}.json", traces[row["episode_id"]]
        )
    atomic_json(root / "rollouts" / f"{'a' * 24}.json", rollout)
    for step, tokens, loss in [(1, 10, 1.0), (2, 90, 3.0)]:
        atomic_json(
            root / "steps" / f"{step:08d}.json",
            {
                "step": step,
                "rollout_id": "a" * 24,
                "tokens": tokens,
                "loss": loss,
                "gradient_norm": 0.5,
                "policy_metrics": None,
            },
        )
    atomic_json(
        root / "index.json",
        {
            "committed_step": 2,
            "steps": [{"step": i, "rollout_id": "a" * 24} for i in (1, 2, 3)],
            "references": [0, 1, 2, 3],
        },
    )
    for step in range(4):
        atomic_json(
            root / "reference" / f"{step:08d}.json",
            {"step": step, "mean_nats": step / 10},
        )
    atomic_json(root / "live.json", {"phase": "backward", "step": 3, "max_steps": 10})


def test_reader_deduplicates_reuse_weights_loss_and_excludes_future_steps(tmp_path):
    """Verify reader deduplicates reuse weights loss and excludes future steps."""
    fixture_files(tmp_path)
    snapshot = Reader(tmp_path).snapshot(20)
    assert snapshot["rollouts"] == 1 and len(snapshot["episodes"]) == 2
    assert snapshot["weighted_loss"]["value"] == pytest.approx(2.8)
    assert snapshot["series"][-1]["rolling_loss"] == pytest.approx(2.8)
    assert [r["step"] for r in snapshot["reference"]] == [0, 1, 2]
    assert snapshot["rates"]["em"] == pair(1, 2)


def test_batch_history_distinguishes_configured_actual_and_reused_batches(tmp_path):
    """Verify batch history distinguishes configured actual and reused batches."""
    root = tmp_path / "observability"
    fixture_files(root)
    atomic_json(
        tmp_path / "run.json",
        {
            "config": {
                "questions_per_update": 2,
                "num_generations": 2,
                "num_iterations": 2,
            }
        },
    )
    snapshot = Reader(root).snapshot(1)
    assert snapshot["training_config"]["questions_per_update"] == 2
    assert snapshot["batch_totals"] == {
        "fresh_batches": 1,
        "accepted_groups": 1,
        "accepted_trajectories": 2,
    }
    batch = snapshot["batches"][0]
    assert batch["optimizer_steps"] == [1, 2]
    assert batch["question_ids"] == [QUESTION.id]
    assert batch["actual_groups"] == 1  # A smaller actual batch, not the target two.
    assert batch["candidate_questions"] is None  # Never invent missing historical data.
    assert len(snapshot["steps"]) == 1  # Windowing does not double-count batch history.


def test_startup_failure_preserves_completed_steps_and_traceback(tmp_path):
    """Verify startup failure preserves completed steps and traceback."""
    import json

    from tinker_training.observability import atomic_json, publish_runtime_failure

    root = tmp_path / "observability"
    atomic_json(
        root / "index.json",
        {"steps": [{"step": 3}], "references": [], "committed_step": 3},
    )
    atomic_json(root / "live.json", {"phase": "completed", "max_steps": 10})
    commits = []
    try:
        raise ValueError("Restored adapter bytes disagree with checkpoint policy")
    except ValueError as error:
        publish_runtime_failure(
            tmp_path, "checkpoint_restore", error, durable=lambda: commits.append(True)
        )
    live = json.loads((root / "live.json").read_text())
    assert live["phase"] == "failed" and live["committed_step"] == 3
    assert live["max_steps"] == 10
    assert live["failure"]["stage"] == "checkpoint_restore"
    assert "ValueError" in live["failure"]["traceback"]
    assert commits == [True]


def test_resume_phase_clears_old_failure_and_stale_counters(tmp_path):
    """Verify resume phase clears old failure and stale counters."""
    import json

    from tinker_training.observability import atomic_json, publish_runtime_phase

    root = tmp_path / "observability"
    atomic_json(
        root / "index.json", {"steps": [], "references": [], "committed_step": 3}
    )
    atomic_json(
        root / "live.json",
        {
            "phase": "failed",
            "failure": {"message": "old failure"},
            "completed": 80,
            "max_steps": 10,
        },
    )
    publish_runtime_phase(tmp_path, "checkpoint_restore")
    live = json.loads((root / "live.json").read_text())
    assert live["phase"] == "checkpoint_restore"
    assert live["committed_step"] == 3 and live["max_steps"] == 10
    assert "failure" not in live and "completed" not in live
    assert live["failure_history"] == [{"message": "old failure"}]
