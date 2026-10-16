"""Pure diagnostics/live reader tests; no GPU, TensorBoard or cloud operations."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from baseline_agent.dashboard import diagnose as dashboard_diagnose
from baseline_agent.diagnostics import diagnose
from grpo_training.advantages import prepare_update
from grpo_training.dashboard import ModalMirror, Reader, create_app
from grpo_training.data import RewardLabel
from grpo_training.metrics import (
    distribution,
    episode_diagnostic,
    merge_rates,
    pair,
    rollout_metrics,
    sampling_metrics,
)
from grpo_training.observability import atomic_json
from grpo_training.replay import QuestionGroup
from grpo_training.trainer import CapturedUpdate
from tests.test_grpo_replay import ANSWER, CONFIG, QUESTION, SEARCH, recorder, submit


def observed_rollout():
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
    assert dashboard_diagnose is diagnose
    prepared, label = observed_rollout()
    metrics, traces = rollout_metrics(prepared, [label])
    assert metrics["rates"]["nm"] == pair(1, 2)
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
    prepared, label = observed_rollout()
    failed = replace(
        prepared.groups[0].episodes[0], status="infrastructure_error", answer=None
    )
    diagnostic, _ = episode_diagnostic(failed, label)
    assert diagnostic["scored"] is False and diagnostic["nm"] is None


def test_rate_merging_uses_counts_not_average_percentages():
    assert merge_rates([pair(1, 1), pair(0, 9)]) == pair(1, 10)
    assert merge_rates([])["value"] is None
    assert distribution([1, 2, 3])["std"] == 1


def test_raw_sampling_cap_count_is_not_inferred_from_capped_weights():
    prepared, _ = observed_rollout()
    old = tuple(
        tuple(v + 2 for v in row["sampling_logprobs"][-len(row["target_ids"]) :])
        for row in prepared.rows()
    )
    # Old logprobs must remain <=0; add only .1 if a toy sampled logprob is too near zero.
    old = tuple(tuple(min(v, -0.01) for v in row) for row in old)
    captured = CapturedUpdate(prepared, old, tuple((2.0,) * len(v) for v in old))
    metrics = sampling_metrics(captured)
    assert metrics["raw_correction_above_cap"]["numerator"] > 0
    assert metrics["applied_correction"]["max"] == 2


def fixture_files(root):
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
    fixture_files(tmp_path)
    snapshot = Reader(tmp_path).snapshot(20)
    assert snapshot["rollouts"] == 1 and len(snapshot["episodes"]) == 2
    assert snapshot["weighted_loss"]["value"] == pytest.approx(2.8)
    assert snapshot["series"][-1]["rolling_loss"] == pytest.approx(2.8)
    assert [r["step"] for r in snapshot["reference"]] == [0, 1, 2]
    assert snapshot["rates"]["nm"] == pair(1, 2)


def test_batch_history_distinguishes_configured_actual_and_reused_batches(tmp_path):
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


def test_live_api_trace_and_csv(tmp_path):
    fixture_files(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/snapshot").json()["committed_step"] == 2
        assert client.get("/api/snapshot?window=-1").status_code == 400
        assert (
            client.get(f"/api/traces/{'b' * 24}").json()["question"]["id"]
            == QUESTION.id
        )
        assert client.get("/api/traces/not-a-key").status_code == 400
        csv = client.get("/api/export/steps")
        assert csv.status_code == 200 and "gradient_norm" in csv.text


def test_remote_mirror_only_reads_new_immutable_files_and_keeps_last_good(tmp_path):
    fixture_files(tmp_path / "remote")
    calls = []

    class Volume:
        def read_file(self, path):
            calls.append(path)
            if path == "run/run.json":
                yield b'{"config": {"questions_per_update": 2, "num_generations": 4, "num_iterations": 2}}'
                return
            relative = path.split("observability/", 1)[1]
            yield (tmp_path / "remote" / relative).read_bytes()

        def iterdir(self, *args, **kwargs):
            return []

    mirror = ModalMirror("run", tmp_path / "local", volume=Volume())
    mirror.sync()
    assert mirror.status["error"] is None
    calls.clear()
    mirror.sync()
    assert len(calls) == 2  # Only mutable live/index JSON on second refresh.
    (tmp_path / "remote" / "index.json").write_text("invalid")
    mirror.sync()
    assert mirror.status["error"]
    assert Reader(tmp_path / "local").snapshot()["committed_step"] == 2


def test_mirror_rejects_unsafe_paths(tmp_path):
    mirror = ModalMirror("run", tmp_path, volume=NS())
    with pytest.raises(ValueError):
        mirror.download("../run.json")


def test_dynamic_candidate_totals_survive_observer_restart(tmp_path):
    from grpo_training.checkpoints import Progress
    from grpo_training.observability import RunObserver

    totals = {
        "groups": 3,
        "accepted": 1,
        "episodes": 12,
        "correct": 7,
        "generated_tokens": 100,
    }
    atomic_json(tmp_path / "observability/live.json", {"dynamic_totals": totals})
    observer = RunObserver(
        NS(root=tmp_path, durable=lambda: None),
        NS(state=NS(global_step=0)),
        NS(config=CONFIG),
        NS(),
        reference=NS(),
        writer=NS(flush=lambda: None),
    )
    observer.reference_step = lambda step: None
    observer.start(Progress())
    observer.event(
        {
            "kind": "dynamic_group",
            "accepted": True,
            "episodes": 4,
            "correct": 2,
            "generated_tokens": 50,
        }
    )
    assert observer.status["dynamic_totals"] == {
        "groups": 4,
        "accepted": 2,
        "episodes": 16,
        "correct": 9,
        "generated_tokens": 150,
    }
