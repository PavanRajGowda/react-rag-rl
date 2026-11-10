"""Trajectory reward and frozen-credit mini-batch contracts; no paid calls."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_grpo_advantages import label, make_group
from tinker_training.advantages import minibatch_update, prepare_update
from tinker_training.config import TrainingConfig


def test_filter_preserves_original_minibatch_denominator_and_difficulty():
    from tinker_training.advantages import filter_constant_groups
    from tinker_training.replay import QuestionGroup

    cfg = TrainingConfig(
        num_generations=2, questions_per_update=2, minibatch_questions=2
    )
    a = make_group(cfg)
    b = make_group(cfg, number=1, answers=("42", "42"))
    a = QuestionGroup(
        tuple(
            replace(e, question=replace(e.question, difficulty="hard"))
            for e in a.episodes
        )
    )
    b = QuestionGroup(
        tuple(
            replace(e, question=replace(e.question, difficulty="medium"))
            for e in b.episodes
        )
    )
    pool = prepare_update((a, b), (label(a), label(b)))
    filtered, report = filter_constant_groups(pool)
    assert filtered.groups == (a,)
    assert filtered.token_count == pool.token_count
    assert len(filtered.rows()) < len(pool.rows())
    assert report["difficulty_original"] == {"hard": 1, "medium": 1}
    assert report["difficulty_surviving"] == {"hard": 1}
    assert report["difficulty_skipped"] == {"medium": 1}
    assert report["surviving_groups"] == 1
    empty, report = filter_constant_groups(prepare_update((b,), (label(b),)))
    assert empty is None and report["surviving_groups"] == 0


def test_empty_minibatches_skip_adam_without_replacement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.test_tinker_migration import TYPES, Service, search
    from tinker_training.runner import run

    cfg = TrainingConfig(
        num_generations=2, questions_per_update=2, minibatch_questions=1, max_steps=2
    )
    groups = tuple(make_group(cfg, number=i, answers=("42", "42")) for i in range(2))
    train = SimpleNamespace(
        questions=tuple(g.episodes[0].question for g in groups),
        labels=tuple(label(g) for g in groups),
    )
    calls = []

    def collect(*args, **kwargs):
        calls.append(1)
        return groups

    monkeypatch.setattr("tinker_training.runner.collect_groups", collect)
    service = Service()
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": "empty"},
        validation_count=0,
    )
    assert state["step"] == 0 and state["skipped_minibatches"] == 2
    assert len(calls) == 1 and service.data == []
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": "empty"},
        validation_count=0,
    )
    assert len(calls) == 1


def test_training_caps_activity_without_truncating_pool(monkeypatch):
    """A smaller active limit must preserve every trajectory slot in the pool."""
    from types import SimpleNamespace

    from tinker_training.trajectory import generate_episodes

    cfg = TrainingConfig(rollout_concurrency=2)
    specs = [
        {
            "episode_id": str(i),
            "group_id": "q",
            "sample_index": i,
            "seed": i,
            "question": SimpleNamespace(id="q", text="Question", source_index=0),
        }
        for i in range(5)
    ]
    seen = []
    monkeypatch.setattr(
        "tinker_training.trajectory.template_digest", lambda _: "template"
    )

    def batched(
        questions, backend, tokenizer, policy, config, search, concurrency, **kwargs
    ):
        seen.append((len(questions), concurrency))
        return iter(())

    monkeypatch.setattr("tinker_training.trajectory.batched_agents", batched)
    list(
        generate_episodes(
            specs,
            cfg,
            SimpleNamespace(template_sha256="template"),
            None,
            None,
            lambda _: {},
        )
    )
    assert seen == [(5, 2)]


def test_minibatches_partition_pool_and_preserve_credit():
    """Each sibling group appears once per epoch; credit is never re-normalized."""
    cfg = TrainingConfig(
        num_generations=2, questions_per_update=3, minibatch_questions=2
    )
    groups = tuple(make_group(cfg, number=i) for i in range(3))
    pool = prepare_update(groups, tuple(label(g) for g in groups))
    first, n = minibatch_update(pool, 0)
    second, _ = minibatch_update(pool, 1)
    again, _ = minibatch_update(pool, 2)
    assert n == 2 and len(first.groups) == 2 and len(second.groups) == 1
    assert first.signals + second.signals == pool.signals
    assert again.signals == first.signals
    assert first.token_count + second.token_count == pool.token_count


def test_new_presets_are_valid_and_matched():
    """Arms share optimizer/rollout settings and differ only in reward type."""
    configs = [
        TrainingConfig(
            **json.loads(Path(f"tinker_training/configs/{name}.json").read_text())
        )
        for name in ("em-30", "process-local-30")
    ]
    assert replace(configs[0], reward="local_process_advantage_v1") == configs[1]
    assert configs[0].episodes_per_update == 2560
    assert configs[0].lora_rank == 32
    assert configs[0].learning_rate == pytest.approx(1e-6 * 10)
    assert configs[0].beta == 0.01


def test_runner_resume_discards_unfinished_in_memory_pool(tmp_path, monkeypatch):
    """A saved mid-pool checkpoint resumes at the next pool without raw artifacts."""
    from types import SimpleNamespace

    from tests.test_tinker_migration import TYPES, Service, search
    from tinker_training.runner import run

    cfg = TrainingConfig(
        num_generations=2,
        questions_per_update=2,
        minibatch_questions=1,
        max_steps=1,
        num_iterations=1,
        checkpoint_interval=1,
    )
    # max_steps is part of the immutable config; simulate interruption after update one.
    cfg = replace(cfg, max_steps=2)
    groups = tuple(make_group(cfg, number=i) for i in range(2))
    train = SimpleNamespace(
        questions=tuple(g.episodes[0].question for g in groups),
        labels=tuple(label(g) for g in groups),
    )
    calls = []

    def collect(*args, **kwargs):
        calls.append(1)
        return groups

    monkeypatch.setattr("tinker_training.runner.collect_groups", collect)
    service = Service()
    service.fail_save_at = 2
    with pytest.raises(RuntimeError, match="interrupted checkpoint"):
        run(
            cfg,
            train,
            train,
            tmp_path,
            service,
            search,
            types=TYPES,
            identity={"test": "mini"},
            validation_count=0,
        )
    state = json.loads((tmp_path / "progress.json").read_text())
    assert state["step"] == 1 and state["pending"] is None and state["iteration"] == 0
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": "mini"},
        validation_count=0,
    )
    assert len(calls) == 2 and state["step"] == 2
    assert state["metrics"]["minibatch"] == 1
    assert not (tmp_path / "raw").exists()
