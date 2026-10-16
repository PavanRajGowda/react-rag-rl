"""Fault injection: retain expensive valid work and isolate recoverable failures."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from grpo_training.advantages import prepare_update
from grpo_training.config import TrainingConfig
from grpo_training.records import generation_seed
from grpo_training.recovery import probability_report, recover_probability_capture
from grpo_training.replay import QuestionGroup
from grpo_training.runner import run_training
from grpo_training.trainer import CapturedUpdate
from tests.test_grpo_advantages import label, make_group
from tests.test_grpo_runner import setup_run


def captured(prepared, bad_id=None):
    vectors = []
    for row in prepared.rows():
        values = [-2.0] * len(row["target_ids"])
        if row["episode_id"] == bad_id:
            values[0] = -5.0
        vectors.append(tuple(values))
    return CapturedUpdate(
        prepared, tuple(vectors), tuple((1.0,) * len(v) for v in vectors)
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_probability_recovery_retains_good_peers_and_unrelated_group(persistent):
    config = TrainingConfig(
        num_generations=2, questions_per_update=2, scale_rewards="batch"
    )
    groups = (
        make_group(config, number=0, answers=("42", "42")),
        make_group(config, number=1),
    )
    labels = tuple(label(g) for g in groups)
    bad = groups[0].episodes[0]
    calls = []
    events = []

    def capture(trainer, prepared, artifact, bundle):
        calls.append(prepared)
        bad_id = prepared.groups[0].episodes[0].episode_id if len(calls) == 1 else None
        if persistent and len(calls) > 1:
            bad_id = next(
                e.episode_id
                for g in prepared.groups
                for e in g.episodes
                if e.question.id == bad.question.id and e.sample_index == 0
            )
        return captured(prepared, bad_id)

    replacement = replace(
        bad,
        episode_id=bad.episode_id + "-retry",
        seed=999,
        generations=tuple(
            replace(
                g,
                request=replace(
                    g.request,
                    episode_id=bad.episode_id + "-retry",
                    seed=generation_seed(999, g.request.turn),
                ),
            )
            for g in bad.generations
        ),
    )

    def regenerate(questions, retained):
        assert questions == (bad.question,)
        assert retained == (groups[0].episodes[1],)
        return (QuestionGroup((replacement, *retained)),)

    result = recover_probability_capture(
        groups,
        labels,
        trainer=NS(state=NS(global_step=0)),
        artifact=None,
        bundle=None,
        capture=capture,
        regenerate=regenerate,
        audit=events.append,
        phase=lambda *a, **kw: None,
        round_index=0,
        mean_limit=0.1,
        max_limit=1,
    )
    assert len(calls) == 2
    assert (
        next(
            g
            for g in result.prepared.groups
            if g.episodes[0].question.id == groups[1].episodes[0].question.id
        )
        is groups[1]
    )
    assert result.prepared.groups[-1].episodes[-1] in (
        groups[0].episodes[1],
        groups[1].episodes[1],
    )
    if persistent:
        assert result.prepared.groups == (groups[1],)
        assert result.prepared.token_count == sum(
            len(r["target_ids"]) for r in result.prepared.rows()
        )
        expected = prepare_update([groups[1]], [labels[1]])
        assert result.prepared.signals == expected.signals
        assert any(e["kind"] == "group_dropped" for e in events)
    else:
        assert len(result.prepared.groups) == 2
        assert any(e is replacement for g in result.prepared.groups for e in g.episodes)
        assert any(
            e is groups[0].episodes[1]
            for g in result.prepared.groups
            for e in g.episodes
        )
    report = events[0]
    assert report["failed_episode_ids"] == [bad.episode_id]
    assert report["outlier_tokens"][0]["token_id"] >= 0


def test_report_localizes_failure_to_one_episode():
    group = make_group()
    update = captured(
        prepare_update([group], [label(group)]), group.episodes[0].episode_id
    )
    report = probability_report(update)
    assert report["failed_episode_ids"] == [group.episodes[0].episode_id]
    assert report["outlier_count"] == 2  # This toy episode contains two sampled turns.
    assert report["checked_tokens"] == update.prepared.token_count


def test_runner_uses_surviving_complete_group_and_actual_token_count(tmp_path):
    config = TrainingConfig(
        num_generations=2, questions_per_update=2, num_iterations=1, max_steps=2
    )
    trainer, bundle, backend, train, store, kwargs, _trace = setup_run(
        tmp_path, config=config
    )
    collect = kwargs["collect_fn"]
    kwargs["collect_fn"] = lambda *a, **kw: collect(*a, **kw)[:1]
    seen = []

    def backward(trainer, update):
        seen.append(update)
        assert len(update.prepared.groups) == 1
        assert update.prepared.token_count == sum(
            len(r["target_ids"]) for r in update.prepared.rows()
        )
        return 0.25

    kwargs["backward_fn"] = backward
    result = run_training(trainer, bundle, backend, train, None, store, **kwargs)
    assert result["status"] == "completed" and len(seen) == 2
    assert all(
        e["groups"] == 1 and e["episodes"] == 2
        for e in store.events
        if e["kind"] == "optimizer_step"
    )


def test_empty_batch_advances_data_but_not_optimizer_then_recovers(tmp_path):
    config = TrainingConfig(
        num_generations=2, questions_per_update=1, num_iterations=1, max_steps=2
    )
    trainer, bundle, backend, train, store, kwargs, _trace = setup_run(
        tmp_path, config=config
    )
    collect = kwargs["collect_fn"]
    calls = []

    def sometimes_empty(*args, **kw):
        calls.append(1)
        return () if len(calls) == 1 else collect(*args, **kw)

    kwargs["collect_fn"] = sometimes_empty
    result = run_training(trainer, bundle, backend, train, None, store, **kwargs)
    assert result["status"] == "completed"
    assert [p.step for p, _ in store.saved] == [0, 0, 1, 2]
    assert (
        store.saved[1][0].cursor == 1
        and store.saved[1][0].consecutive_empty_batches == 1
    )
    assert result["progress"]["round_index"] == 3
    assert _trace.count("backward") == 2


def test_known_model_failures_are_not_technical_retry_candidates(tmp_path):
    # The ordinary collector's context-limit test checks this without generation.
    config = TrainingConfig(
        num_generations=2, questions_per_update=1, num_iterations=1, max_steps=1
    )
    trainer, bundle, backend, train, store, kwargs, _trace = setup_run(
        tmp_path, config=config
    )
    base_collect = kwargs["collect_fn"]

    def wrong(*args, **kw):
        group = base_collect(*args, **kw)[0]
        return (
            QuestionGroup(
                tuple(
                    replace(e, status="search_limit", answer=None)
                    for e in group.episodes
                )
            ),
        )

    kwargs["collect_fn"] = wrong
    result = run_training(trainer, bundle, backend, train, None, store, **kwargs)
    assert result["status"] == "completed"
    assert len([e for e in store.events if e["kind"] == "rollout_captured"]) == 1
    assert not any(e["kind"] == "probability_recovery" for e in store.events)


def test_collector_drops_only_exhausted_group_and_preserves_other_question():
    from grpo_training.collection import collect_groups
    from grpo_training.records import PolicyVersion
    from grpo_training.rollouts import template_digest
    from grpo_training.vllm_backend import BackendFailure, GenerationBatch
    from tests.test_grpo_replay import ToyTokenizer, search
    from tests.test_grpo_runner import ScriptedBackend, inputs

    config = TrainingConfig(num_generations=2, questions_per_update=2)
    train = inputs(config)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )

    class Backend(ScriptedBackend):
        def generate(self, requests):
            batch = super().generate(requests)
            return GenerationBatch(
                tuple(
                    BackendFailure(
                        g.request, TimeoutError("temporary generation error")
                    )
                    if g.request.question_id == train.questions[0].id
                    and g.request.sample_index == 0
                    else g
                    for g in batch.generations
                ),
                batch.seconds,
            )

    backend = Backend(tokenizer)
    events = []
    groups = collect_groups(
        train.questions[:2],
        config,
        policy,
        backend,
        tokenizer,
        search,
        0,
        audit=events.append,
    )
    assert len(groups) == 1 and groups[0].episodes[0].question == train.questions[1]
    assert backend.batch_sizes == [4, 3, 1]
    assert all("retry-0" in e.episode_id for e in groups[0].episodes)
    assert len([e for e in events if e["kind"] == "group_dropped"]) == 1
    assert len([e for e in events if e["kind"] == "trajectory_failed"]) == 2


def test_completed_trajectories_are_audited_before_later_engine_failure():
    from grpo_training.collection import collect_groups
    from grpo_training.records import PolicyVersion
    from grpo_training.rollouts import template_digest
    from grpo_training.vllm_backend import BackendGeneration, GenerationBatch
    from tests.test_grpo_replay import ANSWER, SEARCH, ToyTokenizer, search
    from tests.test_grpo_runner import inputs

    config = TrainingConfig(num_generations=2, questions_per_update=2)
    train = inputs(config)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    calls = []
    events = []

    def generate(requests):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("shared engine OOM")
        items = []
        for request in requests:
            text = (
                SEARCH
                if request.turn == 1 or request.question_id == train.questions[0].id
                else ANSWER
            )
            ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            items.append(
                BackendGeneration(request, ids, (-2.0,) * len(ids), "action", 1)
            )
        return GenerationBatch(tuple(items), 0.01)

    with pytest.raises(RuntimeError, match="engine OOM"):
        collect_groups(
            train.questions[:2],
            config,
            policy,
            NS(generate=generate),
            tokenizer,
            search,
            0,
            audit=events.append,
        )
    completed = [e["episode"] for e in events if e["kind"] == "trajectory_completed"]
    assert len(completed) == 2 and all(
        e["question"]["id"] == train.questions[1].id for e in completed
    )
