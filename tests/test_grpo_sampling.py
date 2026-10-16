from dataclasses import replace

from grpo_training.config import TrainingConfig
from grpo_training.sampling import collect_dynamic
from tests.test_grpo_advantages import make_group
from tests.test_grpo_runner import inputs


def test_replenishes_with_new_questions_preserves_mixed_groups():
    config = TrainingConfig(
        num_generations=2, questions_per_update=2, dynamic_sampling=True
    )
    train = inputs(config)
    calls, events = [], []

    def collect(questions, *args, **kwargs):
        calls.append([q.source_index for q in questions])
        return tuple(
            make_group(
                config,
                number=q.source_index,
                answers=("42", "42") if q.source_index == 0 else ("42", "9"),
            )
            for q in questions
        )

    groups, labels, consumed = collect_dynamic(
        train,
        0,
        config,
        None,
        None,
        None,
        None,
        0,
        collect=collect,
        audit=events.append,
    )
    assert calls == [[0, 1], [2]]
    assert consumed == 3
    assert [g.episodes[0].question.source_index for g in groups] == [1, 2]
    assert [label.id for label in labels] == ["finqa_train_1", "finqa_train_2"]


def test_bounded_uniform_groups_return_empty_without_repeating_questions():
    config = TrainingConfig(
        num_generations=2,
        questions_per_update=1,
        dynamic_sampling=True,
        dynamic_sampling_multiplier=2,
    )
    train = inputs(config)

    def collect(questions, *args, **kwargs):
        return tuple(
            make_group(config, number=q.source_index, answers=("42", "42"))
            for q in questions
        )

    groups, labels, consumed = collect_dynamic(
        train,
        0,
        config,
        None,
        None,
        None,
        None,
        0,
        collect=collect,
        audit=lambda e: None,
    )
    assert (groups, labels, consumed) == ((), (), 2)
    assert replace(config, num_iterations=5).dynamic_sampling


def test_runner_reuses_accepted_batches_and_advances_over_discarded_questions(tmp_path):
    from grpo_training.runner import run_training
    from tests.test_grpo_runner import setup_run

    config = TrainingConfig(
        num_generations=2,
        questions_per_update=2,
        dynamic_sampling=True,
        num_iterations=2,
        max_steps=4,
    )
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(
        tmp_path, config=config
    )
    calls = []

    def collect(questions, config, policy, *args, **kw):
        calls.append((policy.optimizer_step, [q.source_index for q in questions]))
        return tuple(
            make_group(
                config,
                number=q.source_index,
                policy=policy,
                answers=("42", "42") if q.source_index == 0 else ("42", "9"),
            )
            for q in questions
        )

    run_training(
        trainer,
        bundle,
        backend,
        train,
        None,
        store,
        **(kwargs | {"collect_fn": collect}),
    )
    assert calls == [(0, [0, 1]), (0, [2]), (2, [0, 1]), (2, [2])]
    assert trace.count("capture") == 2
    assert trace.count("backward") == 4
    assert [p.iteration for p, _ in store.saved] == [0, 1, 0, 1, 0]
    assert store.saved[-1][0].cursor == 0
