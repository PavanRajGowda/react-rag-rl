from dataclasses import replace

import pytest

from grpo_training.config import TrainingConfig
from grpo_training.validation import select_validation, validation_due
from tests.test_grpo_runner import inputs


def test_fixed_dev_selection_and_alignment():
    prepared = inputs(TrainingConfig())
    dev = replace(
        prepared,
        split="dev",
        questions=tuple(replace(q, split="dev") for q in prepared.questions),
    )
    a = select_validation(dev, 2)
    assert a == select_validation(dev, 2)
    assert len(a.questions) == 2
    assert [q.id for q in a.questions] == [label.id for label in a.labels]
    with pytest.raises(ValueError):
        select_validation(prepared, 2)


def test_validation_schedule():
    assert [s for s in range(51) if validation_due(s, 20, 50)] == [0, 20, 40, 50]


def test_evaluation_runs_full_agent_and_reuses_saved_questions(tmp_path):
    from grpo_training.records import PolicyVersion
    from grpo_training.rollouts import template_digest
    from grpo_training.validation import evaluate
    from tests.test_grpo_environment import response
    from tests.test_grpo_replay import ToyTokenizer
    from tests.test_grpo_runner import ScriptedBackend

    config = TrainingConfig()
    train = inputs(config)
    dev = replace(
        train,
        split="dev",
        questions=tuple(replace(q, split="dev") for q in train.questions),
    )
    dev = select_validation(dev, 2)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    backend = ScriptedBackend(tokenizer)
    summary = evaluate(
        dev, config, policy, backend, tokenizer, lambda p: response(), tmp_path
    )
    assert summary["number_match"] == 1
    assert summary["questions"] == 2
    assert backend.batch_sizes == [1, 1, 1, 1]
    evaluate(dev, config, policy, backend, tokenizer, lambda p: response(), tmp_path)
    assert backend.batch_sizes == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="identity changed"):
        evaluate(
            dev,
            config,
            replace(policy, adapter_sha256="a" * 64),
            backend,
            tokenizer,
            lambda p: response(),
            tmp_path,
        )


def test_runner_validates_initial_periodic_final_without_extra_updates(
    tmp_path, monkeypatch
):
    from grpo_training.runner import run_identity, run_training
    from tests.test_grpo_runner import setup_run

    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)
    dev = replace(
        train,
        split="dev",
        questions=tuple(replace(q, split="dev") for q in train.questions),
    )
    store.identity = run_identity(bundle.config, train, dev)
    observed = []

    def evaluate(dev, config, policy, *args):
        observed.append(policy.optimizer_step)
        return {"step": policy.optimizer_step, "number_match": 1}

    monkeypatch.setattr("grpo_training.validation.evaluate", evaluate)
    run_training(
        trainer,
        bundle,
        backend,
        train,
        None,
        store,
        validation=dev,
        validation_interval=2,
        **kwargs,
    )
    assert observed == [0, 2, 3]
    assert trace.count("backward") == 3
