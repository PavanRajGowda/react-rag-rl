from dataclasses import replace

import pytest

from tests.rollout_fixtures import inputs
from tinker_training.config import TrainingConfig
from tinker_training.validation import select_validation, validation_due


def test_fixed_dev_selection_and_alignment():
    """Verify fixed dev selection and alignment."""
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
    """Verify validation schedule."""
    assert [s for s in range(51) if validation_due(s, 20, 50)] == [0, 20, 40, 50]


def test_validation_technical_failure_is_reported_without_filtering(tmp_path):
    """Verify validation technical failure is reported without filtering."""
    from tests.agent_fixtures import ToyTokenizer
    from tinker_training.backend import Batch as GenerationBatch
    from tinker_training.backend import Failure as BackendFailure
    from tinker_training.records import PolicyVersion
    from tinker_training.policy import template_digest
    from tinker_training.validation import evaluate

    config = TrainingConfig()
    train = inputs(config)
    dev = replace(
        train,
        split="dev",
        questions=tuple(replace(q, split="dev") for q in train.questions),
    )
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )

    class FailedBackend:
        def generate(self, requests):
            """Return failed-generation records to verify validation reports technical
            failures without filtering questions.
            """
            return GenerationBatch(
                tuple(BackendFailure(r, ValueError("bad packet")) for r in requests),
                0.1,
            )

    summary = evaluate(
        dev, config, policy, FailedBackend(), tokenizer, lambda p: {}, tmp_path
    )
    assert summary["technical_failures"] == 3
    assert summary["questions"] == 3
    assert summary["exact_match"] is None
    assert summary["efficiency"]["all"]["questions"] == 3
    assert summary["efficiency"]["correct_only"]["questions"] == 0
    assert summary["efficiency"]["technical_failures"] == 3
    assert summary["efficiency"]["correct_only"]["tokens_per_question"]["value"] is None


def test_evaluation_runs_full_agent_and_reuses_saved_questions(tmp_path):
    """Verify evaluation runs full agent and reuses saved questions."""
    from tests.rollout_fixtures import ScriptedBackend
    from tests.agent_fixtures import response
    from tests.agent_fixtures import ToyTokenizer
    from tinker_training.records import PolicyVersion
    from tinker_training.policy import template_digest
    from tinker_training.validation import evaluate

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
    assert summary["exact_match"] == 1
    assert summary["questions"] == 2
    assert backend.batch_sizes == [1, 1, 1, 1]
    assert summary["efficiency"]["all"]["searches_per_question"]["value"] == 1
    assert summary["efficiency"]["all"]["turns_per_question"]["value"] == 2
    assert summary["efficiency"]["correct_only"]["questions"] == 2
    assert summary["efficiency"]["all"]["turn_limit"]["value"] == 0
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
