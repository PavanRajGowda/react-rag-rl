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
