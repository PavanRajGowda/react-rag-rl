"""Balanced seeded schedules preserve question/label identity across reward arms."""

from collections import Counter
from dataclasses import replace

import pytest

from tinker_training.config import TrainingConfig
from tinker_training.data import PreparedSplit, Question, RewardLabel, training_schedule


def fixture():
    cfg = TrainingConfig(
        questions_per_update=4, training_question_mix="hard_medium_50_50"
    )
    questions = tuple(
        Question(str(i), "Question", "train", i, cfg.dataset_revision, level)
        for i, level in enumerate(["hard"] * 5 + ["medium"] * 8 + ["easy"] * 3)
    )
    return cfg, PreparedSplit(
        "train",
        questions,
        tuple(RewardLabel(q.id, "a") for q in questions),
        (),
        len(questions),
    )


def test_balanced_pools_match_across_arms_and_wrap():
    cfg, source = fixture()
    a = training_schedule(source, cfg)
    b = training_schedule(source, replace(cfg, reward="bounded_process_trajectory_v1"))
    assert a == b and len(a.questions) == 10
    assert len({q.id for q in a.questions}) == 10
    assert all(q.id == label.id for q, label in zip(a.questions, a.labels, strict=True))
    for start in range(0, 30, 4):
        pool = [a.questions[(start + i) % len(a.questions)] for i in range(4)]
        assert Counter(q.difficulty for q in pool) == {"hard": 2, "medium": 2}
    assert training_schedule(source, replace(cfg, seed=43)) != a


def test_balanced_schedule_fails_on_missing_metadata_or_odd_pool():
    cfg, source = fixture()
    with pytest.raises(ValueError, match="even"):
        training_schedule(source, replace(cfg, questions_per_update=3))
    missing = replace(
        source,
        questions=tuple(replace(q, difficulty="unknown") for q in source.questions),
    )
    with pytest.raises(ValueError, match="metadata"):
        training_schedule(missing, cfg)
