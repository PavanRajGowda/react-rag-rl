"""Observation-only per-group metrics and deterministic train sampling."""

import random
from dataclasses import replace

import pytest

from tests.test_grpo_advantages import make_group
from tests.agent_fixtures import ANSWER, CONFIG, QUESTION, SEARCH, recorder, submit
from tinker_training.advantages import prepare_update
from tinker_training.data import PreparedSplit, RewardLabel, sample_training_questions
from tinker_training.metrics import pair, rollout_metrics
from tinker_training.replay import QuestionGroup


def test_retrieval_variation_does_not_change_rewards_or_advantages():
    """Verify retrieval variation does not change rewards or advantages."""
    config = replace(CONFIG, questions_per_update=1)
    a, b = recorder(0, config=config), recorder(1, config=config)
    submit(a, SEARCH)
    submit(a, SEARCH)
    submit(a, ANSWER)
    submit(b, SEARCH)
    submit(b, ANSWER)
    gold = RewardLabel(QUESTION.id, "42", "OLD_EVIDENCE")
    update = prepare_update([QuestionGroup((a.finish(), b.finish()))], [gold])
    before = update.signals
    metrics, _ = rollout_metrics(update, [gold])
    assert update.signals == before
    assert [s.reward for s in update.signals] == [1, 1]
    assert [s.advantage for s in update.signals] == [0, 0]
    group = metrics["groups"][0]
    assert group["all_correct"] and not group["learning_signal"]
    assert group["retrieval"]["ever"] == pair(2, 2)
    assert group["retrieval"]["repeated_requests"] == pair(1, 3)
    assert group["retrieval"]["no_new_context"] == pair(1, 3)
    assert group["answer_diversity"]["distinct_normalized_values"] == 1
    assert group["quality_metrics"]["repeated_request_fraction"]["varies"]
    assert group["quality_metrics"]["repeated_request_fraction"]["mean"] == 0.25
    assert not group["quality_metrics"]["reference_ever"]["varies"]
    assert metrics["rates"]["groups_with_variation_repeated_request_fraction"] == pair(
        1, 1
    )
    assert metrics["rates"]["groups_with_variation_reference_ever"] == pair(0, 1)


def test_no_search_fraction_is_unavailable_not_zero():
    """Verify no search fraction is unavailable not zero."""
    group = make_group()
    gold = RewardLabel(group.episodes[0].question.id, "42", "reference")
    metrics, _ = rollout_metrics(prepare_update([group], [gold]), [gold])
    summary = metrics["groups"][0]["quality_metrics"]["no_new_context_fraction"]
    assert summary["n"] == 0 and summary["mean"] is None
    assert metrics["rates"]["groups_with_variation_no_new_context_fraction"] == pair(
        0, 0
    )


def test_seeded_train_subset_preserves_labels_and_rng():
    """Verify seeded train subset preserves labels and rng."""
    episodes = [make_group(number=i).episodes[0] for i in range(12)]
    prepared = PreparedSplit(
        "train",
        tuple(e.question for e in episodes),
        tuple(RewardLabel(e.question.id, "42", "reference") for e in episodes),
        (),
        12,
    )
    before = random.getstate()
    chosen = sample_training_questions(prepared, 8, 42)
    assert random.getstate() == before
    assert chosen == sample_training_questions(prepared, 8, 42)
    assert chosen != sample_training_questions(prepared, 8, 43)
    assert len({q.id for q in chosen.questions}) == 8
    assert [q.id for q in chosen.questions] == [l.id for l in chosen.labels]
    assert chosen.selected_source_count == 8
    with pytest.raises(ValueError, match="train-only"):
        sample_training_questions(replace(prepared, split="dev"), 8, 42)
