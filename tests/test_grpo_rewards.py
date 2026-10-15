"""Reward parity, terminal-state behavior and label-boundary checks; CPU only."""

import json
import subprocess
import sys

import pytest

from baseline_agent.scoring import numeric_value_match
from grpo_training.data import RewardLabel
from grpo_training.rewards import InvalidRewardLabel, episode_reward


def assess(answer, gold="42", status="answered"):
    return episode_reward("q", status, answer, RewardLabel("q", gold, "reference"))


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        ("42", "42"),
        ("41", "42"),
        ("Yes", "1"),
        ("yEs!", "1"),
        ("NO", "0"),
        ("Yes", "0"),
        ("No", "1"),
        ("Yes", "42"),
        ("-25", "0.25"),
        ("25%", "0.25"),
        ("1,250", "1250"),
        ("(42)", "42"),
        ("$42", "42"),
        ("4.2e1", "42"),
        ("0", "0.009"),
        ("0.01", "0"),
        ("0.009", "0.008"),
        ("1.0099", "1"),
        ("1.01", "1"),
        ("1e1000", "1e-1000"),
        (None, "42"),
        ("", "42"),
        ("unknown", "42"),
        ("NaN", "42"),
        ("Infinity", "42"),
        ("42 or 43", "42"),
        ("The answer is 42", "42"),
    ],
)
def test_reward_exactly_reuses_existing_number_match(answer, gold):
    result = assess(answer, gold)
    assert result.scorable
    assert result.reward == result.number_match == numeric_value_match(answer, gold)
    assert result.reward in (0.0, 1.0)
    assert json.loads(json.dumps(result.to_dict()))["scorable"] is True


def test_sign_scale_quirks_are_preserved_and_visible():
    result = assess("-25", "0.25")
    assert result.reward == 1.0  # Official metric ignores sign and powers of ten.
    assert result.sign_agrees is False
    assert result.normalized_prediction == "-25"
    assert result.normalized_gold == "0.25"


@pytest.mark.parametrize(
    "status",
    [
        "search_limit",
        "turn_limit",
        "context_limit",
        "invalid_protocol",
        "answer_policy_violation",
    ],
)
def test_unfinished_episode_has_zero_reward_even_with_partial_correct_value(status):
    result = assess("42", status=status)
    assert result.reward == 0.0 and result.scorable
    assert result.outcome == "unfinished" and result.number_match is None


@pytest.mark.parametrize("status", ["infrastructure_error", "postprocessing_error"])
def test_technical_failure_is_unscorable_even_with_correct_value(status):
    result = assess("42", status=status)
    assert result.reward is None and result.number_match is None
    assert not result.scorable
    assert result.outcome == "technical_failure"


def test_answer_outcomes_and_recovered_success():
    assert assess("42").outcome == "correct"
    assert assess("41").outcome == "incorrect"
    assert assess("unknown").outcome == "invalid_answer"
    assert assess(None).outcome == assess(" ").outcome == "missing_answer"
    # Only the accepted terminal outcome matters; no format-history bonus/penalty.
    assert assess("42").reward == 1.0
    with pytest.raises(TypeError):
        episode_reward("q", "answered", "42", {"id": "q", "gold_answer": "42"})
    with pytest.raises(ValueError, match="ID"):
        episode_reward("other", "answered", "42", RewardLabel("q", "42", "ref"))
    with pytest.raises(ValueError, match="Unknown"):
        assess("42", status="new_unhandled_failure")
    with pytest.raises(TypeError):
        assess(True)


@pytest.mark.parametrize("gold", ["unknown", "NaN", "Infinity", "", None])
def test_invalid_gold_fails_loudly_instead_of_penalizing_the_model(gold):
    with pytest.raises(InvalidRewardLabel):
        assess("42", gold)


def test_reward_import_does_not_load_model_trainer_or_cloud():
    code = "import sys; import grpo_training.rewards; assert not {'torch','trl','vllm','modal','transformers','datasets'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
