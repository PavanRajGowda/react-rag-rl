import pytest

from baseline_agent.scoring import score_answer, token_f1
from tinker_training.data import RewardLabel
from tinker_training.rewards import episode_reward


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("YES!", "Yes", 1),
        ("No", "Yes", 0),
        ("Sam Bankman-Fried", "sam bankmanfried", 1),
        ("Sam Bankman-Fried", "sam bankman fried", 0),
        ("Sam Altman", "Sam Bankman-Fried", 0),
        ("Insufficient information.", "Insufficient information", 1),
        (None, "Aster", 0),
    ],
)
def test_text_outcome_reward(pred, gold, expected):
    """Verify text outcome reward."""
    r = episode_reward("q", "answered", pred, RewardLabel("q", gold, ("d",)))
    assert r.reward == expected and r.exact_match == expected


def test_partial_name_has_f1_but_not_exact_credit():
    """Verify partial name has f1 but not exact credit."""
    r = score_answer("Bankman-Fried", "Sam Bankman-Fried")
    assert r["score"] == 0 and r["token_f1"] == pytest.approx(2 / 3)
    assert token_f1("no", "yes") == 0


def test_unfinished_is_zero_and_technical_is_unscorable():
    """Verify unfinished is zero and technical is unscorable."""
    label = RewardLabel("q", "Aster", ())
    assert episode_reward("q", "turn_limit", "Aster", label).reward == 0
    assert episode_reward("q", "infrastructure_error", "Aster", label).reward is None
