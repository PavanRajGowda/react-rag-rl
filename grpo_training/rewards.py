"""Terminal outcome reward only. No model, parser, search or trainer execution."""

from dataclasses import asdict, dataclass

from baseline_agent.scoring import extract_numeric_value, score_answer
from grpo_training.data import RewardLabel

MODEL_TERMINATIONS = frozenset(
    {
        "answered",
        "search_limit",
        "turn_limit",
        "context_limit",
        "invalid_protocol",
        "answer_policy_violation",
    }
)
TECHNICAL_TERMINATIONS = frozenset({"infrastructure_error", "postprocessing_error"})


class InvalidRewardLabel(ValueError):
    """Bad gold is a data/configuration problem, never an incorrect prediction."""


@dataclass(frozen=True)
class EpisodeReward:
    question_id: str
    status: str
    outcome: str
    reward: float | None
    number_match: float | None
    prediction_valid: bool | None
    normalized_prediction: str | None
    normalized_gold: str | None
    sign_agrees: bool | None

    @property
    def scorable(self):
        return self.reward is not None

    def to_dict(self):
        return asdict(self) | {"scorable": self.scorable}


def episode_reward(question_id, status, answer, label):
    """Score a controller-accepted final answer, not text scraped from a trace.

    `answered` must mean the episode controller accepted the final answer after
    enforcing its protocol/evidence rules. That controller arrives in Step 3.
    Recovery from earlier format errors does not reduce a correct terminal reward.
    Infrastructure failures return None: never coerce them to zero or include them
    in a question group's baseline. Group retries/filtering belong to later steps.
    """
    if type(label) is not RewardLabel:
        raise TypeError("Expected a separate RewardLabel")
    if not isinstance(question_id, str) or not question_id or question_id != label.id:
        raise ValueError("Episode question ID and reward label ID must agree")
    if (
        not isinstance(status, str)
        or status not in MODEL_TERMINATIONS | TECHNICAL_TERMINATIONS
    ):
        raise ValueError(f"Unknown terminal status: {status!r}")
    if answer is not None and not isinstance(answer, str):
        raise TypeError("The controller answer must be a string or None")
    if extract_numeric_value(label.gold_answer) is None:
        raise InvalidRewardLabel(f"Unsupported numerical gold answer for {question_id}")

    if status in TECHNICAL_TERMINATIONS:
        return EpisodeReward(
            question_id, status, "technical_failure", None, None, None, None, None, None
        )

    if status != "answered":
        # A partial/stale answer attached to an unfinished episode earns no credit.
        # No final accepted prediction was scored, so number_match remains N/A.
        return EpisodeReward(
            question_id, status, "unfinished", 0.0, None, None, None, None, None
        )

    scoring = score_answer(answer, label.gold_answer)
    if answer is None or not answer.strip():
        outcome = "missing_answer"
    elif not scoring["prediction_valid"]:
        outcome = "invalid_answer"
    else:
        outcome = "correct" if scoring["score"] == 1 else "incorrect"
    return EpisodeReward(
        question_id,
        status,
        outcome,
        scoring["score"],
        scoring["number_match"],
        scoring["prediction_valid"],
        scoring.get("normalized_prediction"),
        scoring.get("normalized_gold"),
        scoring["sign_agrees"],
    )


@dataclass(frozen=True)
class GroupOutcome:
    """Terminal rewards need successful siblings; no intermediate token rewards."""

    scores: tuple[EpisodeReward, ...]  # Unchanged Number Match correctness.
    rewards: tuple[float | None, ...]  # Configured training objective.
    turns: tuple[int, ...]
    min_successful_turns: int | None

    @property
    def scorable(self):
        return all(score.scorable for score in self.scores)

    @property
    def all_correct(self):
        return all(score.reward == 1 for score in self.scores)

    @property
    def learning_signal(self):
        return self.scorable and len(set(self.rewards)) > 1


def group_outcome(group, label):
    """Correct: 1-lambda*(turns-min successful turns); incorrect/unfinished: 0.

    A turn is one saved model generation, including malformed actions and the
    final answer. Search calls and unsampled context-budget pseudo-turns are not
    additional turns. Technical errors remain unscorable, never fake negatives.
    """
    episodes = group.episodes
    config = episodes[0].config
    scores = tuple(
        episode_reward(e.question.id, e.status, e.answer, label) for e in episodes
    )
    turns = tuple(len(e.generations) for e in episodes)
    successful_turns = [
        turn for turn, score in zip(turns, scores, strict=True) if score.reward == 1
    ]
    minimum = min(successful_turns) if successful_turns else None
    rewards = tuple(score.reward for score in scores)
    if config.reward == "number_match_turn_efficiency" and minimum is not None:
        rewards = tuple(
            1.0 - config.turn_penalty_lambda * (turn - minimum)
            if score.reward == 1
            else score.reward
            for turn, score in zip(turns, scores, strict=True)
        )
    return GroupOutcome(scores, rewards, turns, minimum)
