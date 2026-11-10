"""Terminal outcome reward only. No model, parser, search or trainer execution."""

import json
from dataclasses import asdict, dataclass

from baseline_agent.scoring import score_answer
from tinker_training.data import RewardLabel

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
    """Terminal answer-scoring result. reward=None denotes an unscorable technical failure
    and must not be converted to an incorrect-answer zero.
    """

    question_id: str
    status: str
    outcome: str
    reward: float | None
    exact_match: float | None
    prediction_valid: bool | None
    normalized_prediction: str | None
    normalized_gold: str | None

    @property
    def scorable(self):
        """Return whether this outcome has a semantic model score rather than a technical
        failure.
        """
        return self.reward is not None

    def to_dict(self):
        """Return a JSON-compatible dictionary of this outcome record."""
        return asdict(self) | {"scorable": self.scorable}


def episode_reward(question_id, status, answer, label):
    """Score one terminal trajectory against its private answer label. Model failures/caps
    receive defined model rewards; technical failures remain unscorable rather than
    becoming incorrect examples. Returns an outcome record.
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
    if not isinstance(label.gold_answer, str) or not label.gold_answer.strip():
        raise InvalidRewardLabel(f"Unsupported text gold answer for {question_id}")

    if status in TECHNICAL_TERMINATIONS:
        return EpisodeReward(
            question_id, status, "technical_failure", None, None, None, None, None
        )

    if status != "answered":
        # A partial/stale answer attached to an unfinished episode earns no credit.
        # No final accepted prediction was scored, so exact_match remains N/A.
        return EpisodeReward(
            question_id, status, "unfinished", 0.0, None, None, None, None
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
        scoring["exact_match"],
        scoring["prediction_valid"],
        scoring.get("normalized_prediction"),
        scoring.get("normalized_gold"),
    )


@dataclass(frozen=True)
class DocumentOutcome:
    """Reference document coverage observed over a complete trajectory. Missing/unknown
    audit coverage remains distinct from confirmed zero retrieval.
    """

    required: int
    retrieved: int | None
    complete: bool | None
    retrieved_document_ids: tuple[str, ...] = ()


def document_outcome(episode, label, *, require_audit=False):
    """Score answer correctness and whether the audited trajectory retrieved every required
    reference document. require_audit rejects missing retrieval history instead of
    inferring coverage from the answer.
    """
    gold = set(label.gold_document_ids)
    if not gold:
        return DocumentOutcome(0, None, None)
    if not episode.controller_audit_json:
        if require_audit:
            raise ValueError("Document reward requires the saved search-response audit")
        return DocumentOutcome(len(gold), None, None)
    audit = json.loads(episode.controller_audit_json)
    if audit.get("question", {}).get("id", episode.question.id) != episode.question.id:
        raise ValueError("Search audit belongs to another question")
    searches = audit.get("searches")
    if not isinstance(searches, list):
        raise TypeError("Search audit must contain a searches list")
    seen = set()
    final_turn = len(episode.generations)
    for search in searches:
        turn = search.get("turn")
        if type(turn) is not int or turn < 1:
            raise ValueError("Search audit lacks a valid generation turn")
        if turn > final_turn or (episode.status == "answered" and turn == final_turn):
            continue
        response = search.get("response") or {}
        if response.get("status") != "ok":
            continue
        for doc in response.get("results", []):
            identifier = doc.get("document_id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Search audit document lacks an article ID")
            seen.add(identifier)
    return DocumentOutcome(
        len(gold), len(gold & seen), gold <= seen, tuple(sorted(seen))
    )


@dataclass(frozen=True)
class GroupOutcome:
    """Aligned sibling answer scores, configured training rewards and document coverage.
    Correctness metrics remain separate from shaped reward values.
    """

    scores: tuple[EpisodeReward, ...]  # Exact-match evaluation stays independent.
    rewards: tuple[float | None, ...]  # Configured training objective.
    documents: tuple[DocumentOutcome, ...]

    @property
    def scorable(self):
        """Return whether every sibling outcome has a semantic score rather than a
        technical failure.
        """
        return all(score.scorable for score in self.scores)

    @property
    def all_correct(self):
        """Return whether all scored sibling rewards correspond to correct answers."""
        return all(score.reward == 1 for score in self.scores)

    @property
    def learning_signal(self):
        """Return whether the group has enough scored, varying rewards to produce relative
        outcome credit.
        """
        return self.scorable and len(set(self.rewards)) > 1


def group_outcome(group, label):
    """Compute configured outcome rewards for all surviving siblings using one matching
    question label. Returns a GroupOutcome with reward values and scoring/variation
    flags.
    """
    episodes = group.episodes
    config = episodes[0].config
    scores = tuple(
        episode_reward(e.question.id, e.status, e.answer, label) for e in episodes
    )
    documents = tuple(
        document_outcome(
            e,
            label,
            require_audit=config.reward == "bounded_process_trajectory_v1",
        )
        if score.scorable
        else DocumentOutcome(len(label.gold_document_ids), None, None)
        for e, score in zip(episodes, scores, strict=True)
    )
    if config.reward == "bounded_process_trajectory_v1":
        from tinker_training.advantages import compact_episode

        rewards = tuple(
            bounded_process_reward(
                score, doc, compact_episode(episode, label)["turns"], config
            )["reward"]
            if score.scorable
            else None
            for episode, score, doc in zip(episodes, scores, documents, strict=True)
        )
        return GroupOutcome(scores, rewards, documents)
    return GroupOutcome(scores, tuple(score.reward for score in scores), documents)


def bounded_process_reward(score, documents, turns, config):
    """Return bounded trajectory reward and interpretable component values.

    Coverage counts distinct reference documents, not successful search turns.
    Q is counted once per parsed search (even when both conditions apply). Parsed
    cap-blocked search attempts count in the search denominator. Invalid protocol
    is divided by actual generated turns, including answer turns. Zero searches
    or turns gives a zero penalty rate; no reference documents gives zero coverage,
    not a vacuous acquisition bonus. Technical failures remain unscorable.

    With nonnegative unit-sum weights, P lies in [-1, 1]. Reward is EM + lambda P;
    lambda < .5 strictly separates all correct from all incorrect trajectories.
    These are raw reward guarantees, not guarantees of positive group advantages.
    """
    if not score.scorable:
        return {"reward": None}
    if documents.required and documents.retrieved is None:
        raise ValueError("Bounded reward requires audited document coverage")
    coverage = documents.retrieved / documents.required if documents.required else 0.0
    searches = [t for t in turns if t["kind"] == "search"]
    redundant = sum(t["Q"] for t in searches) / max(1, len(searches))
    invalid = sum(t["I"] for t in turns) / max(1, len(turns))
    if not all(0 <= value <= 1 for value in (coverage, redundant, invalid)):
        raise ValueError("Invalid bounded reward components")
    quality = (
        config.process_coverage_weight * coverage
        - config.process_redundancy_weight * redundant
        - config.process_invalid_weight * invalid
    )
    return {
        "reward": score.reward + config.process_bonus_scale * quality,
        "coverage": coverage,
        "penalized_search_rate": redundant,
        "protocol_violation_rate": invalid,
        "process_quality": quality,
        "process_contribution": config.process_bonus_scale * quality,
    }
