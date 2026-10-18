"""Complete-episode outcome advantages; no model, Torch or label-bearing prompts."""

import statistics
from dataclasses import dataclass

from grpo_training.data import RewardLabel
from grpo_training.replay import GroupRequiresRetry, QuestionGroup, replay_rows
from grpo_training.rewards import group_outcome


@dataclass(frozen=True)
class EpisodeSignal:
    episode_id: str
    group_id: str
    reward: float
    advantage: float


@dataclass(frozen=True)
class PreparedUpdate:
    groups: tuple[QuestionGroup, ...]
    signals: tuple[EpisodeSignal, ...]

    @property
    def config(self):
        return self.groups[0].episodes[0].config

    @property
    def policy(self):
        return self.groups[0].episodes[0].policy

    def rows(self):
        signals = {s.episode_id: s for s in self.signals}
        return tuple(
            row | {"advantage": signals[row["episode_id"]].advantage}
            for group in self.groups
            for row in replay_rows(group)
        )

    @property
    def token_count(self):
        return sum(len(row["target_ids"]) for row in self.rows())


def group_advantages(values, config, batch_std, *, positive_only=False):
    """Center, scale, then optionally clip negatives; do not re-center afterward."""
    mean = statistics.mean(values)
    raw = tuple(value - mean for value in values)
    std = statistics.stdev(values) if config.scale_rewards == "group" else batch_std
    normalized = (
        tuple(value / (std + 1e-4) for value in raw)
        if config.scale_rewards != "none"
        else raw
    )
    final = (
        tuple(max(0.0, value) for value in normalized) if positive_only else normalized
    )
    return raw, normalized, final


def prepare_update(groups, labels):
    """All G peers contribute to each baseline, even failures/zero-token episodes.

    Technical failures reject the whole update. No zero-variance group filtering,
    extra format reward, truncated-completion masking or partial-group retries.
    """
    groups = tuple(groups)
    labels = tuple(labels)
    if not groups or any(type(g) is not QuestionGroup for g in groups):
        raise TypeError("Expected complete QuestionGroups")
    config, policy = groups[0].episodes[0].config, groups[0].episodes[0].policy
    if not 1 <= len(groups) <= config.questions_per_update:
        raise ValueError("Update must contain 1..questions_per_update complete groups")
    if len({g.episodes[0].group_id for g in groups}) != len(groups) or len(
        {g.episodes[0].question.id for g in groups}
    ) != len(groups):
        raise ValueError("Duplicate group or question in update")
    if any(type(label) is not RewardLabel for label in labels) or len(
        {label.id for label in labels}
    ) != len(labels):
        raise ValueError("Expected unique separate reward labels")
    by_id = {label.id: label for label in labels}
    if set(by_id) != {g.episodes[0].question.id for g in groups}:
        raise ValueError("Labels must match exactly the update's question IDs")
    episodes = tuple(e for g in groups for e in g.episodes)
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("Episode IDs must be unique across the update")
    if any(
        e.config != config or e.policy != policy or e.question.split != "train"
        for e in episodes
    ):
        raise ValueError(
            "Training update needs one config/policy and train-only questions"
        )
    if any(not g.scorable for g in groups):
        raise GroupRequiresRetry(
            "Reject update with a technical failure; never shrink its groups"
        )
    outcomes = tuple(group_outcome(g, by_id[g.episodes[0].question.id]) for g in groups)
    rewards = tuple(value for outcome in outcomes for value in outcome.rewards)
    batch_std = statistics.stdev(rewards)
    signals = []
    for group, outcome in zip(groups, outcomes, strict=True):
        _, _, advantages = group_advantages(
            outcome.rewards,
            config,
            batch_std,
            positive_only=config.reward == "number_match_turn_efficiency"
            and outcome.all_correct,
        )
        for episode, reward, advantage in zip(
            group.episodes, outcome.rewards, advantages, strict=True
        ):
            signals.append(
                EpisodeSignal(episode.episode_id, episode.group_id, reward, advantage)
            )
    prepared = PreparedUpdate(groups, tuple(signals))
    if not prepared.token_count:
        raise ValueError("Update has no sampled tokens to train")
    return prepared
