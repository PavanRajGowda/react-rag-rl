"""Pure-Python grouping and teacher-forcing inputs; no advantages or loss yet."""

from dataclasses import dataclass

from tinker_training.records import EpisodeRecord
from tinker_training.rewards import TECHNICAL_TERMINATIONS


class GroupRequiresRetry(RuntimeError):
    """Do not shrink a question group after a technical failure."""


@dataclass(frozen=True)
class QuestionGroup:
    """Surviving trajectories sampled for one question under one rollout policy. Explicit
    partial groups still need at least two distinct healthy siblings.
    """

    episodes: tuple[EpisodeRecord, ...]
    allow_partial: bool = False

    def __post_init__(self):
        """Validate sibling membership, common policy/config and surviving sample slots.

        Complete groups require all configured siblings; explicitly partial groups need
        at least two surviving distinct trajectories for relative reward calculation.
        """
        episodes = tuple(self.episodes)
        if not episodes or any(
            type(episode) is not EpisodeRecord for episode in episodes
        ):
            raise ValueError("Expected complete episode records")
        first = episodes[0]
        if (
            not self.allow_partial and len(episodes) != first.config.num_generations
        ) or (
            self.allow_partial
            and not 2 <= len(episodes) <= first.config.num_generations
        ):
            raise ValueError("Group must retain exactly G episodes, never G turns")
        for episode in episodes:
            if (
                episode.group_id,
                episode.question,
                episode.config,
                episode.policy,
                episode.sampling,
            ) != (
                first.group_id,
                first.question,
                first.config,
                first.policy,
                first.sampling,
            ):
                raise ValueError(
                    "Mixed question, policy, config or sampling inside a group"
                )
        indices = [episode.sample_index for episode in episodes]
        if (
            len(set(indices)) != len(indices)
            or any(not 0 <= i < first.config.num_generations for i in indices)
            or (
                not self.allow_partial and sorted(indices) != list(range(len(episodes)))
            )
        ):
            raise ValueError("Missing or duplicated sample indices")
        if len({episode.episode_id for episode in episodes}) != len(episodes) or len(
            {episode.seed for episode in episodes}
        ) != len(episodes):
            raise ValueError(
                "Episode IDs and sample seeds must be unique within a group"
            )
        object.__setattr__(
            self,
            "episodes",
            tuple(sorted(episodes, key=lambda episode: episode.sample_index)),
        )

    @property
    def scorable(self):
        """Return whether every retained trajectory is free of technical termination."""
        return not any(
            episode.status in TECHNICAL_TERMINATIONS for episode in self.episodes
        )


def replay_rows(group):
    """Build exact per-turn model inputs, completion targets and direct-loss masks.

    Each row concatenates recorded prompt IDs and sampled output IDs. Prompt/tool
    positions receive mask 0 and ignored labels; output positions receive mask 1.
    logit_positions selects the preceding position predicting each completion token.
    No forward pass or padding occurs here. Unscorable groups raise GroupRequiresRetry.
    """
    if type(group) is not QuestionGroup:
        raise TypeError("Expected a validated QuestionGroup")
    if not group.scorable:
        raise GroupRequiresRetry(
            "Technical failure: retry/exclude the complete group; no partial replay"
        )
    rows = []
    for episode in group.episodes:
        for record in episode.generations:
            # Earlier model output and search evidence are already in this exact
            # prompt. Only the NEW completion is a direct prediction target here;
            # re-encoding decoded text could change token boundaries.
            prefix, output = record.request.prompt_ids, record.completion_ids
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "group_id": episode.group_id,
                    "question_id": episode.question.id,
                    "sample_index": episode.sample_index,
                    "turn": record.request.turn,
                    "policy_id": episode.policy.id,
                    "input_ids": prefix + output,
                    "loss_mask": (0,) * len(prefix) + (1,) * len(output),
                    "labels": (-100,) * len(prefix) + output,
                    "sampling_logprobs": (0.0,) * len(prefix)
                    + record.sampling_logprobs,
                    "logit_positions": tuple(
                        range(len(prefix) - 1, len(prefix) + len(output) - 1)
                    ),
                    "target_ids": output,
                    "finish_reason": record.finish_reason,
                }
            )
    return rows
