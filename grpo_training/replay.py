"""Pure-Python grouping and teacher-forcing inputs; no advantages or loss yet."""

from dataclasses import dataclass

from grpo_training.records import EpisodeRecord
from grpo_training.rewards import TECHNICAL_TERMINATIONS


class GroupRequiresRetry(RuntimeError):
    """Do not shrink a question group after a technical failure."""


@dataclass(frozen=True)
class QuestionGroup:
    episodes: tuple[EpisodeRecord, ...]

    def __post_init__(self):
        episodes = tuple(self.episodes)
        if not episodes or any(
            type(episode) is not EpisodeRecord for episode in episodes
        ):
            raise ValueError("Expected complete episode records")
        first = episodes[0]
        if len(episodes) != first.config.num_generations:
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
        if sorted(episode.sample_index for episode in episodes) != list(
            range(len(episodes))
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
        return not any(
            episode.status in TECHNICAL_TERMINATIONS for episode in self.episodes
        )


def replay_rows(group):
    if type(group) is not QuestionGroup:
        raise TypeError("Expected a validated QuestionGroup")
    if not group.scorable:
        raise GroupRequiresRetry(
            "Technical failure: retry/exclude the complete group; no partial replay"
        )
    rows = []
    for episode in group.episodes:
        for record in episode.generations:
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
