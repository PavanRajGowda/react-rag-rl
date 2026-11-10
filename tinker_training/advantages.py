"""Group-relative outcome advantages with broadcast or local process coefficients."""

import statistics
from dataclasses import dataclass
from functools import cached_property

from tinker_training.data import RewardLabel
from tinker_training.replay import GroupRequiresRetry, QuestionGroup, replay_rows
from tinker_training.rewards import group_outcome


@dataclass(frozen=True)
class EpisodeSignal:
    """One trajectory reward and its group-relative advantage."""

    episode_id: str
    group_id: str
    reward: float
    advantage: float


@dataclass(frozen=True)
class PreparedUpdate:
    """Immutable groups and computed signals for one logical optimizer update. Lazily
    caches replay rows and their completion-token denominator; contains no model tensors
    or gradients.
    """

    groups: tuple[QuestionGroup, ...]
    signals: tuple[EpisodeSignal, ...]
    token_weights: tuple = ()
    original_token_count: int | None = None
    # Freeze eligibility before KL so KL alone cannot retain a task-zero group.
    task_active_group_ids: tuple | None = None

    @property
    def config(self):
        """Return the common TrainingConfig from the first rollout group."""
        return self.groups[0].episodes[0].config

    @property
    def policy(self):
        """Return the frozen rollout policy identity shared by this update."""
        return self.groups[0].episodes[0].policy

    def rows(self):
        """Return cached per-turn replay rows with episode and token credit attached."""
        return self._rows

    @cached_property
    def _rows(self):
        """Build replay rows once, joining each sampled turn to its computed credit.

        Raises ValueError if a turn or token-credit length cannot be matched. The cached
        result avoids rebuilding token positions on every loss or metric access.
        """
        # Join by stable episode/turn identity, never by incidental list order.
        # One trajectory can generate several rows; all share its scalar advantage.
        signals = {s.episode_id: s for s in self.signals}
        rows = []
        for group in self.groups:
            for row in replay_rows(group):
                signal = signals[row["episode_id"]]
                rows.append(row | {"advantage": signal.advantage})
        if self.token_weights:
            if len(self.token_weights) != len(rows):
                raise ValueError("Token-weight row alignment changed")
            rows = [
                r | {"token_advantages": w}
                for r, w in zip(rows, self.token_weights, strict=True)
            ]
        return tuple(rows)

    @cached_property
    def token_count(self):
        """Return the total number of sampled completion tokens across replay rows.

        Prompt and retrieved-evidence tokens do not contribute to this denominator.
        """
        return (
            self.original_token_count
            if self.original_token_count is not None
            else sum(len(row["target_ids"]) for row in self.rows())
        )


def group_advantages(values):
    """Mean-center rewards within each sibling group; do not scale by std or RMS."""
    mean = statistics.mean(values)
    return tuple(value - mean for value in values)


def prepare_update(groups, labels, tokenizer=None):
    """Validate rollout groups and labels, then prepare immutable replay credit.

    Returns PreparedUpdate, containing the original groups and one EpisodeSignal
    per trajectory. Local shaping adds turn/span adjustments to the group-centered
    outcome coefficient; legacy profiles broadcast their trajectory advantage.
    This is pure CPU
    preparation: no model forward pass, gradient computation or optimizer step occurs.
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
    signals = []
    local = config.reward == "local_process_advantage_v1"
    local_rows = {}
    active = set()
    for group, outcome in zip(groups, outcomes, strict=True):
        advantages = group_advantages(outcome.rewards)
        for episode, reward, advantage in zip(
            group.episodes, outcome.rewards, advantages, strict=True
        ):
            signals.append(
                EpisodeSignal(episode.episode_id, episode.group_id, reward, advantage)
            )
            if local:
                from tinker_training.local_credit import local_credit, protocol_mask
                turns = compact_episode(episode, by_id[episode.question.id])["turns"]
                for turn, generation in zip(turns, episode.generations, strict=True):
                    if turn["I"]:
                        if tokenizer is None:
                            raise ValueError("Local protocol masking requires the rollout tokenizer")
                        mask, _ = protocol_mask(list(generation.completion_ids), tokenizer,
                                                generation.request.assistant_prefill)
                    else:
                        mask = [0] * turn["tokens"]
                    weights = tuple(local_credit(advantage, turn | {"protocol_mask":mask},
                                                 config.local_gold_weight,
                                                 config.local_search_penalty_weight,
                                                 config.local_protocol_penalty_weight))
                    local_rows[(episode.episode_id, turn["turn"])] = weights
                    if any(v != 0 for v in weights):
                        active.add(episode.group_id)
    prepared = PreparedUpdate(groups, tuple(signals))
    if local:
        from dataclasses import replace
        prepared = replace(prepared, token_weights=tuple(
            local_rows[(r["episode_id"],r["turn"])] for r in prepared.rows()),
            task_active_group_ids=tuple(sorted(active)))
    if not prepared.token_count:
        raise ValueError("Update has no sampled tokens to train")
    return prepared


def compact_episode(episode, label):
    """Derive turn-level acquisition/error events for trajectory reward scoring.

    Decodes controller audit JSON and derives acquisition/error events. Gold labels
    are used for scoring only; they are not added to the policy prompt.
    """
    import json

    from tinker_training.process_events import events
    from tinker_training.rewards import episode_reward

    if not episode.controller_audit_json and episode.generations:
        raise ValueError("Turn reward requires a controller search audit")
    audit = json.loads(episode.controller_audit_json or "{}")
    generations = [
        {
            "turn": g.request.turn,
            "text": g.controller_output,
            "tokens": len(g.completion_ids),
        }
        for g in episode.generations
    ]
    score = episode_reward(episode.question.id, episode.status, episode.answer, label)
    if not score.scorable:
        raise GroupRequiresRetry("Technical episode has no semantic reward")
    return {
        "episode_id": episode.episode_id,
        "group_id": episode.group_id,
        "question_id": episode.question.id,
        "split": episode.question.split,
        "correct": int(score.reward == 1),
        "status": episode.status,
        "gold_count": len(label.gold_document_ids),
        "turns": events(
            generations, audit, label.gold_document_ids, label.gold_passage_ids
        ),
    }


def minibatch_update(prepared, index):
    """Select a deterministic mini-batch without recomputing pool-frozen credit.

    Siblings remain together. index spans mini-batches and then reuse epochs;
    the returned token denominator is specific to this optimizer update.
    """
    size = prepared.config.minibatch_questions or len(prepared.groups)
    batches = tuple(
        prepared.groups[start : start + size]
        for start in range(0, len(prepared.groups), size)
    )
    groups = batches[index % len(batches)]
    ids = {e.episode_id for group in groups for e in group.episodes}
    return PreparedUpdate(
        groups,
        tuple(s for s in prepared.signals if s.episode_id in ids),
        tuple(
            w
            for row, w in zip(prepared.rows(), prepared.token_weights, strict=True)
            if row["episode_id"] in ids
        )
        if prepared.token_weights
        else (),
        task_active_group_ids=prepared.task_active_group_ids,
    ), len(batches)


def filter_constant_groups(prepared):
    """Skip task-tied groups inside an original minibatch without regrouping/resampling.

    Omission also removes their KL-only contributions. Keep the original sampled
    token denominator so the remaining gradients are not amplified. Return None
    for a fully empty minibatch; callers must not invoke Adam on that batch.
    """
    from collections import Counter

    signals = {s.episode_id: s for s in prepared.signals}
    kept = tuple(
        g
        for g in prepared.groups
        if (g.episodes[0].group_id in prepared.task_active_group_ids
            if prepared.task_active_group_ids is not None
            else len({signals[e.episode_id].reward for e in g.episodes}) > 1)
    )
    kept_ids = {g.episodes[0].group_id for g in kept}
    skipped = tuple(
        g for g in prepared.groups if g.episodes[0].group_id not in kept_ids
    )

    def difficulty(groups):
        return dict(Counter(g.episodes[0].question.difficulty for g in groups))

    ids = {e.episode_id for g in kept for e in g.episodes}
    metrics = {
        "original_groups": len(prepared.groups),
        "surviving_groups": len(kept),
        "skipped_constant_reward_groups": len(skipped),
        "difficulty_original": difficulty(prepared.groups),
        "difficulty_surviving": difficulty(kept),
        "difficulty_skipped": difficulty(skipped),
        "original_token_denominator": prepared.token_count,
        "surviving_tokens": sum(
            len(r["target_ids"]) for r in prepared.rows() if r["episode_id"] in ids
        ),
        "kl_only_groups_omitted": bool(skipped) and prepared.config.beta > 0,
        "filter_rule": "nonzero_local_task_coefficients" if prepared.task_active_group_ids is not None else "nonconstant_trajectory_reward",
        "skipped_task_zero_groups": len(skipped) if prepared.task_active_group_ids is not None else None,
    }
    if not kept:
        return None, metrics
    weights = (
        tuple(
            w
            for r, w in zip(prepared.rows(), prepared.token_weights, strict=True)
            if r["episode_id"] in ids
        )
        if prepared.token_weights
        else ()
    )
    return PreparedUpdate(
        kept,
        tuple(s for s in prepared.signals if s.episode_id in ids),
        weights,
        prepared.token_count,
        prepared.task_active_group_ids,
    ), metrics
