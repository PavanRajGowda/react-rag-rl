"""Batched episodes with bounded single-trajectory replacement and group quarantine."""

import hashlib

from tinker_training.records import generation_seed
from tinker_training.replay import QuestionGroup
from tinker_training.trajectory import generate_episodes


def collect_groups(
    questions,
    config,
    policy,
    backend,
    tokenizer,
    search,
    round_index,
    *,
    group_attempts=None,
    search_workers=4,
    audit=lambda event: None,
    retained_episodes=(),
    retry_round=0,
    allow_partial=False,
):
    """Collect question groups using shared asynchronous generation/search tasks.

    Args:
        questions: Distinct question records; each receives num_generations siblings.
        config: TrainingConfig defining generation limits and bounded retries.
        policy: Frozen policy identity required for all requests in this collection.
        backend: Sampling adapter exposing generate(requests).
        tokenizer: Matching tokenizer used to preserve exact prompt/completion IDs.
        search: Retriever callable, optionally exposing batch(payloads).
        round_index: Logical fresh-rollout index, used in group IDs and audit events.
        group_attempts: Optional total attempts per missing trajectory, not per batch.
        search_workers: Historical validated keyword; async trajectory limits are separate.
        audit: Callback receiving progress, completion and failure dictionaries.
        retained_episodes: Healthy trajectories to reuse without regenerating them.
        retry_round: Seed offset for replacement attempts.
        allow_partial: Accept groups with at least two healthy surviving siblings.

    Returns:
        Tuple of usable QuestionGroup records. Exhausted unusable groups are audited
        and omitted; wrong answers and model protocol mistakes remain model samples.

    Shared policy, identity or backend membership violations propagate. Only technical
    trajectory failures trigger replacement; already healthy siblings are retained.
    """
    questions = tuple(questions)
    attempts = (
        config.max_trajectory_attempts if group_attempts is None else group_attempts
    )
    if not questions or len({q.id for q in questions}) != len(questions):
        raise ValueError("Expected distinct rollout questions")
    if (
        type(attempts) is not int
        or not 1 <= attempts <= 3
        or type(search_workers) is not int
        or not 1 <= search_workers <= 8
    ):
        raise ValueError("Invalid bounded trajectory attempts/search concurrency")
    if type(retry_round) is not int or retry_round < 0:
        raise ValueError("Invalid trajectory retry seed round")
    by_id = {q.id: q for q in questions}
    kept = {q.id: {} for q in questions}
    for episode in retained_episodes:
        if (
            episode.question.id not in by_id
            or episode.question != by_id[episode.question.id]
            or episode.config != config
            or episode.policy != policy
            or episode.status in ("infrastructure_error", "postprocessing_error")
            or episode.sample_index in kept[episode.question.id]
        ):
            raise ValueError(
                "Retained trajectory has wrong identity, state or duplicate slot"
            )
        kept[episode.question.id][episode.sample_index] = episode
    for attempt in range(attempts):
        seed_round = retry_round + attempt
        specs = []
        for question in questions:
            peers = kept[question.id]
            gid = (
                next(iter(peers.values())).group_id
                if peers
                else f"round-{round_index}-{question.id}-attempt-0"
            )
            base = int.from_bytes(
                hashlib.sha256(f"{config.seed}:{gid}:{seed_round}".encode()).digest()[
                    :4
                ],
                "big",
            )
            for index in range(config.num_generations):
                if index not in peers:
                    specs.append(
                        {
                            "question": question,
                            "episode_id": f"{gid}-sample-{index}-retry-{seed_round}",
                            "group_id": gid,
                            "sample_index": index,
                            "seed": generation_seed(base, index + 1),
                        }
                    )
        if not specs:
            break
        completed = 0

        for episode in generate_episodes(
            specs, config, policy, backend, tokenizer, search
        ):
            completed += 1
            technical = episode.status in (
                "infrastructure_error",
                "postprocessing_error",
            )
            audit(
                {
                    "kind": "trajectory_failed"
                    if technical
                    else "trajectory_completed",
                    "round": round_index,
                    "attempt": seed_round,
                    "episode_id": episode.episode_id,
                    "question_id": episode.question.id,
                    "status": episode.status,
                }
            )
            if not technical:
                kept[episode.question.id][episode.sample_index] = episode
        audit(
            {
                "kind": "rollout_attempt",
                "round": round_index,
                "attempt": seed_round,
                "completed": completed,
                "requested": len(specs),
            }
        )
    # Relative credit needs at least two survivors. Partial-group acceptance is
    # explicit; a lone trajectory is audited/dropped rather than scored as a group.
    groups = []
    for question in questions:
        peers = kept[question.id]
        if len(peers) == config.num_generations or (allow_partial and len(peers) >= 2):
            groups.append(
                QuestionGroup(tuple(peers.values()), allow_partial=allow_partial)
            )
        else:
            audit(
                {
                    "kind": "group_dropped",
                    "reason": "trajectory_attempts_exhausted",
                    "round": round_index,
                    "question_id": question.id,
                    "retained_episode_ids": [e.episode_id for e in peers.values()],
                    "missing_sample_indices": [
                        i for i in range(config.num_generations) if i not in peers
                    ],
                }
            )
    return tuple(groups)
