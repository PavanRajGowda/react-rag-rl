"""Batched episodes with bounded single-trajectory replacement and group quarantine."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

from grpo_training.environment import SearchEpisode
from grpo_training.records import SamplingParameters, generation_seed
from grpo_training.replay import QuestionGroup
from grpo_training.rollouts import EpisodeRecorder


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
):
    """Keep healthy peers; retry technical failures only, never model mistakes.

    group_attempts is retained as a compatible keyword but bounds attempts per
    missing trajectory. Unusable groups are logged/dropped, not turned into zeros.
    Shared engine/policy/membership failures still raise rather than being hidden.
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
    sampling = SamplingParameters(config.temperature, config.top_p)
    for attempt in range(attempts):
        started, recorders, finished = time.perf_counter(), [], {}
        seed_round = retry_round + attempt
        for question in questions:
            peers = kept[question.id]
            gid = (
                next(iter(peers.values())).group_id
                if peers
                else f"round-{round_index}-{question.id}-attempt-0"
            )
            for index in range(config.num_generations):
                if index in peers:
                    continue
                base = int.from_bytes(
                    hashlib.sha256(
                        f"{config.seed}:{gid}:{seed_round}".encode()
                    ).digest()[:4],
                    "big",
                )
                recorders.append(
                    EpisodeRecorder(
                        SearchEpisode(question, config, search),
                        episode_id=f"{gid}-sample-{index}-retry-{seed_round}",
                        group_id=gid,
                        sample_index=index,
                        seed=generation_seed(base, index + 1),
                        policy=policy,
                        sampling=sampling,
                    )
                )
        if not recorders:
            break
        if seed_round:
            audit(
                {
                    "kind": "trajectory_retry_started",
                    "round": round_index,
                    "retry_round": seed_round,
                    "trajectories": len(recorders),
                    "episode_ids": [r.episode_id for r in recorders],
                }
            )

        def save_finished(
            recorder, finished=finished, started=started, seed_round=seed_round
        ):
            if recorder.episode_id in finished:
                return
            episode = recorder.finish()
            snapshot = json.loads(episode.controller_audit_json)
            snapshot["collection_seconds"] = time.perf_counter() - started
            episode = replace(episode, controller_audit_json=json.dumps(snapshot))
            finished[episode.episode_id] = episode
            audit(
                {
                    "kind": "trajectory_completed",
                    "round": round_index,
                    "attempt": seed_round,
                    "episode": episode.to_dict(),
                }
            )

        with ThreadPoolExecutor(max_workers=search_workers) as pool:
            while any(not r.episode.done for r in recorders):
                ready = [
                    (r, r.prepare(tokenizer)) for r in recorders if not r.episode.done
                ]
                ready = [(r, request) for r, request in ready if request is not None]
                for r in recorders:
                    if r.episode.done:
                        save_finished(r)
                if not ready:
                    continue
                requests = tuple(request for _, request in ready)
                audit(
                    {
                        "kind": "generation_started",
                        "round": round_index,
                        "attempt": seed_round,
                        "requests": len(requests),
                        "active": len(requests),
                        "completed": sum(r.episode.done for r in recorders),
                        "total": len(recorders),
                        "retained_trajectories": sum(len(p) for p in kept.values()),
                    }
                )
                batch = backend.generate(requests)
                if tuple(g.request for g in batch.generations) != requests:
                    raise ValueError(
                        "Generation backend changed rollout membership/order"
                    )
                audit(
                    {
                        "kind": "searches_started",
                        "round": round_index,
                        "attempt": seed_round,
                        "requests": len(requests),
                    }
                )
                futures = {
                    pool.submit(g.record_into, r, tokenizer): r
                    for (r, _), g in zip(ready, batch.generations, strict=True)
                }
                for future in as_completed(futures):
                    future.result()  # Unknown integration errors are not model failures.
                    r = futures[future]
                    if r.episode.done:
                        save_finished(r)
                    audit(
                        {
                            "kind": "collection_progress",
                            "round": round_index,
                            "attempt": seed_round,
                            "completed": sum(r.episode.done for r in recorders),
                            "total": len(recorders),
                            "active": sum(not r.episode.done for r in recorders),
                        }
                    )
                audit(
                    {
                        "kind": "generation_batch",
                        "round": round_index,
                        "attempt": seed_round,
                        "requests": len(requests),
                        "seconds": batch.seconds,
                        "output_tokens": sum(
                            len(getattr(g, "completion_ids", ()))
                            + getattr(g, "raw_token_count", 0)
                            for g in batch.generations
                        ),
                        "input_tokens": sum(len(r.prompt_ids) for r in requests),
                    }
                )
        episodes = []
        for r in recorders:
            save_finished(r)
            episode = finished[r.episode_id]
            episodes.append(episode)
            if episode.status not in ("infrastructure_error", "postprocessing_error"):
                kept[episode.question.id][episode.sample_index] = episode
            else:
                audit(
                    {
                        "kind": "trajectory_failed",
                        "round": round_index,
                        "attempt": seed_round,
                        "episode_id": episode.episode_id,
                        "question_id": episode.question.id,
                        "sample_index": episode.sample_index,
                        "status": episode.status,
                    }
                )
        audit(
            {
                "kind": "rollout_attempt",
                "round": round_index,
                "attempt": seed_round,
                "episodes": [e.to_dict() for e in episodes],
            }
        )
    groups = []
    for question in questions:
        peers = kept[question.id]
        if len(peers) == config.num_generations:
            groups.append(QuestionGroup(tuple(peers.values())))
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
