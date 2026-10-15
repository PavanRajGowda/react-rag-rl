"""Batched independent episodes, with bounded whole-question search retries."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

from grpo_training.environment import SearchEpisode
from grpo_training.records import SamplingParameters, generation_seed
from grpo_training.replay import GroupRequiresRetry, QuestionGroup
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
    group_attempts=2,
    search_workers=4,
    audit=lambda event: None,
):
    """Retry only terminal search infrastructure failures, never model mistakes.

    Generation/integration exceptions abort: an engine may be unhealthy (e.g. OOM).
    Successful peers within a failed question are also discarded and regenerated.
    """
    questions = tuple(questions)
    if not questions or len({q.id for q in questions}) != len(questions):
        raise ValueError("Expected distinct rollout questions")
    if (
        type(group_attempts) is not int
        or not 1 <= group_attempts <= 3
        or type(search_workers) is not int
        or not 1 <= search_workers <= 8
    ):
        raise ValueError("Invalid bounded group attempts/search concurrency")
    completed, pending = {}, questions
    sampling = SamplingParameters(config.temperature, config.top_p)
    for attempt in range(group_attempts):
        recorders = []
        for question in pending:
            gid = f"round-{round_index}-{question.id}-attempt-{attempt}"
            base = int.from_bytes(
                hashlib.sha256(f"{config.seed}:{gid}".encode()).digest()[:4], "big"
            )
            for index in range(config.num_generations):
                recorders.append(
                    EpisodeRecorder(
                        SearchEpisode(question, config, search),
                        episode_id=f"{gid}-sample-{index}",
                        group_id=gid,
                        sample_index=index,
                        seed=generation_seed(base, index + 1),
                        policy=policy,
                        sampling=sampling,
                    )
                )
        with ThreadPoolExecutor(max_workers=search_workers) as pool:
            while any(not r.episode.done for r in recorders):
                ready = [
                    (r, r.prepare(tokenizer)) for r in recorders if not r.episode.done
                ]
                ready = [(r, request) for r, request in ready if request is not None]
                if not ready:
                    continue
                requests = tuple(request for _, request in ready)
                batch = backend.generate(requests)
                if tuple(g.request for g in batch.generations) != requests:
                    raise ValueError(
                        "Generation backend changed rollout membership/order"
                    )
                futures = [
                    pool.submit(g.record_into, recorder, tokenizer)
                    for (recorder, _), g in zip(ready, batch.generations, strict=True)
                ]
                for future in futures:
                    future.result()  # Exceptions abort, never fabricate a trajectory.
                audit(
                    {
                        "kind": "generation_batch",
                        "round": round_index,
                        "attempt": attempt,
                        "requests": len(requests),
                        "seconds": batch.seconds,
                    }
                )
        episodes = tuple(r.finish() for r in recorders)
        audit(
            {
                "kind": "rollout_attempt",
                "round": round_index,
                "attempt": attempt,
                "episodes": [e.to_dict() for e in episodes],
            }
        )
        failed = []
        for question in pending:
            group = QuestionGroup(
                tuple(e for e in episodes if e.question.id == question.id)
            )
            if group.scorable:
                completed[question.id] = group
            else:
                failed.append(question)
        pending = tuple(failed)
        if not pending:
            return tuple(completed[q.id] for q in questions)
    raise GroupRequiresRetry(
        "Search failures exhausted whole-group attempts; no update performed"
    )
