"""Fixed dev evaluation, separate from training groups and gradient replay."""

import hashlib
import random
import time
from dataclasses import asdict, replace

from tinker_training.observability import atomic_json
from tinker_training.provenance import split_record
from tinker_training.rewards import episode_reward
from tinker_training.trajectory import generate_episodes


def select_validation(dev, count=50, seed=1729):
    """Choose a fixed, seeded subset of development questions while preserving aligned
    labels. Returns a PreparedSplit; the same selected questions can be reused across
    checkpoints.
    """
    if (
        dev.split != "dev"
        or type(count) is not int
        or not 0 < count <= len(dev.questions)
    ):
        raise ValueError("Validation requires a nonempty dev-only selection")
    split_record(dev)  # Check question/label alignment before selection.
    indices = random.Random(seed).sample(range(len(dev.questions)), count)
    return replace(
        dev,
        questions=tuple(dev.questions[i] for i in indices),
        labels=tuple(dev.labels[i] for i in indices),
    )


def validation_due(step, interval, final_step):
    """Return True at step zero, each positive interval, and the final optimizer step."""
    return step == 0 or step == final_step or step % interval == 0


def evaluate(
    dev,
    config,
    policy,
    backend,
    tokenizer,
    search,
    root,
    *,
    seed=1729,
    trajectory_attempts=1,
):
    """Evaluate a policy snapshot on fixed development questions and persist scored
    results. Uses the shared recorder/scheduler, with bounded whole-trajectory technical
    retries. Existing checkpoint validation artifacts support resume; labels are read
    only after generation for scoring.
    """
    if not 1 <= trajectory_attempts <= 3:
        raise ValueError("Trajectory attempts must be 1..3")
    split_record(dev)
    if dev.split != "dev":
        raise ValueError("Evaluation must use dev questions")
    root = root / "observability" / "validation" / f"{policy.optimizer_step:08d}"
    identity = {
        "selection": split_record(dev),
        "policy": asdict(policy),
        "config": config.to_dict(),
        "seed": seed,
        "trajectory_attempts": trajectory_attempts,
    }
    import json

    identity_path = root / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Validation identity changed on resume")
    else:
        atomic_json(identity_path, identity)
    rows, recorders = [], []
    started = time.perf_counter()
    labels = {label.id: label for label in dev.labels}
    for question in dev.questions:
        path = root / f"{question.id}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        key = f"validation-{question.id}"
        recorders.append(
            {
                "question": question,
                "episode_id": key,
                "group_id": key,
                "sample_index": 0,
                "seed": int(
                    hashlib.sha256(f"{seed}:{question.id}".encode()).hexdigest()[:8], 16
                ),
            }
        )
    for start in range(0, len(recorders), 8):
        chunk = recorders[start : start + 8]
        chunk_started = time.perf_counter()
        for episode in _evaluate_chunk(
            chunk, backend, tokenizer, search, trajectory_attempts, config, policy, seed
        ):
            question, label = episode.question, labels[episode.question.id]
            score = episode_reward(question.id, episode.status, episode.answer, label)
            row = {
                "question_id": question.id,
                "gold_answer": label.gold_answer,
                "gold_document_ids": label.gold_document_ids,
                "score": score.to_dict(),
                "seconds": time.perf_counter() - chunk_started,
                "timing_definition": "shared wall time in synchronous evaluation chunk; not additive across episodes",
                "episode": asdict(episode),
            }
            atomic_json(root / f"{question.id}.json", row)
            rows.append(row)
    failures = sum(not row["score"]["scorable"] for row in rows)
    summary = {
        "step": policy.optimizer_step,
        "questions": len(rows),
        "correct": sum(row["score"]["reward"] == 1 for row in rows),
        "technical_failures": failures,
        "seconds_this_invocation": time.perf_counter() - started,
    }
    summary["exact_match"] = summary["correct"] / len(rows) if not failures else None
    from tinker_training.validation_metrics import validation_efficiency

    summary["efficiency"] = validation_efficiency(rows)
    atomic_json(root.parent / f"{policy.optimizer_step:08d}.json", summary)
    return summary


def _evaluate_chunk(chunk, backend, tokenizer, search, attempts, config, policy, seed):
    """Generate one validation chunk, retry only technical trajectory failures, and retain
    healthy peers. Returns completed episode records; wrong answers remain valid
    evaluated outcomes rather than retry candidates.
    """
    for attempt in range(attempts):
        failed = []
        by_id = {spec["episode_id"]: spec for spec in chunk}
        for episode in generate_episodes(
            chunk, config, policy, backend, tokenizer, search
        ):
            if (
                episode.status not in ("infrastructure_error", "postprocessing_error")
                or attempt + 1 == attempts
            ):
                yield episode
            else:
                question = episode.question
                key = f"validation-{question.id}-retry-{attempt + 1}"
                failed.append(
                    by_id[episode.episode_id]
                    | {
                        "episode_id": key,
                        "group_id": key,
                        "seed": int(
                            hashlib.sha256(
                                f"{seed}:{question.id}:{attempt + 1}".encode()
                            ).hexdigest()[:8],
                            16,
                        ),
                    }
                )
        chunk = failed
        if not chunk:
            break
