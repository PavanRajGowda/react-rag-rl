"""Fixed dev evaluation, separate from training groups and gradient replay."""

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

from grpo_training.environment import SearchEpisode
from grpo_training.observability import atomic_json
from grpo_training.provenance import split_record
from grpo_training.records import SamplingParameters
from grpo_training.rewards import episode_reward
from grpo_training.rollouts import EpisodeRecorder


def select_validation(dev, count=50, seed=1729):
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
    return step == 0 or step == final_step or step % interval == 0


def evaluate(dev, config, policy, backend, tokenizer, search, root, *, seed=1729):
    """One seeded trajectory per question; no reward filtering or optimizer calls.

    Failures stay in the report. Accuracy is unavailable if any technical failure
    remains, rather than silently reporting accuracy on easier surviving questions.
    Completed individual records survive interruption and are reused on resume.
    """
    split_record(dev)
    if dev.split != "dev":
        raise ValueError("Evaluation must use dev questions")
    root = root / "observability" / "validation" / f"{policy.optimizer_step:08d}"
    identity = {
        "selection": split_record(dev),
        "policy": asdict(policy),
        "config": config.to_dict(),
        "seed": seed,
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
            EpisodeRecorder(
                SearchEpisode(question, config, search),
                episode_id=key,
                group_id=key,
                sample_index=0,
                seed=int(
                    hashlib.sha256(f"{seed}:{question.id}".encode()).hexdigest()[:8], 16
                ),
                policy=policy,
                sampling=SamplingParameters(config.temperature, config.top_p),
            )
        )
    # Same independent trajectories, batched inference and four concurrent searches.
    # Never manufacture G-sized training groups or filter by validation outcomes.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(recorders), 8):
            chunk = recorders[start : start + 8]
            chunk_started, saved = time.perf_counter(), set()
            while len(saved) < len(chunk):
                ready = []
                for recorder in chunk:
                    if not recorder.episode.done:
                        request = recorder.prepare(tokenizer)
                        if request is not None:
                            ready.append((recorder, request))
                if ready:
                    requests = tuple(request for _, request in ready)
                    batch = backend.generate(requests)
                    if tuple(g.request for g in batch.generations) != requests:
                        raise ValueError("Validation generation membership changed")
                    futures = [
                        pool.submit(g.record_into, recorder, tokenizer)
                        for (recorder, _), g in zip(
                            ready, batch.generations, strict=True
                        )
                    ]
                    for future in futures:
                        future.result()
                for recorder in chunk:
                    if not recorder.episode.done or recorder.episode_id in saved:
                        continue
                    episode = recorder.finish()
                    question, label = episode.question, labels[episode.question.id]
                    score = episode_reward(
                        question.id, episode.status, episode.answer, label
                    )
                    row = {
                        "question_id": question.id,
                        "gold_answer": label.gold_answer,
                        "gold_context_id": label.gold_context_id,
                        "score": score.to_dict(),
                        "seconds": time.perf_counter() - chunk_started,
                        "timing_definition": "wall time in evaluation chunk, including concurrent work",
                        "episode": asdict(episode),
                    }
                    atomic_json(root / f"{question.id}.json", row)
                    rows.append(row)
                    saved.add(recorder.episode_id)
    failures = sum(not row["score"]["scorable"] for row in rows)
    summary = {
        "step": policy.optimizer_step,
        "questions": len(rows),
        "correct": sum(row["score"]["reward"] == 1 for row in rows),
        "technical_failures": failures,
        "seconds_this_invocation": time.perf_counter() - started,
    }
    summary["number_match"] = summary["correct"] / len(rows) if not failures else None
    from grpo_training.validation_metrics import validation_efficiency

    summary["efficiency"] = validation_efficiency(rows)
    atomic_json(root.parent / f"{policy.optimizer_step:08d}.json", summary)
    return summary
