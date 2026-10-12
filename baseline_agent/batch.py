"""Checkpoint and bounded retry utilities; no GPU dependencies."""

import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path


class PersistenceError(RuntimeError):
    """Continuing without a durable result would silently lose evaluation data."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def failure_result(question, config, error, status="infrastructure_error"):
    return {
        "question": question,
        "config": asdict(replace(config, seed=config.seed + question["example_index"])),
        "status": status,
        "answer": None,
        "turns": [],
        "searches": [],
        "events": [],
        "error": f"{type(error).__name__}: {error}",
    }


def retry_call(operation, attempts=3, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise
            sleep(2**attempt)


def run_batches(
    questions, invoke, save, failed, log, batch_size=10, attempts=3, sleep=time.sleep
):
    """A broken stream retries only questions not yet saved successfully."""
    if not 1 <= batch_size <= 50 or not 1 <= attempts <= 5:
        raise ValueError("batch_size must be 1–50; attempts must be 1–5")
    for offset in range(0, len(questions), batch_size):
        pending = questions[offset : offset + batch_size]
        last_error = RuntimeError("Worker returned before finishing its chunk")
        for attempt in range(attempts):
            if not pending:
                break
            completed = set()
            try:
                for result in invoke(pending):
                    key = result["question"]["id"]
                    if key in completed:
                        continue
                    if key not in {q["id"] for q in pending}:
                        raise PersistenceError(f"Unexpected result ID {key}")
                    save(result)
                    completed.add(key)
            except PersistenceError:
                raise
            except Exception as exc:
                logging.getLogger(__name__).exception(
                    "GPU chunk stream failed; checkpointed results will not be repeated"
                )
                last_error = exc
                log(
                    {
                        "attempt": attempt + 1,
                        "error": f"{type(exc).__name__}: {exc}",
                        "pending_ids": [
                            q["id"] for q in pending if q["id"] not in completed
                        ],
                    }
                )
            pending = [q for q in pending if q["id"] not in completed]
            if pending and attempt + 1 < attempts:
                sleep(2**attempt)
        for question in pending:
            save(failed(question, last_error))
        if pending and len(pending) == len(questions[offset : offset + batch_size]):
            log(
                {
                    "attempt": attempts,
                    "error": "Circuit breaker: a whole chunk failed after bounded retries; recording the remaining rows without more GPU calls",
                    "pending_ids": [q["id"] for q in questions[offset + batch_size :]],
                }
            )
            for question in questions[offset + batch_size :]:
                save(failed(question, last_error))
            return
