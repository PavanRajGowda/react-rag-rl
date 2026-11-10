"""Atomic result writes and technical-failure records; no GPU dependencies."""

import json
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path


def write_json(path, value):
    """Atomically replace a JSON file at path with the serializable value.

    Creates parent directories and a unique same-directory temporary file. Briefly
    retries only local Windows sharing/access conflicts during atomic replacement;
    never retries a model/search call. Persistent filesystem failures still raise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Readers/antivirus can briefly hold a Windows handle that denies replace.
        # Do not treat unrelated OS errors as transient or use a non-atomic fallback.
        for attempt in range(7):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 6:
                    raise
                time.sleep(0.02 * 2**attempt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def failure_result(question, config, error, status="infrastructure_error"):
    """Build a failed trajectory result from a question, AgentConfig and exception.

    Returns an empty-turn result dictionary with the requested failure status."""
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
