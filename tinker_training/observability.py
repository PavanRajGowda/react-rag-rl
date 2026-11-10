"""Backend-independent atomic dashboard state and error publication."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from baseline_agent.persistence import write_json


def atomic_json(path, value):
    """Write a JSON artifact through the shared atomic persistence helper.
    Filesystem/serialization errors propagate to the caller.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def publish_runtime_phase(run_root, phase, *, max_steps=None, durable=lambda: None):
    """Publish a fresh live phase record for a run, clearing stale failure information.
    max_steps optionally updates progress limits; durable is an injected flush callback
    invoked after publication.
    """
    root = Path(run_root) / "observability"
    index_path = root / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"steps": [], "references": [], "validation": [], "committed_step": 0}
    )
    live_path = root / "live.json"
    previous = (
        json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    )
    history = previous.get("failure_history", [])
    if previous.get("failure"):
        history = [*history, previous["failure"]]
        atomic_json(
            Path(run_root) / "events" / f"previous-failure-{time.time_ns()}.json",
            {"kind": "previous_attempt_failure", "failure": previous["failure"]},
        )
    now = datetime.now(UTC).isoformat()
    live = {
        "phase": phase,
        "committed_step": index["committed_step"],
        "step": index["committed_step"],
        "max_steps": max_steps if max_steps is not None else previous.get("max_steps"),
        "training_config": previous.get("training_config", {}),
        "failure_history": history,
        "phase_started_at_utc": now,
        "updated_at_utc": now,
    }
    atomic_json(index_path, index)
    atomic_json(live_path, live)
    durable()


def publish_runtime_failure(run_root, stage, error, *, durable=lambda: None):
    """Publish a failed runtime state with stage, exception text and traceback, then call
    the injected durability callback. This exposes startup and training errors to the
    dashboard.
    """
    import traceback

    root = Path(run_root) / "observability"
    index_path = root / "index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"steps": [], "references": [], "validation": [], "committed_step": 0}
    )
    live_path = root / "live.json"
    live = (
        json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    )
    failure = {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }
    live.update(
        phase="interrupted"
        if (
            isinstance(error, (KeyboardInterrupt, SystemExit))
            or type(error).__name__ == "InputCancellation"
        )
        else "failed",
        committed_step=index["committed_step"],
        failure=failure,
        updated_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(index_path, index)
    atomic_json(live_path, live)
    durable()
