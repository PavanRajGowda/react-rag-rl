"""Transient Windows file locks must not regenerate paid trajectories."""

import json
from pathlib import Path

import pytest

from baseline_agent.persistence import write_json


def test_atomic_replace_retries_only_local_sharing_conflicts(tmp_path, monkeypatch):
    """Retry replacement of the same serialized file, keeping old JSON valid meanwhile."""
    target = tmp_path / "status.json"
    write_json(target, {"saved": 1})
    original = Path.replace
    attempts, sleeps = [], []

    def locked(source, destination):
        """Simulate two short Windows sharing violations before successful replacement."""
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(target.read_text()) == {"saved": 1}
            error = PermissionError("test sharing lock")
            error.winerror = 5
            raise error
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("baseline_agent.persistence.time.sleep", sleeps.append)
    write_json(target, {"saved": 2})
    assert json.loads(target.read_text()) == {"saved": 2}
    assert len(set(attempts)) == 1 and sleeps == [0.02, 0.04]
    assert not list(tmp_path.glob("*.tmp"))


def test_permanent_write_error_preserves_old_file(tmp_path, monkeypatch):
    """Stop after bounded local retries and leave the old complete artifact intact."""
    target = tmp_path / "status.json"
    write_json(target, {"saved": 1})
    calls = []

    def denied(source, destination):
        """Keep denying replacement to exercise bounded failure cleanup."""
        calls.append(source)
        error = PermissionError("test")
        error.winerror = 32
        raise error

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr("baseline_agent.persistence.time.sleep", lambda _: None)
    with pytest.raises(PermissionError):
        write_json(target, {"saved": 2})
    assert len(calls) == 7
    assert json.loads(target.read_text()) == {"saved": 1}
    assert not list(tmp_path.glob("*.tmp"))
