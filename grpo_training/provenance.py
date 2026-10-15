"""Content-addressed preflight records, not execution or training checkpoints."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from grpo_training.config import TrainingConfig
from grpo_training.data import policy_messages

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_record(prepared):
    if len(prepared.questions) != len(prepared.labels):
        raise ValueError("Question/label counts disagree")
    if any(
        q.id != label.id
        for q, label in zip(prepared.questions, prepared.labels, strict=True)
    ):
        raise ValueError("Question/label IDs disagree")
    return {
        "split": prepared.split,
        "selected_source_count": prepared.selected_source_count,
        "question_count": len(prepared.questions),
        "question_ids": [q.id for q in prepared.questions],
        "skipped_blank_ids": list(prepared.skipped_blank_ids),
        "policy_inputs_sha256": digest(
            [{"id": q.id, "messages": policy_messages(q)} for q in prepared.questions]
        ),
        "reward_labels_sha256": digest(
            [
                {
                    "id": label.id,
                    "gold_answer": label.gold_answer,
                    "gold_context_id": label.gold_context_id,
                }
                for label in prepared.labels
            ]
        ),
    }


def git_snapshot(root):
    def command(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": command("rev-parse", "HEAD"),
        "working_tree_status": command("status", "--porcelain"),
    }


def build_preflight(config: TrainingConfig, train, dev, input_files=None, root=ROOT):
    if train.split != "train" or dev.split != "dev":
        raise ValueError("Expected separate train and dev selections")
    train_record, dev_record = split_record(train), split_record(dev)
    if set(train_record["question_ids"]) & set(dev_record["question_ids"]):
        raise ValueError("Train/dev question IDs overlap")
    source_paths = [
        *sorted((root / "grpo_training").glob("*.py")),
        root / "grpo_training/pyproject.toml",
        root / "grpo_training/uv.lock",
        root / "baseline_agent/prompt.py",
        root / "baseline_agent/protocol.py",
    ]
    source_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    identity = {
        "config": config.to_dict(),
        "train": train_record,
        "dev": dev_record,
        "input_files": input_files or {},
        "source_sha256": source_hashes,
    }
    return {
        "schema_version": 1,
        "kind": "grpo_preflight_only",
        "execution_status": "not_started",
        "experiment_id": digest(identity)[:16],
        **identity,
        "git": git_snapshot(root),
        "inspection_python": sys.version.split()[0],
        "environment_validation": "uv_lock_resolved_only; GPU imports and runtime unverified",
        "data_boundary": "Only question text enters model messages. Reward answers and reference IDs are separate.",
        "split_policy": "QA train/dev only. Retrieval corpus retains all indexed document splits, as in baseline; this is not a report-disjoint evaluation.",
        "logical_episodes_per_update": config.episodes_per_update,
    }


def write_preflight(record, output_root):
    output = Path(output_root) / record["experiment_id"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preflight.json"
    # Never overwrite an earlier review record or masquerade as a resumed run.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return path
