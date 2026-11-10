"""Content-addressed preflight records, not execution or training checkpoints."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tinker_training.config import TrainingConfig
from tinker_training.data import policy_messages

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    """Return a stable SHA-256 hash of canonical JSON metadata; reject nonfinite JSON
    numbers.
    """
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_record(prepared):
    """Describe a PreparedSplit using question IDs and separate hashes of policy inputs and
    private reward labels. Raises ValueError if questions and labels are misaligned.
    """
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
                    "gold_document_ids": label.gold_document_ids,
                    "gold_passage_ids": label.gold_passage_ids,
                }
                for label in prepared.labels
            ]
        ),
    }


def git_snapshot(root):
    """Return the repository HEAD and working-tree status for diagnostic provenance. Git
    failures produce None values rather than claiming a clean revision.
    """

    def command(*args):
        """Run a read-only Git command in the selected root and return stripped stdout, or
        None when Git reports failure.
        """
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": command("rev-parse", "HEAD"),
        "working_tree_status": command("status", "--porcelain"),
    }


def training_source_hashes(root=ROOT):
    """Hash active training/inference code and dependency files using repository-relative
    paths. Excludes changing Git status and absolute workspace paths from identity.
    """
    source_paths = [
        *sorted((root / "tinker_training").glob("*.py")),
        *sorted((root / "batched_inference").glob("*.py")),
        root / "tinker_training/pyproject.toml",
        root / "tinker_training/uv.lock",
        root / "retrieval_service/dataset.py",
        root / "retrieval_service/hotpotqa.py",
        root / "baseline_agent/evidence.py",
        root / "retrieval_service/debug_subset.py",
        root / "baseline_agent/prompt.py",
        root / "baseline_agent/protocol.py",
        root / "baseline_agent/scoring.py",
        root / "baseline_agent/diagnostics.py",
        root / "baseline_agent/retrieval_diagnostics.py",
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }


def build_preflight(config: TrainingConfig, train, dev, input_files=None, root=ROOT):
    """Construct a content-addressed experiment record from config, train/dev selections
    and optional input-file hashes. Rejects split overlap or mismatched split roles;
    performs no model training or cloud calls.
    """
    if train.split != "train" or dev.split != "dev":
        raise ValueError("Expected separate train and dev selections")
    train_record, dev_record = split_record(train), split_record(dev)
    if set(train_record["question_ids"]) & set(dev_record["question_ids"]):
        raise ValueError("Train/dev question IDs overlap")
    source_hashes = training_source_hashes(root)
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
        "environment_validation": "Tinker environment locked; live service verification pending",
        "data_boundary": "Only question text enters model messages. Reward answers and reference IDs are separate.",
        "split_policy": "QA train/dev only. Retrieval corpus retains all indexed document splits, as in baseline; this is not a report-disjoint evaluation.",
        "logical_episodes_per_update": config.episodes_per_update,
    }


def write_preflight(record, output_root):
    """Write a preflight artifact beneath its experiment identity and return its path.
    Existing records are not overwritten; callers must explicitly use a new identity or
    destination.
    """
    output = Path(output_root) / record["experiment_id"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preflight.json"
    # Never overwrite an earlier review record or masquerade as a resumed run.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return path
