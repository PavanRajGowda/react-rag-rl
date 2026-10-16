"""Train/dev selection with a question-only boundary to model inputs."""

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

from baseline_agent.prompt import messages
from grpo_training.config import TrainingConfig


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    split: str
    source_index: int
    dataset_revision: str


@dataclass(frozen=True)
class RewardLabel:
    id: str
    gold_answer: str
    gold_context_id: (
        str  # Training reward/diagnostics only; never a retrieval argument.
    )


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    questions: tuple[Question, ...]
    labels: tuple[RewardLabel, ...]
    skipped_blank_ids: tuple[str, ...]
    selected_source_count: int


def policy_messages(question):
    """Only this allowlisted object may enter the prompt-building interface.

    Labels can live in the trainer process later, but never in its model input.
    No company, year, gold context, answer or program is supplied as a hint.
    """
    if type(question) is not Question:
        raise TypeError("policy_messages accepts Question only, never a raw row/label")
    return messages(question.text)


def validate_selection(split, start, count):
    if split not in ("train", "dev"):
        raise ValueError(
            "Training preparation permits train/dev only; test is held out"
        )
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("count must be a positive integer or None")


def prepare_rows(rows, split, config: TrainingConfig, start=0, count=None):
    validate_selection(split, start, count)
    end = len(rows) if count is None else start + count
    if start >= len(rows) or end > len(rows):
        raise ValueError("Requested selection exceeds the split")
    questions, labels, skipped, seen = [], [], [], set()
    for index in range(start, end):
        row = rows[index]
        expected = f"finqa_{split}_{index}"
        if row.get("id") != expected or row.get("split", split) != split:
            raise ValueError(f"Unexpected ID/split at {split} row {index}")
        if row["id"] in seen:
            raise ValueError("Duplicate question ID")
        seen.add(row["id"])
        text = row.get("question")
        if text is None or (isinstance(text, str) and not text.strip()):
            skipped.append(row["id"])
            continue
        if not isinstance(text, str):
            raise TypeError(f"Question must be text: {row['id']}")
        gold = row.get("program_answer")
        context_id = row.get("context_id")
        if (
            isinstance(gold, bool)
            or not isinstance(gold, (str, int, float))
            or not str(gold).strip()
        ):
            raise ValueError(f"Missing or unsupported reward label: {row['id']}")
        if isinstance(gold, float) and not math.isfinite(gold):
            raise ValueError(f"Nonfinite reward label: {row['id']}")
        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError(f"Missing context ID: {row['id']}")
        questions.append(
            Question(row["id"], text.strip(), split, index, config.dataset_revision)
        )
        labels.append(RewardLabel(row["id"], str(gold), context_id))
    if not questions:
        raise ValueError("Selection contains no nonblank questions")
    return PreparedSplit(
        split, tuple(questions), tuple(labels), tuple(skipped), end - start
    )


def load_pinned_split(split, config, start=0, count=None):
    validate_selection(split, start, count)  # Reject test BEFORE any network access.
    from datasets import load_dataset

    rows = load_dataset(
        config.dataset_id,
        config.dataset_config,
        split=split,
        revision=config.dataset_revision,
    )
    return prepare_rows(rows, split, config, start, count)


def load_local_splits(corpus_dir, config, train_count=None, dev_count=None):
    """Read prepared QA labels, never document text, for offline inspection.

    This checks provenance declarations and file structure, not authenticity of
    arbitrary local files. File hashes are recorded in the preflight artifact.
    """
    corpus_dir = Path(corpus_dir)
    manifest_path, labels_path = (
        corpus_dir / "manifest.json",
        corpus_dir / "labels.jsonl",
    )
    manifest_bytes, label_bytes = manifest_path.read_bytes(), labels_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {
        "dataset": config.dataset_id,
        "revision": config.dataset_revision,
        "corpus_version": config.corpus_version,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Local manifest {key} disagrees with training configuration"
            )
    splits = {"train": [], "dev": []}
    for line in label_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") in splits:
            splits[row["split"]].append(row)
    for split, rows in splits.items():
        if len(rows) != manifest.get("split_questions", {}).get(split):
            raise ValueError(f"Local {split} count disagrees with manifest")
    train = prepare_rows(splits["train"], "train", config, count=train_count)
    dev = prepare_rows(splits["dev"], "dev", config, count=dev_count)
    hashes = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
    }
    return train, dev, hashes


def sample_training_questions(prepared, count, seed):
    """Seeded sampling without replacement; labels travel only in their own tuple."""
    if (
        prepared.split != "train"
        or type(count) is not int
        or not 1 <= count <= len(prepared.questions)
    ):
        raise ValueError("Sampling requires a valid nonempty train-only subset")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid subset seed")
    if len(prepared.questions) != len(prepared.labels):
        raise ValueError("Question/label lengths disagree")
    if any(
        q.id != label.id
        for q, label in zip(prepared.questions, prepared.labels, strict=True)
    ):
        raise ValueError("Question/label order disagrees")
    indices = random.Random(seed).sample(range(len(prepared.questions)), count)
    return replace(
        prepared,
        questions=tuple(prepared.questions[i] for i in indices),
        labels=tuple(prepared.labels[i] for i in indices),
        skipped_blank_ids=(),
        selected_source_count=count,
    )
