"""Train/dev selection with a question-only boundary to model inputs."""

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

from baseline_agent.data import verify_prepared_files
from baseline_agent.prompt import messages
from retrieval_service.dataset import DATA_DIR
from tinker_training.config import TrainingConfig


@dataclass(frozen=True)
class Question:
    """Public policy input and source identity for a labeled question. Reference answers
    and evidence are deliberately stored separately in RewardLabel.
    """

    id: str
    text: str
    split: str
    source_index: int
    dataset_revision: str
    difficulty: str = "unknown"  # Reporting metadata, never rendered in policy prompts.


@dataclass(frozen=True)
class RewardLabel:
    """Private answer and reference evidence metadata for retrospective scoring. Never
    passed to the search tool or inserted into generation messages.
    """

    id: str
    gold_answer: str
    gold_document_ids: tuple[
        str, ...
    ] = ()  # Diagnostics only; never a search argument.

    gold_passage_ids: tuple[str, ...] = ()  # Legacy MultiHopRAG artifacts only.

    def __post_init__(self):
        """Freeze supporting evidence ID collections as tuples. A single document-ID string
        becomes a one-element tuple. Source-row answer/evidence validation is performed
        by prepare_rows, not by this constructor.
        """
        object.__setattr__(self, "gold_passage_ids", tuple(self.gold_passage_ids))
        object.__setattr__(
            self,
            "gold_document_ids",
            (self.gold_document_ids,)
            if isinstance(self.gold_document_ids, str)
            else tuple(self.gold_document_ids),
        )


@dataclass(frozen=True)
class PreparedSplit:
    """Aligned question/label tuples and selection provenance for one official split. Also
    records skipped blank question IDs and original selected-source count.
    """

    split: str
    questions: tuple[Question, ...]
    labels: tuple[RewardLabel, ...]
    skipped_blank_ids: tuple[str, ...]
    selected_source_count: int


def policy_messages(question):
    """Build system/user messages from a Question without reference answers or evidence."""
    if type(question) is not Question:
        raise TypeError("policy_messages accepts Question only, never a raw row/label")
    return messages(question.text)


def validate_selection(split, start, count):
    """Validate a labeled split name and optional start/count slice.

    Rejects unsupported splits, negative offsets and invalid counts before loading.
    """
    if split not in ("train", "dev"):
        raise ValueError(
            "Training preparation permits train/dev only; test is held out"
        )
    if type(start) is not int or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("count must be a positive integer or None")


def prepare_rows(rows, split, config: TrainingConfig, start=0, count=None):
    """Separate source rows into aligned question-only inputs and private reward labels.

    Applies the requested split and slice, skips blank questions with an audit list,
    and returns PreparedSplit. Gold answers and evidence stay in RewardLabel, never
    in Question text or policy messages. Invalid source identities raise ValueError.
    """
    validate_selection(split, start, count)
    end = len(rows) if count is None else start + count
    if start >= len(rows) or end > len(rows):
        raise ValueError("Requested selection exceeds the split")
    questions, labels, skipped, seen = [], [], [], set()
    for index in range(start, end):
        row = rows[index]
        if not row.get("id") or row.get("split", split) != split:
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
        gold = row.get("answer")
        document_ids = row.get("gold_document_ids")
        if (
            isinstance(gold, bool)
            or not isinstance(gold, (str, int, float))
            or not str(gold).strip()
        ):
            raise ValueError(f"Missing or unsupported reward label: {row['id']}")
        if isinstance(gold, float) and not math.isfinite(gold):
            raise ValueError(f"Nonfinite reward label: {row['id']}")
        if not isinstance(document_ids, list) or any(
            not isinstance(x, str) or not x for x in document_ids
        ):
            raise ValueError(f"Invalid evidence document IDs: {row['id']}")
        questions.append(
            Question(
                row["id"],
                text.strip(),
                split,
                index,
                config.dataset_revision,
                row.get("level", "unknown"),
            )
        )
        labels.append(
            RewardLabel(
                row["id"],
                str(gold),
                tuple(document_ids),
                ()
                if config.dataset_id == "hotpotqa/hotpot_qa"
                else tuple(
                    sorted(
                        {
                            p
                            for e in row.get("evidence", [])
                            for p in e.get("passage_ids", [])
                        }
                    )
                ),
            )
        )
    if not questions:
        raise ValueError("Selection contains no nonblank questions")
    return PreparedSplit(
        split, tuple(questions), tuple(labels), tuple(skipped), end - start
    )


def load_pinned_split(split, config, start=0, count=None):
    """Load an official labeled HotpotQA split at the configured dataset revision.

    Uses the Hugging Face cache/download mechanism, then delegates selection and
    question/label separation to prepare_rows. Network or cache errors propagate.
    """
    validate_selection(split, start, count)  # Reject test BEFORE any network access.
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != config.dataset_revision:
        raise ValueError("Prepared dataset revision mismatch")
    verify_prepared_files(DATA_DIR, manifest)
    rows = [
        json.loads(line)
        for line in (DATA_DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return prepare_rows(
        [r for r in rows if r["split"] == split], split, config, start, count
    )


def load_local_splits(corpus_dir, config, train_count=None, dev_count=None):
    """Verify prepared corpus files and return train/dev selections with input hashes.

    corpus_dir must contain a completed manifest and matching label artifacts.
    Optional counts limit a local diagnostic selection; the training CLI uses the
    full train split. Returns (train, dev, hashes) for provenance construction.
    """
    corpus_dir = Path(corpus_dir)
    manifest_path, labels_path = (
        corpus_dir / "manifest.json",
        corpus_dir / "labels.jsonl",
    )
    manifest_bytes, label_bytes = manifest_path.read_bytes(), labels_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    verify_prepared_files(corpus_dir, manifest)
    if (
        manifest.get("labels_sha256")
        and hashlib.sha256(label_bytes).hexdigest() != manifest["labels_sha256"]
    ):
        raise ValueError("Prepared label hash disagrees with manifest")
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
    # Prepared labels omit difficulty; join original source metadata for reporting.
    source = corpus_dir.parent / "hotpotqa-source" / "fullwiki"
    files = sorted(source.glob("train-*.parquet")) + sorted(
        source.glob("validation-*.parquet")
    )
    if files:
        import pandas as pd

        levels = pd.concat([pd.read_parquet(p, columns=["id", "level"]) for p in files])
        lookup = dict(zip(levels["id"], levels["level"], strict=True))
        for rows in splits.values():
            for row in rows:
                row["level"] = lookup.get(row["id"]) or "unknown"
    train = prepare_rows(splits["train"], "train", config, count=train_count)
    dev = prepare_rows(splits["dev"], "dev", config, count=dev_count)
    hashes = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
    }
    return train, dev, hashes


def sample_training_questions(prepared, count, seed):
    """Select a reproducible training-only question/label subset using a local RNG.

    Returns a new PreparedSplit with labels kept aligned; does not alter the global
    random generator or overwrite official split assignments.
    """
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


def training_schedule(prepared, config):
    """Build a reward-independent seeded schedule with labels kept aligned.

    Balanced mode independently shuffles hard and medium questions, retains equal
    counts without replacement, then interleaves them. Every even-sized contiguous
    rollout pool (including cyclic wraparound) has a 50/50 mix. Easy questions are
    excluded. Missing difficulty metadata is an error, not silent random sampling.
    """
    if prepared.split != "train" or any(
        q.id != label.id
        for q, label in zip(prepared.questions, prepared.labels, strict=True)
    ):
        raise ValueError("Training schedule requires aligned train-only data")
    rng = random.Random(config.seed)
    if config.training_question_mix == "all":
        order = list(range(len(prepared.questions)))
        rng.shuffle(order)
    else:
        if config.questions_per_update % 2:
            raise ValueError("50/50 sampling requires an even questions_per_update")
        bins = {
            level: [
                i for i, q in enumerate(prepared.questions) if q.difficulty == level
            ]
            for level in ("hard", "medium")
        }
        if any(
            q.difficulty not in ("easy", "medium", "hard") for q in prepared.questions
        ):
            raise ValueError("50/50 sampling requires source difficulty metadata")
        for indices in bins.values():
            rng.shuffle(indices)
        count = min(map(len, bins.values()))
        if count * 2 < config.questions_per_update:
            raise ValueError(
                "Insufficient distinct hard/medium questions for a full rollout pool"
            )
        order = [
            i
            for pair in zip(bins["hard"][:count], bins["medium"][:count], strict=True)
            for i in pair
        ]
    return replace(
        prepared,
        questions=tuple(prepared.questions[i] for i in order),
        labels=tuple(prepared.labels[i] for i in order),
        selected_source_count=len(order),
    )
