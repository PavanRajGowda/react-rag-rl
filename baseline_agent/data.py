"""Question-only boundary over official HotpotQA splits; score validation only."""

import hashlib
import json
from pathlib import Path

from retrieval_service.dataset import DATA_DIR, DATASET
from retrieval_service.dataset import REVISION as DATASET_REVISION


def verify_prepared_files(root, manifest):
    """Verify declared local artifacts before selecting any training/eval rows."""
    root = Path(root)
    for name in (
        "labels",
        "split_assignments",
        "debug_subset",
        "partition_audit",
        "test_questions",
    ):
        expected = manifest.get(name + "_sha256")
        if expected:
            path = root / (
                name + (".jsonl" if name in ("labels", "test_questions") else ".json")
            )
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Prepared {name} hash disagrees with manifest")
    if manifest.get("split_assignments_sha256"):
        rows = [
            json.loads(line)
            for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        fixed = json.loads(
            (root / "split_assignments.json").read_text(encoding="utf-8")
        )["assignments"]
        expected = {
            r["id"]: r["split"]
            for r in fixed
            if not (manifest.get("hidden_test_labels") and r["split"] == "test")
        }
        actual = {r["id"]: r["split"] for r in rows}
        if (
            len({r["id"] for r in fixed}) != len(fixed)
            or len(actual) != len(rows)
            or expected != actual
        ):
            raise ValueError("Prepared labels disagree with fixed split assignments")
        if manifest.get("hidden_test_labels"):
            test = [
                json.loads(line)
                for line in (root / "test_questions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            expected_test = {r["id"]: r["split"] for r in fixed if r["split"] == "test"}
            actual_test = {r["id"]: r["split"] for r in test}
            if (
                len(actual_test) != len(test)
                or expected_test != actual_test
                or actual.keys() & actual_test.keys()
            ):
                raise ValueError(
                    "Prepared test questions disagree with fixed split assignments"
                )


def question_fields(row, example_index):
    """Extract public question metadata from a prepared label row at example_index. Returns
    an ID/text/source dictionary without the private gold answer.
    """
    return {
        "id": row["id"],
        "question": row["question"].strip(),
        "example_index": example_index,
        "split": row.get("split", "dev"),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
    }


def policy_question(row, example_index):
    """Return only the question fields allowed in the policy input, excluding evaluation
    labels.
    """
    result = question_fields(row, example_index)
    if not result["question"]:
        raise ValueError("Blank question")
    return result


def _test_rows():
    """Legacy internal name: returns labeled official validation rows, never test."""
    root = DATA_DIR
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["revision"] != DATASET_REVISION:
        raise ValueError("Dataset revision mismatch")
    verify_prepared_files(root, manifest)
    rows = [
        json.loads(s)
        for s in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [r for r in rows if r["split"] == "dev"]


def load_evaluation_examples(start=0, count=5):
    """Load a bounded slice of labeled development examples for scored baseline evaluation.
    Returns question inputs alongside private gold answer/evidence metadata; this does
    not use the hidden-answer test split.
    """
    rows = _test_rows()
    end = len(rows) if count is None else start + count
    if start < 0 or end <= start or end > len(rows):
        raise ValueError("Invalid validation selection")
    return [
        {
            "question": question_fields(rows[i], i),
            "gold_answer": rows[i]["answer"],
            "gold_document_ids": rows[i]["gold_document_ids"],
            "question_type": rows[i]["question_type"],
            "skip_reason": None if rows[i]["question"].strip() else "blank_question",
        }
        for i in range(start, end)
    ]


def load_question(example_index=0):
    """Return the single development question at example_index using the shared evaluation
    loader.
    """
    return load_evaluation_examples(example_index, 1)[0]["question"]


def load_prediction_questions():
    """Official hidden test inputs, deliberately without fabricated labels."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    path = DATA_DIR / "test_questions.jsonl"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest["test_questions_sha256"]
    ):
        raise ValueError("Hidden test question hash disagrees with manifest")
    return [
        policy_question(json.loads(line), index)
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
    ]
