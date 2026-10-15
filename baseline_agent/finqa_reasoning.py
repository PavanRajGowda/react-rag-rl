"""Join original FinQA reasoning annotations for offline evaluation only."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from baseline_agent.batch import write_json

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SPLITS = ("train", "dev", "test")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_answer(value):
    return " ".join(str(value if value is not None else "").casefold().split())


def source_file(value):
    return str(value).replace("\\", "/").removeprefix("pdf/")


def program_operations(program):
    if not isinstance(program, str) or not program.strip():
        raise ValueError("Missing gold reasoning program")
    matches = list(re.finditer(r"([a-z_]+)\(([^()]*)\)", program))
    remaining = re.sub(r"[a-z_]+\([^()]*\)", "", program)
    if not matches or remaining.strip(" ,\t\r\n"):
        raise ValueError("Unsupported gold reasoning program syntax")
    return [match.group(1) for match in matches]


def join_row(label, document, original_rows):
    """Use positional IDs only with independent document/answer verification."""
    result = {
        "id": label["id"],
        "split": label["split"],
        "status": "unmatched",
        "t2_question": label["question"],
        "t2_gold_answer": label["program_answer"],
    }
    match = re.fullmatch(r"finqa_(train|dev|test)_(\d+)", label["id"])
    if not match or match[1] != label["split"]:
        return result | {"reason": "unsupported_t2_id"}
    index = int(match[2])
    if index >= len(original_rows):
        return result | {"reason": "original_index_out_of_range"}
    original = original_rows[index]
    qa = original.get("qa", {})
    if not document or source_file(document.get("source_file")) != source_file(
        original.get("filename")
    ):
        return result | {"reason": "source_report_page_mismatch"}
    if normalized_answer(label.get("original_answer")) != normalized_answer(
        qa.get("answer")
    ):
        return result | {"reason": "original_answer_mismatch"}
    try:
        operations = program_operations(qa.get("program"))
    except ValueError as error:
        return result | {"reason": str(error)}
    if len(operations) != len(qa.get("steps", [])):
        return result | {"reason": "program_and_steps_disagree"}
    support = qa.get("gold_inds", {})
    return result | {
        "status": "matched",
        "join_method": "split_index_verified_by_source_report_page_and_original_answer",
        "original_index": index,
        "original_id": original["id"],
        "source_file": original["filename"],
        "original_question": qa.get("question"),
        "original_answer": qa.get("answer"),
        "original_execution_answer": qa.get("exe_ans"),
        "program": qa["program"],
        "operations": operations,
        "operation_count": len(operations),
        "reasoning_bucket": "1"
        if len(operations) == 1
        else "2"
        if len(operations) == 2
        else "3+",
        "supporting_fact_ids": list(support),
        "supporting_fact_count": len(support),
        "supporting_sources": sorted({key.split("_", 1)[0] for key in support}),
        "question_reformulation_semantically_verified": False,
    }


def build(original_dir, corpus_dir, output, report):
    labels_path, contexts_path = (
        corpus_dir / "labels.jsonl",
        corpus_dir / "contexts.jsonl",
    )
    labels = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
    ]
    documents = {
        d["context_id"]: d
        for d in map(json.loads, contexts_path.read_text(encoding="utf-8").splitlines())
    }
    originals = {
        split: json.loads((original_dir / f"{split}.json").read_text(encoding="utf-8"))
        for split in SPLITS
    }
    revision = (original_dir / "revision.txt").read_text(encoding="utf-8-sig").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A pinned original FinQA revision is required")
    ids = [label["id"] for label in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate T2 question IDs")
    rows = [
        join_row(label, documents.get(label["context_id"]), originals[label["split"]])
        for label in labels
    ]
    matched_ids = [
        (r["split"], r["original_id"]) for r in rows if r["status"] == "matched"
    ]
    if len(matched_ids) != len(set(matched_ids)):
        raise ValueError("Join is not one-to-one")
    provenance = {
        "original_repository": "https://github.com/czyssrs/FinQA",
        "original_revision": revision,
        "original_file_sha256": {
            split: sha256(original_dir / f"{split}.json") for split in SPLITS
        },
        "t2_revision": json.loads(
            (corpus_dir / "manifest.json").read_text(encoding="utf-8")
        )["revision"],
        "labels_sha256": sha256(labels_path),
        "contexts_sha256": sha256(contexts_path),
    }
    summary = {
        "provenance": provenance,
        "definition": "Count of operations in original sequential gold program, verified against steps length. Not a multihop retrieval label. Reformulated-question semantics have not been verified.",
        "join_checks": [
            "same split and positional ID",
            "exact source report/page path",
            "original answer equality with null/blank normalization",
            "one-to-one matched original IDs",
            "program syntax and steps count agreement",
        ],
        "splits": {
            split: {
                "total": sum(r["split"] == split for r in rows),
                "matched": sum(
                    r["split"] == split and r["status"] == "matched" for r in rows
                ),
                "operation_counts": dict(
                    sorted(
                        Counter(
                            r["operation_count"]
                            for r in rows
                            if r["split"] == split and r["status"] == "matched"
                        ).items()
                    )
                ),
            }
            for split in SPLITS
        },
        "unmatched": [r for r in rows if r["status"] != "matched"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "provenance": provenance,
            "definition": summary["definition"],
            "rows": {r["id"]: r for r in rows},
        },
    )
    write_json(report, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download public FinQA annotations at the pinned revision",
    )
    parser.add_argument(
        "--original-dir", type=Path, default=ROOT / "data/finqa_original"
    )
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data/finqa")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/finqa/reasoning_annotations.json"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/finqa-reasoning-join.json"
    )
    args = parser.parse_args()
    if args.download:
        args.original_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            url = f"https://raw.githubusercontent.com/czyssrs/FinQA/{ORIGINAL_REVISION}/dataset/{split}.json"
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            json.loads(payload)
            (args.original_dir / f"{split}.json").write_bytes(payload)
        (args.original_dir / "revision.txt").write_text(
            ORIGINAL_REVISION, encoding="utf-8"
        )
    print(
        json.dumps(
            build(args.original_dir, args.corpus_dir, args.output, args.report),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
