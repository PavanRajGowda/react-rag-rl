"""Audit FinQA and export intact contexts separately from QA labels."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "G4KMU/t2-ragbench"
FIELDS = {
    "context_id": "context_id",
    "text": "context",
    "table_markdown": "table",
    "pre_text": "pre_text",
    "post_text": "post_text",
    "company_name": "company_name",
    "ticker": "company_symbol",
    "report_year": "report_year",
    "page_number": "page_number",
    "source_file": "file_name",
    "company_cik": "company_cik",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(dataset, revision, output):
    documents, labels, memberships, counts = {}, [], {}, {}
    for split, rows in dataset.items():
        counts[split] = len(rows)
        for row in rows:
            doc = {target: row[source] for target, source in FIELDS.items()}
            doc["report_year"] = int(doc["report_year"])
            doc["page_number"] = int(doc["page_number"])
            doc["ticker"] = doc["ticker"].strip().upper()
            cik = str(doc["company_cik"]).strip()
            doc["company_id"] = (
                f"cik:{int(cik)}" if cik and int(cik) > 0 else f"ticker:{doc['ticker']}"
            )
            doc["report_id"] = f"{doc['company_id']}:{doc['report_year']}"
            key = doc["context_id"]
            if key in documents and documents[key] != doc:
                raise ValueError(f"Conflicting context contents or metadata: {key}")
            documents[key] = doc
            memberships.setdefault(key, set()).add(split)
            labels.append(
                {
                    k: row[k]
                    for k in (
                        "id",
                        "context_id",
                        "question",
                        "program_answer",
                        "original_answer",
                    )
                }
                | {
                    "split": split,
                    "company_id": doc["company_id"],
                    "ticker": doc["ticker"],
                    "report_year": doc["report_year"],
                    "report_id": doc["report_id"],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    corpus = "".join(
        json.dumps(documents[k], ensure_ascii=False, sort_keys=True) + "\n"
        for k in sorted(documents)
    )
    digest = hashlib.sha256(corpus.encode()).hexdigest()
    version = f"finqa-{digest[:12]}"
    (output / "contexts.jsonl").write_bytes(corpus.encode("utf-8"))
    (output / "labels.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels),
        encoding="utf-8",
    )
    lengths = sorted(len(x["text"].split()) for x in documents.values())
    reports = {}
    for label in labels:
        reports.setdefault(label["report_id"], set()).add(label["split"])
    manifest = {
        "dataset": REPO,
        "revision": revision,
        "corpus_version": version,
        "sha256": digest,
        "split_questions": counts,
        "blank_question_ids": [x["id"] for x in labels if not x["question"].strip()],
        "unique_contexts": len(documents),
        "unique_reports": len(reports),
        "duplicate_context_rows": len(labels) - len(documents),
        "context_split_overlap": dict(
            Counter("+".join(sorted(v)) for v in memberships.values())
        ),
        "report_split_overlap": dict(
            Counter("+".join(sorted(v)) for v in reports.values())
        ),
        "whitespace_words": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int((len(lengths) - 1) * 0.95)],
            "max": lengths[-1],
        },
        "chunking": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("data/finqa"))
    args = parser.parse_args()
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    dataset = load_dataset(REPO, "FinQA", revision=revision)
    print(json.dumps(prepare(dataset, revision, args.output), indent=2))


if __name__ == "__main__":
    main()
