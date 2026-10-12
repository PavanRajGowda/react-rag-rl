"""Paired comparison of question-only and RAG outcomes on identical questions."""

import argparse
import csv
import json
from pathlib import Path

from baseline_agent.batch import write_json


def compare_runs(rag_directory, pretrained_directory, output, num_examples=None):
    directories = [Path(rag_directory), Path(pretrained_directory)]
    manifests = [
        json.loads((path / "run.json").read_text(encoding="utf-8"))
        for path in directories
    ]
    if (
        manifests[0].get("evaluation_mode", "rag") != "rag"
        or manifests[1].get("evaluation_mode") != "pretrained_only"
    ):
        raise ValueError("Expected a RAG run and a pretrained-only run")
    for key in ("model_id", "model_revision", "gpu", "dtype", "dataset_revision"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Runs differ in {key}")
    for key in ("seed", "temperature", "max_new_tokens"):
        if manifests[0]["base_config"][key] != manifests[1]["base_config"][key]:
            raise ValueError(f"Runs differ in generation setting {key}")
    tables = []
    for path in directories:
        with (path / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed = {row["id"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate question IDs")
        if num_examples is not None:
            if num_examples < 1 or len(rows) < num_examples:
                raise ValueError(
                    "Both runs must contain the requested comparison count"
                )
            selected = sorted(rows, key=lambda row: int(row["example_index"]))[
                :num_examples
            ]
            indexed = {row["id"]: row for row in selected}
        tables.append(indexed)
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Runs must contain identical question IDs")
    pairs, skipped = [], []
    for key in sorted(tables[0], key=lambda key: int(tables[0][key]["example_index"])):
        rag, pretrained = tables[0][key], tables[1][key]
        for field in ("question", "correct_answer", "example_index"):
            if rag[field] != pretrained[field]:
                raise ValueError(f"Runs differ in {field} for {key}")
        if not rag["score"] or not pretrained["score"]:
            skipped.append(key)
            continue
        pairs.append(
            {
                "id": key,
                "question": rag["question"],
                "gold_answer": rag["correct_answer"],
                "rag_answer": rag["our_answer"],
                "pretrained_only_answer": pretrained["our_answer"],
                "rag_score": float(rag["score"]),
                "pretrained_only_score": float(pretrained["score"]),
                "rag_status": rag["status"],
                "pretrained_only_status": pretrained["status"],
            }
        )
    n = len(pairs)
    summary = {
        "rag_experiment": directories[0].name,
        "pretrained_only_experiment": directories[1].name,
        "model_id": manifests[0]["model_id"],
        "model_revision": manifests[0]["model_revision"],
        "paired_questions": n,
        "skipped_ids": skipped,
        "rag_number_match": sum(p["rag_score"] for p in pairs) / n if n else None,
        "pretrained_only_number_match": sum(p["pretrained_only_score"] for p in pairs)
        / n
        if n
        else None,
        "both_correct": sum(
            p["rag_score"] == p["pretrained_only_score"] == 1 for p in pairs
        ),
        "rag_only_correct": sum(
            p["rag_score"] == 1 and p["pretrained_only_score"] == 0 for p in pairs
        ),
        "pretrained_only_correct": sum(
            p["rag_score"] == 0 and p["pretrained_only_score"] == 1 for p in pairs
        ),
        "both_incorrect": sum(
            p["rag_score"] == p["pretrained_only_score"] == 0 for p in pairs
        ),
        "technical_failure_rows": sum(
            p[field] in ("infrastructure_error", "postprocessing_error")
            for p in pairs
            for field in ("rag_status", "pretrained_only_status")
        ),
        "limitations": [
            "Question-only success can reflect facts/numbers present in the question, reasoning or pretraining knowledge; it does not prove memorization.",
            "Low question-only accuracy cannot rule out all pretraining contamination.",
            "Number Match ignores signs and powers of ten; RAG permits multiple generations while question-only uses one.",
        ],
    }
    output = Path(output)
    write_json(output.with_suffix(".json"), summary)
    if pairs:
        with output.with_suffix(".csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", default="reports/pretrained-only-pilot-comparison")
    parser.add_argument("--num-examples", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_runs(args.rag, args.pretrained, args.output, args.num_examples),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
