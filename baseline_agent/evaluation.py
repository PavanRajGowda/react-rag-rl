"""Local-only scoring and exports; gold answers never cross to the GPU."""

import csv
import json
from collections import Counter
from pathlib import Path

from baseline_agent.persistence import write_json
from baseline_agent.retrieval_diagnostics import annotate_retrieval
from baseline_agent.scoring import score_answer

CSV_FIELDS = [
    "source_question_id",
    "repetition_index",
    "repetitions",
    "sampling_seed",
    "evaluation_mode",
    "example_index",
    "id",
    "question",
    "correct_answer",
    "our_answer",
    "normalized_prediction",
    "normalized_gold",
    "status",
    "match_type",
    "score",
    "exact_match",
    "token_f1",
    "prediction_valid",
    "inference_seconds",
    "generation_seconds",
    "search_seconds",
    "search_attempts",
    "generation_turns",
    "input_tokens",
    "generated_tokens",
    "trace_file",
    "model_revision",
    "context_mode",
    "gold_document_ids",
    "gold_context_retrieved_ever",
]


def record_result(
    result, gold, directory, gold_document_ids=None, *, write_artifacts=True
):
    """Score and flatten one raw trajectory using its private gold answer/document IDs.
    Writes per-trajectory evaluation artifacts beneath directory and returns a metrics
    row; the original model behavior is not regenerated. Set write_artifacts=False
    when the authoritative raw trajectory already exists to avoid duplicate writes.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    skipped = result["status"].startswith("skipped_")
    if (
        gold_document_ids is not None
        and not skipped
        and result.get("evaluation_mode", "rag") == "rag"
    ):
        result = annotate_retrieval(result, gold_document_ids)
    scoring = score_answer(result["answer"], gold)
    if skipped:
        scoring = scoring | {
            "score": None,
            "exact_match": None,
            "match_type": "skipped",
        }
    result = result | {
        "gold_answer": gold,
        "validation": scoring,
        "answer_validation": "performed_locally",
    }
    path = directory / (result["question"]["id"] + ".json")
    if write_artifacts:
        write_json(path, result)
        path.with_suffix(".txt").write_text(
            "Question: "
            + result["question"]["question"]
            + "\nGold answer: "
            + str(gold)
            + "\nOur answer: "
            + str(result["answer"])
            + "\n\nRetrieval diagnostics (local only):\n"
            + "\n".join(
                f"Turn {turn.get('turn', '?')}: "
                + json.dumps(turn.get("retrieval_diagnostic", {}))
                for turn in result["turns"]
            )
            + "\n\n"
            + "\n\n".join(x["text"] for x in result["events"]),
            encoding="utf-8",
        )
    else:
        path = directory / "raw" / (result["question"]["id"] + ".json")
    row = {
        **{
            k: result["question"][k]
            for k in (
                "source_question_id",
                "repetition_index",
                "repetitions",
                "sampling_seed",
            )
            if k in result["question"]
        },
        "evaluation_mode": result.get("evaluation_mode", "rag"),
        "example_index": result["question"]["example_index"],
        "id": result["question"]["id"],
        "question": result["question"]["question"],
        "correct_answer": gold,
        "our_answer": result["answer"],
        "status": result["status"],
        **scoring,
        "inference_seconds": result["inference_seconds"],
        "generation_seconds": sum(
            x.get("generation_seconds", 0) for x in result["turns"]
        ),
        "search_seconds": sum(x["seconds"] for x in result["searches"]),
        "search_attempts": len(result["searches"]),
        "generation_turns": len(result["turns"]),
        "input_tokens": sum(x.get("input_tokens", 0) for x in result["turns"]),
        "generated_tokens": sum(x.get("generated_tokens", 0) for x in result["turns"]),
        "trace_file": str(path.resolve()),
        "model_revision": result["model_revision"],
        "context_mode": result["config"]["context_mode"],
        "gold_document_ids": gold_document_ids,
        "gold_context_retrieved_ever": result.get("retrieval_validation", {}).get(
            "gold_context_retrieved_ever"
        ),
        "_turn_diagnostics": [
            {
                "id": result["question"]["id"],
                "turn": turn["turn"],
                **turn["retrieval_diagnostic"],
            }
            for turn in result["turns"]
            if "retrieval_diagnostic" in turn
        ],
    }
    return row


def export_rows(rows, directory):
    """Write aggregate baseline metrics and tabular/JSON exports for completed rows.
    Returns the summary used by progress displays and dashboards.
    """
    path = Path(directory) / "results.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )
    temporary.replace(path)
    scored = [x["score"] for x in rows if x["score"] is not None]
    turn_rows = []
    for row in rows:
        if "_turn_diagnostics" in row:
            turn_rows.extend(dict(turn) for turn in row["_turn_diagnostics"])
            continue
        trace = json.loads(Path(row["trace_file"]).read_text(encoding="utf-8"))
        for turn in trace["turns"]:
            if "retrieval_diagnostic" in turn:
                turn_rows.append(
                    {
                        "id": row["id"],
                        "turn": turn["turn"],
                        **turn["retrieval_diagnostic"],
                    }
                )
    if turn_rows:
        for row in turn_rows:
            row["retrieved_document_ids"] = json.dumps(row["retrieved_document_ids"])
        turns_path = Path(directory) / "turns.csv"
        temporary_turns = turns_path.with_suffix(".csv.tmp")
        with temporary_turns.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(turn_rows[0]))
            writer.writeheader()
            writer.writerows(turn_rows)
        temporary_turns.replace(turns_path)
    summary = {
        "metric": "Normalized exact match",
        "examples_completed": len(rows),
        "examples_skipped": sum(row["status"].startswith("skipped_") for row in rows),
        "infrastructure_errors": sum(
            row["status"] in ("infrastructure_error", "postprocessing_error")
            for row in rows
        ),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "examples_scored": len(scored),
        "answer_match_accuracy": sum(scored) / len(scored) if scored else None,
        "inference_seconds": sum(x["inference_seconds"] for x in rows),
        "timing_definition": "Sum of per-trajectory client-side agent-loop wall times, including hosted generation and search waits. Concurrent trajectories overlap, so this is not elapsed evaluation wall time; excludes setup before the agent loop.",
        "answer_normalization": "casefold, punctuation/articles removal, whitespace",
    }
    assessed = [
        row["gold_context_retrieved_ever"]
        for row in rows
        if row.get("gold_context_retrieved_ever") is not None
    ]
    if assessed:
        summary["gold_context_retrieval_rate"] = sum(assessed) / len(assessed)
    if rows and all("source_question_id" in row for row in rows):
        from baseline_agent.sampling import summarize_repetitions

        repeated = summarize_repetitions(rows)
        write_json(Path(directory) / "question_results.json", repeated["per_question"])
        summary["repeated_evaluation"] = {
            k: v for k, v in repeated.items() if k != "per_question"
        }
    write_json(Path(directory) / "summary.json", summary)
    return summary
