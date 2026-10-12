"""Local-only scoring and exports; gold answers never cross to the GPU."""

import csv
import json
from pathlib import Path

from baseline_agent.retrieval_diagnostics import annotate_retrieval
from baseline_agent.scoring import score_answer

CSV_FIELDS = [
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
    "number_match",
    "sign_agrees",
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
    "gold_context_id",
    "gold_context_retrieved_ever",
]


def record_result(result, gold, directory, gold_context_id=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if gold_context_id is not None:
        result = annotate_retrieval(result, gold_context_id)
    scoring = score_answer(result["answer"], gold)
    result = result | {
        "gold_answer": gold,
        "validation": scoring,
        "numeric_validation": "performed_locally",
    }
    path = directory / (result["question"]["id"] + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
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
    row = {
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
        "gold_context_id": gold_context_id,
        "gold_context_retrieved_ever": result.get("retrieval_validation", {}).get(
            "gold_context_retrieved_ever"
        ),
    }
    return row


def export_rows(rows, directory):
    path = Path(directory) / "results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scored = [x["score"] for x in rows if x["score"] is not None]
    turn_rows = []
    for row in rows:
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
            row["retrieved_context_ids"] = json.dumps(row["retrieved_context_ids"])
        with (Path(directory) / "turns.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(turn_rows[0]))
            writer.writeheader()
            writer.writerows(turn_rows)
    summary = {
        "metric": "T2-RAGBench Number Match",
        "epsilon": 0.01,
        "boolean_mapping": {"Yes": 1, "No": 0},
        "examples_completed": len(rows),
        "examples_scored": len(scored),
        "answer_match_accuracy": sum(scored) / len(scored) if scored else None,
        "inference_seconds": sum(x["inference_seconds"] for x in rows),
        "timing_definition": "Per-question agent loop on GPU, including generation and tool waits; excludes image build, model download/load and readiness warmup",
        "number_match_ignores_sign_and_power_of_ten": True,
    }
    assessed = [
        row["gold_context_retrieved_ever"]
        for row in rows
        if row.get("gold_context_retrieved_ever") is not None
    ]
    if assessed:
        summary["gold_context_retrieval_rate"] = sum(assessed) / len(assessed)
    (Path(directory) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
