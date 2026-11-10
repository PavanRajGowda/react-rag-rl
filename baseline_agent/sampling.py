"""Reproducible repeated evaluation; independent of model/backend and gold labels."""

import hashlib
from collections import defaultdict

from tinker_training.records import generation_seed


def evaluation_seed(master_seed, question_id, repetition):
    """Derive a reproducible integer seed from master_seed, question_id and repetition.

    Repetition is a nonnegative index; distinct repetitions use distinct streams."""
    if type(repetition) is not int or repetition < 0:
        raise ValueError("Repetition must be a nonnegative integer")
    key = int.from_bytes(hashlib.sha256(question_id.encode()).digest()[:4], "big")
    return generation_seed(generation_seed(master_seed, key), repetition + 1)


def repeat_examples(examples, repetitions=3, master_seed=42):
    """Expand example dictionaries into independently seeded evaluation repetitions.

    Inputs contain a question dictionary; returns a new list with unique sample IDs,
    source_question_id, repetition metadata and sampling_seed. Gold labels are unchanged."""
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("Repetitions must be 1–100")
    if len({e["question"]["id"] for e in examples}) != len(examples):
        raise ValueError("Duplicate source questions")
    result = []
    for example in examples:
        question = example["question"]
        for repetition in range(repetitions):
            result.append(
                example
                | {
                    "question": question
                    | {
                        "id": f"{question['id']}--sample-{repetition}",
                        "source_question_id": question["id"],
                        "repetition_index": repetition,
                        "repetitions": repetitions,
                        "sampling_seed": evaluation_seed(
                            master_seed, question["id"], repetition
                        ),
                    }
                }
            )
    return result


def summarize_repetitions(rows):
    """Aggregate scored trajectory rows by source_question_id.

    Returns run-level metrics plus per_question rows, equally weighting questions.
    Incomplete or technical failures disable final accuracy metrics; pass_at_n is
    an oracle any-correct diagnostic, not the accuracy of a selected answer."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_question_id"]].append(row)
    per_question = []
    technical = ("infrastructure_error", "postprocessing_error")
    for qid, peers in sorted(groups.items()):
        expected = peers[0]["repetitions"]
        indices = [r["repetition_index"] for r in peers]
        if len(set(indices)) != len(indices) or any(
            r["repetitions"] != expected for r in peers
        ):
            raise ValueError("Duplicate repetitions or mixed evaluation schedules")
        complete = sorted(indices) == list(range(expected))
        valid = complete and all(
            r["status"] not in technical and r["score"] is not None for r in peers
        )
        item = {
            "question_id": qid,
            "repetitions": expected,
            "saved": len(peers),
            "complete": complete,
            "metrics_available": valid,
            "mean_exact_match": sum(r["exact_match"] for r in peers) / expected
            if valid
            else None,
            "mean_token_f1": sum(r["token_f1"] for r in peers) / expected
            if valid
            else None,
            "pass_at_n": int(any(r["exact_match"] for r in peers)) if valid else None,
        }
        for metric in (
            "search_attempts",
            "generated_tokens",
            "generation_turns",
            "inference_seconds",
        ):
            item["mean_" + metric] = (
                sum(r[metric] for r in peers) / expected if complete else None
            )
        per_question.append(item)
    available = all(r["metrics_available"] for r in per_question)
    summary = {
        "questions": len(groups),
        "trajectories": len(rows),
        "complete_questions": sum(r["complete"] for r in per_question),
        "metrics_available": available,
        "aggregation": "Equal question weight; mean of all attempts. Pass@N is oracle any-correct, not primary accuracy. Incomplete/technical failures make final metrics unavailable.",
    }
    for metric in (
        "mean_exact_match",
        "mean_token_f1",
        "pass_at_n",
        "mean_search_attempts",
        "mean_generated_tokens",
        "mean_generation_turns",
        "mean_inference_seconds",
    ):
        summary[metric] = (
            sum(r[metric] for r in per_question) / len(per_question)
            if available and per_question
            else None
        )
    return summary | {"per_question": per_question}
