"""Post-evaluation diagnostics; no inference, training or reward changes."""

from tinker_training.data import RewardLabel
from tinker_training.metrics import episode_diagnostic, pair
from tinker_training.records import EpisodeRecord


def validation_efficiency(rows):
    """Reconstruct validation episode diagnostics and aggregate retrieval/length
    efficiency. rows contains persisted episode and score records; returns summaries for
    all questions and selected outcome cohorts.
    """
    diagnostics = []
    for row in rows:
        episode = EpisodeRecord.from_dict({"schema_version": 1, **row["episode"]})
        label = RewardLabel(
            row["question_id"], row["gold_answer"], row["gold_document_ids"]
        )
        diagnostic, _ = episode_diagnostic(episode, label)
        diagnostic["correct"] = row["score"]["reward"] == 1
        diagnostics.append(diagnostic)

    def aggregate(selected):
        """Summarize the selected validation cohort with explicit question/search
        denominators. Empty cohorts retain undefined rates as None.
        """
        n = len(selected)
        searches = sum(e["searches"] for e in selected)
        attempts = sum(sum(e["format_attempts"].values()) for e in selected)
        generations = sum(e["generations"] for e in selected)
        answered = [e for e in selected if e["status"] == "answered"]
        result = {"questions": n}
        for name, key in (
            ("searches_per_question", "searches"),
            ("tokens_per_question", "output_tokens"),
            ("turns_per_question", "generations"),
        ):
            result[name] = pair(sum(e[key] for e in selected), n)
        result["searches_per_question"] = pair(
            searches + sum(e["infrastructure_failures"] for e in selected), n
        )
        result.update(
            turn_limit=pair(sum(e["status"] == "turn_limit" for e in selected), n),
            token_limit_turns=pair(
                sum(e["length_truncated_generations"] for e in selected), generations
            ),
            token_limit_questions=pair(
                sum(e["length_truncated_generations"] > 0 for e in selected), n
            ),
            repeated_requests=pair(
                sum(e["repeated_requests"] for e in selected), searches
            ),
            no_new_context=pair(sum(e["no_new_context"] for e in selected), searches),
            reference_ever=pair(
                sum(bool(e["ever"]) for e in selected if e["ever"] is not None),
                sum(e["ever"] is not None for e in selected),
            ),
            reference_active_at_answer=pair(
                sum(bool(e["active"]) for e in answered if e["active"] is not None),
                sum(e["active"] is not None for e in answered),
            ),
            format_errors=pair(
                sum(sum(e["format_failures"].values()) for e in selected), attempts
            ),
        )
        return result

    return {
        "all": aggregate(diagnostics),
        "correct_only": aggregate([e for e in diagnostics if e["correct"]]),
        "technical_failures": sum(not row["score"]["scorable"] for row in rows),
        "definition": "All includes every saved validation question, including failures. Correct-only uses reward=1. Search redundancy rates use returned searches; failed unreturned calls are counted separately in raw traces. Reference coverage uses the full supporting-document set; null questions are excluded from retrieval denominators. Token-limit means finish_reason=length, not merely reaching a token count.",
    }
