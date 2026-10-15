"""Shared pure baseline/training diagnostics; no web or model dependencies."""

import json
import re
from collections import Counter
from decimal import Decimal

from baseline_agent.protocol import ProtocolError, parse_action
from baseline_agent.scoring import extract_numeric_value


def diagnose(trace, reference_metadata=None):
    """Derive diagnostics without modifying scores, traces or model inputs."""
    turns, searches = trace.get("turns", []), trace.get("searches", [])
    rag = trace.get("evaluation_mode", "rag") == "rag"
    gold = trace.get("retrieval_validation", {}).get("gold_context_id")
    attempts, failures = Counter(), Counter()
    format_errors = []
    answer_turn = None
    for turn in turns:
        if "model_output" not in turn:
            continue
        output = turn.get("model_output", "")
        action_text = output.rsplit("</reasoning>", 1)[-1]
        kind = (
            "search"
            if re.search(r"<search\b", action_text)
            else "answer"
            if re.search(r"<answer\b", action_text)
            else "other"
        )
        attempts[kind] += 1
        try:
            action, _ = parse_action(output)
            if not rag and action != "answer":
                raise ProtocolError(
                    "Search is unavailable in question-only evaluation."
                )
            if action == "answer":
                answer_turn = turn
        except ProtocolError as error:
            failures[kind] += 1
            format_errors.append(
                {"turn": turn.get("turn"), "kind": kind, "message": str(error)}
            )
    seen_requests, seen_sets, seen_ids = set(), set(), set()
    repeated_requests = repeated_sets = no_new = after_hit = 0
    first_hit = first_turn = first_rank = None
    reasons = Counter()
    for index, search in enumerate(searches, 1):
        request, response = search.get("request", {}), search.get("response", {})
        key = json.dumps(
            {k: " ".join(str(v).casefold().split()) for k, v in request.items()},
            sort_keys=True,
        )
        repeated_requests += key in seen_requests
        seen_requests.add(key)
        docs = response.get("results", [])
        ids = [d.get("context_id") for d in docs if d.get("context_id")]
        group = tuple(sorted(set(ids)))
        repeated_sets += bool(group) and group in seen_sets
        if group:
            seen_sets.add(group)
        no_new += not (set(ids) - seen_ids)
        seen_ids.update(ids)
        after_hit += first_hit is not None
        if gold and gold in ids and first_hit is None:
            first_hit, first_turn, first_rank = (
                index,
                search.get("turn"),
                ids.index(gold) + 1,
            )
        if response.get("status") == "error":
            reasons[response.get("code", "unspecified_api_error")] += 1
        elif not ids:
            reasons["empty_results"] += 1
        if response.get("match_method") == "broad_query":
            reasons["company_unresolved_broad_fallback"] += 1
        # Evaluation only: compare applied filters with canonical document metadata.
        # Broad fallback has no issuer/year filter and must not count as exclusion.
        filters = response.get("applied_filters", {})
        if reference_metadata and rag:
            if filters.get("report_year") is not None and filters[
                "report_year"
            ] != reference_metadata.get("report_year"):
                reasons["reference_excluding_year_filter"] += 1
            if (
                filters.get("company_ids")
                and reference_metadata.get("company_id") not in filters["company_ids"]
            ):
                reasons["reference_excluding_company_filter"] += 1
    if repeated_requests:
        reasons["repeated_request"] = repeated_requests
    if format_errors:
        reasons["invalid_protocol"] = len(format_errors)
    if trace.get("status") in ("infrastructure_error", "postprocessing_error"):
        reasons[trace["status"]] += 1
    pred, reference = (
        extract_numeric_value(trace.get("answer")),
        extract_numeric_value(trace.get("gold_answer")),
    )
    strict = bool(
        pred is not None
        and reference is not None
        and abs(pred - reference)
        <= max(Decimal("0.000001"), abs(reference) * Decimal("0.01"))
    )
    scored = trace.get("validation", {}).get("score") is not None
    active = (
        answer_turn.get("retrieval_diagnostic", {}).get("gold_context_in_prompt")
        if answer_turn and rag
        else None
    )
    values = lambda key: [t[key] for t in turns if t.get(key) is not None]
    return {
        "id": trace["question"]["id"],
        "question": trace["question"].get("question", ""),
        "index": trace["question"].get("example_index"),
        "mode": trace.get("evaluation_mode", "rag"),
        "status": trace.get("status"),
        "gold": trace.get("gold_answer"),
        "answer": trace.get("answer"),
        "scored": scored,
        "nm": trace.get("validation", {}).get("score"),
        "exact": bool(
            trace.get("answer") is not None
            and " ".join(str(trace["answer"]).split())
            == " ".join(str(trace.get("gold_answer")).split())
        )
        if scored
        else None,
        "strict": strict if scored else None,
        "normalized_prediction": str(pred) if pred is not None else None,
        "normalized_gold": str(reference) if reference is not None else None,
        "sign_disagreement": pred is not None
        and reference is not None
        and pred * reference < 0,
        "scale_disagreement": bool(
            pred is not None
            and reference is not None
            and reference != 0
            and trace.get("validation", {}).get("score") == 1
            and abs(abs(pred / reference) - 1) > Decimal("0.01")
        ),
        "gold_id": gold,
        "ever": first_hit is not None if rag and gold else None,
        "reference_metadata": reference_metadata,
        "active": active,
        "first_hit_search": first_hit,
        "first_hit_turn": first_turn,
        "first_hit_rank": first_rank,
        "searches": len(searches),
        "generations": len(turns),
        "seconds": trace.get("inference_seconds"),
        "generation_seconds": sum(values("generation_seconds"))
        if values("generation_seconds")
        else None,
        "search_seconds": sum(s["seconds"] for s in searches if "seconds" in s)
        if searches
        else None,
        "output_tokens": sum(values("generated_tokens"))
        if values("generated_tokens")
        else None,
        "input_tokens": sum(values("input_tokens")) if values("input_tokens") else None,
        "peak_input_tokens": max(values("input_tokens"), default=None),
        "format_attempts": dict(attempts),
        "format_failures": dict(failures),
        "format_errors": format_errors,
        "recovered_format": bool(
            format_errors
            and answer_turn
            and answer_turn.get("turn", 0) > format_errors[-1]["turn"]
        ),
        "repeated_requests": repeated_requests,
        "repeated_sets": repeated_sets,
        "no_new_context": no_new,
        "searches_after_hit": after_hit,
        "reasons": dict(reasons),
    }
