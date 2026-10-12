"""Local post-run diagnostics. Gold context IDs never enter model prompts."""

from copy import deepcopy


def annotate_retrieval(result, gold_context_id):
    annotated = deepcopy(result)
    search_by_turn = {}
    searches = iter(annotated["searches"])
    turn_number = 0
    for event in annotated["events"]:
        if event["kind"] == "model":
            turn_number += 1
        elif event["kind"] == "information":
            search_by_turn[turn_number] = next(searches)
    if next(searches, None) is not None:
        raise ValueError("Search events and recorded searches do not agree")
    active, ever = set(), False
    for turn in annotated["turns"]:
        search = search_by_turn.get(turn["turn"])
        ids = []
        before = gold_context_id in active
        if search is not None:
            response = search["response"]
            if response.get("status") == "ok":
                ids = [doc["context_id"] for doc in response.get("results", [])]
            active = (
                active | set(ids)
                if annotated["config"]["context_mode"] == "append"
                else set(ids)
            )
            ever |= gold_context_id in ids
            search["turn"] = turn["turn"]
            search["retrieval_diagnostic"] = {
                "gold_context_id": gold_context_id,
                "retrieved_context_ids": ids,
                "gold_context_retrieved": gold_context_id in ids,
            }
        turn["retrieval_diagnostic"] = {
            "gold_context_id": gold_context_id,
            "gold_context_in_prompt": before,
            "search_performed": search is not None,
            "retrieved_context_ids": ids,
            "gold_context_retrieved": gold_context_id in ids
            if search is not None
            else None,
            "gold_context_active_after_turn": gold_context_id in active,
            "gold_context_retrieved_ever": ever,
        }
    annotated["retrieval_validation"] = {
        "gold_context_id": gold_context_id,
        "gold_context_retrieved_ever": ever,
        "definition": "Exact dataset context-ID match; alternative valid evidence may exist. Assessed locally after inference and never fed to the model.",
    }
    return annotated
