"""Post-run evidence-set coverage. Labels never enter prompts or API requests."""

from copy import deepcopy


def annotate_retrieval(result, gold_document_ids):
    """Attach reference-document acquisition, visibility and search-efficiency annotations
    to a trajectory. gold_document_ids are retrospective labels, never search filters or
    policy inputs. Returns the annotated trace for evaluation.
    """
    annotated = deepcopy(result)
    gold = (
        {gold_document_ids}
        if isinstance(gold_document_ids, str)
        else set(gold_document_ids or ())
    )
    active = set()
    seen = set()
    searches = {}
    iterator = iter(annotated["searches"])
    number = 0
    for event in annotated.get("events", []):
        if event["kind"] == "model":
            number += 1
        elif event["kind"] == "information":
            item = next(iterator)
            item["turn"] = number
            searches[number] = item
    if not searches:
        searches = {s["turn"]: s for s in annotated["searches"]}
    for turn in annotated["turns"]:
        search = searches.get(turn["turn"])
        if "visible_evidence" in turn:
            active = {d["document_id"] for d in turn["visible_evidence"]}
        before = gold <= active if gold else None
        ids = []
        if search is not None:
            ids = [
                d["document_id"]
                for d in (search.get("response") or {}).get("results", [])
            ]
            seen.update(ids)
            if annotated["config"]["context_mode"] == "bounded":
                observation = next(
                    (
                        e
                        for e in annotated.get("events", [])
                        if e.get("search_turn") == turn["turn"]
                    ),
                    None,
                )
                if observation is not None:
                    active = {d["document_id"] for d in observation["visible_evidence"]}
            else:
                active = (
                    active | set(ids)
                    if annotated["config"]["context_mode"] == "append"
                    else set(ids)
                )
            search["retrieval_diagnostic"] = {
                "gold_document_ids": sorted(gold),
                "retrieved_document_ids": ids,
                "gold_context_retrieved": gold <= set(ids) if gold else None,
            }
        turn["retrieval_diagnostic"] = {
            "gold_document_ids": sorted(gold),
            "gold_context_in_prompt": before,
            "search_performed": search is not None,
            "retrieved_document_ids": ids,
            "gold_context_retrieved": gold <= set(ids)
            if gold and search is not None
            else None,
            "gold_context_active_after_turn": gold <= active if gold else None,
            "gold_context_retrieved_ever": gold <= seen if gold else None,
            "evidence_recall": len(gold & seen) / len(gold) if gold else None,
        }
    annotated["retrieval_validation"] = {
        "gold_document_ids": sorted(gold),
        "gold_context_retrieved_ever": gold <= seen if gold else None,
        "evidence_recall": len(gold & seen) / len(gold) if gold else None,
        "definition": "Complete supporting-document coverage; empty gold set is N/A, not success. Document retrieval does not imply supporting-fact coverage; alternative valid evidence may exist.",
    }
    return annotated
