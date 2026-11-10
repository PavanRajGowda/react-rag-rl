"""Deterministic retrieval and protocol flags for trajectory rewards."""

from baseline_agent.protocol import ProtocolError, parse_action


def events(generations, audit, gold_articles, gold_passages=()):
    """Reconstruct per-turn acquisition, search-quality and protocol indicators.

    Args:
        generations: Sampled turns with text, token counts and optional token regions.
        audit: Controller history with actual search requests and returned documents.
        gold_articles: Reference document IDs used only for retrospective scoring.
        gold_passages: Historical passage IDs; empty for document-only HotpotQA.

    Returns:
        Chronological turn dictionaries with g (new gold acquisition), Q (no-new
        evidence or search after coverage), and I (invalid protocol).

    Accumulated evidence is updated only after scoring the current search, so a
    document earns acquisition credit only on its first retrieval. Transport failures
    are distinguished from successful searches returning no new documents.
    """
    legacy_passages = bool(gold_passages) or any(
        "context_id" in doc
        for item in audit.get("searches", [])
        for doc in (item.get("response") or {}).get("results", [])
    )
    searches = {}
    for search in audit.get("searches", []):
        if search["turn"] in searches:
            raise ValueError("Duplicate search decision for a generation")
        searches[search["turn"]] = search
    outputs = {t["turn"]: t.get("model_output") for t in audit.get("turns", [])}
    # These sets contain evidence seen BEFORE the current search. HotpotQA uses
    # document IDs only; the separate passage path reads historical artifacts.
    articles, passages = set(), set()
    gold_articles, gold_passages = set(gold_articles), set(gold_passages)
    rows = []
    for gen in sorted(generations, key=lambda x: x["turn"]):
        text = outputs.get(gen["turn"]) or gen["text"]
        try:
            kind, _ = parse_action(text)
        except ProtocolError:
            kind = "invalid"
        search = searches.get(gen["turn"], {})
        response = search.get("response") or {}
        ok = response.get("status") == "ok"
        results = response.get("results", []) if ok else []
        returned_a, returned_p = set(), set()
        for result in results:
            if not result.get("document_id") or (
                legacy_passages and not result.get("context_id")
            ):
                raise ValueError("Reward audit lacks document/passage identity")
            returned_a.add(result["document_id"])
            returned_p.add(
                result["context_id"] if legacy_passages else result["document_id"]
            )
        article = int(bool((returned_a & gold_articles) - articles))
        passage = int(bool((returned_p & gold_passages) - passages))
        # A successful empty result also adds nothing new. A failed transport is
        # different: it must not be mislabeled as an unproductive model search.
        no_new = bool(ok and not (returned_p - passages))
        technical = bool(search and not response) or response.get("code") in (
            "infrastructure_error",
            "transport_error",
        )
        after = bool(
            kind == "search"
            and not technical
            and (gold_passages if legacy_passages else gold_articles)
            and (
                gold_passages <= passages
                if legacy_passages
                else gold_articles <= articles
            )
        )
        rows.append(
            {
                "turn": gen["turn"],
                "kind": kind,
                "tokens": gen["tokens"],
                **(
                    {"article_acquisition": article, "passage_acquisition": passage}
                    if legacy_passages
                    else {"document_acquisition": article}
                ),
                "g": article + passage if legacy_passages else article,
                "Q": int(no_new or after),
                "I": int(kind == "invalid"),
                **(
                    {"no_new_passages": no_new}
                    if legacy_passages
                    else {"no_new_documents": no_new}
                ),
                "after_coverage": after,
                "search_ok": ok,
            }
        )
        # Update after computing indicators so a first acquisition can earn a
        # bonus and later retrieval of the same evidence cannot earn it again.
        if ok:
            articles.update(returned_a)
            passages.update(returned_p)
    if set(searches) - {r["turn"] for r in rows}:
        raise ValueError("Search audit has no generated turn")
    return rows
