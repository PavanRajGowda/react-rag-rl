"""Training-only debugging selection; never creates or changes dataset splits."""

import random
from collections import defaultdict


def stratum(row):
    """Return the joint question-type/evidence-count category used for debugging subset
    diversity.
    """
    return f"{row['question_type']}|documents={len(row['gold_document_ids'])}"


def select_debug_subset(rows, seed=42, count=24):
    """Select a deterministic small, stratified training-only debugging subset. Uses local
    seeded randomness and round-robin strata; returns question identifiers without
    changing official train/dev/test assignments.
    """
    # Cycle over question-type/evidence-count categories for debugging variety.
    train_by_stratum = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            train_by_stratum[stratum(row)].append(row)
    rng = random.Random(seed + 1)
    for candidates in train_by_stratum.values():
        rng.shuffle(candidates)
    debug, used = [], set()
    while len(debug) < count:
        added = False
        for key in sorted(train_by_stratum):
            candidates = train_by_stratum[key]
            while candidates and candidates[-1]["id"] in used:
                candidates.pop()
            if candidates and len(debug) < count:
                row = candidates.pop()
                used.add(row["id"])
                debug.append(row["id"])
                added = True
        if not added:
            break
    return {
        "split": "train",
        "seed": seed + 1,
        "question_ids": debug,
        "purpose": "Prompt/tool debugging only; excluded from dev/test by construction, still part of training.",
    }
