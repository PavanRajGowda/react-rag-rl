"""Normalized short-answer exact match and token F1; no numeric scale matching."""

import re
import string
from collections import Counter


def normalize_text(value):
    """Canonicalize an answer using the evaluator's case, punctuation, article and
    whitespace rules. Used for answer metrics rather than modifying generated output.
    """
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def boolean_value(value):
    """Return normalized "yes" or "no", or None when the answer is not Boolean text."""
    text = normalize_text(value)
    return text if text in {"yes", "no"} else None


def token_f1(pred, gold):
    """Return multiset token-overlap F1 after answer normalization. A mismatched
    yes/no/noanswer response or an empty token sequence receives 0.0; repeated tokens
    are counted with their multiplicity.
    """
    normalized_p, normalized_g = normalize_text(pred), normalize_text(gold)
    if (
        normalized_p in {"yes", "no", "noanswer"}
        or normalized_g in {"yes", "no", "noanswer"}
    ) and normalized_p != normalized_g:
        return 0.0
    p, g = normalized_p.split(), normalized_g.split()
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    return 2 * overlap / (len(p) + len(g))


def score_answer(pred, gold):
    """Return exact/normalized match and token-overlap metrics for predicted versus gold
    answer text. This is deterministic answer scoring, not an LLM-judge call.
    """
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("Nonblank text gold required")
    p, g = normalize_text(pred), normalize_text(gold)
    exact = float(p == g and bool(p))
    return {
        "match_type": "exact_match",
        "score": exact,
        "exact_match": exact,
        "token_f1": token_f1(pred, gold),
        "prediction_valid": bool(p),
        "normalized_prediction": p,
        "normalized_gold": g,
    }
