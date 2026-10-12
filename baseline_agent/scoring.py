"""T2-RAGBench Number Match; Yes/No predictions normalize to 1/0."""

import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext


def boolean_value(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", value, re.IGNORECASE)
    return match.group(1).casefold() if match else None


def extract_numeric_value(value):
    if value is None or isinstance(value, bool):
        return None
    boolean = boolean_value(value)
    if boolean is not None:
        return Decimal(1) if boolean == "yes" else Decimal(0)
    text = str(value).strip().replace("−", "-")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    match = re.fullmatch(
        r"[$€£]?\s*([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(%)?",
        text,
    )
    if not match:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
        if not number.is_finite():
            return None
        if negative:
            number = -abs(number)
        return number / 100 if match.group(2) else number
    except InvalidOperation:
        return None


def numeric_value_match(pred, gold, epsilon=1e-2):
    pred_value, gold_value = extract_numeric_value(pred), extract_numeric_value(gold)
    if pred_value is None or gold_value is None:
        return 0.0
    threshold = Decimal(str(epsilon))
    if not threshold.is_finite() or threshold <= 0:
        raise ValueError("epsilon must be finite and positive")
    pred_abs, gold_abs = abs(pred_value), abs(gold_value)
    if pred_abs < threshold and gold_abs < threshold:
        return 1.0
    if not pred_abs or not gold_abs:
        return 0.0
    # Algebraically equivalent to ratio * 10**(-round(log10(ratio))),
    # using log differences to avoid overflow/underflow of extreme ratios.
    with localcontext() as context:
        context.prec = 40
        log_ratio = pred_abs.log10() - gold_abs.log10()
        power = log_ratio.to_integral_value(rounding=ROUND_HALF_EVEN)
        normalized = Decimal(10) ** (log_ratio - power)
        return float(abs(normalized - 1) < threshold)


def score_answer(pred, gold):
    gold_number = extract_numeric_value(gold)
    if gold_number is None:
        return {
            "match_type": "unsupported_gold",
            "score": None,
            "number_match": None,
            "sign_agrees": None,
            "prediction_valid": False,
        }
    predicted = extract_numeric_value(pred)
    score = numeric_value_match(pred, gold)
    return {
        "match_type": "number_match",
        "score": score,
        "number_match": score,
        "sign_agrees": (predicted >= 0) == (gold_number >= 0)
        if predicted is not None
        else None,
        "prediction_valid": predicted is not None,
        "normalized_prediction": str(predicted) if predicted is not None else None,
        "normalized_gold": str(gold_number),
    }
