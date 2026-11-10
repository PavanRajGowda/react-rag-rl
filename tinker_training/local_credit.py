"""Local process coefficients and deterministic sampled-token protocol masks."""

import re


def protocol_mask(ids, tokenizer, prefill):
    """Locate action spans on exact sampled tokens; conservative whole-turn fallback.

    Prefix decoding must align monotonically with the final decoded text. Ambiguous
    decoding or absent action markup receives the agreed whole-turn penalty.
    Missing closes extend the identifiable action span to the end of the turn.
    All action spans are included for multiple-action violations.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    whole = prefill + text
    spans = []
    for match in re.finditer(r"</?(search|answer)(?=[\s/>])", whole):
        close = whole.find(f"</{match.group(1)}>", match.end())
        end = len(whole) if close < 0 else close + len(f"</{match.group(1)}>")
        spans.append((max(0, match.start() - len(prefill)), max(0, end - len(prefill))))
    if not spans:
        return [1] * len(ids), "no_identifiable_action"
    bounds = [0]
    for k in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:k], skip_special_tokens=True)
        if not text.startswith(prefix) or len(prefix) < bounds[-1]:
            return [1] * len(ids), "ambiguous_token_alignment"
        bounds.append(len(prefix))
    mask = [int(any(a < right and b > left for a, b in spans))
            for left, right in zip(bounds[:-1], bounds[1:], strict=True)]
    if not any(mask):
        return [1] * len(ids), "no_sampled_action_tokens"
    return mask, "action_span"


def local_credit(outcome_advantage, turn, wg, wq, wp):
    """Return unscaled per-token local coefficients, without group process centering."""
    mask = turn["protocol_mask"]
    if turn["tokens"] != len(mask) or any(m not in (0, 1) for m in mask):
        raise ValueError("Invalid protocol token mask")
    return [outcome_advantage + wg*turn["g"] - wq*turn["Q"] - wp*turn["I"]*m for m in mask]
