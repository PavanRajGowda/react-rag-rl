"""Frozen-reference sampled KL adjustment, computed once per rollout pool."""

import hashlib
import json
import math
import statistics
from dataclasses import replace

from baseline_agent.persistence import write_json


def apply_reference_kl(prepared, reference, types, path=None):
    """Score exact completion tokens; freeze token weights across minibatches/resume.

    The base sampler is immutable. k = rollout logprob - reference logprob is a
    signed sampled log-ratio, not a full-vocabulary KL. Center k across sampled
    completion tokens in this rollout pool, then add beta * (mean(k) - k).
    Prompt/tool predictions are excluded. Cached records validate exact token,
    policy, reward-advantage and reference-model identity before reuse.
    """
    rows = prepared.rows()
    identity = (
        hashlib.sha256(
            json.dumps(
                {
                    "schema": 1,
                    "reference": prepared.config.model_id,
                    "beta": prepared.config.beta,
                    "policy": prepared.policy.id,
                    "rows": rows,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if path is not None
        else None
    )
    if path is not None and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Cached reference KL identity changed")
        refs = saved["reference_logprobs"]
    else:
        refs = []
        # Bound outstanding scoring calls independently of the logical pool size.
        for start in range(0, len(rows), prepared.config.rollout_concurrency):
            chunk = rows[start : start + prepared.config.rollout_concurrency]
            pending = [
                reference.compute_logprobs(
                    types.ModelInput.from_ints(list(row["input_ids"])),
                    prompt_logprobs_last_n=len(row["target_ids"]),
                )
                for row in chunk
            ]
            for row, future in zip(chunk, pending, strict=True):
                values = future.result()
                if len(values) != len(row["input_ids"]):
                    raise ValueError("Reference token alignment changed")
                refs.append(values[-len(row["target_ids"]) :])
        _validate(rows, refs)
        if path is not None:
            write_json(path, {"identity": identity, "reference_logprobs": refs})
    _validate(rows, refs)
    ks = [
        tuple(
            old - ref
            for old, ref in zip(
                row["sampling_logprobs"][-len(row["target_ids"]) :], values, strict=True
            )
        )
        for row, values in zip(rows, refs, strict=True)
    ]
    mean = statistics.mean(k for seq in ks for k in seq)
    adjustments = tuple(
        tuple(prepared.config.beta * (mean - k) for k in seq) for seq in ks
    )
    weights = tuple(
        tuple(a + d for a, d in zip(
            row.get("token_advantages", (row["advantage"],)*len(delta)), delta, strict=True))
        for row, delta in zip(rows, adjustments, strict=True)
    )
    metrics = {
        "beta": prepared.config.beta,
        "reference_model": prepared.config.model_id,
        "sampled_logratio_mean": mean,
        "sampled_logratio_note": "signed rollout-reference estimate; not exact vocabulary KL",
        "adjustment_abs_mean": statistics.mean(
            abs(d) for seq in adjustments for d in seq
        ),
        "task_advantage_abs_token_mean": statistics.mean(
            abs(a) for row in rows for a in row.get(
                "token_advantages", (row["advantage"],)*len(row["target_ids"]))
        ),
    }
    return replace(prepared, token_weights=weights), metrics


def _validate(rows, refs):
    """Reject incomplete/nonfinite scoring rather than fabricating reference values."""
    if len(rows) != len(refs):
        raise ValueError("Reference row count changed")
    for row, values in zip(rows, refs, strict=True):
        if len(values) != len(row["target_ids"]) or any(
            v is None or not math.isfinite(v) or v > 0 for v in values
        ):
            raise ValueError("Invalid reference completion log probabilities")
