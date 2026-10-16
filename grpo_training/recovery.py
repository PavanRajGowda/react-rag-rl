"""Pre-update, trajectory-scoped probability recovery; no optimizer or GPU imports."""

import time

from grpo_training.advantages import prepare_update
from grpo_training.trainer import CapturedUpdate


def probability_report(captured, mean_limit=0.1, max_limit=1.0):
    by_episode, differences, outliers = {}, [], []
    for row, values in zip(
        captured.prepared.rows(), captured.old_logprobs, strict=True
    ):
        summary = by_episode.setdefault(
            row["episode_id"],
            {
                "episode_id": row["episode_id"],
                "question_id": row["question_id"],
                "tokens": 0,
                "absolute_sum": 0.0,
                "max_absolute_difference": 0.0,
            },
        )
        for index, (old, sampled) in enumerate(
            zip(values, row["sampling_logprobs"][-len(values) :], strict=True)
        ):
            delta = abs(old - sampled)
            differences.append(delta)
            summary["tokens"] += 1
            summary["absolute_sum"] += delta
            summary["max_absolute_difference"] = max(
                summary["max_absolute_difference"], delta
            )
            if delta > max_limit:
                outliers.append(
                    {
                        "episode_id": row["episode_id"],
                        "turn": row["turn"],
                        "target_position": index,
                        "token_id": row["target_ids"][index],
                        "old_logprob": old,
                        "sampling_logprob": sampled,
                        "absolute_difference": delta,
                    }
                )
    for summary in by_episode.values():
        summary["mean_absolute_difference"] = (
            summary.pop("absolute_sum") / summary["tokens"]
        )
    failed = [
        s["episode_id"]
        for s in by_episode.values()
        if s["mean_absolute_difference"] > mean_limit
        or s["max_absolute_difference"] > max_limit
    ]
    return {
        "mean_absolute_logprob_difference": sum(differences) / len(differences),
        "max_absolute_logprob_difference": max(differences),
        "failed_episode_ids": failed,
        "episodes": list(by_episode.values()),
        "outlier_tokens": sorted(
            outliers, key=lambda r: r["absolute_difference"], reverse=True
        )[:20],
        "outlier_count": len(outliers),
        "checked_tokens": len(differences),
    }


def subset_capture(captured, groups, labels):
    """Reuse valid immutable old probabilities; recompute advantages/denominator."""
    ids = {g.episodes[0].question.id for g in groups}
    prepared = prepare_update(groups, [l for l in labels if l.id in ids])
    saved = {
        (row["episode_id"], row["turn"]): (old, correction)
        for row, old, correction in zip(
            captured.prepared.rows(),
            captured.old_logprobs,
            captured.corrections,
            strict=True,
        )
    }
    vectors = [saved[(row["episode_id"], row["turn"])] for row in prepared.rows()]
    return CapturedUpdate(
        prepared, tuple(v[0] for v in vectors), tuple(v[1] for v in vectors)
    )


def recover_probability_capture(
    groups,
    labels,
    *,
    trainer,
    artifact,
    bundle,
    capture,
    regenerate,
    audit,
    phase,
    round_index,
    mean_limit,
    max_limit,
):
    """One replacement attempt for bad trajectories; then drop only affected groups.

    Runtime errors/invariant violations inside capture still propagate. We never
    downgrade weight/policy corruption or nonfinite numerics into model failures.
    """
    groups = tuple(groups)
    for attempt in range(2):
        if not groups:
            return None
        selected = {g.episodes[0].question.id for g in groups}
        selected_labels = [l for l in labels if l.id in selected]
        if not any(e.generations for g in groups for e in g.episodes):
            audit(
                {
                    "kind": "empty_replay",
                    "reason": "no_sampled_tokens",
                    "round": round_index,
                }
            )
            return None
        phase("old_policy_capture", step=trainer.state.global_step)
        started = time.perf_counter()
        captured = capture(
            trainer, prepare_update(groups, selected_labels), artifact, bundle
        )
        report = probability_report(captured, mean_limit, max_limit)
        audit(
            {
                "kind": "rollout_captured",
                "step": trainer.state.global_step,
                "round": round_index,
                "capture_attempt": attempt,
                "seconds": time.perf_counter() - started,
                "tokens": captured.prepared.token_count,
                **report,
            }
        )
        if not report["failed_episode_ids"]:
            return captured
        bad = set(report["failed_episode_ids"])
        bad_questions = {
            e.question.id for g in groups for e in g.episodes if e.episode_id in bad
        }
        if attempt == 0:
            affected = [g for g in groups if g.episodes[0].question.id in bad_questions]
            retained = tuple(
                e for g in affected for e in g.episodes if e.episode_id not in bad
            )
            audit(
                {
                    "kind": "probability_recovery",
                    "round": round_index,
                    "failed_episode_ids": sorted(bad),
                    "retained_trajectories": sum(len(g.episodes) for g in groups)
                    - len(bad),
                }
            )
            replacements = regenerate(
                tuple(g.episodes[0].question for g in affected), retained
            )
            original = {g.episodes[0].question.id: g for g in groups}
            for qid in bad_questions:
                original.pop(qid)
            original.update({g.episodes[0].question.id: g for g in replacements})
            groups = tuple(original.values())
        else:
            for qid in sorted(bad_questions):
                audit(
                    {
                        "kind": "group_dropped",
                        "reason": "probability_agreement_retries_exhausted",
                        "round": round_index,
                        "question_id": qid,
                    }
                )
            remaining = tuple(
                g for g in groups if g.episodes[0].question.id not in bad_questions
            )
            return subset_capture(captured, remaining, labels) if remaining else None
    raise AssertionError("Unreachable recovery state")
