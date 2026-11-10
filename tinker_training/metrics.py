"""Pure training diagnostics and mergeable numerators/denominators."""

import hashlib
import json
import math
import statistics
from collections import Counter

from baseline_agent.diagnostics import diagnose
from baseline_agent.retrieval_diagnostics import annotate_retrieval
from baseline_agent.scoring import normalize_text, score_answer
from tinker_training.advantages import group_advantages
from tinker_training.rewards import group_outcome


def distribution(values):
    """Return count, sample standard deviation, extrema and interpolated percentiles for
    numeric values. Empty input returns None for undefined statistics; singleton
    standard deviation is also None.
    """
    values = sorted(values)
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }

    def quantile(q):
        """Interpolate a quantile between neighboring values in the enclosing sorted
        sample.
        """
        position = (len(values) - 1) * q
        low = math.floor(position)
        return values[low] + (values[math.ceil(position)] - values[low]) * (
            position - low
        )

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": values[0],
        "max": values[-1],
        "p50": quantile(0.5),
        "p95": quantile(0.95),
    }


def pair(numerator, denominator):
    """Return numerator, denominator and their ratio; use None when the denominator is
    zero. Retaining both counts allows correct aggregation across batches.
    """
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def merge_rates(rates):
    """Pool rate numerators and denominators before division, rather than averaging
    percentages.
    """
    return pair(
        sum(r["numerator"] for r in rates), sum(r["denominator"] for r in rates)
    )


def episode_diagnostic(episode, label, reference_metadata=None):
    """Join a terminal episode with private labels to compute answer, retrieval and
    behavior diagnostics. Returns (diagnostic, trace); technical episodes are excluded
    from accuracy denominators. This runs after generation and never changes the prompt.
    """
    trace = (
        json.loads(episode.controller_audit_json)
        if episode.controller_audit_json
        else {"turns": [], "searches": [], "events": []}
    )
    trace.update(
        question={
            "id": episode.question.id,
            "question": episode.question.text,
            "example_index": episode.question.source_index,
        },
        config=episode.config.to_dict(),
        status=episode.status,
        answer=episode.answer,
        gold_answer=label.gold_answer,
        evaluation_mode="rag",
        validation=score_answer(episode.answer, label.gold_answer),
    )
    trace["inference_seconds"] = trace.get("collection_seconds")
    # Exact replay records live once in the saved rollout pool, not UI exports.
    records = {g.request.turn: g for g in episode.generations}
    if not trace["turns"]:
        trace["turns"] = [{"turn": turn} for turn in records]
    # Context-limit/error pseudo-turns have no sampled output and no fabricated timing.
    for turn in trace["turns"]:
        record = records.get(turn.get("turn"))
        if record:
            turn.update(
                model_input=record.request.prompt_text,
                model_output=record.controller_output,
                input_tokens=len(record.request.prompt_ids),
                generated_tokens=len(record.completion_ids),
                generation_seconds=record.generation_seconds,
            )
    # A terminal search infrastructure failure may have no corresponding information
    # event. Exclude only that unreturned call from evidence-history annotation.
    failed = [s for s in trace["searches"] if s.get("response") is None]
    trace["searches"] = [s for s in trace["searches"] if s.get("response") is not None]
    trace = annotate_retrieval(trace, label.gold_document_ids)
    diagnostic = diagnose(trace, reference_metadata)
    diagnostic.update(
        output_sha256=hashlib.sha256(
            json.dumps([g.completion_text for g in episode.generations]).encode()
        ).hexdigest(),
        episode_id=episode.episode_id,
        group_id=episode.group_id,
        sample_index=episode.sample_index,
        infrastructure_failures=len(failed),
        length_truncated_generations=sum(
            g.finish_reason == "length" for g in episode.generations
        ),
        generations=len(episode.generations),
        input_tokens=sum(len(g.request.prompt_ids) for g in episode.generations),
        output_tokens=sum(len(g.completion_ids) for g in episode.generations),
    )
    # Infrastructure episodes are not part of accuracy denominators.
    if episode.status in ("infrastructure_error", "postprocessing_error"):
        diagnostic.update(scored=False, em=None, strict=None, exact=None)
    return diagnostic, trace


def rollout_metrics(prepared, labels, reference_catalog=None, *, include_traces=True):
    """Aggregate a PreparedUpdate and aligned labels into dashboard metrics and episode
    traces. Returns (metrics, traces), retaining fresh group identities so reused
    optimizer updates need not be counted as new samples.
    """
    labels = {label.id: label for label in labels}
    episodes, traces = [], {}
    for group in prepared.groups:
        for episode in group.episodes:
            label = labels[episode.question.id]
            row, trace = episode_diagnostic(episode, label, None)
            episodes.append(row)
            if include_traces:
                traces[episode.episode_id] = trace
    rates = {}
    for metric in ("em", "strict", "exact", "ever", "active"):
        eligible = [r for r in episodes if r[metric] is not None]
        rates[metric] = pair(sum(bool(r[metric]) for r in eligible), len(eligible))
    searches = sum(r["searches"] for r in episodes)
    for metric in (
        "repeated_requests",
        "repeated_sets",
        "no_new_context",
        "searches_after_hit",
    ):
        rates[metric] = pair(sum(r[metric] for r in episodes), searches)
    for kind in ("search", "answer", "other"):
        attempts = sum(r["format_attempts"].get(kind, 0) for r in episodes)
        failures = sum(r["format_failures"].get(kind, 0) for r in episodes)
        rates[f"format_{kind}"] = pair(attempts - failures, attempts)
    by_episode = {row["episode_id"]: row for row in episodes}
    groups = []
    signals = {s.episode_id: s for s in prepared.signals}
    for group in prepared.groups:
        outcome = group_outcome(group, labels[group.episodes[0].question.id])
        rewards = [signals[e.episode_id].reward for e in group.episodes]
        rows = [by_episode[e.episode_id] for e in group.episodes]
        advantages = group_advantages(rewards)
        for index, (episode, row) in enumerate(zip(group.episodes, rows, strict=True)):
            row.update(
                training_reward=rewards[index],
                advantage=advantages[index],
                all_required_documents=outcome.documents[index].complete,
                required_documents=outcome.documents[index].required,
                retrieved_required_documents=outcome.documents[index].retrieved,
            )
            if include_traces:
                traces[episode.episode_id]["training_signal"] = {
                    key: row[key]
                    for key in (
                        "training_reward",
                        "advantage",
                        "all_required_documents",
                        "required_documents",
                        "retrieved_required_documents",
                    )
                } | {
                    "credit_schema": "local_process" if prepared.task_active_group_ids is not None else "trajectory",
                    "reward_type": prepared.config.reward,
                    "reward_values": {
                        "process_bonus_scale": prepared.config.process_bonus_scale,
                        "coverage_weight": prepared.config.process_coverage_weight,
                        "redundancy_weight": prepared.config.process_redundancy_weight,
                        "invalid_weight": prepared.config.process_invalid_weight,
                        "local_gold_weight": prepared.config.local_gold_weight,
                        "local_search_penalty_weight": prepared.config.local_search_penalty_weight,
                        "local_protocol_penalty_weight": prepared.config.local_protocol_penalty_weight,
                    },
                }
        answer_rewards = [float(r["em"] or 0) for r in rows]
        answers = [r["answer"] for r in rows if r["answer"] is not None]
        normalized_answers = {normalize_text(a) for a in answers}
        groups.append(
            {
                "group_id": group.episodes[0].group_id,
                "question_id": group.episodes[0].question.id,
                "reward": distribution(rewards),
                "reward_type": prepared.config.reward,
                "document_complete": tuple(d.complete for d in outcome.documents),
                "advantage": distribution(
                    signals[e.episode_id].advantage for e in group.episodes
                ),
                "answer_reward": distribution(answer_rewards),
                "all_correct": all(r == 1 for r in answer_rewards),
                "all_wrong": all(r == 0 for r in answer_rewards),
                "answer_learning_signal": len(set(answer_rewards)) > 1,
                "nonzero_advantage": pair(
                    sum(signals[e.episode_id].advantage != 0 for e in group.episodes),
                    len(rows),
                ),
                "answer_diversity": {
                    "samples": len(rows),
                    "accepted_answers": len(answers),
                    "missing_answers": len(rows) - len(answers),
                    "distinct_answer_strings": len(set(answers)),
                    "distinct_normalized_values": len(normalized_answers),
                    "distinct_full_outputs": len({r["output_sha256"] for r in rows}),
                },
                "quality_metrics": group_quality_metrics(rows),
                "retrieval": group_retrieval_metrics(rows),
                "learning_signal": (group.episodes[0].group_id in prepared.task_active_group_ids
                                    if prepared.task_active_group_ids is not None else len(set(rewards)) > 1),
            }
        )
    rates["groups_with_signal"] = pair(
        sum(g["learning_signal"] for g in groups), len(groups)
    )
    rates["groups_with_answer_signal"] = pair(
        sum(g["answer_learning_signal"] for g in groups), len(groups)
    )
    for key in groups[0]["quality_metrics"]:
        eligible_groups = [g for g in groups if g["quality_metrics"][key]["n"] >= 2]
        rates[f"groups_with_variation_{key}"] = pair(
            sum(g["quality_metrics"][key]["varies"] for g in eligible_groups),
            len(eligible_groups),
        )
    matrix = {
        f"{evidence}_{outcome}": 0
        for evidence in ("retrieved", "not_retrieved", "unknown")
        for outcome in ("correct", "incorrect", "no_answer")
    }
    for row in episodes:
        evidence = (
            "unknown"
            if row["ever"] is None
            else "retrieved"
            if row["ever"]
            else "not_retrieved"
        )
        outcome = (
            "no_answer"
            if row["answer"] is None
            else "correct"
            if row["em"]
            else "incorrect"
        )
        matrix[f"{evidence}_{outcome}"] += 1
    reasons = Counter()
    for row in episodes:
        reasons.update(row["reasons"])
    first_hits = [r for r in episodes if r["first_hit_search"] is not None]
    first_misses = [
        r
        for r in episodes
        if r["ever"] is not None and r["searches"] and r["first_hit_search"] != 1
    ]
    rates["later_search_recovery"] = pair(
        sum(r["first_hit_search"] is not None for r in first_misses), len(first_misses)
    )
    return {
        "credit": ({"schema":"local_process_advantage_v1",
                    "task_token_advantage":distribution(
                        v for row in prepared.rows() for v in row["token_advantages"]),
                    "episode_advantage_semantics":"group-centered EM before local shaping"}
                   if prepared.task_active_group_ids is not None else None),
        "episodes": episodes,
        "groups": groups,
        "rates": rates,
        "evidence_outcome": matrix,
        "reasons": dict(reasons),
        "reward": distribution(s.reward for s in prepared.signals),
        "advantage": distribution(s.advantage for s in prepared.signals),
        "searches_to_hit": distribution(r["first_hit_search"] for r in first_hits),
        "generations_to_hit": distribution(
            r["first_hit_turn"] for r in first_hits if r["first_hit_turn"] is not None
        ),
        "first_hit_reciprocal_rank": pair(
            sum(1 / r["first_hit_rank"] for r in first_hits), len(first_hits)
        ),
        "lengths": {
            key: distribution(r[key] for r in episodes if r.get(key) is not None)
            for key in (
                "input_tokens",
                "output_tokens",
                "peak_input_tokens",
                "generations",
                "searches",
                "search_seconds",
            )
        },
    }, traces


def group_retrieval_metrics(rows):
    """Aggregate reference-hit and search-efficiency diagnostics within one question group,
    preserving event counts and undefined denominators.
    """
    searches = sum(r["searches"] for r in rows)
    hits = [r for r in rows if r["first_hit_search"] is not None]
    misses = [
        r
        for r in rows
        if r["ever"] is not None and r["searches"] and r["first_hit_search"] != 1
    ]
    result = {
        "first_search_hit": pair(
            sum(r["first_hit_search"] == 1 for r in rows if r["ever"] is not None),
            sum(r["ever"] is not None for r in rows),
        ),
        "later_search_recovery": pair(
            sum(r["first_hit_search"] is not None for r in misses), len(misses)
        ),
        "searches_to_hit": distribution(r["first_hit_search"] for r in hits),
        "generations_to_hit": distribution(
            r["first_hit_turn"] for r in hits if r["first_hit_turn"] is not None
        ),
        "first_hit_reciprocal_rank": pair(
            sum(1 / r["first_hit_rank"] for r in hits), len(hits)
        ),
        "searches": distribution(r["searches"] for r in rows),
    }
    for key in ("ever", "active"):
        eligible = [r for r in rows if r[key] is not None]
        result[key] = pair(sum(bool(r[key]) for r in eligible), len(eligible))
    for key in (
        "repeated_requests",
        "repeated_sets",
        "no_new_context",
        "searches_after_hit",
    ):
        result[key] = pair(sum(r[key] for r in rows), searches)
    return result


def group_quality_metrics(rows):
    """Measure whether process/retrieval features vary among sibling trajectories.
    Variation describes potential learning signal; it does not prove improved held-out
    performance.
    """
    values = {
        key: []
        for key in (
            "reference_ever",
            "reference_active",
            "first_search_hit",
            "searches",
            "repeated_request_fraction",
            "repeated_set_fraction",
            "no_new_context_fraction",
            "searches_after_hit_fraction",
            "first_hit_search",
            "first_hit_generation",
            "later_search_recovery",
            "first_hit_reciprocal_rank",
        )
    }
    for row in rows:
        for key, source in (("reference_ever", "ever"), ("reference_active", "active")):
            if row[source] is not None:
                values[key].append(float(row[source]))
        if row["ever"] is not None:
            values["first_search_hit"].append(float(row["first_hit_search"] == 1))
        values["searches"].append(row["searches"])
        for key, source in (
            ("repeated_request_fraction", "repeated_requests"),
            ("repeated_set_fraction", "repeated_sets"),
            ("no_new_context_fraction", "no_new_context"),
            ("searches_after_hit_fraction", "searches_after_hit"),
        ):
            if row["searches"]:
                values[key].append(row[source] / row["searches"])
        if row["searches"] and row["first_hit_search"] != 1:
            values["later_search_recovery"].append(
                float(row["first_hit_search"] is not None)
            )
        if row["first_hit_search"] is not None:
            if row["first_hit_turn"] is not None:
                values["first_hit_generation"].append(row["first_hit_turn"])
            values["first_hit_search"].append(row["first_hit_search"])
            values["first_hit_reciprocal_rank"].append(1 / row["first_hit_rank"])
    return {
        key: distribution(v) | {"varies": len(set(v)) > 1} for key, v in values.items()
    }
