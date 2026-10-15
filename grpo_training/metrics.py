"""Pure training diagnostics and mergeable numerators/denominators."""

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict

from baseline_agent.diagnostics import diagnose
from baseline_agent.retrieval_diagnostics import annotate_retrieval
from baseline_agent.scoring import score_answer


def distribution(values):
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
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def merge_rates(rates):
    return pair(
        sum(r["numerator"] for r in rates), sum(r["denominator"] for r in rates)
    )


def episode_diagnostic(episode, label, reference_metadata=None):
    """Post-rollout only. Gold/reference metadata never enter generation requests."""
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
    trace["generation_records"] = asdict(episode)["generations"]
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
    trace = annotate_retrieval(trace, label.gold_context_id)
    diagnostic = diagnose(trace, reference_metadata)
    diagnostic.update(
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
        diagnostic.update(scored=False, nm=None, strict=None, exact=None)
    return diagnostic, trace


def rollout_metrics(prepared, labels, reference_catalog=None):
    labels = {label.id: label for label in labels}
    episodes, traces = [], {}
    for group in prepared.groups:
        for episode in group.episodes:
            label = labels[episode.question.id]
            row, trace = episode_diagnostic(
                episode, label, (reference_catalog or {}).get(label.gold_context_id)
            )
            episodes.append(row)
            traces[episode.episode_id] = trace
    rates = {}
    for metric in ("nm", "strict", "exact", "ever", "active"):
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
    groups = []
    signals = {s.episode_id: s for s in prepared.signals}
    for group in prepared.groups:
        rewards = [signals[e.episode_id].reward for e in group.episodes]
        groups.append(
            {
                "group_id": group.episodes[0].group_id,
                "question_id": group.episodes[0].question.id,
                "reward": distribution(rewards),
                "all_correct": all(r == 1 for r in rewards),
                "all_wrong": all(r == 0 for r in rewards),
                "learning_signal": len(set(rewards)) > 1,
            }
        )
    rates["groups_with_signal"] = pair(
        sum(g["learning_signal"] for g in groups), len(groups)
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
            if row["nm"]
            else "incorrect"
        )
        matrix[f"{evidence}_{outcome}"] += 1
    reasons = Counter()
    for row in episodes:
        reasons.update(row["reasons"])
    first_hits = [r for r in episodes if r["first_hit_search"] is not None]
    first_misses = [r for r in episodes if r["searches"] and r["first_hit_search"] != 1]
    rates["later_search_recovery"] = pair(
        sum(r["first_hit_search"] is not None for r in first_misses), len(first_misses)
    )
    return {
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


def sampling_metrics(captured):
    diffs, corrections = [], []
    capped = 0
    for row, old, correction in zip(
        captured.prepared.rows(),
        captured.old_logprobs,
        captured.corrections,
        strict=True,
    ):
        sampled = row["sampling_logprobs"][-len(old) :]
        for a, b in zip(old, sampled, strict=True):
            diffs.append(a - b)
            capped += a - b > math.log(
                captured.prepared.config.vllm_importance_sampling_cap
            )
        corrections.extend(correction)
    return {
        "training_minus_sampler_logprob": distribution(diffs),
        "applied_correction": distribution(corrections),
        "raw_correction_above_cap": pair(capped, len(diffs)),
        "sampler_to_training_kl_sample_estimate": pair(-sum(diffs), len(diffs)),
    }


def reference_catalog(path, context_ids):
    """CPU-side diagnostic metadata only; never retain/pass document text."""
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            doc = json.loads(line)
            if doc["context_id"] in context_ids:
                result[doc["context_id"]] = {
                    k: doc[k] for k in ("company_id", "report_year")
                }
    return result
