"""Offline paired behavioral audit of frozen baseline and final-policy evaluations."""

import ast
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from baseline_agent.protocol import ProtocolError, parse_action
from tinker_training.process_events import events
from tinker_training.rewards import bounded_process_reward, DocumentOutcome, EpisodeReward
from tinker_training.config import TrainingConfig


def load(run):
    """Read scored outcomes and derive action metrics from exact saved trajectories."""
    with (run / "validation/results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def inspect(row):
        raw = json.loads(
            (run / "validation/raw" / f"{row['id']}.json").read_text(encoding="utf-8")
        )
        turns = raw["turns"]
        searches = raw["searches"]
        gold = set(ast.literal_eval(row["gold_document_ids"]))
        if row["status"] in ("infrastructure_error", "postprocessing_error"):
            raise ValueError("Technical failures cannot be scored as behavioral errors")
        flags = events([
            {"turn": t["turn"], "text": t["model_output"],
             "tokens": t["generated_tokens"]} for t in turns
        ], raw, gold)
        seen = set()
        queries = set()
        redundant = after = repeat = new_gold_search = miss = 0
        first = None
        late_new_gold = 0
        for number, s in enumerate(searches, 1):
            if gold and gold <= seen:
                after += 1
            docs = {d["document_id"] for d in s["response"].get("results", [])}
            novel = docs - seen
            gains = novel & gold
            redundant += not bool(novel)
            new_gold_search += bool(gains)
            late_new_gold += bool(gains) and number > 2
            miss += bool(novel) and not bool(gains)
            query = " ".join(s["request"]["query"].casefold().split())
            repeat += query in queries
            queries.add(query)
            seen |= docs
            if first is None and gold and gold <= seen:
                first = number
        invalid = 0
        reason_chars = 0
        valid_queries = []
        for t in turns:
            text = t.get("model_output", "")
            try:
                kind, action = parse_action(text)
                if kind == "search":
                    valid_queries.append(len(action["query"]))
            except ProtocolError:
                invalid += 1
            match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
            if match:
                reason_chars += len(match.group(1))
        correct = float(row["score"])
        parsed_searches = [t for t in flags if t["kind"] == "search"]
        q_count = sum(t["Q"] for t in parsed_searches)
        components = bounded_process_reward(
            EpisodeReward(row["source_question_id"], row["status"], "saved_grade",
                          correct, correct, None, None, None),
            DocumentOutcome(len(gold), len(gold & seen), bool(gold) and gold <= seen),
            flags, TrainingConfig(reward="bounded_process_trajectory_v1"))
        m = {
            "em": correct,
            "f1": float(row["token_f1"]),
            "searches": len(searches),
            "turns": len(turns),
            "tokens": float(row["generated_tokens"]),
            "input_tokens": float(row["input_tokens"]),
            "reasoning_chars": reason_chars,
            "no_new_searches": redundant,
            "repeated_queries": repeat,
            "searches_after_coverage": after,
            "new_gold_searches": new_gold_search,
            "novel_without_new_gold": miss,
            "gold_recall": len(gold & seen) / len(gold) if gold else 0,
            "full_gold_coverage": int(bool(gold) and gold <= seen),
            "unique_documents": len(seen),
            "protocol_errors": invalid,
            "any_protocol_error": int(invalid > 0),
            "length_limit_turns": sum(
                t.get("finish_reason") == "length" for t in turns
            ),
            "turn_limit": int(row["status"] == "turn_limit"),
            "search_limit": int(row["status"] == "search_limit"),
            "answered": int(row["status"] == "answered"),
            "new_gold_after_search2": int(late_new_gold > 0),
            "incorrect_incomplete": int(not correct and not (gold and gold <= seen)),
            "incorrect_complete": int(not correct and bool(gold) and gold <= seen),
            "correct_by3": int(correct and len(turns) <= 3),
            "parsed_search_actions": len(parsed_searches),
            "cap_blocked_search_actions": len(parsed_searches) - len(searches),
            "penalized_search_actions": q_count,
            "trajectory_Q_rate": components["penalized_search_rate"],
            "trajectory_I_rate": components["protocol_violation_rate"],
            "process_contribution": components["process_contribution"],
            "combined_reward": components["reward"],
        }
        return {
            "question_id": row["source_question_id"],
            "repetition": int(row["repetition_index"]),
            "metrics": m,
            "status": row["status"],
            "first_full_search": first,
            "query_chars": sum(valid_queries),
            "parsed_queries": len(valid_queries),
        }

    with ThreadPoolExecutor(max_workers=12) as workers:
        return list(workers.map(inspect, rows))
