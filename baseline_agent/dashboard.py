"""Read-only evaluation dashboard. Run: python -m baseline_agent.dashboard."""

import argparse
import csv
import io
import json
import re
from collections import Counter
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from baseline_agent.protocol import ProtocolError, parse_action
from baseline_agent.scoring import extract_numeric_value

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def diagnose(trace, reference_metadata=None):
    """Derive diagnostics without modifying scores, traces or model inputs."""
    turns, searches = trace.get("turns", []), trace.get("searches", [])
    rag = trace.get("evaluation_mode", "rag") == "rag"
    gold = trace.get("retrieval_validation", {}).get("gold_context_id")
    attempts, failures = Counter(), Counter()
    format_errors = []
    answer_turn = None
    for turn in turns:
        if "model_output" not in turn:
            continue
        output = turn.get("model_output", "")
        action_text = output.rsplit("</reasoning>", 1)[-1]
        kind = (
            "search"
            if re.search(r"<search\b", action_text)
            else "answer"
            if re.search(r"<answer\b", action_text)
            else "other"
        )
        attempts[kind] += 1
        try:
            action, _ = parse_action(output)
            if not rag and action != "answer":
                raise ProtocolError(
                    "Search is unavailable in question-only evaluation."
                )
            if action == "answer":
                answer_turn = turn
        except ProtocolError as error:
            failures[kind] += 1
            format_errors.append(
                {"turn": turn.get("turn"), "kind": kind, "message": str(error)}
            )
    seen_requests, seen_sets, seen_ids = set(), set(), set()
    repeated_requests = repeated_sets = no_new = after_hit = 0
    first_hit = first_turn = first_rank = None
    reasons = Counter()
    for index, search in enumerate(searches, 1):
        request, response = search.get("request", {}), search.get("response", {})
        key = json.dumps(
            {k: " ".join(str(v).casefold().split()) for k, v in request.items()},
            sort_keys=True,
        )
        repeated_requests += key in seen_requests
        seen_requests.add(key)
        docs = response.get("results", [])
        ids = [d.get("context_id") for d in docs if d.get("context_id")]
        group = tuple(sorted(set(ids)))
        repeated_sets += bool(group) and group in seen_sets
        if group:
            seen_sets.add(group)
        no_new += not (set(ids) - seen_ids)
        seen_ids.update(ids)
        after_hit += first_hit is not None
        if gold and gold in ids and first_hit is None:
            first_hit, first_turn, first_rank = (
                index,
                search.get("turn"),
                ids.index(gold) + 1,
            )
        if response.get("status") == "error":
            reasons[response.get("code", "unspecified_api_error")] += 1
        elif not ids:
            reasons["empty_results"] += 1
        if response.get("match_method") == "broad_query":
            reasons["company_unresolved_broad_fallback"] += 1
        # Evaluation only: compare applied filters with canonical document metadata.
        # Broad fallback has no issuer/year filter and must not count as exclusion.
        filters = response.get("applied_filters", {})
        if reference_metadata and rag:
            if filters.get("report_year") is not None and filters[
                "report_year"
            ] != reference_metadata.get("report_year"):
                reasons["reference_excluding_year_filter"] += 1
            if (
                filters.get("company_ids")
                and reference_metadata.get("company_id") not in filters["company_ids"]
            ):
                reasons["reference_excluding_company_filter"] += 1
    if repeated_requests:
        reasons["repeated_request"] = repeated_requests
    if format_errors:
        reasons["invalid_protocol"] = len(format_errors)
    if trace.get("status") in ("infrastructure_error", "postprocessing_error"):
        reasons[trace["status"]] += 1
    pred, reference = (
        extract_numeric_value(trace.get("answer")),
        extract_numeric_value(trace.get("gold_answer")),
    )
    strict = bool(
        pred is not None
        and reference is not None
        and abs(pred - reference)
        <= max(Decimal("0.000001"), abs(reference) * Decimal("0.01"))
    )
    scored = trace.get("validation", {}).get("score") is not None
    active = (
        answer_turn.get("retrieval_diagnostic", {}).get("gold_context_in_prompt")
        if answer_turn and rag
        else None
    )
    values = lambda key: [t[key] for t in turns if t.get(key) is not None]
    return {
        "id": trace["question"]["id"],
        "question": trace["question"].get("question", ""),
        "index": trace["question"].get("example_index"),
        "mode": trace.get("evaluation_mode", "rag"),
        "status": trace.get("status"),
        "gold": trace.get("gold_answer"),
        "answer": trace.get("answer"),
        "scored": scored,
        "nm": trace.get("validation", {}).get("score"),
        "exact": bool(
            trace.get("answer") is not None
            and " ".join(str(trace["answer"]).split())
            == " ".join(str(trace.get("gold_answer")).split())
        )
        if scored
        else None,
        "strict": strict if scored else None,
        "normalized_prediction": str(pred) if pred is not None else None,
        "normalized_gold": str(reference) if reference is not None else None,
        "sign_disagreement": pred is not None
        and reference is not None
        and pred * reference < 0,
        "scale_disagreement": bool(
            pred is not None
            and reference is not None
            and reference != 0
            and trace.get("validation", {}).get("score") == 1
            and abs(abs(pred / reference) - 1) > Decimal("0.01")
        ),
        "gold_id": gold,
        "ever": first_hit is not None if rag and gold else None,
        "reference_metadata": reference_metadata,
        "active": active,
        "first_hit_search": first_hit,
        "first_hit_turn": first_turn,
        "first_hit_rank": first_rank,
        "searches": len(searches),
        "generations": len(turns),
        "seconds": trace.get("inference_seconds"),
        "generation_seconds": sum(values("generation_seconds"))
        if values("generation_seconds")
        else None,
        "search_seconds": sum(s["seconds"] for s in searches if "seconds" in s)
        if searches
        else None,
        "output_tokens": sum(values("generated_tokens"))
        if values("generated_tokens")
        else None,
        "input_tokens": sum(values("input_tokens")) if values("input_tokens") else None,
        "peak_input_tokens": max(values("input_tokens"), default=None),
        "format_attempts": dict(attempts),
        "format_failures": dict(failures),
        "format_errors": format_errors,
        "recovered_format": bool(
            format_errors
            and answer_turn
            and answer_turn.get("turn", 0) > format_errors[-1]["turn"]
        ),
        "repeated_requests": repeated_requests,
        "repeated_sets": repeated_sets,
        "no_new_context": no_new,
        "searches_after_hit": after_hit,
        "reasons": dict(reasons),
    }


def create_app(experiments=None, corpus=None, reasoning=None):
    base = Path(experiments or ROOT / "data/baseline/experiments").resolve()
    app = FastAPI(title="FinanceRAG evaluation", docs_url=None, redoc_url=None)

    corpus_path = Path(corpus or ROOT / "data/finqa/contexts.jsonl")
    reasoning_path = Path(reasoning or ROOT / "data/finqa/reasoning_annotations.json")

    @lru_cache(maxsize=2)
    def reasoning_catalog(signature):
        if not reasoning_path.is_file():
            return {}
        return read_json(reasoning_path)

    def reasoning_signature():
        return reasoning_path.stat().st_mtime_ns if reasoning_path.is_file() else None

    @lru_cache(maxsize=2)
    def reference_catalog(signature):
        catalog = {}
        if corpus_path.is_file():
            with corpus_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        doc = json.loads(line)
                        catalog[doc["context_id"]] = {
                            k: doc.get(k)
                            for k in ("company_id", "company_name", "report_year")
                        }
                    except (ValueError, KeyError):
                        continue
        return catalog

    def catalog():
        return reference_catalog(
            corpus_path.stat().st_mtime_ns if corpus_path.is_file() else None
        )

    def evaluate(trace):
        gold = trace.get("retrieval_validation", {}).get("gold_context_id")
        result = diagnose(trace, catalog().get(gold))
        annotations = reasoning_catalog(reasoning_signature())
        annotation = annotations.get("rows", {}).get(result["id"])
        valid = (
            annotation
            and annotation.get("status") == "matched"
            and annotations.get("provenance", {}).get("t2_revision")
            == trace["question"].get("dataset_revision")
            and annotation.get("t2_question", "").strip() == result["question"].strip()
            and str(annotation.get("t2_gold_answer")) == str(result["gold"])
        )
        result["reasoning_annotation"] = annotation if valid else None
        result["reasoning_bucket"] = (
            annotation["reasoning_bucket"] if valid else "unknown"
        )
        result["depth_bucket"] = (
            annotation.get("depth_bucket", "unknown") if valid else "unknown"
        )
        result["later_search_recovery"] = (
            result["first_hit_search"] > 1
            if result["ever"] is not None and result["first_hit_search"] is not None
            else False
            if result["ever"] is not None
            else None
        )
        return result

    def directory(run):
        path = (base / run).resolve()
        if path.parent != base or not path.is_dir():
            raise HTTPException(404, "Unknown run")
        return path

    @lru_cache(maxsize=12)
    def load(run, signature):
        rows, errors = [], []
        for path in sorted(directory(run).glob("*.json")):
            if path.name in ("run.json", "summary.json"):
                continue
            try:
                trace = read_json(path)
                if "question" in trace and "turns" in trace:
                    rows.append(evaluate(trace))
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append({"file": path.name, "error": str(error)})
        rows.sort(key=lambda r: (r["index"] is None, r["index"] or 0))
        return {"rows": rows, "read_errors": errors}

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("dashboard.html"))

    @app.get("/api/runs")
    def runs():
        found = []
        for path in sorted(base.glob("*/run.json")):
            try:
                found.append({"id": path.parent.name, "metadata": read_json(path)})
            except (ValueError, OSError):
                continue
        return found

    @app.get("/api/runs/{run}")
    def run_data(run: str):
        path = directory(run)
        signature = (
            reasoning_signature(),
            corpus_path.stat().st_mtime_ns if corpus_path.is_file() else None,
            tuple(
                (p.name, p.stat().st_mtime_ns, p.stat().st_size)
                for p in sorted(path.glob("*.json"))
            ),
        )
        return {
            "id": run,
            "metadata": read_json(path / "run.json"),
            **load(run, signature),
        }

    @app.get("/api/runs/{run}/questions/{question}")
    def question_data(run: str, question: str):
        path = directory(run)
        target = (path / (question + ".json")).resolve()
        if (
            target.parent != path
            or not target.is_file()
            or target.name in ("run.json", "summary.json")
        ):
            raise HTTPException(404, "Unknown question")
        trace = read_json(target)
        return {"diagnostics": evaluate(trace), "trace": trace}

    @app.get("/api/runs/{run}/export.csv")
    def export(run: str):
        rows = run_data(run)["rows"]
        output = io.StringIO(newline="")
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                safe = {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
                # Prevent formula execution when opening model text in spreadsheets.
                writer.writerow(
                    {
                        k: "'" + v
                        if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                        else v
                        for k, v in safe.items()
                    }
                )
        return Response(
            output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{run}-diagnostics.csv"'
            },
        )

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--experiments", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Optional local canonical-document metadata for evaluation diagnostics only",
    )
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.experiments, args.corpus), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
