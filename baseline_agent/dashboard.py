"""Read-only evaluation dashboard. Run: python -m baseline_agent.dashboard."""

import argparse
import csv
import io
import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from baseline_agent.diagnostics import diagnose

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
