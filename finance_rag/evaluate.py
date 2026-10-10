"""Oracle-filter retrieval baseline; not an end-to-end agent evaluation."""

import argparse
import json
from pathlib import Path

from finance_rag.index import client
from finance_rag.search import Registry, SearchRequest, load_docs, query_body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/finqa"))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument(
        "--output", type=Path, default=Path("data/retrieval-baseline.json")
    )
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    registry = Registry(load_docs(args.data))
    labels = [
        json.loads(line)
        for line in (args.data / "labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    results, excluded = [], []
    with client() as es:
        for row in labels:
            if row["split"] != args.split:
                continue
            if not row["question"].strip():
                excluded.append({"id": row["id"], "reason": "blank_question"})
                continue
            request = SearchRequest(
                ticker=row["ticker"],
                year=row["report_year"],
                query=row["question"],
                top_k=10,
            )
            company = registry.resolve(request)
            response = es.search(
                index=manifest["corpus_version"], **query_body(request, company)
            )
            if response.get("timed_out") or response.get("_shards", {}).get(
                "failed", 0
            ):
                raise RuntimeError("Incomplete retrieval; refusing to score")
            ids = [x["_source"]["context_id"] for x in response["hits"]["hits"]]
            rank = (
                ids.index(row["context_id"]) + 1 if row["context_id"] in ids else None
            )
            results.append(
                {"id": row["id"], "rank": rank, "retrieved_context_ids": ids}
            )
    count = len(results)
    if not count:
        raise RuntimeError("No valid questions to evaluate")
    metrics = {
        "corpus_version": manifest["corpus_version"],
        "split": args.split,
        "questions": count,
        "excluded_questions": len(excluded),
        "protocol": "oracle company/year filters; question as query",
        "recall_at_5": sum(x["rank"] is not None and x["rank"] <= 5 for x in results)
        / count,
        "recall_at_10": sum(x["rank"] is not None for x in results) / count,
        "mrr_at_10": sum(1 / x["rank"] if x["rank"] else 0 for x in results) / count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"metrics": metrics, "results": results, "excluded": excluded}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
