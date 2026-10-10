"""Benchmark real Modal retrieval; labels stay on the calling machine."""

import asyncio
import json
import time


async def evaluate_service(service, root, split):
    if split not in ("train", "dev", "test"):
        raise ValueError("Unknown split")
    labels = [
        json.loads(x)
        for x in (root / "data/finqa/labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    semaphore = asyncio.Semaphore(8)

    async def retrieve(row):
        if not row["question"].strip():
            return {"id": row["id"], "excluded": "blank_question"}
        async with semaphore:
            started = time.perf_counter()
            response = await service.search.remote.aio(
                {
                    "ticker": row["ticker"],
                    "year": row["report_year"],
                    "query": row["question"],
                    "top_k": 10,
                }
            )
            elapsed = time.perf_counter() - started
        if response["status"] != "ok":
            raise RuntimeError(f"Search failed for {row['id']}: {response}")
        ids = [x["context_id"] for x in response["results"]]
        rank = ids.index(row["context_id"]) + 1 if row["context_id"] in ids else None
        return {
            "id": row["id"],
            "rank": rank,
            "seconds": elapsed,
            "retrieved_context_ids": ids,
        }

    results = await asyncio.gather(
        *(retrieve(x) for x in labels if x["split"] == split)
    )
    eligible = [x for x in results if "excluded" not in x]
    count = len(eligible)
    latencies = sorted(x["seconds"] for x in eligible)
    manifest = json.loads((root / "data/finqa/manifest.json").read_text())
    metrics = {
        "corpus_version": manifest["corpus_version"],
        "split": split,
        "questions": count,
        "excluded_questions": len(results) - count,
        "protocol": "oracle company/year filters; question as query; concurrency 8",
        "recall_at_5": sum(x["rank"] is not None and x["rank"] <= 5 for x in eligible)
        / count,
        "recall_at_10": sum(x["rank"] is not None for x in eligible) / count,
        "mrr_at_10": sum(1 / x["rank"] if x["rank"] else 0 for x in eligible) / count,
        "request_latency_seconds": {
            "median": latencies[count // 2],
            "p95": latencies[int((count - 1) * 0.95)],
        },
    }
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / f"modal-retrieval-{split}.json").write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
