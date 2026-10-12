"""Private Modal CPU Elasticsearch service, callable from GPU training."""

import asyncio
import json
import time
from pathlib import Path

import modal

root = Path(__file__).parent
image = (
    modal.Image.from_registry(
        "docker.elastic.co/elasticsearch/elasticsearch:8.19.3", add_python="3.11"
    )
    .entrypoint([])
    .pip_install_from_requirements(str(root / "requirements.lock.txt"))
    .env({"ELASTICSEARCH_URL": "http://127.0.0.1:9200", "ES_REPLICAS": "0"})
    .add_local_python_source("finance_rag")
    .add_local_file(root / "data/finqa/contexts.jsonl", "/corpus/contexts.jsonl")
    .add_local_file(root / "data/finqa/manifest.json", "/corpus/manifest.json")
    .add_local_file(
        root / "data/finqa/organizations.json", "/corpus/organizations.json"
    )
)
app = modal.App("finance-rag-search")


@app.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    cpu=0.25,
    memory=256,
    timeout=180,
)
async def verify_training_call():
    """Check a deployed service from another Modal container, without a GPU."""
    deployed = modal.Cls.from_name("finance-rag-search", "SearchService")()
    request = {"ticker": "AAPL", "year": 2002, "query": "net sales 2001 2000"}
    cold_started = time.perf_counter()
    first = await deployed.search.remote.aio(request)
    first_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    second = await deployed.search.remote.aio(request)
    warm_seconds = time.perf_counter() - warm_started
    if first["status"] != "ok" or not first["results"] or first != second:
        raise RuntimeError("Deployed service failed deterministic search verification")
    return {
        "corpus_version": first["corpus_version"],
        "modal_to_modal": True,
        "results": len(first["results"]),
        "first_call_seconds": first_seconds,
        "warm_call_seconds": warm_seconds,
    }


@app.cls(
    image=image,
    cpu=2,
    memory=4096,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    startup_timeout=300,
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class SearchService:
    @modal.enter()
    def start(self):
        from finance_rag.embedded import EmbeddedSearch

        self.engine = EmbeddedSearch(Path("/corpus"))
        self.engine.start()

    @modal.method()
    async def search(self, request: dict):
        return await asyncio.to_thread(self.engine.search, request)

    @modal.exit()
    def stop(self):
        if hasattr(self, "engine"):
            self.engine.stop()


@app.local_entrypoint()
def main(
    evaluate: bool = False,
    split: str = "dev",
    verify_deployed: bool = False,
    verify_companies: bool = False,
    verify_resolution: bool = False,
):
    if verify_resolution:
        service = SearchService()
        requests = [
            {"company_name": "Apple", "year": 2002, "query": "net sales"},
            {
                "company_name": "Alltel Information Services",
                "year": 2010,
                "query": "acquisition",
            },
            {
                "company_name": "Metavante",
                "year": 2010,
                "query": "Metavante stock awards total purchase price",
            },
            {
                "company_name": "Metavante",
                "year": 2018,
                "query": "Metavante stock awards total purchase price",
            },
            {
                "company_name": "ZZZZ_UNLISTED",
                "year": 2018,
                "query": "Metavante stock awards total purchase price",
            },
        ]
        checks = []
        for request in requests:
            result = service.search.remote(request)
            if result.get("status") == "ok":
                if not result["results"] or len(result["results"]) > 2:
                    raise RuntimeError(f"Unexpected retrieval: {result}")
                if result["match_method"] != "broad_query" and any(
                    doc["report_year"] != request["year"] for doc in result["results"]
                ):
                    raise RuntimeError("Report year filter was not respected")
            elif result.get("code") != "report_year_unavailable":
                raise RuntimeError(f"Unexpected error: {result}")
            checks.append({"request": request, "response": result})
        if (
            checks[0]["response"].get("match_method") != "company_name"
            or checks[1]["response"].get("match_method") != "company_synonyms"
            or checks[-1]["response"].get("match_method") != "broad_query"
        ):
            raise RuntimeError("Resolution precedence or broad fallback failed")
        for check in checks:
            if (
                check["request"]["company_name"] == "Metavante"
                and check["response"].get("status") == "ok"
                and not any(
                    doc["context_id"] == "finqa_test_ctx_77"
                    for doc in check["response"]["results"]
                )
            ):
                raise RuntimeError("Metavante acquisition table was not retrieved")
        (root / "reports/elasticsearch-resolution-smoke.json").write_text(
            json.dumps(checks, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                [
                    {
                        "request": x["request"],
                        "status": x["response"]["status"],
                        "match_method": x["response"].get("match_method"),
                        "code": x["response"].get("code"),
                        "results": len(x["response"].get("results", [])),
                    }
                    for x in checks
                ],
                indent=2,
            )
        )
        return
    if verify_companies:
        service = SearchService()
        request = {"company_name": "apple", "year": 2002, "query": "net sales"}
        apple = service.search.remote(request)
        if (
            apple.get("status") != "ok"
            or apple["resolved_company"]["ticker"] != "AAPL"
            or not apple["results"]
        ):
            raise RuntimeError("Short company name failed to resolve Apple")
        ambiguous = service.search.remote(request | {"company_name": "American"})
        if (
            ambiguous.get("code") != "ambiguous_company"
            or len(ambiguous.get("candidates", [])) < 2
        ):
            raise RuntimeError("Ambiguous company did not return candidates")
        if any(
            not {"company_name", "company_id", "ticker"} <= x.keys()
            for x in ambiguous["candidates"]
        ):
            raise RuntimeError("Candidate metadata is incomplete")
        refined = service.search.remote(
            request | {"company_name": "APPLE, INC.", "ticker": "aapl"}
        )
        if (
            refined.get("status") != "ok"
            or refined["resolved_company"]["ticker"] != "AAPL"
        ):
            raise RuntimeError("Refined company name did not resolve")
        report = {
            "corpus_version": apple["corpus_version"],
            "short_name": "apple",
            "resolved_company": apple["resolved_company"],
            "results": len(apple["results"]),
            "ambiguity": ambiguous,
            "refined_name_and_ticker": True,
        }
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports/company-resolution-smoke.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return
    if verify_deployed:
        verification = verify_training_call.remote()
        (root / "reports").mkdir(exist_ok=True)
        (root / "reports/modal-deployed.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        print(json.dumps(verification, indent=2))
        return
    service = SearchService()
    started = time.perf_counter()
    result = service.search.remote(
        {"ticker": "AAPL", "year": 2002, "query": "net sales 2001 2000", "top_k": 5}
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "elapsed_including_startup_seconds": elapsed,
                "status": result.get("status"),
                "corpus_version": result.get("corpus_version"),
                "results": len(result.get("results", [])),
            },
            indent=2,
        )
    )
    if result.get("status") != "ok" or not result.get("results"):
        raise RuntimeError(f"Smoke test failed: {result}")
    from finance_rag.search import load_docs

    original = {x["context_id"]: x for x in load_docs(root / "data/finqa")}
    for hit in result["results"]:
        if {
            k: v for k, v in hit.items() if k not in ("score", "company_synonyms")
        } != original[hit["context_id"]]:
            raise RuntimeError("Retrieved context changed from source corpus")
        if hit["ticker"] != "AAPL" or hit["report_year"] != 2002:
            raise RuntimeError("Metadata filter failed")
    invalid = service.search.remote({"ticker": "AAPL", "query": "net sales"})
    unknown = service.search.remote(
        {"ticker": "ZZZZ_UNKNOWN", "year": 2002, "query": "sales"}
    )
    if (
        invalid.get("code") != "invalid_request"
        or unknown.get("code") != "unknown_company"
    ):
        raise RuntimeError("Search argument error handling failed")
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/modal-smoke.json").write_text(
        json.dumps(
            {
                "corpus_version": result["corpus_version"],
                "elapsed_including_startup_seconds": elapsed,
                "results": len(result["results"]),
                "context_preservation": True,
                "metadata_filters": True,
                "argument_error_handling": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if evaluate:
        from finance_rag.modal_benchmark import evaluate_service

        asyncio.run(evaluate_service(service, root, split))
