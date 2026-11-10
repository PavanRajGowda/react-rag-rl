"""Private Modal CPU Elasticsearch service, callable from GPU training."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import modal

from retrieval_service.dataset import CORPUS_VERSION, DATA_DIR, SEARCH_APP
from retrieval_service.search import SEARCH_SCHEMA_VERSION

root = Path(__file__).parent
image = (
    modal.Image.from_registry(
        "docker.elastic.co/elasticsearch/elasticsearch:8.19.3", add_python="3.11"
    )
    .entrypoint([])
    .pip_install_from_requirements(str(root / "requirements.lock.txt"))
    .env(
        {
            "ELASTICSEARCH_URL": "http://127.0.0.1:9200",
            "ES_REPLICAS": "0",
            "ES_JAVA_OPTS": "-Xms4g -Xmx4g -XX:+StartAttachListener",
        }
    )
    .add_local_python_source("retrieval_service")
)
app = modal.App(SEARCH_APP)

# Evaluated at deployment, so configuration is fixed for each published version.
from retrieval_service.dynamic_batching import BatchSettings

DYNAMIC_BATCH = BatchSettings.from_env(os.environ)

corpus = modal.Volume.from_name(
    "hotpotqa-fullwiki-documents-corpus", create_if_missing=True, version=2
)
index_storage = modal.Volume.from_name(
    "hotpotqa-fullwiki-documents-index", create_if_missing=True, version=2
)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("pydantic==2.14.0")
    .add_local_python_source("retrieval_service"),
    cpu=0.25,
    memory=256,
    timeout=900,
)
async def verify_training_call():
    """Check a deployed service from another Modal container, without a GPU."""
    deployed = modal.Cls.from_name(SEARCH_APP, "SearchService")()
    request = {"query": "Arthur's Magazine"}
    cold_started = time.perf_counter()
    first = await deployed.search.remote.aio(request)
    first_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    second = await deployed.search.remote.aio(request)
    warm_seconds = time.perf_counter() - warm_started
    if first["status"] != "ok" or not first["results"] or first != second:
        raise RuntimeError("Deployed service failed deterministic search verification")
    if (
        first.get("corpus_version") != CORPUS_VERSION
        or first.get("search_schema_version") != SEARCH_SCHEMA_VERSION
        or any("context_id" in doc or "chunk_index" in doc for doc in first["results"])
    ):
        raise RuntimeError("Deployed service is not using document-only evidence")
    identity = await deployed.identity.remote.aio()
    if (
        not identity["index_reused"]
        or identity["documents"] != identity["expected_documents"]
    ):
        raise RuntimeError(
            "Deployed service did not restore the complete fullwiki index"
        )
    return {
        "corpus_version": first["corpus_version"],
        "search_schema_version": first["search_schema_version"],
        "evidence_identity": "document_id",
        "document_ids": [doc["document_id"] for doc in first["results"]],
        "service_identity": identity,
        "top_titles": [doc["title"] for doc in first["results"]],
        "modal_to_modal": True,
        "results": len(first["results"]),
        "first_call_seconds": first_seconds,
        "warm_call_seconds": warm_seconds,
    }


@app.cls(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/corpus": corpus, "/search-data": index_storage},
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    startup_timeout=900,
    timeout=600,
)
@modal.concurrent(max_inputs=16)
class SearchService:
    @modal.enter()
    def start(self):
        """Start the container's embedded CPU Elasticsearch service from the completed
        persisted snapshot. Disallows rebuilding the full corpus during a serving cold
        start.
        """
        from retrieval_service.embedded import EmbeddedSearch
        from retrieval_service.search import index_name

        manifest = json.loads(Path("/corpus/manifest.json").read_text())
        self.engine = EmbeddedSearch(
            Path("/corpus"),
            snapshot_dir=Path("/search-data") / index_name(manifest),
            allow_index=False,
        )
        self.engine.start()

    @modal.method()
    async def search(self, request: dict):
        """Run one synchronous embedded search in a worker thread so the Modal async method
        does not block its event loop. Returns the same response contract as the batch
        method.
        """
        return await asyncio.to_thread(self.engine.search, request)

    @modal.method()
    def search_batch(self, requests: list[dict], trace_id: str | None = None):
        """Delegate an ordered payload list to the embedded msearch implementation. A
        shared transport error propagates to the caller; individual responses retain
        input ordering.
        """
        if trace_id is None:
            return self.engine.search_batch(requests)
        timing = {"batch_id": trace_id, "container": os.environ.get("MODAL_TASK_ID")}
        responses = self.engine.search_batch(requests, timing=timing)
        return {"responses": responses, "timing": timing}

    @modal.method()
    def identity(self):
        """Return deployed corpus/index counts, reuse status and container identity for
        verification. Queries Elasticsearch metadata but performs no generation or
        training.
        """
        from retrieval_service.search import index_name

        return {
            "corpus_version": self.engine.manifest["corpus_version"],
            "expected_documents": self.engine.manifest["unique_contexts"],
            "documents": self.engine.es.count(index=index_name(self.engine.manifest))[
                "count"
            ],
            "index_reused": self.engine.index_reused,
            "container": __import__("os").environ.get("MODAL_TASK_ID"),
        }

    @modal.exit()
    def stop(self):
        """Stop the container's embedded Elasticsearch runtime if startup created one.
        Persisted corpus and snapshot volumes remain intact.
        """
        if hasattr(self, "engine"):
            self.engine.stop()

    @modal.method()
    def cache_stats(self):
        """Expose native request-cache hit/miss/memory/eviction metrics for monitoring."""
        return self.engine.cache_stats()


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("pydantic==2.14.0")
    .env(
        {
            "SEARCH_DYNAMIC_BATCH_SIZE": str(DYNAMIC_BATCH.max_batch_size),
            "SEARCH_DYNAMIC_WAIT_MS": str(DYNAMIC_BATCH.wait_ms),
        }
    )
    .add_local_python_source("retrieval_service"),
    cpu=0.125,
    memory=256,
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
    timeout=600,
    retries=0,
)
@modal.batched(
    max_batch_size=DYNAMIC_BATCH.max_batch_size, wait_ms=DYNAMIC_BATCH.wait_ms
)
async def dynamic_search(requests: list[dict]) -> list[dict]:
    """Accumulate individual calls natively, delegate one batch to persistent ES.

    A separate function avoids Modal's batched-class/ordinary-method restriction.
    The existing SearchService remains the sole index owner. No new index or GPU
    is allocated. Logs expose realized sizes and RPC latency for later experiments;
    latency excludes Modal's batching wait and any queueing before invocation.
    """
    from retrieval_service.dynamic_batching import dispatch_envelopes

    entered = (time.perf_counter_ns(), time.time_ns())
    service = modal.Cls.from_name(SEARCH_APP, "SearchService")()
    responses, metrics = await dispatch_envelopes(
        requests,
        service.search_batch.remote.aio,
        entered=entered,
        settings=DYNAMIC_BATCH,
    )
    print(json.dumps({"kind": "dynamic_search_batch", **metrics}), flush=True)
    return responses


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=14400,
    volumes={"/corpus": corpus, "/search-data": index_storage},
    max_containers=1,
)
def build_fullwiki_index():
    """Build and persist the fullwiki Elasticsearch index once on CPU. Reuses an existing
    completed marker; otherwise indexes local data, creates a supported ES snapshot,
    publishes its archive and commits the Volume. Never allocates a GPU.
    """
    from retrieval_service.embedded import EmbeddedSearch
    from retrieval_service.persistence import (
        check_marker,
        create_snapshot,
        publish_repository,
    )
    from retrieval_service.search import index_name

    manifest = json.loads(Path("/corpus/manifest.json").read_text())
    target = Path("/search-data") / index_name(manifest)
    if (target / "ready.json").exists():
        return check_marker(target, manifest) | {"already_built": True}
    with tempfile.TemporaryDirectory(prefix="fullwiki-build-") as runtime:
        engine = EmbeddedSearch("/corpus", runtime_dir=runtime)
        try:
            engine.start()
            repository = create_snapshot(engine.es, runtime, manifest)
        finally:
            engine.stop()
        marker = publish_repository(repository, target, manifest)
        index_storage.commit()
        return marker


@app.function(image=image, cpu=2, memory=8192, timeout=600)
def verify_snapshot_roundtrip():
    """Build a small CPU test index, publish a supported snapshot and restore it into a
    fresh runtime. Returns verification evidence; validates persistence mechanics
    without indexing the full corpus.
    """
    import hashlib
    import os

    from retrieval_service.embedded import EmbeddedSearch
    from retrieval_service.persistence import create_snapshot, publish_repository

    with tempfile.TemporaryDirectory(prefix="snapshot-smoke-") as work:
        work = Path(work)
        os.chmod(work, 0o755)
        docs = work / "corpus"
        docs.mkdir()
        rows = [
            {
                "document_id": "wiki-1",
                "title": "Arthur's Magazine",
                "text": "Arthur's Magazine was an American literary periodical.",
                "url": "https://en.wikipedia.org/wiki/Arthur%27s_Magazine",
            },
            {
                "document_id": "wiki-2",
                "title": "First for Women",
                "text": "First for Women is a magazine.",
                "url": "https://en.wikipedia.org/wiki/First_for_Women",
            },
        ]
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (docs / "contexts.jsonl").write_bytes(content)
        manifest = {
            "corpus_version": "snapshot-smoke",
            "sha256": hashlib.sha256(content).hexdigest(),
            "unique_contexts": 2,
        }
        (docs / "manifest.json").write_text(json.dumps(manifest))
        first = EmbeddedSearch(docs, runtime_dir=work / "first")
        try:
            first.start()
            expected = first.search({"query": "Arthur's Magazine", "top_k": 2})
            repository = create_snapshot(first.es, first.runtime, manifest)
        finally:
            first.stop()
        marker = publish_repository(repository, work / "saved", manifest)
        second = EmbeddedSearch(
            docs,
            runtime_dir=work / "second",
            allow_index=False,
            snapshot_dir=work / "saved",
        )
        try:
            second.start()
            actual = second.search({"query": "Arthur's Magazine", "top_k": 2})
            if not second.index_reused or actual != expected:
                raise RuntimeError("Fresh-node restore changed retrieval results")
            return marker | {"fresh_node_restore": True, "same_search_results": True}
        finally:
            second.stop()


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=900,
    volumes={"/corpus": corpus, "/search-data": index_storage},
    single_use_containers=True,
)
def verify_fullwiki_restore():
    """Restore the completed fullwiki snapshot in an independent CPU runtime and verify
    document counts and representative searches. Does not regenerate the corpus.
    """
    import os

    from retrieval_service.embedded import EmbeddedSearch
    from retrieval_service.search import index_name

    manifest = json.loads(Path("/corpus/manifest.json").read_text())
    engine = EmbeddedSearch(
        "/corpus",
        snapshot_dir=Path("/search-data") / index_name(manifest),
        allow_index=False,
    )
    started = time.perf_counter()
    try:
        engine.start()
        query = {"query": "Arthur's Magazine", "top_k": 2}
        first, second = engine.search(query), engine.search(query)
        if first != second or not first["results"] or not engine.index_reused:
            raise RuntimeError("Fullwiki persistence/search verification failed")
        properties = engine.es.indices.get_mapping(index=index_name(manifest))[
            index_name(manifest)
        ]["mappings"]["properties"]
        if "context_id" in properties or "chunk_index" in properties:
            raise RuntimeError("Restored index still contains legacy chunk fields")
        if any("context_id" in doc or "chunk_index" in doc for doc in first["results"]):
            raise RuntimeError("Restored search results still contain chunk fields")
        for doc in first["results"]:
            stored = engine.es.get(index=index_name(manifest), id=doc["document_id"])
            if stored["_source"]["document_id"] != doc["document_id"]:
                raise RuntimeError("Elasticsearch identity differs from document_id")
        probes = []
        for title in ("Arthur's Magazine", "Anarchism", "Eiffel Tower"):
            response = engine.search({"query": title, "top_k": 2})
            if (
                response != engine.search({"query": title, "top_k": 2})
                or not response["results"]
            ):
                raise RuntimeError(
                    "Retrieval probe is empty or nondeterministic: " + title
                )
            probes.append(
                {
                    "query": title,
                    "top_titles": [doc["title"] for doc in response["results"]],
                    "document_ids": [doc["document_id"] for doc in response["results"]],
                }
            )
        return {
            "retrieval_probes": probes,
            "evidence_identity": "document_id",
            "mapping_fields": sorted(properties),
            "documents": engine.es.count(index=index_name(manifest))["count"],
            "index_reused": True,
            "container": os.environ.get("MODAL_TASK_ID"),
            "startup_seconds": time.perf_counter() - started,
            "top_titles": [doc["title"] for doc in first["results"]],
            "same_search_results": True,
        }
    finally:
        engine.stop()


@app.local_entrypoint()
def main(
    verify_deployed: bool = False,
    upload: bool = False,
    build_index: bool = False,
    verify_persistence: bool = False,
    verify_fullwiki: bool = False,
):
    """Parse local Modal entrypoint flags for corpus upload, index build and deployment
    verification. Executes only the requested CPU operation and prints/saves its result;
    it is separate from the Tinker training CLI.
    """
    if verify_fullwiki:
        result = verify_fullwiki_restore.remote()
        (root / "data/retrieval-audits").mkdir(parents=True, exist_ok=True)
        (root / "data/retrieval-audits/hotpotqa-index-restore.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return
    if verify_persistence:
        print(json.dumps(verify_snapshot_roundtrip.remote(), indent=2))
        return
    if upload:
        if not (DATA_DIR / "manifest.json").is_file():
            raise RuntimeError("Prepare HotpotQA locally before uploading")
        with corpus.batch_upload(force=False) as batch:
            for name in ("contexts.jsonl", "manifest.json"):
                batch.put_file(DATA_DIR / name, "/" + name)
        print("Uploaded fullwiki corpus. Next: modal run modal_app.py --build-index")
        return
    if build_index:
        result = build_fullwiki_index.remote()
        (root / "data/retrieval-audits").mkdir(parents=True, exist_ok=True)
        (root / "data/retrieval-audits/hotpotqa-index-build.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return
    if verify_deployed:
        result = verify_training_call.remote()
        (root / "data/retrieval-audits").mkdir(parents=True, exist_ok=True)
        (root / "data/retrieval-audits/hotpotqa-search-verification.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return
    service = SearchService()
    result = service.search.remote({"query": "Arthur's Magazine", "top_k": 2})
    if result.get("status") != "ok" or not result.get("results"):
        raise RuntimeError("HotpotQA search smoke failed")
    print(json.dumps(result, indent=2))
