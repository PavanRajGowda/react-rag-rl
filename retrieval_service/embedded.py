"""An isolated Elasticsearch process and index per Modal container."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from retrieval_service.dataset import file_sha256
from retrieval_service.index import client, index_corpus
from retrieval_service.retrieval import format_response
from retrieval_service.search import SearchError, SearchRequest, index_name, query_body


class EmbeddedSearch:
    """Container-local Elasticsearch process with versioned corpus/index validation.
    Restores immutable persisted snapshots into writable local disk instead of sharing
    active Lucene files across containers.
    """

    def __init__(
        self,
        directory,
        *,
        runtime_dir=None,
        allow_index=True,
        snapshot_dir=None,
    ):
        """Configure an isolated Elasticsearch runtime without starting it. directory holds
        corpus/manifest files; runtime_dir optionally selects local working storage;
        snapshot_dir supplies a completed persisted index. allow_index controls whether
        a missing index may be built from raw documents.
        """
        self.directory = Path(directory)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.runtime = None
        self.index_reused = False
        self.runtime_dir = Path(runtime_dir) if runtime_dir else None
        self.allow_index = allow_index
        self.process = None
        self.es = None
        self.workspace = None
        self.log = None

    def start(self):
        """Launch a local Elasticsearch subprocess and verify or restore the requested
        corpus index. Waits for readiness, checks index metadata/document count, and
        optionally builds a missing index. Serving with allow_index=False fails instead
        of silently rebuilding millions of documents. On startup failure prints the
        local log, stops resources and re-raises. Intended for the Linux Modal CPU
        image.
        """
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        # Snapshot serving checks the archive checksum and restored index metadata.
        # It does not read the 2.36 GB raw corpus again at every cold start.
        if self.snapshot_dir is None:
            digest = file_sha256(self.directory / "contexts.jsonl")
            if digest != self.manifest["sha256"]:
                raise RuntimeError("Corpus does not match manifest")
        # Elasticsearch writes mutable Lucene files to container-local disk. The
        # shared Volume stores only immutable completed snapshot archives.
        if self.runtime_dir is None:
            self.workspace = tempfile.TemporaryDirectory(prefix="rag-es-")
            runtime = Path(self.workspace.name)
        else:
            runtime = self.runtime_dir
            runtime.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        os.chown(runtime, 1000, 1000)
        self.log = (runtime / "startup.log").open("w")
        es_env = os.environ | {
            "ES_JAVA_OPTS": os.environ.get(
                "ES_JAVA_OPTS", "-Xms1g -Xmx1g -XX:+StartAttachListener"
            )
        }
        self.process = subprocess.Popen(
            [
                "/usr/share/elasticsearch/bin/elasticsearch",
                "-Ediscovery.type=single-node",
                "-Enetwork.host=127.0.0.1",
                "-Expack.security.enabled=false",
                "-Expack.security.http.ssl.enabled=false",
                "-Expack.security.autoconfiguration.enabled=false",
                "-Expack.ml.enabled=false",
                "-Eingest.geoip.downloader.enabled=false",
                "-Enode.store.allow_mmap=false",
                f"-Epath.data={runtime / 'data'}",
                f"-Epath.logs={runtime / 'logs'}",
                f"-Epath.repo={runtime / 'repository'}",
            ],
            cwd="/usr/share/elasticsearch",
            # Start the attach socket eagerly: gVisor does not expose the
            # process signal state expected by Java's lazy attach handshake.
            env=es_env,
            user=1000,
            group=1000,
            extra_groups=(),
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            self.es = client()
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("Elasticsearch exited during startup")
                if self.es.ping():
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Elasticsearch startup exceeded 180 seconds")
            index = index_name(self.manifest)
            if self.snapshot_dir is not None and not self.es.indices.exists(
                index=index
            ):
                from retrieval_service.persistence import restore_index

                restore_index(self.es, runtime, self.snapshot_dir, self.manifest)
            if self.es.indices.exists(index=index):
                self.index_reused = True
                metadata = self.es.indices.get_mapping(index=index)[index][
                    "mappings"
                ].get("_meta", {})
                if (
                    metadata.get("sha256") != self.manifest["sha256"]
                    or self.es.count(index=index)["count"]
                    != self.manifest["unique_contexts"]
                ):
                    raise RuntimeError(
                        "Persistent search index is incomplete or has the wrong corpus"
                    )
            elif self.allow_index:
                index_corpus(self.directory, es=self.es)
            else:
                raise RuntimeError(
                    "Fullwiki index is not built. Run modal run modal_app.py --build-index first."
                )
        except Exception:
            self.log.flush()
            print((runtime / "startup.log").read_text()[-8000:])
            self.stop()
            raise

    def search(self, payload):
        """Validate one payload, execute a BM25 query and return the public retrieval
        response. Invalid requests/search-domain errors become status=error
        dictionaries; unexpected transport/runtime errors propagate for trajectory retry
        handling.
        """
        try:
            request = SearchRequest.model_validate(payload)

            response = self.es.search(
                index=index_name(self.manifest),
                request_cache=True,
                **query_body(request),
            )
            return format_response(request, response, self.manifest["corpus_version"])
        except ValidationError as exc:
            return {"status": "error", "code": "invalid_request", "message": str(exc)}
        except SearchError as exc:
            return {"status": "error"} | exc.as_dict()

    def search_batch(self, payloads, *, timing=None):
        """Optionally measure service stages without changing returned evidence."""
        started = time.perf_counter()
        if timing is not None:
            timing.update(
                service_entered_utc_ns=time.time_ns(),
                query_took_ms=[None] * len(payloads),
            )
        try:
            return self._search_batch(payloads, timing=timing)
        finally:
            if timing is not None:
                timing.update(
                    service_total_seconds=time.perf_counter() - started,
                    service_returning_utc_ns=time.time_ns(),
                )

    def _search_batch(self, payloads, *, timing=None):
        """Run ordered queries with one synchronous Elasticsearch msearch request. Returns
        one response per input, keeping invalid-request and individual-query errors in
        their original slots. A shared transport failure returns structured errors.
        The batch limit is
        separate from generation trajectory concurrency.
        """
        if not payloads:
            return []
        if len(payloads) > 512:
            raise ValueError("Search batch exceeds 512 requests")
        prepared = time.perf_counter()
        output, valid, searches = [None] * len(payloads), [], []
        for index, payload in enumerate(payloads):
            try:
                request = SearchRequest.model_validate(payload)
            except ValidationError as exc:
                output[index] = {
                    "status": "error",
                    "code": "invalid_request",
                    "message": str(exc),
                }
            else:
                valid.append((index, request))
                # Explicit opt-in is required for top-K (size > 0) request caching.
                # Elasticsearch owns eviction and refresh/mapping invalidation.
                searches.extend(({"request_cache": True}, query_body(request)))
        if timing is not None:
            timing.update(
                validation_prepare_seconds=time.perf_counter() - prepared,
                valid_queries=len(valid),
            )
        if valid:
            started = time.perf_counter()
            try:
                response = self.es.msearch(
                    index=index_name(self.manifest),
                    searches=searches,
                    # Paired saved-query tests favored 16 over 4/8 on this node.
                    max_concurrent_searches=int(
                        os.environ.get("ES_SEARCH_CONCURRENCY", "16")
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- serialize transport failures as data
                # Modal callers need no elastic_transport package to decode this.
                error = {
                    "status": "error",
                    "code": "search_unavailable",
                    "message": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - started,
                    "batch_queries": len(valid),
                    "retryable": True,
                }
                for index, _ in valid:
                    output[index] = dict(error)
                return output
            finally:
                if timing is not None:
                    timing["msearch_seconds"] = time.perf_counter() - started
            formatted = time.perf_counter()
            items = response["responses"]
            if len(items) != len(valid):
                raise RuntimeError("Elasticsearch changed batch membership")
            for (index, request), item in zip(valid, items, strict=True):
                if timing is not None:
                    timing["query_took_ms"][index] = item.get("took")
                try:
                    output[index] = format_response(
                        request, item, self.manifest["corpus_version"]
                    )
                except (RuntimeError, KeyError, TypeError):
                    output[index] = {
                        "status": "error",
                        "code": "search_unavailable",
                        "message": "Elasticsearch query failed or returned incomplete results",
                    }
            if timing is not None:
                timing["format_seconds"] = time.perf_counter() - formatted
        return output

    def cache_stats(self):
        """Return native shard-cache counters and serving limits without clearing caches."""
        stats = self.es.indices.stats(
            index=index_name(self.manifest), metric="request_cache"
        )
        return {
            "request_cache": dict(stats["_all"]["total"]["request_cache"]),
            "explicit_top_k_cache": True,
            "exact_hit_counting": False,
            "max_concurrent_searches": int(
                os.environ.get("ES_SEARCH_CONCURRENCY", "16")
            ),
            "request_timeout_seconds": float(
                os.environ.get("ELASTICSEARCH_REQUEST_TIMEOUT", "90")
            ),
            "transport_retries": 0,
            "cache_lifetime": "warm CPU container; cleared on restart",
        }

    def stop(self):
        """Close the Elasticsearch client, terminate its subprocess and release local
        temporary/log resources. Does not delete the persisted Modal snapshot or corpus
        volumes.
        """
        if self.es is not None:
            self.es.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
        if self.workspace is not None:
            self.workspace.cleanup()
