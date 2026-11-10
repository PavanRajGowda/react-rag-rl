import argparse
import json
import os
import time
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

from retrieval_service.dataset import DATA_DIR, file_sha256
from retrieval_service.search import SEARCH_SCHEMA_VERSION, index_name, iter_docs


def client():
    """Create a synchronous Elasticsearch client for the local runtime with bounded
    connection/retry settings. No remote Modal method is invoked here.
    """
    return Elasticsearch(
        os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"),
        api_key=os.environ.get("ELASTIC_API_KEY"),
        # Keep the ES deadline below the enclosing 120-second Modal RPC deadline.
        # Do not repeat an expensive msearch implicitly after a timeout.
        request_timeout=float(os.environ.get("ELASTICSEARCH_REQUEST_TIMEOUT", "90")),
        max_retries=0,
        retry_on_timeout=False,
    )


def main():
    """Parse the corpus-directory option and invoke local corpus indexing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(index_corpus(args.data)))


def index_corpus(directory, es=None):
    """Build a versioned Elasticsearch index from streamed corpus documents. directory
    supplies a completed manifest; es optionally injects a client. Verifies corpus
    identity, creates document-only mappings, bulk indexes records and checks the final
    count before returning index metadata. This is a one-time build, not a per-search
    operation.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if file_sha256(directory / "contexts.jsonl") != manifest["sha256"]:
        raise ValueError("Corpus hash disagrees with manifest")
    first = next(iter_docs(directory), None)
    if first is None or manifest["unique_contexts"] <= 0:
        raise ValueError("Empty corpus")
    index = index_name(manifest)
    owns_client = es is None
    es = es if es is not None else client()
    try:
        if es.indices.exists(index=index):
            raise RuntimeError(
                f"Index {index} already exists; use it or create a new version"
            )
        properties = {k: {"type": "keyword"} for k in first}
        for key in ("title", "text"):
            properties[key] = {"type": "text", "analyzer": "standard"}
        properties["published_at"] = {"type": "date"}
        for key in ("chunk_index", "start_char", "end_char", "token_count"):
            if key in properties:
                properties[key] = {"type": "integer"}
        es.indices.create(
            index=index,
            settings={
                "number_of_shards": 1,
                "refresh_interval": "-1",
                "number_of_replicas": int(os.environ.get("ES_REPLICAS", "1")),
            },
            mappings={
                "dynamic": "strict",
                "_meta": manifest | {"search_schema_version": SEARCH_SCHEMA_VERSION},
                "properties": properties,
            },
        )
        started = time.perf_counter()

        def actions():
            """Yield one Elasticsearch bulk-index action per corpus document, using
            document_id as the stable record key. Streaming bounds Python memory
            independently of corpus size.
            """
            for number, doc in enumerate(iter_docs(directory), 1):
                if number % 100000 == 0:
                    print(
                        f"Indexing {number:,}/{manifest['unique_contexts']:,} ({number / manifest['unique_contexts']:.1%}); {time.perf_counter() - started:.0f}s",
                        flush=True,
                    )
                yield {
                    "_index": index,
                    "_id": doc.get("context_id", doc["document_id"]),
                    "_source": doc,
                }

        helpers.bulk(
            es.options(request_timeout=120),
            actions(),
            chunk_size=1000,
            max_chunk_bytes=16 * 1024 * 1024,
        )
        es.indices.put_settings(index=index, settings={"refresh_interval": "1s"})
        es.indices.refresh(index=index)
        count = es.count(index=index)["count"]
        if count != manifest["unique_contexts"]:
            raise RuntimeError(
                f"Indexed {count}, expected {manifest['unique_contexts']}"
            )
        es.indices.put_settings(index=index, settings={"index.blocks.write": True})
        return {"index": index, "documents": count}
    finally:
        if owns_client:
            es.close()


if __name__ == "__main__":
    main()
