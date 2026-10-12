import argparse
import json
import os
from pathlib import Path

from elasticsearch import Elasticsearch, helpers

from finance_rag.organizations import MODEL, MODEL_VERSION, indexed_documents
from finance_rag.search import SEARCH_SCHEMA_VERSION, index_name, load_docs


def client():
    return Elasticsearch(
        os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"),
        api_key=os.environ.get("ELASTIC_API_KEY"),
        request_timeout=15,
        max_retries=2,
        retry_on_timeout=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/finqa"))
    args = parser.parse_args()
    print(json.dumps(index_corpus(args.data)))


def index_corpus(directory, es=None, nlp=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    docs = load_docs(directory)
    index = index_name(manifest)
    owns_client = es is None
    es = es if es is not None else client()
    try:
        if es.indices.exists(index=index):
            raise RuntimeError(
                f"Index {index} already exists; use it or create a new version"
            )
        docs = indexed_documents(directory, docs, nlp=nlp)
        properties = {k: {"type": "keyword"} for k in docs[0]}
        properties["company_name"] = {
            "type": "text",
            "analyzer": "standard",
            "fields": {"keyword": {"type": "keyword"}},
        }
        properties["company_synonyms"] = {
            "type": "text",
            "analyzer": "standard",
            "fields": {"keyword": {"type": "keyword"}},
        }
        for key in ("text", "table_markdown", "pre_text", "post_text"):
            properties[key] = {
                "type": "text",
                "index": key in ("text", "table_markdown"),
            }
        for key in ("report_year", "page_number"):
            properties[key] = {"type": "integer"}
        es.indices.create(
            index=index,
            settings={
                "number_of_shards": 1,
                "number_of_replicas": int(os.environ.get("ES_REPLICAS", "1")),
            },
            mappings={
                "dynamic": "strict",
                "_meta": manifest
                | {
                    "search_schema_version": SEARCH_SCHEMA_VERSION,
                    "organization_model": MODEL,
                    "organization_model_version": MODEL_VERSION,
                },
                "properties": properties,
            },
        )
        helpers.bulk(
            es, ({"_index": index, "_id": d["context_id"], "_source": d} for d in docs)
        )
        es.indices.refresh(index=index)
        count = es.count(index=index)["count"]
        if count != len(docs):
            raise RuntimeError(f"Indexed {count}, expected {len(docs)}")
        es.indices.put_settings(index=index, settings={"index.blocks.write": True})
        return {"index": index, "documents": count}
    finally:
        if owns_client:
            es.close()


if __name__ == "__main__":
    main()
