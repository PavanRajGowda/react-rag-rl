"""Index-time organization extraction; never used during retrieval."""

import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"


def enrich_documents(docs, nlp=None):
    if nlp is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import spacy

        nlp = spacy.load(
            MODEL, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
        )
        if nlp.meta["version"] != MODEL_VERSION:
            raise RuntimeError("Organization extraction requires en_core_web_sm 3.8.0")
    names = {}
    # Preserve original text: normalization applies only to extracted entities.
    for source, parsed in zip(
        docs, nlp.pipe((d["text"] for d in docs), batch_size=16), strict=True
    ):
        issuer = " ".join(source["company_name"].casefold().split())
        bucket = names.setdefault(source["company_id"], set())
        for entity in parsed.ents:
            name = " ".join(entity.text.casefold().split())
            if entity.label_ == "ORG" and name != issuer and 2 <= len(name) <= 200:
                bucket.add(name)
    return [
        doc | {"company_synonyms": sorted(names[doc["company_id"]])} for doc in docs
    ]


def indexed_documents(directory, docs, nlp=None):
    """Consume a verified index-input cache, or extract when none is present."""
    path = Path(directory) / "organizations.json"
    if not path.exists():
        return enrich_documents(docs, nlp)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        (Path(directory) / "contexts.jsonl").read_bytes()
    ).hexdigest()
    if (
        artifact.get("corpus_sha256") != digest
        or artifact.get("model") != MODEL
        or artifact.get("model_version") != MODEL_VERSION
    ):
        raise RuntimeError(
            "Organization artifact does not match corpus/model; regenerate it"
        )
    names = artifact["companies"]
    if set(names) != {d["company_id"] for d in docs} or any(
        not isinstance(values, list)
        or any(not isinstance(name, str) for name in values)
        for values in names.values()
    ):
        raise RuntimeError(
            "Organization artifact has incomplete or invalid company lists"
        )
    return [doc | {"company_synonyms": names[doc["company_id"]]} for doc in docs]


def main():
    from finance_rag.search import load_docs

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/finqa"))
    args = parser.parse_args()
    docs = load_docs(args.data)
    print(f"Extracting organizations from {len(docs)} contexts...", flush=True)
    enriched = enrich_documents(docs)
    artifact = {
        "corpus_sha256": hashlib.sha256(
            (args.data / "contexts.jsonl").read_bytes()
        ).hexdigest(),
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "companies": {d["company_id"]: d["company_synonyms"] for d in enriched},
    }
    (args.data / "organizations.json").write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Saved organization lists for {len(artifact['companies'])} issuers.",
        flush=True,
    )


if __name__ == "__main__":
    main()
