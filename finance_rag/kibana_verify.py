"""Verify a deployed Kibana login, corpus count, and exact source integrity."""

import argparse
import hashlib
import json
from pathlib import Path

import requests

from finance_rag.search import index_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    credentials = json.loads(Path("data/kibana-credentials.json").read_text())
    manifest = json.loads(Path("data/finqa/manifest.json").read_text())
    organizations = json.loads(Path("data/finqa/organizations.json").read_text())
    if organizations["corpus_sha256"] != manifest["sha256"]:
        raise RuntimeError("Organization artifact does not match prepared corpus")
    response = requests.get(base + "/api/data_views", timeout=600)
    if response.status_code != 401:
        raise RuntimeError(
            f"Unauthenticated API returned {response.status_code}, expected 401"
        )
    with requests.Session() as browser:
        login = browser.post(
            base + "/internal/security/login",
            headers={"kbn-xsrf": "verification"},
            json={
                "providerType": "basic",
                "providerName": "basic",
                "currentURL": base + "/login",
                "params": {
                    "username": credentials["KIBANA_USERNAME"],
                    "password": credentials["KIBANA_PASSWORD"],
                },
            },
            timeout=60,
        )
        login.raise_for_status()
        browser_access = browser.get(base + "/api/data_views", timeout=60)
        browser_access.raise_for_status()
    with requests.Session() as session:
        session.auth = (credentials["KIBANA_USERNAME"], credentials["KIBANA_PASSWORD"])
        session.headers["kbn-xsrf"] = "verification"
        views = session.get(base + "/api/data_views", timeout=60)
        views.raise_for_status()
        if not any(x["id"] == "finqa-contexts" for x in views.json()["data_view"]):
            raise RuntimeError("FinQA data view missing")
        count_response = session.post(
            base + "/api/console/proxy",
            params={"path": index_name(manifest) + "/_count", "method": "GET"},
            timeout=60,
        )
        count_response.raise_for_status()
        count = count_response.json()["count"]
        search = session.post(
            base + "/api/console/proxy",
            params={"path": index_name(manifest) + "/_search", "method": "POST"},
            json={"size": 3000, "query": {"match_all": {}}},
            timeout=60,
        )
        search.raise_for_status()
        response = search.json()
        if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Partial search results")
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            if (
                doc.get("company_synonyms")
                != organizations["companies"][doc["company_id"]]
            ):
                raise RuntimeError(
                    "Indexed organization lists do not match prepared artifact"
                )
        docs = sorted(
            (
                {
                    key: value
                    for key, value in x["_source"].items()
                    if key != "company_synonyms"
                }
                for x in response["hits"]["hits"]
            ),
            key=lambda x: x["context_id"],
        )
        corpus = "".join(
            json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in docs
        )
        digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        if (
            count != manifest["unique_contexts"]
            or len(docs) != count
            or digest != manifest["sha256"]
        ):
            raise RuntimeError("Indexed contents do not match prepared corpus")
    result = {
        "url": base,
        "corpus_version": manifest["corpus_version"],
        "documents": count,
        "authenticated_access": True,
        "browser_cookie_login": True,
        "unauthenticated_api_status": 401,
        "data_view": "FinQA contexts",
        "exact_corpus_integrity": True,
        "organization_integrity": True,
    }
    Path("reports/kibana-verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
