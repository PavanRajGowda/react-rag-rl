import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from finance_rag.api import create_app
from finance_rag.prepare import prepare
from finance_rag.search import (
    Registry,
    SearchError,
    SearchRequest,
    index_name,
    query_body,
)
from tests.helpers import identity_response, sample


@pytest.mark.parametrize(
    "args",
    [
        {"year": 2002, "query": "x"},
        {"ticker": "AAPL", "query": "x"},
        {"ticker": "AAPL", "year": 2002, "query": " "},
        {"ticker": "AAPL", "year": "2002", "query": "x"},
        {"ticker": "AAPL", "year": 2002, "query": "x", "top_k": 11},
    ],
)
def test_validation(args):
    with pytest.raises(ValidationError):
        SearchRequest(**args)


def test_registry_and_filters(tmp_path):
    prepare({"train": [sample()]}, "commit", tmp_path)
    doc = json.loads((tmp_path / "contexts.jsonl").read_text())
    registry = Registry(
        [
            doc,
            doc
            | {"company_id": "other", "ticker": "MSFT", "company_name": "Microsoft"},
        ]
    )
    request = SearchRequest(
        company_name="APPLE, INC", ticker="aapl", year=2002, query="revenue"
    )
    company = registry.resolve(request, name_candidates={"cik:320193"})
    body = query_body(request, company)
    assert body["query"]["bool"]["filter"] == [
        {"term": {"company_id": "cik:320193"}},
        {"term": {"report_year": 2002}},
    ]
    for args, code in [
        ({"ticker": "ZZZ"}, "unknown_company"),
        ({"ticker": "MSFT", "company_name": "Apple Inc."}, "company_mismatch"),
        ({"ticker": "AAPL", "year": 2003}, "report_year_unavailable"),
    ]:
        with pytest.raises(SearchError) as exc:
            registry.resolve(
                SearchRequest(**({"year": 2002, "query": "x"} | args)),
                name_candidates={"cik:320193"} if args.get("company_name") else None,
            )
        assert exc.value.code == code


def test_api_auth_preservation_empty_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_API_TOKEN", "test-token")
    manifest = prepare({"train": [sample()]}, "commit", tmp_path)
    doc = json.loads((tmp_path / "contexts.jsonl").read_text())
    es = SimpleNamespace(
        count=AsyncMock(return_value={"count": manifest["unique_contexts"]}),
        indices=SimpleNamespace(
            get_mapping=AsyncMock(
                return_value={
                    index_name(manifest): {
                        "mappings": {"_meta": manifest | {"search_schema_version": 3}}
                    }
                }
            )
        ),
        search=AsyncMock(
            return_value={
                "hits": {
                    "total": {"value": 1},
                    "hits": [{"_source": doc, "_score": 1.0}],
                }
            }
        ),
    )
    with TestClient(create_app(tmp_path, es)) as api:
        payload = {"ticker": "AAPL", "year": 2002, "query": "revenue"}
        assert api.post("/v1/search", json=payload).status_code == 401
        headers = {"Authorization": "Bearer test-token"}
        es.search.side_effect = [identity_response(doc), es.search.return_value]
        response = api.post("/v1/search", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["results"][0]["text"] == sample()["context"]
        es.search.side_effect = [
            identity_response(doc),
            {"hits": {"total": {"value": 0}, "hits": []}},
        ]
        assert (
            api.post("/v1/search", json=payload, headers=headers).json()["results"]
            == []
        )
        es.search.side_effect = TimeoutError()
        assert api.post("/v1/search", json=payload, headers=headers).status_code == 503
