import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from finance_rag.api import create_app
from finance_rag.embedded import EmbeddedSearch
from finance_rag.prepare import prepare
from finance_rag.search import (
    Registry,
    SearchError,
    SearchRequest,
    company_candidates,
    index_name,
)
from tests.helpers import identity_response, sample


def response(*ids):
    return {
        "aggregations": {
            "companies": {
                "buckets": [{"key": x} for x in ids],
                "sum_other_doc_count": 0,
                "doc_count_error_upper_bound": 0,
            }
        }
    }


def registry_docs(tmp_path):
    prepare({"train": [sample()]}, "commit", tmp_path)
    apple = json.loads((tmp_path / "contexts.jsonl").read_text())
    other = apple | {
        "company_id": "other",
        "company_name": "Apple Holdings",
        "ticker": "APH",
    }
    return apple, other


def test_short_name_uses_analyzed_match_and_deduplicates(tmp_path):
    apple, _ = registry_docs(tmp_path)
    registry = Registry([apple, apple | {"context_id": "another"}])
    es = SimpleNamespace(search=Mock(return_value=response(apple["company_id"])))
    resolved = registry.resolve(
        SearchRequest(company_name="apple", year=2002, query="sales"),
        es=es,
        index="test-index",
    )
    assert resolved["ticker"] == "AAPL"
    assert es.search.call_args.kwargs["query"] == {
        "match": {"company_name": {"query": "apple", "operator": "and"}}
    }
    assert es.search.call_args.kwargs["aggs"]["companies"]["terms"]["size"] == 1


def test_ambiguity_contains_names_tickers_and_can_be_refined(tmp_path):
    apple, other = registry_docs(tmp_path)
    registry = Registry([apple, other])
    ids = {apple["company_id"], other["company_id"]}
    with pytest.raises(SearchError) as exc:
        registry.resolve(
            SearchRequest(company_name="apple", year=2002, query="sales"),
            name_candidates=ids,
        )
    detail = exc.value.as_dict()
    assert detail["code"] == "ambiguous_company"
    assert {x["company_name"] for x in detail["candidates"]} == {
        "Apple Inc.",
        "Apple Holdings",
    }
    assert all(
        set(x) == {"company_name", "company_id", "ticker"} for x in detail["candidates"]
    )
    assert (
        registry.resolve(
            SearchRequest(
                company_name="apple", ticker="aapl", year=2002, query="sales"
            ),
            name_candidates=ids,
        )["ticker"]
        == "AAPL"
    )


@pytest.mark.parametrize(
    "result",
    [
        {"timed_out": True},
        {"_shards": {"failed": 1}},
        {"aggregations": {"companies": {"buckets": [], "sum_other_doc_count": 1}}},
    ],
)
def test_incomplete_candidates_are_backend_failures(result):
    with pytest.raises(RuntimeError):
        company_candidates(result)


def test_both_interfaces_return_candidate_feedback(tmp_path, monkeypatch):
    # Include both companies in the local corpus used by HTTP startup.
    rows = [
        sample(),
        sample(context_id="c2", ticker="APH", cik=123)
        | {"company_name": "Apple Holdings"},
    ]
    manifest = prepare({"train": rows}, "commit", tmp_path)
    from finance_rag.search import load_docs

    docs = load_docs(tmp_path)
    es = SimpleNamespace(
        count=AsyncMock(return_value={"count": 2}),
        indices=SimpleNamespace(
            get_mapping=AsyncMock(
                return_value={
                    index_name(manifest): {
                        "mappings": {"_meta": manifest | {"search_schema_version": 4}}
                    }
                }
            )
        ),
        search=AsyncMock(return_value=identity_response(*docs)),
    )
    monkeypatch.setenv("SEARCH_API_TOKEN", "test-token")
    payload = {"company_name": "apple", "year": 2002, "query": "sales"}
    with TestClient(create_app(tmp_path, es)) as api:
        result = api.post(
            "/v1/search", json=payload, headers={"Authorization": "Bearer test-token"}
        )
        assert result.status_code == 422
        assert len(result.json()["detail"]["candidates"]) == 2
        es.search.side_effect = [
            identity_response(docs[0]),
            {
                "hits": {
                    "total": {"value": 1},
                    "hits": [{"_source": docs[0], "_score": 1.0}],
                }
            },
        ]
        resolved = api.post(
            "/v1/search",
            json=payload | {"company_name": "Apple Inc."},
            headers={"Authorization": "Bearer test-token"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["resolved_company"]["ticker"] == "AAPL"
        es.search.side_effect = TimeoutError()
        assert (
            api.post(
                "/v1/search",
                json=payload,
                headers={"Authorization": "Bearer test-token"},
            ).status_code
            == 503
        )
    engine = EmbeddedSearch(tmp_path)
    engine.manifest = manifest
    engine.registry = Registry(docs)
    engine.es = SimpleNamespace(search=Mock(return_value=identity_response(*docs)))
    result = engine.search(payload)
    assert result["code"] == "ambiguous_company"
    assert len(result["candidates"]) == 2


def test_unknown_company_suggestions_do_not_resolve_automatically(tmp_path):
    apple, _ = registry_docs(tmp_path)
    entergy = apple | {
        "company_id": "entergy",
        "company_name": "Entergy",
        "ticker": "ETR",
    }
    registry = Registry([entergy, entergy | {"context_id": "duplicate"}])
    for name in ("Entergy Louisiana", "Entenergy Louisiana", "Entergy Corporation"):
        with pytest.raises(SearchError) as exc:
            registry.resolve(
                SearchRequest(company_name=name, year=2002, query="revenue"),
                name_candidates=set(),
            )
        assert exc.value.code == "unknown_company"
        assert exc.value.as_dict()["candidates"] == [
            {"company_id": "entergy", "company_name": "Entergy", "ticker": "ETR"}
        ]
    assert registry.suggest("zzzzzzzz") == []
    assert registry.suggest("Corporation") == []
