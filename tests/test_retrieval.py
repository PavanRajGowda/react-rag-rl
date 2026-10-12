import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from baseline_agent.protocol import format_information
from finance_rag.organizations import (
    MODEL,
    MODEL_VERSION,
    enrich_documents,
    indexed_documents,
)
from finance_rag.retrieval import identities, retrieve
from finance_rag.search import SearchError, SearchRequest
from tests.helpers import identity_response

FIS = {
    "company_id": "fis",
    "company_name": "Fidelity National Information Services",
    "ticker": "FIS",
    "report_year": 2010,
}
OTHER = FIS | {"company_id": "other", "ticker": "OTHER", "company_name": "Other Issuer"}
HITS = {
    "hits": {
        "total": {"value": 1},
        "hits": [
            {
                "_score": 1,
                "_source": FIS
                | {
                    "context_id": "c1",
                    "page_number": 70,
                    "text": "| Stock awards | 121.4 |\n| Total price | 4187.8 |",
                },
            }
        ],
    }
}


def run(responses, name="Metavante", year=2010):
    search = AsyncMock(side_effect=responses)
    result = asyncio.run(
        retrieve(
            SearchRequest(
                company_name=name, year=year, query="stock awards purchase price"
            ),
            search,
            "corpus",
        )
    )
    return result, search


def test_primary_match_has_priority_and_never_queries_synonyms():
    result, search = run([identity_response(FIS), HITS], name="Fidelity")
    assert result["match_method"] == "company_name"
    assert search.await_count == 2
    assert result["applied_filters"] == {"company_ids": ["fis"], "report_year": 2010}


def test_synonym_returns_full_context_and_explanation_visible_to_agent():
    result, search = run([identity_response(), identity_response(FIS), HITS])
    assert search.await_args_list[1].kwargs["query"] == {
        "match_phrase": {"company_synonyms": "Metavante"}
    }
    assert result["match_method"] == "company_synonyms"
    assert result["results"][0]["text"] == HITS["hits"]["hits"][0]["_source"]["text"]
    assert "company_synonyms" in format_information(result)
    assert "121.4" in format_information(result)


@pytest.mark.parametrize("synonym", [False, True])
def test_missing_year_does_not_relax_filters(synonym):
    responses = (
        [identity_response(), identity_response(FIS)]
        if synonym
        else [identity_response(FIS)]
    )
    with pytest.raises(SearchError) as exc:
        run(responses, year=2018)
    assert exc.value.code == "report_year_unavailable"
    assert exc.value.candidates[0]["report_years"] == [2010]


def test_primary_ambiguity_returns_candidates_without_evidence():
    with pytest.raises(SearchError) as exc:
        run([identity_response(FIS, OTHER)])
    assert exc.value.code == "ambiguous_company"
    assert len(exc.value.candidates) == 2


def test_synonym_ambiguity_returns_evidence_with_year_and_total_limit():
    result, search = run([identity_response(), identity_response(FIS, OTHER), HITS])
    assert result["ambiguous"]
    assert len(result["candidates"]) == 2
    body = search.await_args_list[-1].kwargs
    assert body["size"] == 2
    assert body["query"]["bool"]["filter"][1] == {"term": {"report_year": 2010}}


def test_broad_fallback_concat_and_explicitly_relaxes_year():
    result, search = run([identity_response(), identity_response(), HITS], year=2018)
    assert result["match_method"] == "broad_query"
    assert result["applied_filters"] == {}
    assert result["year"] is None and result["requested_year"] == 2018
    query = search.await_args_list[-1].kwargs["query"]["bool"]
    assert query["filter"] == []
    assert (
        query["must"][0]["multi_match"]["query"]
        == "Metavante stock awards purchase price"
    )


def test_no_hits_and_partial_response_are_distinct():
    result, _ = run(
        [
            identity_response(),
            identity_response(),
            {"hits": {"total": {"value": 0}, "hits": []}},
        ]
    )
    assert result["code"] == "no_results"
    with pytest.raises(RuntimeError, match="Incomplete"):
        run([{"timed_out": True}])


def test_composite_resolution_reads_all_pages():
    first = identity_response(FIS)
    first["aggregations"]["companies"]["after_key"] = {"id": "fis"}
    search = AsyncMock(side_effect=[first, identity_response(OTHER)])
    result = asyncio.run(identities(search, {"match_all": {}}))
    assert len(result) == 2
    assert search.await_args_list[1].kwargs["aggs"]["companies"]["composite"][
        "after"
    ] == {"id": "fis"}


def test_extraction_unions_orgs_by_issuer_without_modifying_context():
    docs = [FIS | {"text": "original table"}, FIS | {"text": "second table"}]
    entities = [
        SimpleNamespace(text=name, label_=label)
        for name, label in [
            ("Metavante", "ORG"),
            ("Metavante", "ORG"),
            (FIS["company_name"], "ORG"),
            ("John", "PERSON"),
        ]
    ]
    nlp = SimpleNamespace(
        pipe=lambda texts, **kwargs: iter(
            [SimpleNamespace(ents=entities), SimpleNamespace(ents=[])]
        )
    )
    enriched = enrich_documents(docs, nlp)
    assert all(doc["company_synonyms"] == ["metavante"] for doc in enriched)
    assert enriched[0]["text"] == docs[0]["text"]
    assert "company_synonyms" not in docs[0]


def test_installed_spacy_model_extracts_real_organizations():
    docs = [FIS | {"text": "Microsoft acquired GitHub in 2018."}]
    enriched = enrich_documents(docs)
    assert "microsoft" in enriched[0]["company_synonyms"]
    assert "github" in enriched[0]["company_synonyms"]


def test_index_input_cache_is_validated_and_never_runs_ner(tmp_path):
    source = tmp_path / "contexts.jsonl"
    source.write_text("corpus bytes", encoding="utf-8")
    artifact = {
        "corpus_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": MODEL,
        "model_version": MODEL_VERSION,
        "companies": {"fis": ["metavante"]},
    }
    cache = tmp_path / "organizations.json"
    cache.write_text(json.dumps(artifact), encoding="utf-8")
    docs = [FIS | {"text": "unchanged table"}]
    result = indexed_documents(tmp_path, docs, nlp=object())
    assert result[0]["company_synonyms"] == ["metavante"]
    assert result[0]["text"] == "unchanged table"
    source.write_text("different corpus", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        indexed_documents(tmp_path, docs)
