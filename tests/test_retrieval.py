import asyncio
from unittest.mock import AsyncMock

import pytest

from baseline_agent.protocol import format_information
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


def test_primary_match_preserves_company_and_year():
    result, search = run([identity_response(FIS), HITS], name="Fidelity")
    assert result["match_method"] == "company_name"
    assert search.await_count == 2
    assert result["applied_filters"] == {"company_ids": ["fis"], "report_year": 2010}


def test_missing_year_does_not_relax_filters():
    with pytest.raises(SearchError) as exc:
        run([identity_response(FIS)], year=2018)
    assert exc.value.code == "report_year_unavailable"
    assert exc.value.candidates[0]["report_years"] == [2010]


def test_primary_ambiguity_returns_candidates_without_evidence():
    with pytest.raises(SearchError) as exc:
        run([identity_response(FIS, OTHER)])
    assert exc.value.code == "ambiguous_company"
    assert len(exc.value.candidates) == 2


def test_broad_fallback_concat_and_explicitly_relaxes_year():
    result, search = run([identity_response(), HITS], year=2018)
    assert search.await_count == 2
    assert result["match_method"] == "broad_query"
    assert "No report issuer matched" in format_information(result)
    assert "121.4" in format_information(result)
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
