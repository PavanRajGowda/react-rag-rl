import asyncio
from unittest.mock import AsyncMock

import pytest

from retrieval_service.retrieval import retrieve
from retrieval_service.search import SearchRequest
from tests.helpers import passage


def test_bm25_titles_text_and_deterministic_tie_breaking():
    """Verify bm25 titles text and deterministic tie breaking."""
    search = AsyncMock(
        return_value={
            "hits": {
                "total": {"value": 1},
                "hits": [{"_source": passage(), "_score": 2.0}],
            }
        }
    )
    r = asyncio.run(retrieve(SearchRequest(query="Aster launch"), search, "v"))
    assert r["results"][0]["document_id"] == "a" and r["search_schema_version"] == 6
    body = search.call_args.kwargs
    assert body["query"]["multi_match"]["fields"] == ["title", "text"]
    assert "filter" not in str(body) and body["sort"][-1] == {"document_id": "asc"}


@pytest.mark.parametrize("response", [{"timed_out": True}, {"_shards": {"failed": 1}}])
def test_partial_results_are_infrastructure_errors(response):
    """Verify partial results are infrastructure errors."""
    with pytest.raises(RuntimeError, match="Incomplete"):
        asyncio.run(
            retrieve(SearchRequest(query="x"), AsyncMock(return_value=response), "v")
        )


def test_empty_results_are_valid_feedback():
    """Verify empty results are valid feedback."""
    r = asyncio.run(
        retrieve(
            SearchRequest(query="x"),
            AsyncMock(return_value={"hits": {"total": {"value": 0}, "hits": []}}),
            "v",
        )
    )
    assert r["code"] == "no_results" and r["results"] == []
