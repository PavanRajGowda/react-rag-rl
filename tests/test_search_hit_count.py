"""Top-K retrieval does not require counting all matching documents."""

from retrieval_service.retrieval import format_response
from retrieval_service.search import SearchRequest, query_body


def test_disabled_exact_count_preserves_hits_and_unknown_total():
    """Keep ranked documents and represent an uncounted total as None, not zero."""
    request = SearchRequest(query="example", top_k=2)
    assert query_body(request)["track_total_hits"] is False
    hit = {
        "_source": {"document_id": "wiki-1", "title": "Example", "text": "Text"},
        "_score": 3.0,
    }
    result = format_response(request, {"hits": {"hits": [hit]}}, "test")
    assert result["total_hits"] is None
    assert result["results"][0]["document_id"] == "wiki-1"
    assert result["results"][0]["score"] == 3.0
