"""Elasticsearch-only BM25 article passage retrieval."""

from retrieval_service.search import SEARCH_SCHEMA_VERSION, query_body


async def retrieve(request, search, corpus_version):
    """Await an injected Elasticsearch search callable using query_body(request), then
    format the response. This optional async adapter does not define or deploy an HTTP
    service.
    """
    response = await search(**query_body(request))
    return format_response(request, response, corpus_version)


def format_response(request, response, corpus_version):
    """Validate an Elasticsearch hit response and return the versioned public result
    dictionary. Includes query, hit count, scored document records and corpus/schema
    identity. Timeouts, failed shards or query errors raise instead of being treated as
    valid empty retrieval.
    """
    if response.get("error"):
        raise RuntimeError("Elasticsearch query failed")
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Incomplete Elasticsearch response")
    results = [
        hit["_source"] | {"score": hit["_score"]} for hit in response["hits"]["hits"]
    ]
    return {
        "status": "ok",
        "code": "results" if results else "no_results",
        "corpus_version": corpus_version,
        "search_schema_version": SEARCH_SCHEMA_VERSION,
        "query": request.query,
        # Exact counting is disabled for speed; missing totals are unknown, not 0.
        "total_hits": response["hits"].get("total", {}).get("value"),
        "message": "BM25 search across article titles and passage text."
        if results
        else "No matching passages. Reformulate the search.",
        "results": results,
    }
