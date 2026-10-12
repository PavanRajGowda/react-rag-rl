"""Elasticsearch-only resolution and retrieval shared by both interfaces."""

from finance_rag.search import SearchError


def checked(response):
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Incomplete Elasticsearch response")
    return response


async def identities(search, query):
    # Composite pagination avoids silently dropping ambiguous issuers.
    companies, after = [], None
    while True:
        composite = {
            "size": 100,
            "sources": [{"id": {"terms": {"field": "company_id"}}}],
        }
        if after:
            composite["after"] = after
        response = checked(
            await search(
                size=0,
                query=query,
                aggs={
                    "companies": {
                        "composite": composite,
                        "aggs": {
                            "identity": {
                                "top_hits": {
                                    "size": 1,
                                    "_source": ["company_id", "company_name", "ticker"],
                                }
                            },
                            "years": {"terms": {"field": "report_year", "size": 201}},
                        },
                    }
                },
            )
        )
        result = response["aggregations"]["companies"]
        for bucket in result["buckets"]:
            years = bucket["years"]
            if years.get("sum_other_doc_count", 0):
                raise RuntimeError("Incomplete report years")
            companies.append(
                bucket["identity"]["hits"]["hits"][0]["_source"]
                | {"report_years": sorted(x["key"] for x in years["buckets"])}
            )
        after = result.get("after_key")
        if not after or not result["buckets"]:
            return companies


async def retrieve(request, search, corpus_version):
    method = "ticker" if not request.company_name else "company_name"
    if request.company_name:
        companies = await identities(
            search,
            {
                "match": {
                    "company_name": {"query": request.company_name, "operator": "and"}
                }
            },
        )
        if not companies:
            method = "company_synonyms"
            # Phrase matching prevents tokens from different array entries matching.
            companies = await identities(
                search, {"match_phrase": {"company_synonyms": request.company_name}}
            )
    else:
        companies = await identities(
            search,
            {"term": {"ticker": {"value": request.ticker, "case_insensitive": True}}},
        )
    if request.ticker:
        ticker_matches = (
            await identities(
                search,
                {
                    "term": {
                        "ticker": {"value": request.ticker, "case_insensitive": True}
                    }
                },
            )
            if request.company_name
            else companies
        )
        if not ticker_matches:
            raise SearchError("unknown_company", "Ticker is not in the corpus", 404)
        if request.company_name:
            ids = {x["company_id"] for x in ticker_matches}
            companies = [x for x in companies if x["company_id"] in ids]
            if not companies:
                raise SearchError(
                    "company_mismatch",
                    "Company name and ticker refer to different companies",
                )
    if len(companies) > 1 and method == "company_name":
        raise SearchError(
            "ambiguous_company",
            "Multiple issuers matched the company name. Choose a fuller name or ticker.",
            candidates=companies,
        )
    candidates = companies
    if method == "company_synonyms":
        candidates = companies = [
            company | {"matched_synonym": request.company_name} for company in companies
        ]
    if companies:
        companies = [x for x in companies if request.year in x["report_years"]]
        if not companies:
            raise SearchError(
                "report_year_unavailable",
                f"Matched via {method}, but requested report year {request.year} is unavailable. Choose an available report year.",
                404,
                candidates=candidates,
            )
        filters = [
            {"terms": {"company_id": [x["company_id"] for x in companies]}},
            {"term": {"report_year": request.year}},
        ]
        query = request.query
        message = "Matched report issuer and applied the requested report year."
        if method == "company_synonyms":
            message = "No issuer name matched. Matched organization names extracted into company_synonyms; these may be aliases, acquisitions, subsidiaries, partners, or other mentioned organizations. Applied the requested report year."
            if len(candidates) > 1:
                message += " Multiple issuers matched; inspect the returned contexts and refine the issuer if necessary."
    else:
        method, filters = "broad_query", []
        query = f"{request.company_name} {request.query}"
        message = "Neither issuer name nor company_synonyms matched. Searched company name plus query across all issuers and report years; the supplied company and year filters were relaxed. Check each result's actual issuer and year."
    response = checked(
        await search(
            size=request.top_k,
            track_total_hits=True,
            query={
                "bool": {
                    "filter": filters,
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text", "table_markdown"],
                                "type": "best_fields",
                            }
                        }
                    ],
                }
            },
            sort=[{"_score": "desc"}, {"context_id": "asc"}],
        )
    )
    results = [
        hit["_source"] | {"score": hit["_score"]} for hit in response["hits"]["hits"]
    ]
    return {
        "status": "ok",
        "code": "results" if results else "no_results",
        "corpus_version": corpus_version,
        "match_method": method,
        "message": message
        + (" No matching evidence; reformulate the query." if not results else ""),
        "requested_company": request.company_name or request.ticker,
        "requested_year": request.year,
        "applied_filters": {
            "company_ids": [x["company_id"] for x in companies],
            "report_year": request.year,
        }
        if filters
        else {},
        "ambiguous": len(candidates) > 1,
        "candidates": candidates,
        "resolved_company": {
            k: companies[0][k] for k in ("company_id", "company_name", "ticker")
        }
        if len(companies) == 1
        else None,
        "year": request.year if filters else None,
        "total_hits": response["hits"]["total"]["value"],
        "results": results,
    }
