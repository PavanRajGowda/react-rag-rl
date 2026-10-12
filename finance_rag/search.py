"""Company resolution and exact-filter BM25 search."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SEARCH_SCHEMA_VERSION = 4


def index_name(manifest):
    return f"{manifest['corpus_version']}-search-v{SEARCH_SCHEMA_VERSION}"


def company_name_query(name, company_count):
    return {
        "size": 0,
        "query": {"match": {"company_name": {"query": name, "operator": "and"}}},
        "aggs": {
            "companies": {
                "terms": {"field": "company_id", "size": max(1, company_count)}
            }
        },
    }


def company_candidates(response):
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Incomplete company-name search")
    result = response["aggregations"]["companies"]
    if result.get("sum_other_doc_count", 0) or result.get(
        "doc_count_error_upper_bound", 0
    ):
        raise RuntimeError("Incomplete company candidates")
    return {bucket["key"] for bucket in result["buckets"]}


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company_name: str | None = Field(default=None, max_length=200)
    ticker: str | None = Field(default=None, max_length=30)
    year: int = Field(strict=True, ge=1900, le=2100)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=2, strict=True, ge=1, le=10)

    @field_validator("company_name", "ticker", "query")
    @classmethod
    def nonempty(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Must not be blank")
        return value

    @model_validator(mode="after")
    def identity(self):
        if not self.company_name and not self.ticker:
            raise ValueError("company_name or ticker is required")
        return self


class SearchError(Exception):
    def __init__(self, code, message, status=422, candidates=None):
        self.code, self.message, self.status = code, message, status
        self.candidates = candidates

    def as_dict(self):
        result = {"code": self.code, "message": self.message}
        if self.candidates is not None:
            result["candidates"] = self.candidates
        return result


class Registry:
    def __init__(self, docs):
        self.tickers, self.companies, self.years = {}, {}, {}
        for doc in docs:
            key = doc["company_id"]
            self.tickers.setdefault(doc["ticker"].upper(), set()).add(key)
            self.companies[key] = {
                k: doc[k] for k in ("company_id", "company_name", "ticker")
            }
            self.years.setdefault(key, set()).add(doc["report_year"])

    def suggest(self, name, limit=5):
        """Rank nearby identities for feedback only; never resolve a suggestion."""
        legal = {
            "inc",
            "incorporated",
            "corp",
            "corporation",
            "ltd",
            "limited",
            "plc",
            "company",
            "co",
            "the",
        }

        def tokens(value):
            return set(re.findall(r"[^\W_]+", value.casefold())) - legal

        wanted = tokens(name)
        if not wanted:
            return []
        ranked = []
        for company in self.companies.values():
            available = tokens(company["company_name"])
            if not available:
                continue
            overlap = len(wanted & available) / min(len(wanted), len(available))
            similarity = max(
                SequenceMatcher(None, a, b).ratio() for a in wanted for b in available
            )
            if overlap or similarity >= 0.7:
                ranked.append((overlap, similarity, company))
        ranked.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2]["company_name"].casefold(),
                x[2]["company_id"],
            )
        )
        if any(x[0] for x in ranked):
            ranked = [x for x in ranked if x[0]]
        return [x[2] for x in ranked[:limit]]

    def resolve(self, request, es=None, index=None, name_candidates=None):
        matches = []
        if request.company_name:
            if name_candidates is None:
                if es is None or index is None:
                    raise RuntimeError("Company-name resolution requires Elasticsearch")
                name_candidates = company_candidates(
                    es.search(
                        index=index,
                        **company_name_query(request.company_name, len(self.companies)),
                    )
                )
            if not name_candidates:
                suggestions = self.suggest(request.company_name)
                raise SearchError(
                    "unknown_company",
                    "Company not found. Retry with one of these issuer names or tickers."
                    if suggestions
                    else "Company not found and no similar names were found. Try a shorter company name or check its spelling.",
                    404,
                    candidates=suggestions,
                )
            if not name_candidates <= self.companies.keys():
                raise RuntimeError("Company candidates do not match local registry")
            matches.append(name_candidates)
        if request.ticker:
            candidates = self.tickers.get(request.ticker.upper(), set())
            if not candidates:
                raise SearchError("unknown_company", "Ticker is not in the corpus", 404)
            matches.append(candidates)
        candidates = set.intersection(*matches)
        if not candidates:
            raise SearchError(
                "company_mismatch",
                "Company name and ticker refer to different companies",
            )
        if len(candidates) != 1:
            raise SearchError(
                "ambiguous_company",
                "Multiple companies matched; use a fuller company name or one of the candidate tickers",
                candidates=sorted(
                    (self.companies[key] for key in candidates),
                    key=lambda x: (
                        x["company_name"].casefold(),
                        x["ticker"],
                        x["company_id"],
                    ),
                ),
            )
        key = next(iter(candidates))
        if request.year not in self.years[key]:
            raise SearchError(
                "report_year_unavailable",
                f"Available report years: {sorted(self.years[key])}",
                404,
            )
        return self.companies[key]


def load_docs(directory):
    return [
        json.loads(line)
        for line in (Path(directory) / "contexts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def query_body(request, company):
    return {
        "size": request.top_k,
        "track_total_hits": True,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"company_id": company["company_id"]}},
                    {"term": {"report_year": request.year}},
                ],
                "must": [
                    {
                        "multi_match": {
                            "query": request.query,
                            "fields": ["text", "table_markdown"],
                            "type": "best_fields",
                        }
                    }
                ],
            }
        },
        "sort": [{"_score": "desc"}, {"context_id": "asc"}],
    }
