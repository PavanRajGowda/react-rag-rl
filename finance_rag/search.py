"""Company resolution and exact-filter BM25 search."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company_name: str | None = Field(default=None, max_length=200)
    ticker: str | None = Field(default=None, max_length=30)
    year: int = Field(strict=True, ge=1900, le=2100)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, strict=True, ge=1, le=10)

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
    def __init__(self, code, message, status=422):
        self.code, self.message, self.status = code, message, status


class Registry:
    def __init__(self, docs):
        self.names, self.tickers, self.companies, self.years = {}, {}, {}, {}
        for doc in docs:
            key = doc["company_id"]
            self.names.setdefault(normalize(doc["company_name"]), set()).add(key)
            self.tickers.setdefault(doc["ticker"].upper(), set()).add(key)
            self.companies[key] = {
                k: doc[k] for k in ("company_id", "company_name", "ticker")
            }
            self.years.setdefault(key, set()).add(doc["report_year"])

    def resolve(self, request):
        matches = []
        for value, mapping in (
            (
                normalize(request.company_name) if request.company_name else None,
                self.names,
            ),
            (request.ticker.upper() if request.ticker else None, self.tickers),
        ):
            if value is not None:
                candidates = mapping.get(value, set())
                if not candidates:
                    raise SearchError(
                        "unknown_company",
                        "Company name or ticker is not in the corpus",
                        404,
                    )
                matches.append(candidates)
        candidates = set.intersection(*matches)
        if not candidates:
            raise SearchError(
                "company_mismatch",
                "Company name and ticker refer to different companies",
            )
        if len(candidates) != 1:
            raise SearchError(
                "ambiguous_company", "Company identifier matches multiple companies"
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
