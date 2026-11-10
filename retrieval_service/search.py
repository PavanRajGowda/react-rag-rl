"""Flat Wikipedia query contract and BM25 query construction."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

SEARCH_SCHEMA_VERSION = 6


def index_name(manifest):
    """Derive the versioned Elasticsearch index name from corpus identity and search
    schema.
    """
    return f"{manifest['corpus_version']}-search-v{SEARCH_SCHEMA_VERSION}"


class SearchRequest(BaseModel):
    """Validated flat query and retrieval top-k contract. Forbids unknown fields; no
    company/year filters or nested query protocol is required.
    """
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=2, strict=True, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def nonempty(cls, value):
        """Strip surrounding query whitespace and reject an empty result during Pydantic
        validation.
        """
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank")
        return value


class SearchError(Exception):
    """Structured retrieval-domain exception with public code/message and status metadata."""
    def __init__(self, code, message, status=422):
        """Create a structured search-domain error with a machine-readable code, message
        and suggested HTTP-style status. Does not send any response itself.
        """
        self.code, self.message, self.status = code, message, status

    def as_dict(self):
        """Return the public error code/message dictionary for a SearchError."""
        return {"code": self.code, "message": self.message}


def iter_docs(directory):
    """Yield decoded document records from contexts.jsonl with bounded memory. Raises on
    malformed JSON; it does not load the full Wikipedia corpus at once.
    """
    with (Path(directory) / "contexts.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_docs(directory):
    """Return all streamed documents as a list for small fixtures/inspection. Production
    fullwiki indexing uses iter_docs to avoid this memory cost.
    """
    return list(iter_docs(directory))


def query_body(request):
    """Build a BM25 multi_match query across title and text with request.top_k results.
    Uses descending relevance and document_id tie-breaking for deterministic equal-score
    order. This is lexical retrieval; there is no embedding-model tokenization here.
    """
    return {
        "size": request.top_k,
        # The agent consumes top-K evidence, not the exact number of all matches.
        "track_total_hits": False,
        "query": {
            "multi_match": {
                "query": request.query,
                "fields": ["title", "text"],
                "type": "best_fields",
            }
        },
        "sort": [{"_score": "desc"}, {"document_id": "asc"}],
    }
