import pytest
from pydantic import ValidationError

from retrieval_service.search import SearchRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " "},
        {"query": "x", "company_name": "Aster"},
        {"query": "x", "year": 2023},
        {"query": "x", "top_k": True},
        {"query": "x", "top_k": 11},
        {"query": "x" * 2001},
    ],
)
def test_flat_request_rejects_invalid_or_legacy_arguments(payload):
    """Verify flat request rejects invalid or legacy arguments."""
    with pytest.raises(ValidationError):
        SearchRequest(**payload)
