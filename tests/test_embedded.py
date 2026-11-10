from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from retrieval_service.embedded import EmbeddedSearch


def test_private_api_validation_and_transport_failure(tmp_path):
    """Verify private api validation and transport failure."""
    engine = EmbeddedSearch(tmp_path)
    engine.manifest = {"corpus_version": "v"}
    engine.es = SimpleNamespace(search=Mock(return_value={"timed_out": True}))
    assert engine.search({"query": "x", "year": 2023})["code"] == "invalid_request"
    with pytest.raises(RuntimeError, match="Incomplete"):
        engine.search({"query": "Aster"})
    assert engine.es.search.call_args.kwargs["request_cache"] is True
