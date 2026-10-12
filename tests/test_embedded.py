import json
from types import SimpleNamespace

import pytest

from finance_rag.embedded import EmbeddedSearch
from finance_rag.prepare import prepare
from finance_rag.search import Registry
from tests.helpers import identity_response, sample


def test_private_modal_argument_errors_and_partial_failure(tmp_path):
    from unittest.mock import Mock

    manifest = prepare({"train": [sample()]}, "commit", tmp_path)
    doc = json.loads((tmp_path / "contexts.jsonl").read_text())
    engine = EmbeddedSearch(tmp_path)
    engine.manifest = manifest
    engine.registry = Registry([doc])
    engine.es = SimpleNamespace(search=Mock(return_value={"timed_out": True}))
    assert (
        engine.search({"ticker": "AAPL", "query": "sales"})["code"] == "invalid_request"
    )
    engine.es.search.return_value = identity_response()
    assert (
        engine.search({"ticker": "ZZZ", "year": 2002, "query": "sales"})["code"]
        == "unknown_company"
    )
    engine.es.search.return_value = {"timed_out": True}
    with pytest.raises(RuntimeError, match="Incomplete"):
        engine.search({"ticker": "AAPL", "year": 2002, "query": "sales"})
