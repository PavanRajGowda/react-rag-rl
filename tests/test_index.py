import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from finance_rag.index import index_corpus
from finance_rag.prepare import prepare
from finance_rag.search import index_name
from tests.helpers import sample


def test_indexing_keeps_injected_authenticated_client_open(tmp_path):
    manifest = prepare({"train": [sample()]}, "commit", tmp_path)
    es = MagicMock()
    es.indices.exists.return_value = False
    es.count.return_value = {"count": 1}
    with patch("finance_rag.index.helpers.bulk") as bulk:
        nlp = SimpleNamespace(
            pipe=lambda texts, **kwargs: (SimpleNamespace(ents=[]) for _ in texts)
        )
        result = index_corpus(tmp_path, es=es, nlp=nlp)
        indexed = list(bulk.call_args.args[1])
    assert result == {"index": index_name(manifest), "documents": 1}
    assert es.indices.create.call_args.kwargs["mappings"]["properties"][
        "company_name"
    ] == {
        "type": "text",
        "analyzer": "standard",
        "fields": {"keyword": {"type": "keyword"}},
    }
    assert indexed[0]["_source"] == json.loads(
        (tmp_path / "contexts.jsonl").read_text()
    ) | {"company_synonyms": []}
    mapping = es.indices.create.call_args.kwargs["mappings"]
    assert mapping["properties"]["company_synonyms"]["type"] == "text"
    assert mapping["_meta"]["organization_model_version"] == "3.8.0"
    assert mapping["_meta"]["search_schema_version"] == 3
    es.close.assert_not_called()
