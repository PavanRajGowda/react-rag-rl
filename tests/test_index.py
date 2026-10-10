import json
from unittest.mock import MagicMock, patch

from finance_rag.index import index_corpus
from finance_rag.prepare import prepare
from tests.helpers import sample


def test_indexing_keeps_injected_authenticated_client_open(tmp_path):
    manifest = prepare({"train": [sample()]}, "commit", tmp_path)
    es = MagicMock()
    es.indices.exists.return_value = False
    es.count.return_value = {"count": 1}
    with patch("finance_rag.index.helpers.bulk") as bulk:
        result = index_corpus(tmp_path, es=es)
        indexed = list(bulk.call_args.args[1])
    assert result == {"index": manifest["corpus_version"], "documents": 1}
    assert indexed[0]["_source"] == json.loads(
        (tmp_path / "contexts.jsonl").read_text()
    )
    es.close.assert_not_called()
