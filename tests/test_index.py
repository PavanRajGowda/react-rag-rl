import json
from unittest.mock import MagicMock, patch

from retrieval_service.dataset import file_sha256
from retrieval_service.index import index_corpus
from retrieval_service.search import index_name


def test_indexing_preserves_metadata_and_authenticated_client(tmp_path):
    """Verify indexing preserves metadata and authenticated client."""
    document = {"document_id":"wiki-1", "title":"Example", "text":"A fact.", "url":"https://example.org"}
    (tmp_path / "contexts.jsonl").write_text(json.dumps(document)+"\n",encoding="utf-8")
    m = {"corpus_version":"test", "unique_contexts":1,
         "sha256":file_sha256(tmp_path / "contexts.jsonl")}
    (tmp_path / "manifest.json").write_text(json.dumps(m),encoding="utf-8")
    es = MagicMock()
    es.indices.exists.return_value = False
    es.count.return_value = {"count": m["unique_contexts"]}
    with patch("retrieval_service.index.helpers.bulk") as bulk:
        result = index_corpus(tmp_path, es)
        docs = list(bulk.call_args.args[1])
    assert result == {"index": index_name(m), "documents": m["unique_contexts"]}
    mapping = es.indices.create.call_args.kwargs["mappings"]
    assert mapping["_meta"]["search_schema_version"] == 6
    assert mapping["properties"]["title"]["type"] == "text"
    assert mapping["properties"]["document_id"]["type"] == "keyword"
    assert mapping["properties"]["published_at"]["type"] == "date"
    assert docs[0]["_source"] == json.loads(
        (tmp_path / "contexts.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert es.indices.put_settings.call_args.kwargs["settings"] == {"index.blocks.write": True}
    es.close.assert_not_called()
