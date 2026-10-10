import hashlib
import json

import pytest

from finance_rag.prepare import prepare
from tests.helpers import sample


def test_preservation_dedup_and_label_separation(tmp_path):
    row = sample()
    manifest = prepare({"train": [row, row], "test": [row]}, "commit", tmp_path)
    assert manifest["unique_contexts"] == 1
    assert (
        hashlib.sha256((tmp_path / "contexts.jsonl").read_bytes()).hexdigest()
        == manifest["sha256"]
    )
    assert manifest["context_split_overlap"] == {"test+train": 1}
    doc = json.loads((tmp_path / "contexts.jsonl").read_text())
    assert doc["text"] == row["context"]
    assert doc["table_markdown"] == row["table"]
    assert not {"question", "program_answer", "original_answer", "split"} & doc.keys()


def test_conflicting_context_fails(tmp_path):
    row = sample()
    with pytest.raises(ValueError, match="Conflicting context"):
        prepare({"train": [row, row | {"table": "different"}]}, "commit", tmp_path)
