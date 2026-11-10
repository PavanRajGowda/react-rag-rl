from unittest.mock import MagicMock

import pytest

from retrieval_service.persistence import (
    check_marker,
    publish_repository,
    restore_index,
)


def test_only_complete_snapshot_is_published_and_is_immutable(tmp_path):
    """Verify only complete snapshot is published and is immutable."""
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "index-0").write_text("completed snapshot")
    manifest = {"corpus_version": "test", "sha256": "corpus-hash", "unique_contexts": 2}
    target = tmp_path / "saved"
    marker = publish_repository(repo, target, manifest)
    assert check_marker(target, manifest) == marker
    assert marker["method"] == "elasticsearch-snapshot-api"
    assert (target / "repository.tar").is_file()
    with pytest.raises(FileExistsError):
        publish_repository(repo, target, manifest)
    with pytest.raises(ValueError, match="identity"):
        check_marker(target, manifest | {"sha256": "wrong-corpus"})


def test_corrupted_snapshot_never_reaches_elasticsearch(tmp_path):
    """Verify corrupted snapshot never reaches elasticsearch."""
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "index-0").write_text("snapshot")
    manifest = {"corpus_version": "test", "sha256": "corpus-hash", "unique_contexts": 2}
    target = tmp_path / "saved"
    publish_repository(repo, target, manifest)
    (target / "repository.tar").write_bytes(b"corrupted archive")
    es = MagicMock()
    with pytest.raises(ValueError, match="checksum"):
        restore_index(es, tmp_path / "runtime", target, manifest)
    es.snapshot.create_repository.assert_not_called()
