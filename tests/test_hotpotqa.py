import bz2
import io
import json
import tarfile

import pytest

from baseline_agent.data import verify_prepared_files
from baseline_agent.evidence import EvidenceStore
from retrieval_service.dataset import REVISION, file_sha256
from retrieval_service.hotpotqa import paragraph, prepare, verify_archive, wiki_rows
from tinker_training.config import TrainingConfig
from tinker_training.data import load_local_splits
from tinker_training.search_contract import checked_response


def wiki(title="Alpha", identity="1"):
    """Construct a source-format Wikipedia abstract row for preparation tests."""
    return {
        "id": identity,
        "title": title,
        "url": "https://en.wikipedia.org/wiki?curid=" + identity,
        "text": ["Alpha was founded in 1900.", " It is a company."],
    }


def question(identity):
    """Construct an official-layout HotpotQA question with supporting title/sentence
    annotations.
    """
    return {
        "id": identity,
        "question": "When was Alpha founded?",
        "answer": "1900",
        "type": "bridge",
        "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
        "context": {"title": ["Alpha"], "sentences": [wiki()["text"]]},
    }


@pytest.fixture
def prepared(tmp_path):
    """Write a tiny nested Wikipedia archive and convert it with fixture questions."""
    archive = tmp_path / "mini.tar.bz2"
    raw = bz2.compress((json.dumps(wiki()) + "\n").encode())
    with tarfile.open(archive, "w:bz2") as tar:
        member = tarfile.TarInfo("abstracts/AA/wiki_00.bz2")
        member.size = len(raw)
        tar.addfile(member, io.BytesIO(raw))
    source = [
        ("train", question("train-one")),
        ("dev", question("dev-one")),
        ("test", {"id": "test-one", "question": "Hidden question?"}),
    ]
    output = tmp_path / "prepared"
    manifest = prepare(archive, source, output, verify=False)
    return archive, output, manifest


def test_stream_nested_bz2_and_preserve_official_splits(prepared):
    """Verify stream nested bz2 and preserve official splits."""
    archive, root, manifest = prepared
    assert list(wiki_rows(archive)) == [wiki()]
    assert manifest["split_questions"] == {"train": 1, "dev": 1, "test": 1}
    verify_prepared_files(root, manifest)
    labels = [
        json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in labels} == {"train", "dev"}
    assert all("split_group_id" not in row for row in labels)
    assignments = json.loads((root / "split_assignments.json").read_text())[
        "assignments"
    ]
    assert all(set(row) == {"id", "split"} for row in assignments)
    assert labels[0]["evidence"][0]["sentence_mapping_valid"] is True
    assert labels[0]["evidence"][0]["sentence_id"] == 0
    test = json.loads((root / "test_questions.jsonl").read_text())
    assert "answer" not in test and "evidence" not in test
    assert "split_group_id" not in test
    docs = (root / "contexts.jsonl").read_text()
    assert "supporting_facts" not in docs and "question" not in docs
    assert file_sha256(root / "contexts.jsonl") == manifest["sha256"]
    train, dev, _ = load_local_splits(root, TrainingConfig())
    assert train.questions[0].dataset_revision == REVISION
    assert dev.labels[0].gold_document_ids == ("wiki-1",)
    assert dev.labels[0].gold_passage_ids == ()


def test_wrong_archive_rejected_before_preparing(prepared):
    """Verify wrong archive rejected before preparing."""
    archive, _, _ = prepared
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(archive)


def test_wikipedia_response_needs_no_news_dates():
    """Verify wikipedia response needs no news dates."""
    config = TrainingConfig()
    response = {
        "status": "ok",
        "corpus_version": config.corpus_version,
        "search_schema_version": config.search_schema_version,
        "results": [paragraph(wiki())],
    }
    checked, rendered = checked_response(response, config)
    assert "Alpha" in rendered and "published=" not in rendered
    memory = EvidenceStore(12, 12000)
    memory.update(checked)
    assert memory.visible()[0]["document_id"] == "wiki-1"
    assert "context_id" not in memory.visible()[0]


def test_huggingface_loader_preserves_pinned_official_splits(monkeypatch, tmp_path):
    """Verify huggingface loader preserves pinned official splits."""
    import datasets

    from retrieval_service.dataset import CONFIG, DATASET
    from retrieval_service.hotpotqa import source_questions

    def load(name, config, **kwargs):
        """Return fake official Hugging Face splits while checking requested dataset
        revision/configuration.
        """
        assert (name, config) == (DATASET, CONFIG)
        assert kwargs == {"revision": REVISION, "cache_dir": str(tmp_path)}
        return {
            "train": [{"id": "train-one"}],
            "validation": [{"id": "dev-one"}],
            "test": [{"id": "test-one"}],
        }

    monkeypatch.setattr(datasets, "load_dataset", load)
    assert list(source_questions(tmp_path)) == [
        ("train", {"id": "train-one"}),
        ("dev", {"id": "dev-one"}),
        ("test", {"id": "test-one"}),
    ]


def test_hidden_test_split_assignment_is_verified(prepared):
    """Verify hidden test split assignment is verified."""
    _, root, manifest = prepared
    path = root / "test_questions.jsonl"
    row = json.loads(path.read_text())
    row["split"] = "dev"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest["test_questions_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="test questions disagree"):
        verify_prepared_files(root, manifest)
