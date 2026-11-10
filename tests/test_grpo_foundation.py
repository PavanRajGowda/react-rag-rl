"""CPU-only tests for Commit 1; no trainer, reward, model or tool execution."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from baseline_agent.prompt import messages
from tinker_training.config import TrainingConfig
from tinker_training.data import (
    load_local_splits,
    load_pinned_split,
    policy_messages,
    prepare_rows,
)
from tinker_training.provenance import build_preflight, write_preflight


def rows(split="train"):
    """Return source-like train/dev question rows with controlled private labels."""
    return [
        {
            "id": f"multihop_{split}_0",
            "split": split,
            "question": "  What changed?  ",
            "answer": "GOLD_SENTINEL",
            "gold_document_ids": ["REFERENCE_SENTINEL"],
            "document_id": "REFERENCE_SENTINEL",
            "title": "Test article",
            "source": "Example",
            "published_at": "2023-10-01T00:00:00+00:00",
            "company_name": "ISSUER_SENTINEL",
            "report_year": 2099,
            "context": "EVIDENCE_SENTINEL",
            "program": "PROGRAM_SENTINEL",
            "prompt": "INJECTED_PROMPT_SENTINEL",
            "system": "INJECTED_SYSTEM_SENTINEL",
        }
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_generations": 1},
        {"num_generations": True},
        {"questions_per_update": 0},
        {"max_context_tokens": 1024},
        {"learning_rate": float("nan")},
        {"beta": float("inf")},
        {"temperature": 0},
        {"warmup_ratio": 1},
        {"top_p": 1.1},
        {"seed": -1},
        {"seed": True},
        {"max_turns": 21},
        {"context_mode": "unknown"},
        {"model_revision": "main"},
        {"dataset_revision": "latest"},
        {"loss_type": []},
        {"reward": "format"},
    ],
)
def test_invalid_configuration_fails_before_execution(overrides):
    """Verify invalid configuration fails before execution."""
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(overrides)


def test_config_roundtrip_and_logical_batch_units():
    """Verify config roundtrip and logical batch units."""
    config = TrainingConfig()
    assert config.episodes_per_update == 32
    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert (
        replace(config, num_generations=4, questions_per_update=2).episodes_per_update
        == 8
    )
    with pytest.raises(ValueError, match="Unknown"):
        TrainingConfig.from_dict({"learning_ratte": 1e-5})


def test_labels_and_raw_metadata_cannot_enter_policy_messages():
    """Verify labels and raw metadata cannot enter policy messages."""
    prepared = prepare_rows(rows(), "train", TrainingConfig())
    assert policy_messages(prepared.questions[0]) == messages("What changed?")
    payload = json.dumps(policy_messages(prepared.questions[0]))
    for sentinel in (
        "GOLD_SENTINEL",
        "REFERENCE_SENTINEL",
        "ISSUER_SENTINEL",
        "EVIDENCE_SENTINEL",
        "PROGRAM_SENTINEL",
        "INJECTED_PROMPT_SENTINEL",
        "INJECTED_SYSTEM_SENTINEL",
    ):
        assert sentinel not in payload
    assert prepared.labels[0].gold_answer == "GOLD_SENTINEL"
    assert prepared.labels[0].gold_document_ids == ("REFERENCE_SENTINEL",)
    for forbidden in (rows()[0], prepared.labels[0]):
        with pytest.raises(TypeError):
            policy_messages(forbidden)


def test_pinned_loader_rejects_test_before_accessing_dataset(monkeypatch):
    """Verify pinned loader rejects test before accessing dataset."""
    with pytest.raises(ValueError, match="held out"):
        load_pinned_split("test", TrainingConfig())
    # Split loading is local and uses pinned prepared assignments, not HF's single split.
    result = load_pinned_split("dev", TrainingConfig(), count=1)
    assert result.questions[0].split == "dev"


def test_blank_selection_and_invalid_rows_are_explicit():
    """Verify blank selection and invalid rows are explicit."""
    config = TrainingConfig()
    items = rows() + [rows()[0] | {"id": "multihop_train_1", "question": None}]
    result = prepare_rows(items, "train", config)
    assert result.selected_source_count == 2 and len(result.questions) == 1
    assert result.skipped_blank_ids == ("multihop_train_1",)
    for invalid in (
        {"id": ""},
        {"split": "test"},
        {"question": 123},
        {"answer": None},
        {"answer": float("nan")},
        {"gold_document_ids": ""},
    ):
        with pytest.raises((ValueError, TypeError)):
            prepare_rows([rows()[0] | invalid], "train", config)
    with pytest.raises(ValueError, match="no nonblank"):
        prepare_rows([rows()[0] | {"question": " "}], "train", config)
    with pytest.raises(ValueError):
        prepare_rows(rows(), "train", config, count=2)


def local_corpus(tmp_path, config):
    """Write a small prepared corpus manifest/label fixture for provenance and split
    validation.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": config.dataset_id,
                "revision": config.dataset_revision,
                "corpus_version": config.corpus_version,
                "split_questions": {"train": 1, "dev": 1},
            }
        )
    )
    items = (
        rows() + rows("dev") + [rows()[0] | {"id": "multihop_test_0", "split": "test"}]
    )
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in items))


def test_local_loader_manifest_and_provenance(tmp_path):
    """Verify local loader manifest and provenance."""
    config = TrainingConfig()
    local_corpus(tmp_path, config)
    train, dev, hashes = load_local_splits(tmp_path, config)
    first = build_preflight(config, train, dev, hashes)
    assert first["execution_status"] == "not_started"
    assert first["train"]["question_ids"] == ["multihop_train_0"]
    assert first["dev"]["question_ids"] == ["multihop_dev_0"]
    assert "GOLD_SENTINEL" not in json.dumps(first)
    assert (
        build_preflight(config, train, dev, hashes)["experiment_id"]
        == first["experiment_id"]
    )
    changed_gold = replace(
        train, labels=(replace(train.labels[0], gold_answer="DIFFERENT"),)
    )
    second = build_preflight(config, changed_gold, dev, hashes)
    assert second["experiment_id"] != first["experiment_id"]
    assert (
        second["train"]["policy_inputs_sha256"]
        == first["train"]["policy_inputs_sha256"]
    )
    assert (
        second["train"]["reward_labels_sha256"]
        != first["train"]["reward_labels_sha256"]
    )
    assert (
        build_preflight(replace(config, seed=43), train, dev, hashes)["experiment_id"]
        != first["experiment_id"]
    )
    path = write_preflight(first, tmp_path / "preflight")
    assert json.loads(path.read_text())["experiment_id"] == first["experiment_id"]
    with pytest.raises(FileExistsError):
        write_preflight(first, tmp_path / "preflight")
    with pytest.raises(ValueError):
        build_preflight(config, dev, train)
    with pytest.raises(ValueError):
        load_local_splits(tmp_path, replace(config, dataset_revision="a" * 40))


def test_foundation_imports_do_not_load_training_or_cloud_libraries():
    """Verify foundation imports do not load training or cloud libraries."""
    code = "import sys; import tinker_training.records; import tinker_training.config; assert not {'torch', 'trl', 'vllm', 'modal', 'transformers', 'datasets'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
