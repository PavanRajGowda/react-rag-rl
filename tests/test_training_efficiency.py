"""No hosted calls: pool reuse, checkpoint cadence and reference filtering."""

import json
from types import SimpleNamespace

import pytest

from tests.test_grpo_advantages import label, make_group
from tests.test_tinker_migration import TYPES, Client, Future, Service, search
from tinker_training.config import TrainingConfig
from tinker_training.runner import run


def harness(monkeypatch, cfg, *, tied_second=False):
    groups = tuple(
        make_group(
            cfg,
            number=i,
            answers=("42", "42") if i == 1 and tied_second else ("42", "9"),
        )
        for i in range(2)
    )
    train = SimpleNamespace(
        questions=tuple(g.episodes[0].question for g in groups),
        labels=tuple(label(g) for g in groups),
    )
    counts = {"collect": 0, "prepare": 0, "export": 0, "scored": 0, "saves": []}

    def collect(*args, **kwargs):
        counts["collect"] += 1
        return groups

    monkeypatch.setattr("tinker_training.runner.collect_groups", collect)
    from tinker_training.advantages import prepare_update

    def prepare(*args, **kwargs):
        counts["prepare"] += 1
        return prepare_update(*args, **kwargs)

    monkeypatch.setattr("tinker_training.runner.prepare_update", prepare)
    export, save = Client.save_weights_and_get_sampling_client, Client.save_state

    def exporting(self):
        counts["export"] += 1
        return export(self)

    def saving(self, name):
        counts["saves"].append(name)
        return save(self, name)

    monkeypatch.setattr(Client, "save_weights_and_get_sampling_client", exporting)
    monkeypatch.setattr(Client, "save_state", saving)
    service = Service()

    def reference(base_model):
        assert base_model == cfg.model_id

        def score(prompt, prompt_logprobs_last_n):
            counts["scored"] += 1
            n = prompt_logprobs_last_n
            return Future([None] * (len(prompt) - n) + [-2.0] * n)

        return SimpleNamespace(compute_logprobs=score)

    service.create_sampling_client = reference
    return service, train, counts


def test_pool_cached_and_sampler_exported_only_for_new_pool(tmp_path, monkeypatch):
    cfg = TrainingConfig(
        num_generations=2,
        questions_per_update=2,
        minibatch_questions=1,
        max_steps=7,
        checkpoint_interval=5,
    )
    service, train, counts = harness(monkeypatch, cfg)
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"case": "cadence"},
    )
    assert state["step"] == 7
    assert counts["collect"] == counts["prepare"] == counts["export"] == 4
    assert counts["saves"] == ["initial", "step-00000005", "step-00000007"]
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "observability/validation").exists()
    batches = list((tmp_path / "observability/rollouts").glob("*.json"))
    assert len(batches) == 4
    for path in batches:
        row = json.loads(path.read_text())
        assert "episodes" not in row and "groups" not in row


def test_reference_scoring_skips_tied_groups_and_reuses_pool(tmp_path, monkeypatch):
    cfg = TrainingConfig(
        beta=0.01,
        num_generations=2,
        questions_per_update=2,
        minibatch_questions=1,
        num_iterations=2,
        max_steps=4,
    )
    service, train, counts = harness(monkeypatch, cfg, tied_second=True)
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"case": "kl"},
    )
    assert state["step"] == 2 and state["skipped_minibatches"] == 2
    assert counts["scored"] == 3  # Only informative group: fixture has two turns + one.
    assert counts["collect"] == counts["export"] == counts["prepare"] == 1
    assert state["metrics"]["reference_kl"]["centering_population"].startswith(
        "surviving"
    )
    assert not (tmp_path / "raw").exists()


def test_resume_rolls_back_metrics_and_discards_midpool(tmp_path, monkeypatch):
    cfg = TrainingConfig(
        num_generations=2,
        questions_per_update=2,
        minibatch_questions=1,
        max_steps=7,
        checkpoint_interval=5,
    )
    service, train, counts = harness(monkeypatch, cfg)
    service.fail_save_at = 7
    with pytest.raises(RuntimeError, match="interrupted checkpoint"):
        run(
            cfg,
            train,
            train,
            tmp_path,
            service,
            search,
            types=TYPES,
            identity={"case": "resume"},
        )
    pointer = json.loads((tmp_path / "progress.json").read_text())
    assert (
        pointer["step"] == 5
        and pointer["iteration"] == 0
        and pointer["pending"] is None
    )
    assert pointer["discarded_pool_on_resume"]
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"case": "resume"},
    )
    assert state["step"] == 7 and service.restored == [pointer["checkpoint"]]
    assert counts["collect"] == 5  # Four attempted pools plus one fresh resume pool.
    events = [
        json.loads(s) for s in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["kind"] == "resume_rollback" for e in events)
    index = json.loads((tmp_path / "observability/index.json").read_text())
    assert index["committed_step"] == 7
