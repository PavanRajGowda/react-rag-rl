"""Real CPU Torch/PEFT checkpoint tests; optional in the dependency-light root env."""

import json
import random
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

from grpo_training.advantages import prepare_update
from grpo_training.checkpoints import Progress, RunStore, restore_rng
from grpo_training.config import TrainingConfig
from grpo_training.model import TrainingModel, export_adapter
from grpo_training.runner import optimizer_step, run_identity
from grpo_training.trainer import CapturedUpdate
from tests.test_grpo_advantages import label, make_group
from tests.test_grpo_runner import inputs

TARGETS = (
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.0.self_attn.q_proj",
)


class TinyBase(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=8)
        self.config.save_pretrained(path)
        self.name_or_path = str(path)
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(8, 8, bias=False)
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([layer])

    def forward(self, x):
        layer = self.model.language_model.layers[0]
        return layer.mlp.down_proj(layer.self_attn.q_proj(x))


def make_training(tmp_path, config):
    torch.manual_seed(7)
    base = TinyBase(tmp_path / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            target_modules=list(TARGETS),
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
        ),
    )
    bundle = TrainingModel(
        model, NS(chat_template="checkpoint-test"), tmp_path, config, TARGETS
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1 / (step + 1)
    )
    trainer = NS(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        state=NS(global_step=0),
        accelerator=NS(
            clip_grad_norm_=torch.nn.utils.clip_grad_norm_,
            optimizer_step_was_skipped=False,
        ),
    )
    return trainer, bundle


@pytest.fixture
def setup(tmp_path):
    config = TrainingConfig(num_generations=2, questions_per_update=1, num_iterations=2)
    trainer, bundle = make_training(tmp_path, config)
    commits = []
    store = RunStore(
        tmp_path / "runs",
        "test",
        run_identity(config, inputs(config)),
        durable=lambda: commits.append(1),
    )
    artifact = export_adapter(bundle, store.root / "transfers", 0)
    return trainer, bundle, store, artifact, commits


def update(trainer, config):
    trainer.model(torch.ones(1, 8)).square().sum().backward()
    return optimizer_step(trainer, config)


def test_real_adapter_optimizer_scheduler_and_rng_roundtrip(setup):
    trainer, bundle, store, _, commits = setup
    update(trainer, bundle.config)
    artifact = export_adapter(bundle, store.root / "transfers", 1)
    original = [p.detach().clone() for p in trainer.model.parameters()]
    original_lr = trainer.optimizer.param_groups[0]["lr"]
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    store.save(trainer, artifact, Progress(1, 1, 1))
    expected = (random.random(), np.random.random(), torch.rand(3))
    update(trainer, bundle.config)
    progress, pending, rng = store.restore(
        trainer, bundle, inputs(bundle.config).labels
    )
    assert progress == Progress(1, 1, 1) and pending is None
    assert (
        trainer.state.global_step == 1
        and trainer.optimizer.param_groups[0]["lr"] == original_lr
    )
    for parameter, before in zip(trainer.model.parameters(), original, strict=True):
        torch.testing.assert_close(parameter, before)
    restore_rng(rng)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2])
    assert len(commits) == 2


def test_pending_rollout_reuse_survives_checkpoint(setup):
    trainer, bundle, store, artifact, _ = setup
    group = make_group(bundle.config, policy=artifact.policy)
    prepared = prepare_update([group], [label(group)])
    vectors = tuple((-2.0,) * len(r["target_ids"]) for r in prepared.rows())
    pending = CapturedUpdate(prepared, vectors, tuple((1.0,) * len(v) for v in vectors))
    update(trainer, bundle.config)
    artifact = export_adapter(bundle, store.root / "transfers", 1)
    store.save(trainer, artifact, Progress(1, 1, 1, 1), pending)
    progress, restored, _ = store.restore(trainer, bundle, inputs(bundle.config).labels)
    assert progress.iteration == 1 and restored == pending


def test_resume_matches_uninterrupted_next_optimizer_update(setup):
    trainer, bundle, store, _, _ = setup
    update(trainer, bundle.config)
    artifact = export_adapter(bundle, store.root / "transfers", 1)
    store.save(trainer, artifact, Progress(1, 1, 1))
    update(trainer, bundle.config)
    expected = [p.detach().clone() for p in trainer.model.parameters()]
    expected_lr = trainer.optimizer.param_groups[0]["lr"]
    store.restore(trainer, bundle, inputs(bundle.config).labels)
    update(trainer, bundle.config)
    for parameter, before in zip(trainer.model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert trainer.optimizer.param_groups[0]["lr"] == expected_lr


@pytest.mark.parametrize(
    "file", ["training.pt", "state.json", "adapter/adapter_model.safetensors"]
)
def test_corrupt_checkpoint_is_rejected_before_loading(setup, file):
    trainer, _, store, artifact, _ = setup
    directory = store.save(trainer, artifact, Progress())
    (directory / file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity failure"):
        store.latest()


def test_incomplete_checkpoint_never_replaces_latest(setup, monkeypatch):
    import grpo_training.checkpoints as module

    trainer, bundle, store, artifact, _ = setup
    complete = store.save(trainer, artifact, Progress())
    update(trainer, bundle.config)
    artifact = export_adapter(bundle, store.root / "transfers", 1)

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(
        module.torch if hasattr(module, "torch") else torch, "save", fail
    )
    with pytest.raises(OSError, match="disk failure"):
        store.save(trainer, artifact, Progress(1, 1, 1))
    assert store.latest()[0] == complete


def test_invalid_checkpoint_pointer_and_reuse_state_rejected(setup):
    trainer, _, store, artifact, _ = setup
    with pytest.raises(ValueError, match="pending-reuse"):
        store.save(trainer, artifact, Progress(iteration=1))
    (store.root / "latest.json").write_text(json.dumps({"checkpoint": "../escape"}))
    with pytest.raises(ValueError, match="pointer"):
        store.latest()


def test_nonfinite_gradient_cannot_step_or_advance_scheduler(setup):
    trainer, bundle, _, _, _ = setup
    parameter = next(p for p in trainer.model.parameters() if p.requires_grad)
    parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(RuntimeError, match="Nonfinite"):
        optimizer_step(trainer, bundle.config)
    assert trainer.state.global_step == 0 and trainer.lr_scheduler.last_epoch == 0
    assert all(p.grad is None for p in trainer.model.parameters())


def test_checkpoint_rejects_policy_step_mismatch(setup):
    trainer, _, store, artifact, _ = setup
    with pytest.raises(ValueError, match="step mismatch"):
        store.save(trainer, artifact, Progress(step=1))
