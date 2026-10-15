"""Exact CPU reference KL and unmodified-gradient instrumentation checks."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from grpo_training.observability import observe_loss
from grpo_training.reference import ReferenceProbe
from grpo_training.trainer import backward_update, capture_update
from tests.test_grpo_trainer import setup as trl_setup_fixture

setup = trl_setup_fixture


class Tokenizer:
    chat_template = "probe-test"

    def apply_chat_template(self, *args, **kwargs):
        return "prompt"

    def encode(self, *args, **kwargs):
        return [1, 2]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer("base", torch.tensor([1.0, 2.0, 3.0]))
        self.enabled = True

    def forward(self, **kwargs):
        assert kwargs["logits_to_keep"] == 1 and not torch.is_grad_enabled()
        return NS(
            logits=(self.base + (self.adapter if self.enabled else 0))[None, None]
        )

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def test_reference_kl_exact_positive_fixed_and_rng_preserved(tmp_path):
    from grpo_training.config import TrainingConfig
    from grpo_training.data import Question

    model, config = Policy(), TrainingConfig(reference_probe_questions=1)
    bundle = NS(config=config, tokenizer=Tokenizer())
    trainer = NS(model=model, accelerator=NS(unwrap_model=lambda m: m))
    questions = [
        Question("finqa_train_0", "Question?", "train", 0, config.dataset_revision)
    ]
    probe = ReferenceProbe(trainer, bundle, questions, tmp_path)
    assert probe.measure()["mean_nats"] == pytest.approx(0, abs=1e-7)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        model.adapter.add_(torch.tensor([2.0, 0.0, 0.0]))
    result = probe.measure()
    current = ((model.base + model.adapter) / config.temperature).log_softmax(-1)
    reference = (model.base / config.temperature).log_softmax(-1)
    expected = (current.exp() * (current - reference)).sum().item()
    assert result["mean_nats"] == pytest.approx(expected) and expected > 0
    assert model.training and model.enabled and model.adapter.requires_grad
    torch.testing.assert_close(torch.get_rng_state(), before)
    restored = ReferenceProbe(trainer, bundle, questions, tmp_path)
    assert restored.measure()["mean_nats"] == pytest.approx(expected)
    assert model.adapter.grad is None


def test_reference_cache_tampering_is_rejected(tmp_path):
    from grpo_training.config import TrainingConfig
    from grpo_training.data import Question

    config = TrainingConfig(reference_probe_questions=1)
    model = Policy()
    args = (
        NS(model=model, accelerator=NS(unwrap_model=lambda m: m)),
        NS(config=config, tokenizer=Tokenizer()),
        [Question("finqa_train_0", "Q", "train", 0, config.dataset_revision)],
        tmp_path,
    )
    ReferenceProbe(*args)
    (tmp_path / "reference_probe/reference.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceProbe(*args)


def test_policy_metrics_preserve_gradients_and_use_completion_denominators(setup):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    baseline_loss = backward_update(trainer, captured)
    gradients = [p.grad.clone() for p in trainer.model.parameters()]
    with observe_loss(trainer, bundle.config) as collector:
        observed_loss = backward_update(trainer, captured)
    assert observed_loss == pytest.approx(baseline_loss)
    for parameter, expected in zip(trainer.model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, expected)
    summary = collector.summary()
    assert summary["tokens"] == prepared.token_count == 6
    assert summary["ratio_outside_bounds"]["value"] == 0
    assert summary["current_over_old_ratio"]["mean"] == pytest.approx(1)
    assert trainer._loss_metrics is None and collector.inputs is None


def test_reference_failure_restores_adapter_and_training_mode(tmp_path):
    from grpo_training.config import TrainingConfig
    from grpo_training.data import Question

    model = Policy()

    def fail(**kwargs):
        raise RuntimeError("probe error")

    model.forward = fail
    config = TrainingConfig(reference_probe_questions=1)
    with pytest.raises(RuntimeError, match="probe error"):
        ReferenceProbe(
            NS(model=model, accelerator=NS(unwrap_model=lambda m: m)),
            NS(config=config, tokenizer=Tokenizer()),
            [Question("finqa_train_0", "Q", "train", 0, config.dataset_revision)],
            tmp_path,
        )
    assert model.training and model.enabled


def test_real_tensorboard_and_resume_do_not_duplicate_fresh_rollouts(setup, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from grpo_training.checkpoints import Progress, RunStore
    from grpo_training.observability import RunObserver
    from grpo_training.runner import run_identity
    from tests.test_grpo_runner import inputs

    trainer, prepared, artifact, bundle = setup
    train = inputs(bundle.config)
    store = RunStore(tmp_path / "runs", "observed", run_identity(bundle.config, train))

    class Reference:
        def __init__(self):
            self.calls = 0

        def measure(self):
            self.calls += 1
            return {
                "mean_nats": 0.1 * self.calls,
                "sum_nats": 0.2 * self.calls,
                "positions": 2,
                "max_nats": 0.1 * self.calls,
            }

    reference = Reference()
    observer = RunObserver(store, trainer, bundle, train, reference=reference)
    observer.start(Progress())
    captured = capture_update(trainer, prepared, artifact, bundle)
    observer.rollout(captured, fresh=True, round_index=0)
    with observer.loss(trainer) as policy:
        loss = backward_update(trainer, captured)
    observer.reference_step(1)
    metrics = {
        "step": 1,
        "loss": loss,
        "gradient_norm": 0.5,
        "tokens": prepared.token_count,
        "policy_metrics": policy.summary(),
    }
    observer.step(metrics)
    observer.close("completed")
    trainer.state.global_step = 1
    resumed = RunObserver(store, trainer, bundle, train, reference=reference)
    resumed.start(Progress(step=1))
    resumed.rollout(captured, fresh=False, round_index=0)
    resumed.close("completed")
    assert reference.calls == 2  # initial and post-step, not a duplicate on resume.
    events = EventAccumulator(str(store.root / "tensorboard")).Reload()
    assert [s.step for s in events.Scalars("reference/kl_current_to_initial_nats")] == [
        0,
        1,
    ]
    assert len(events.Scalars("rollout/nm")) == 1
    index = __import__("json").loads(
        (store.root / "observability/index.json").read_text()
    )
    assert index["committed_step"] == 1 and len(index["steps"]) == 1


def test_uncommitted_reference_measurement_is_replaced_on_retry(setup, tmp_path):
    from grpo_training.checkpoints import RunStore
    from grpo_training.observability import RunObserver, atomic_json
    from grpo_training.runner import run_identity
    from tests.test_grpo_runner import inputs

    trainer, _, _, bundle = setup
    trainer.state.global_step = 1
    store = RunStore(
        tmp_path, "probe-retry", run_identity(bundle.config, inputs(bundle.config))
    )

    class Writer:
        def add_scalar(self, *a):
            pass

        def flush(self):
            pass

    reference = NS(measure=lambda: {"mean_nats": 0.2, "positions": 1, "sum_nats": 0.2})
    observer = RunObserver(
        store,
        trainer,
        bundle,
        inputs(bundle.config),
        reference=reference,
        writer=Writer(),
    )
    atomic_json(observer.root / "reference/00000002.json", {"mean_nats": 999})
    assert observer.reference_step(2)["mean_nats"] == 0.2
