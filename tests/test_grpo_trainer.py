"""Real TRL DAPO + CPU autograd tests, in the optional isolated CPU environment.

The tiny probability model replaces Qwen/fused kernels; TRL's loss is unmodified.
"""

import json
import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from grpo_training.advantages import prepare_update
from grpo_training.config import TrainingConfig
from grpo_training.model import TrainingModel, export_adapter
from grpo_training.trainer import (
    backward_update,
    capture_update,
    collate_segments,
    loss_microbatches,
    replay_trainer_class,
    validate_replay_config,
)
from tests.test_grpo_advantages import label, make_group

TARGETS = (
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
)


class Tokenizer:
    chat_template = "cpu-test"
    pad_token_id = 0


class ToyPolicy(torch.nn.Module):
    """Frozen base logits plus two trainable adapter matrices; synthetic serialization."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.register_buffer("base", torch.randn(8, 8, generator=generator) / 5)
        self.a = torch.nn.Parameter(torch.randn(8, 2, generator=generator) / 10)
        self.b = torch.nn.Parameter(torch.randn(2, 8, generator=generator) / 10)

    def forward(self, ids, mask):
        contexts = ((ids * mask).cumsum(-1) % 8).long()
        features = torch.nn.functional.one_hot(contexts, 8).float()
        return self.base[contexts] + features @ self.a @ self.b

    def named_parameters(self, *args, **kwargs):
        return iter(
            [
                (f"base_model.model.{TARGETS[0]}.lora_A.default.weight", self.a),
                (f"base_model.model.{TARGETS[0]}.lora_B.default.weight", self.b),
            ]
        )

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        (path / "adapter_model.safetensors").write_bytes(
            self.a.detach().numpy().tobytes() + self.b.detach().numpy().tobytes()
        )
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "bias": "none",
                    "target_modules": TARGETS,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.0,
                }
            )
        )


@pytest.fixture
def setup(tmp_path):
    config = TrainingConfig(num_generations=2, questions_per_update=1, num_iterations=2)
    model = ToyPolicy()
    bundle = TrainingModel(model, Tokenizer(), tmp_path, config, TARGETS)
    artifact = export_adapter(bundle, tmp_path / "adapters", 0)
    group = make_group(config, policy=artifact.policy)
    prepared = prepare_update([group], [label(group)])
    native_class = replay_trainer_class()

    class CPUTrainer(native_class):
        def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep, **kwargs
        ):
            self.forward_grad_modes.append(torch.is_grad_enabled())
            logits = (
                model(input_ids, attention_mask)[:, -logits_to_keep - 1 : -1]
                / self.temperature
            )
            logps = torch.log_softmax(logits, dim=-1)
            targets = input_ids[:, -logits_to_keep:]
            chosen = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            return chosen, -(logps.exp() * logps).sum(-1), None

    # Isolate the real native objective from constructor/model-kernel/GPU setup.
    trainer = object.__new__(CPUTrainer)
    trainer.model = model
    trainer.processing_class = bundle.tokenizer
    trainer.use_vllm = False
    trainer.beta = 0.0
    trainer.loss_type = "dapo"
    trainer.temperature = config.temperature
    trainer.epsilon_low, trainer.epsilon_high = config.epsilon_low, config.epsilon_high
    trainer.current_gradient_accumulation_steps = 1
    trainer.top_entropy_quantile = 1.0
    trainer._entropy_bonus_enabled = trainer.aux_loss_enabled = False
    trainer.off_policy_mask_threshold = None
    trainer.importance_sampling_level = "token"
    trainer.args = NS(steps_per_generation=1, delta=None, report_to=[])
    trainer.state = NS(global_step=0)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.forward_grad_modes = []
    trainer.accelerator = NS(
        num_processes=1,
        is_main_process=True,
        device=torch.device("cpu"),
        unwrap_model=lambda m: m,
        autocast=nullcontext,
        gather=lambda t: t,
        reduce=lambda t, **kw: t,
        backward=lambda loss: loss.backward(),
    )
    trainer.compute_loss_context_manager = nullcontext
    return trainer, prepared, artifact, bundle


def test_old_capture_has_no_graph_and_backward_trains_only_adapters(setup):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    assert trainer.forward_grad_modes == [False] * 3
    assert all(p.grad is None for p in trainer.model.parameters())
    assert all(isinstance(v, float) for row in captured.old_logprobs for v in row)
    value = backward_update(trainer, captured)
    assert math.isfinite(value) and trainer.forward_grad_modes[-3:] == [True] * 3
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in trainer.model.parameters()
    )
    assert trainer.model.base.grad is None
    assert sum(p.grad.abs().sum() for p in trainer.model.parameters()) > 0


def test_unequal_turn_microbatches_match_one_batch_gradients(setup):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    accumulated_loss = backward_update(trainer, captured)
    accumulated_grads = [p.grad.clone() for p in trainer.model.parameters()]
    trainer.model.zero_grad(set_to_none=True)
    rows = prepared.rows()
    inputs = collate_segments(rows, 0, "cpu")
    pad = torch.nn.utils.rnn.pad_sequence
    inputs["old_per_token_logps"] = pad(
        [torch.tensor(v) for v in captured.old_logprobs], batch_first=True
    )
    inputs["advantages"] = pad(
        [
            row["advantage"] * torch.tensor(c)
            for row, c in zip(rows, captured.corrections, strict=True)
        ],
        batch_first=True,
    )
    inputs["num_items_in_batch"] = torch.tensor(prepared.token_count)
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    assert loss.item() == pytest.approx(accumulated_loss, abs=1e-6)
    for actual, expected in zip(
        trainer.model.parameters(), accumulated_grads, strict=True
    ):
        torch.testing.assert_close(actual.grad, expected, atol=1e-6, rtol=1e-5)


def test_native_trl_clips_positive_and_negative_advantages(setup, monkeypatch):
    trainer, _, _, _ = setup
    log_ratios = torch.tensor([1.5, 1.5, 0.5, 0.5]).log().requires_grad_()
    monkeypatch.setattr(
        trainer,
        "_get_per_token_logps_and_entropies",
        lambda *a, **kw: ((log_ratios - 2).unsqueeze(0), torch.ones(1, 4), None),
    )
    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 3, 2, 3]]),
        "completion_mask": torch.ones(1, 4),
        "old_per_token_logps": torch.full((1, 4), -2.0),
        "advantages": torch.tensor([[1.0, -1.0, 1.0, -1.0]]),
        "num_items_in_batch": torch.tensor(4),
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    assert loss.item() == pytest.approx(-(1.28 - 1.5 + 0.5 - 0.8) / 4, abs=1e-6)
    torch.testing.assert_close(log_ratios.grad, torch.tensor([0.0, 1.5, -0.5, 0.0]) / 4)


def test_correction_matches_old_over_sampling_and_is_applied_once(setup):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    for row, old, correction in zip(
        prepared.rows(), captured.old_logprobs, captured.corrections, strict=True
    ):
        assert correction == pytest.approx([min(math.exp(a + 2), 2.0) for a in old])
    for inputs, row in zip(
        loss_microbatches(trainer, captured), prepared.rows(), strict=True
    ):
        index = list(prepared.rows()).index(row)
        assert inputs["advantages"][0].tolist() == pytest.approx(
            [row["advantage"] * c for c in captured.corrections[index]]
        )
        assert not inputs["old_per_token_logps"].requires_grad
        assert inputs["num_items_in_batch"].item() == 6


def test_correction_cap_handles_extreme_logprob_gap(setup):
    trainer, prepared, artifact, bundle = setup
    groups = []
    for group in prepared.groups:
        groups.append(
            replace(
                group,
                episodes=tuple(
                    replace(
                        e,
                        generations=tuple(
                            replace(
                                g, sampling_logprobs=(-10000.0,) * len(g.completion_ids)
                            )
                            for g in e.generations
                        ),
                    )
                    for e in group.episodes
                ),
            )
        )
    prepared = prepare_update(groups, [label(groups[0])])
    captured = capture_update(trainer, prepared, artifact, bundle)
    assert all(v == 2.0 for row in captured.corrections for v in row)


def test_sampled_eos_is_active_even_when_it_equals_padding_id(setup):
    _, prepared, _, _ = setup
    rows = prepared.rows()
    rows = [rows[0] | {"target_ids": (0,), "input_ids": (1, 1, 0)}, rows[1]]
    inputs = collate_segments(rows, 0, "cpu")
    assert inputs["completion_ids"][0].tolist() == [0, 0, 0]
    assert inputs["completion_mask"][0].tolist() == [1, 0, 0]
    assert inputs["prompt_mask"][0].tolist() == [0, 1, 1]


def test_old_snapshot_cannot_be_captured_after_weights_change(setup):
    trainer, prepared, artifact, bundle = setup
    with torch.no_grad():
        trainer.model.a.add_(1.0)
    with pytest.raises(ValueError, match="Live training adapter"):
        capture_update(trainer, prepared, artifact, bundle)


def test_reuse_keeps_old_logprobs_fixed_and_respects_iteration_limit(setup):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    old = captured.old_logprobs
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    backward_update(trainer, captured)
    optimizer.step()
    trainer.state.global_step = 1
    backward_update(trainer, captured)
    assert captured.old_logprobs == old
    trainer.state.global_step = 2
    with pytest.raises(ValueError, match="reuse limit"):
        backward_update(trainer, captured)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_failed_microbatch_discards_partial_gradients(setup, monkeypatch):
    trainer, prepared, artifact, bundle = setup
    captured = capture_update(trainer, prepared, artifact, bundle)
    original = trainer.compute_loss
    calls = []

    def fail(model, inputs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated failure")
        return original(model, inputs)

    monkeypatch.setattr(trainer, "compute_loss", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        backward_update(trainer, captured)
    assert all(p.grad is None for p in trainer.model.parameters())


def test_unsupported_modes_and_builtin_training_are_blocked(setup):
    trainer, prepared, _, _ = setup
    for config in [
        replace(prepared.config, top_p=0.9),
        replace(prepared.config, beta=0.01),
        replace(prepared.config, lora_dropout=0.1),
    ]:
        with pytest.raises(ValueError):
            validate_replay_config(config)
    with pytest.raises(RuntimeError, match="orchestration"):
        trainer.train()
    with pytest.raises(RuntimeError, match="SearchEpisode"):
        trainer._generate_and_score_completions([])


def test_zero_variance_group_has_zero_policy_gradient(setup):
    trainer, prepared, artifact, bundle = setup
    group = make_group(bundle.config, policy=artifact.policy, answers=("42", "42"))
    prepared = prepare_update([group], [label(group)])
    captured = capture_update(trainer, prepared, artifact, bundle)
    assert backward_update(trainer, captured) == 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in trainer.model.parameters())


def test_disabling_correction_uses_unit_weights(setup):
    trainer, _, artifact, bundle = setup
    bundle.config = replace(bundle.config, vllm_importance_sampling_correction=False)
    artifact = export_adapter(bundle, artifact.path.parent / "uncorrected", 0)
    group = make_group(bundle.config, policy=artifact.policy)
    prepared = prepare_update([group], [label(group)])
    captured = capture_update(trainer, prepared, artifact, bundle)
    assert all(v == 1.0 for row in captured.corrections for v in row)


def test_failed_old_capture_restores_mode(setup, monkeypatch):
    trainer, prepared, artifact, bundle = setup
    monkeypatch.setattr(
        trainer,
        "_get_per_token_logps_and_entropies",
        lambda *a, **kw: (torch.full((1, 2), float("nan")), None, None),
    )
    with pytest.raises(ValueError, match="Invalid teacher-forcing"):
        capture_update(trainer, prepared, artifact, bundle)
    assert trainer.model.training
    assert all(p.grad is None for p in trainer.model.parameters())


@pytest.mark.parametrize(
    "setting", ["distributed", "double_correction", "accumulation"]
)
def test_incompatible_trainer_settings_rejected(setup, setting):
    trainer, prepared, artifact, bundle = setup
    if setting == "distributed":
        trainer.accelerator.num_processes = 2
    elif setting == "double_correction":
        trainer.use_vllm = True
    else:
        trainer.current_gradient_accumulation_steps = 2
    with pytest.raises(ValueError):
        capture_update(trainer, prepared, artifact, bundle)


def test_factory_keeps_labels_out_and_disables_second_engine(
    setup, monkeypatch, tmp_path
):
    import trl

    import grpo_training.trainer as module

    trainer, _, _, bundle = setup
    calls = {}

    def factory(**kwargs):
        calls.update(kwargs)
        return trainer

    monkeypatch.setattr(module, "check_runtime", lambda: None)
    monkeypatch.setattr(module, "replay_trainer_class", lambda: factory)
    monkeypatch.setattr(trl, "GRPOConfig", NS)
    question = make_group(bundle.config).episodes[0].question
    assert module.create_replay_trainer(bundle, [question], tmp_path) is trainer
    args = calls["args"]
    assert args.use_vllm is False and args.beta == 0 and args.loss_type == "dapo"
    assert args.epsilon == 0.2 and args.epsilon_high == 0.28
    assert args.mask_truncated_completions is False
    assert args.gradient_accumulation_steps == args.steps_per_generation == 1
    assert args.report_to == [] and args.save_strategy == "no"
    assert list(calls["train_dataset"].features) == ["prompt"]
    assert "diagnostic-only" not in str(calls["train_dataset"].to_list())
    with pytest.raises(RuntimeError, match="complete"):
        calls["reward_funcs"][0]()


def test_real_trl_constructor_and_forward_wrapper_cpu(tmp_path, monkeypatch):
    """Native constructor/wrapper, with a reference CPU kernel replacing Triton."""
    import trl.trainer.utils as trl_utils
    from datasets import Dataset
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig

    from grpo_training.runner import initialize_optimizer, optimizer_step

    class CPUReferenceKernel:
        @staticmethod
        def apply(
            hidden,
            weight,
            bias,
            labels,
            temperature,
            chunk_size,
            softcap,
            scale,
            outputs,
        ):
            assert softcap is None and scale == 1.0
            logits = (
                torch.nn.functional.linear(hidden, weight, bias).float() / temperature
            )
            logps = logits.log_softmax(-1)
            return (
                logps.gather(-1, labels[:, None]).squeeze(-1),
                -(logps.exp() * logps).sum(-1),
                None,
                None,
                None,
            )

    monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", CPUReferenceKernel)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=FastTokenizer(
            WordLevel({"[PAD]": 0, "[EOS]": 1, "x": 2, "y": 3}, unk_token="x")
        ),
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=2, n_positions=32)
    )
    args = GRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        per_device_train_batch_size=2,
        num_generations=2,
        loss_type="dapo",
        beta=0.0,
        temperature=0.7,
        epsilon=0.2,
        epsilon_high=0.28,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        report_to=[],
        max_steps=1,
    )
    trainer = replay_trainer_class()(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=[lambda **kw: [0.0, 1.0]],
        train_dataset=Dataset.from_list([{"prompt": "x"}]),
    )
    trainer.current_gradient_accumulation_steps = 1
    config = TrainingConfig(num_generations=2, questions_per_update=1, max_steps=1)
    initialize_optimizer(trainer, config)
    inputs = {
        "prompt_ids": torch.tensor([[2]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[3, 1]]),
        "completion_mask": torch.tensor([[1, 1]]),
        "advantages": torch.tensor([1.0]),
        "num_items_in_batch": torch.tensor(2),
    }
    loss = trainer.compute_loss(model, inputs)
    trainer.accelerator.backward(loss)
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    assert math.isfinite(optimizer_step(trainer, config))
    assert trainer.state.global_step == 1


def test_partial_complete_group_batch_matches_original_native_gradients(
    setup, tmp_path
):
    trainer, prepared, artifact, bundle = setup
    original = capture_update(trainer, prepared, artifact, bundle)
    expected_loss = backward_update(trainer, original)
    expected_grads = [p.grad.clone() for p in trainer.model.parameters()]
    # Nominally request two groups, but only one survived technical recovery.
    bundle.config = replace(bundle.config, questions_per_update=2)
    artifact = export_adapter(bundle, tmp_path / "partial-adapters", 0)
    group = make_group(bundle.config, policy=artifact.policy)
    partial = prepare_update([group], [label(group)])
    actual = capture_update(trainer, partial, artifact, bundle)
    assert actual.prepared.token_count == original.prepared.token_count
    assert backward_update(trainer, actual) == pytest.approx(expected_loss, abs=1e-6)
    for parameter, expected in zip(
        trainer.model.parameters(), expected_grads, strict=True
    ):
        torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)
