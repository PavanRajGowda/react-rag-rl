"""Explicit per-turn replay bridge to TRL 1.15's native DAPO objective.

No GPU libraries load on import. This is not a Modal runner or a rollout loop.
TRL owns the clipped objective; this bridge owns episode grouping and exact
replacement-history inputs. Built-in single-transcript generation is disabled.
"""

import math
from dataclasses import dataclass

from grpo_training.advantages import PreparedUpdate
from grpo_training.model import AdapterArtifact, check_runtime, export_adapter


def validate_replay_config(config):
    if config.beta != 0 or config.loss_type != "dapo":
        raise ValueError("Initial replay bridge supports DAPO with beta=0 only")
    if config.top_p != 1 or config.lora_dropout != 0:
        raise ValueError(
            "Replay requires top_p=1 and zero adapter dropout for probability agreement"
        )


def collate_segments(rows, pad_token_id, device):
    """Left-pad prompts, right-pad completions; masks depend on lengths, not IDs."""
    import torch

    rows = tuple(rows)
    if not rows or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Expected nonempty replay rows and a valid padding ID")
    prompts = [row["input_ids"][: -len(row["target_ids"])] for row in rows]
    completions = [row["target_ids"] for row in rows]
    pwidth, cwidth = max(map(len, prompts)), max(map(len, completions))
    data = {
        "prompt_ids": [],
        "prompt_mask": [],
        "completion_ids": [],
        "completion_mask": [],
    }
    for prompt, completion in zip(prompts, completions, strict=True):
        ppad, cpad = pwidth - len(prompt), cwidth - len(completion)
        data["prompt_ids"].append((pad_token_id,) * ppad + prompt)
        data["prompt_mask"].append((0,) * ppad + (1,) * len(prompt))
        data["completion_ids"].append(completion + (pad_token_id,) * cpad)
        data["completion_mask"].append((1,) * len(completion) + (0,) * cpad)
    return {
        key: torch.tensor(value, dtype=torch.long, device=device)
        for key, value in data.items()
    }


@dataclass(frozen=True)
class CapturedUpdate:
    prepared: PreparedUpdate
    old_logprobs: tuple[tuple[float, ...], ...]
    corrections: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        rows = self.prepared.rows()
        for name in ("old_logprobs", "corrections"):
            vectors = tuple(tuple(v) for v in getattr(self, name))
            if len(vectors) != len(rows):
                raise ValueError("Captured row count changed")
            for row, values in zip(rows, vectors, strict=True):
                if len(values) != len(row["target_ids"]) or any(
                    isinstance(v, bool)
                    or not isinstance(v, (float, int))
                    or not math.isfinite(v)
                    for v in values
                ):
                    raise ValueError("Invalid captured token probabilities")
                if name == "old_logprobs" and any(v > 0 for v in values):
                    raise ValueError("Logprobs must be nonpositive")
                if name == "corrections" and any(
                    not 0 <= v <= self.prepared.config.vllm_importance_sampling_cap
                    for v in values
                ):
                    raise ValueError("Correction outside configured bounds")
            object.__setattr__(self, name, vectors)


def _guard_trainer(trainer, config):
    validate_replay_config(config)
    # TRL's own vLLM engine must be off: Step 5 owns sampling and weight transfer.
    # Positive fixed correction factors are folded into token advantages below.
    if trainer.use_vllm or trainer.beta != 0 or trainer.loss_type != "dapo":
        raise ValueError("Trainer must use external sampling, DAPO and beta=0")
    if (
        trainer.accelerator.num_processes != 1
        or trainer.current_gradient_accumulation_steps != 1
        or trainer.args.steps_per_generation != 1
    ):
        raise ValueError(
            "Initial replay bridge requires single-process explicit token accumulation"
        )
    if (
        trainer.temperature != config.temperature
        or trainer.epsilon_low != config.epsilon_low
        or trainer.epsilon_high != config.epsilon_high
    ):
        raise ValueError(
            "Trainer probability/clipping settings disagree with experiment"
        )
    if (
        trainer.top_entropy_quantile != 1
        or trainer._entropy_bonus_enabled
        or trainer.aux_loss_enabled
        or trainer.off_policy_mask_threshold is not None
        or trainer.args.delta is not None
        or trainer.importance_sampling_level != "token"
    ):
        raise ValueError("Unexpected extra loss terms, masks or sequence-level ratios")


def capture_update(trainer, prepared, artifact, bundle):
    """Before ANY optimizer update: teacher-force old policy without an autograd graph."""
    import torch

    if type(prepared) is not PreparedUpdate or type(artifact) is not AdapterArtifact:
        raise TypeError(
            "Expected prepared complete groups and exported policy artifact"
        )
    artifact.verify()
    _guard_trainer(trainer, prepared.config)
    if (
        artifact.policy != prepared.policy
        or trainer.state.global_step != prepared.policy.optimizer_step
    ):
        raise ValueError(
            "Old-policy capture must precede updates and match the rollout policy"
        )
    if (
        trainer.accelerator.unwrap_model(trainer.model) is not bundle.model
        or bundle.config != prepared.config
    ):
        raise ValueError("Trainer and exported TrainingModel disagree")
    current = export_adapter(bundle, artifact.path.parent, trainer.state.global_step)
    if current.policy != artifact.policy:
        raise ValueError("Live training adapter no longer matches the sampled policy")
    if any(
        e.sampling.top_k != 0 or e.sampling.repetition_penalty != 1
        for g in prepared.groups
        for e in g.episodes
    ):
        raise ValueError(
            "Initial replay supports no top-k filter or repetition penalty"
        )
    rows, old, corrections = prepared.rows(), [], []
    size = prepared.config.replay_microbatch_segments
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), size):
                chunk = rows[start : start + size]
                inputs = collate_segments(
                    chunk,
                    trainer.processing_class.pad_token_id,
                    trainer.accelerator.device,
                )
                logps, _, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.model,
                    torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1),
                    torch.cat(
                        [inputs["prompt_mask"], inputs["completion_mask"]], dim=1
                    ),
                    inputs["completion_ids"].shape[1],
                )
                for row, values in zip(chunk, logps, strict=True):
                    values = tuple(
                        values[: len(row["target_ids"])].float().cpu().tolist()
                    )
                    if len(values) != len(row["target_ids"]) or any(
                        not math.isfinite(v) or v > 0 for v in values
                    ):
                        raise ValueError("Invalid teacher-forcing old logprobs")
                    sampling = row["sampling_logprobs"][-len(values) :]
                    cap = math.log(prepared.config.vllm_importance_sampling_cap)
                    ratios = (
                        tuple(
                            math.exp(min(a - b, cap))
                            for a, b in zip(values, sampling, strict=True)
                        )
                        if prepared.config.vllm_importance_sampling_correction
                        else (1.0,) * len(values)
                    )
                    old.append(values)
                    corrections.append(ratios)
    finally:
        trainer.model.train(was_training)
    return CapturedUpdate(prepared, tuple(old), tuple(corrections))


def loss_microbatches(trainer, captured):
    """Whole-update token denominator, even when turns have unequal lengths."""
    import torch

    config = captured.prepared.config
    _guard_trainer(trainer, config)
    if (
        not 0
        <= trainer.state.global_step - captured.prepared.policy.optimizer_step
        < config.num_iterations
    ):
        raise ValueError(
            "Captured rollout reuse limit exceeded or policy step rolled back"
        )
    rows = captured.prepared.rows()
    if len(rows) != len(captured.old_logprobs) or len(rows) != len(
        captured.corrections
    ):
        raise ValueError("Captured replay row count changed")
    for start in range(0, len(rows), config.replay_microbatch_segments):
        chunk = rows[start : start + config.replay_microbatch_segments]
        inputs = collate_segments(
            chunk, trainer.processing_class.pad_token_id, trainer.accelerator.device
        )
        shape = inputs["completion_ids"].shape
        old = torch.zeros(shape, device=trainer.accelerator.device)
        advantages = torch.zeros_like(old)
        for index, row in enumerate(chunk):
            values, correction = (
                captured.old_logprobs[start + index],
                captured.corrections[start + index],
            )
            if len(values) != len(row["target_ids"]) or len(correction) != len(values):
                raise ValueError("Captured token/probability length changed")
            old[index, : len(values)] = torch.tensor(values, device=old.device)
            advantages[index, : len(values)] = row["advantage"] * torch.tensor(
                correction, device=old.device
            )
        # For fixed positive c: min(r*A, clip(r)*A)*c == min(r*(A*c), clip(r)*(A*c)).
        # This preserves TRL's objective while avoiding a second vLLM engine/correction.
        inputs.update(
            old_per_token_logps=old,
            advantages=advantages,
            num_items_in_batch=torch.tensor(
                captured.prepared.token_count, device=old.device
            ),
        )
        yield inputs


def backward_update(trainer, captured):
    """Accumulate one logical update's gradients; caller owns optimizer/scheduler step.

    No automatic division by number of microbatches: DAPO's shared token denominator
    already normalizes the sum. On failure discard partial gradients, then raise.
    """
    import torch

    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    total = 0.0
    try:
        for inputs in loss_microbatches(trainer, captured):
            if getattr(trainer, "_loss_metrics", None) is not None:
                trainer._loss_metrics.inputs = inputs
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(trainer.model, inputs)
            if loss.ndim or not torch.isfinite(loss).item() or not loss.requires_grad:
                raise ValueError("Expected finite differentiable scalar GRPO loss")
            trainer.accelerator.backward(loss)
            total += loss.detach().float().item()
    except Exception:
        trainer.model.zero_grad(set_to_none=True)
        raise
    return total


def replay_trainer_class():
    """Import TRL only when explicitly requested; keep its native loss/forward."""
    from trl import GRPOTrainer

    class ReplayGRPOTrainer(GRPOTrainer):
        def train(self, *args, **kwargs):
            raise RuntimeError(
                "Use explicit replay orchestration; built-in transcript training is disabled"
            )

        def _generate_and_score_completions(self, *args, **kwargs):
            raise RuntimeError(
                "Use independent SearchEpisode rollouts, never stitched transcripts"
            )

    return ReplayGRPOTrainer


def create_replay_trainer(bundle, questions, output_dir):
    """Explicit GPU setup after loading a TrainingModel; launches no job or rollout."""
    check_runtime()
    validate_replay_config(bundle.config)
    from datasets import Dataset
    from trl import GRPOConfig

    from grpo_training.data import Question, policy_messages

    questions = tuple(questions)
    if not questions or any(
        type(q) is not Question
        or q.split != "train"
        or q.dataset_revision != bundle.config.dataset_revision
        for q in questions
    ):
        raise ValueError(
            "Trainer dataset must contain pinned train-only Question objects"
        )
    config = bundle.config
    args = GRPOConfig(
        output_dir=str(output_dir),
        use_vllm=False,
        bf16=True,
        per_device_train_batch_size=config.num_generations,
        num_generations=config.num_generations,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
        max_completion_length=config.max_new_tokens,
        temperature=config.temperature,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dapo",
        scale_rewards=config.scale_rewards,
        num_iterations=config.num_iterations,
        epsilon=config.epsilon_low,
        epsilon_high=config.epsilon_high,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.scheduler,
        optim="adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config.seed,
        report_to=[],
        save_strategy="no",
        mask_truncated_completions=False,
        disable_dropout=True,
        remove_unused_columns=False,
    )

    def unused_reward(**kwargs):
        raise RuntimeError(
            "Rewards must come from complete controller-accepted episodes"
        )

    trainer = replay_trainer_class()(
        model=bundle.model,
        args=args,
        processing_class=bundle.tokenizer,
        reward_funcs=[unused_reward],
        train_dataset=Dataset.from_list(
            [{"prompt": policy_messages(q)} for q in questions]
        ),
    )
    trainer.current_gradient_accumulation_steps = 1
    _guard_trainer(trainer, config)
    return trainer
