"""Explicit single-GPU GRPO loop. Importing this module performs no execution."""

import math
import time
from dataclasses import asdict, replace

from grpo_training.advantages import prepare_update
from grpo_training.checkpoints import Progress, restore_rng
from grpo_training.collection import collect_groups
from grpo_training.data import PreparedSplit
from grpo_training.model import export_adapter
from grpo_training.provenance import split_record, training_source_hashes
from grpo_training.rewards import episode_reward
from grpo_training.trainer import (
    backward_update,
    capture_update,
    validate_replay_config,
)


def run_identity(config, train):
    validate_replay_config(config)
    if (
        type(train) is not PreparedSplit
        or train.split != "train"
        or len(train.questions) < config.questions_per_update
    ):
        raise ValueError(
            "Runner requires enough pinned train questions for one complete update"
        )
    if len({q.id for q in train.questions}) != len(train.questions):
        raise ValueError("Duplicate training questions")
    for question, label in zip(train.questions, train.labels, strict=True):
        if (
            question.split != "train"
            or question.dataset_revision != config.dataset_revision
        ):
            raise ValueError("Question split/revision mismatch")
        episode_reward(
            question.id, "context_limit", None, label
        )  # Validate gold before compute.
    return {
        "schema_version": 1,
        "config": config.to_dict(),
        "train": split_record(train),
        "source_sha256": training_source_hashes(),
    }


def initialize_optimizer(trainer, config):
    """Do not invoke Trainer.train(); prepare its optimizer/scheduler explicitly."""
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.create_optimizer_and_scheduler(num_training_steps=config.max_steps)
    trainer.optimizer = trainer.accelerator.prepare_optimizer(trainer.optimizer)
    trainer.lr_scheduler = trainer.accelerator.prepare_scheduler(trainer.lr_scheduler)


def optimizer_step(trainer, config):
    import torch

    parameters = [p for p in trainer.model.parameters() if p.requires_grad]
    if not parameters or not any(p.grad is not None for p in parameters):
        raise RuntimeError("No trainable policy gradients")
    norm = trainer.accelerator.clip_grad_norm_(parameters, config.max_grad_norm)
    if not torch.isfinite(norm).item():
        trainer.optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Nonfinite gradient norm; optimizer step refused")
    trainer.optimizer.step()
    if trainer.accelerator.optimizer_step_was_skipped:
        raise RuntimeError("Optimizer step skipped; refusing to advance/checkpoint")
    trainer.lr_scheduler.step()
    trainer.state.global_step += 1
    trainer.optimizer.zero_grad(set_to_none=True)
    return norm.item()


def probability_agreement(captured):
    differences = [
        abs(old - sampled)
        for row, values in zip(
            captured.prepared.rows(), captured.old_logprobs, strict=True
        )
        for old, sampled in zip(
            values, row["sampling_logprobs"][-len(values) :], strict=True
        )
    ]
    return {
        "mean_absolute_logprob_difference": sum(differences) / len(differences),
        "max_absolute_logprob_difference": max(differences),
    }


def run_training(
    trainer,
    bundle,
    backend,
    train,
    search,
    store,
    *,
    progress=None,
    pending=None,
    rng=None,
    group_attempts=2,
    search_workers=4,
    mean_logprob_tolerance=0.1,
    max_logprob_tolerance=1.0,
    emit=print,
    step_fn=optimizer_step,
    capture_fn=capture_update,
    backward_fn=backward_update,
    export_fn=export_adapter,
    collect_fn=collect_groups,
    release_memory=None,
):
    """One caller owns the GPU/run lock. Checkpoint after every successful step.

    Injectable boundaries permit CPU orchestration tests; production uses the
    actual model, TRL, vLLM, search transport and optimizer functions above.
    """
    progress = Progress() if progress is None else progress
    config = bundle.config
    validate_replay_config(config)
    if store.identity != run_identity(config, train):
        raise ValueError("Run store identity disagrees with current inputs/source")
    if (
        progress.step != trainer.state.global_step
        or progress.step > config.max_steps
        or progress.cursor >= len(train.questions)
    ):
        raise ValueError("Invalid resume step/data cursor")
    if (pending is None) != (progress.iteration == 0):
        raise ValueError("Invalid pending-reuse state")
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0
        for v in (mean_logprob_tolerance, max_logprob_tolerance)
    ):
        raise ValueError("Invalid logprob agreement tolerance")
    if release_memory is None:

        def release_memory():
            import torch

            torch.cuda.empty_cache()

    if rng is not None:
        restore_rng(rng)  # Model/trainer/vLLM must already exist.
    artifact = export_fn(bundle, store.root / "transfers", progress.step)
    awake = True  # Newly constructed vLLM starts awake, with no registered adapter.
    try:
        if progress.step == 0 and pending is None:
            store.save(trainer, artifact, progress)
        if pending is not None:
            backend.sleep()
            awake = False
        while progress.step < config.max_steps:
            if pending is None:
                release_memory()
                if not awake:
                    backend.wake_up()
                    awake = True
                backend.synchronize(artifact)
                indices = [
                    (progress.cursor + i) % len(train.questions)
                    for i in range(config.questions_per_update)
                ]
                questions = tuple(train.questions[i] for i in indices)
                labels = tuple(train.labels[i] for i in indices)
                started = time.perf_counter()
                groups = collect_fn(
                    questions,
                    config,
                    artifact.policy,
                    backend,
                    bundle.tokenizer,
                    search,
                    progress.round_index,
                    group_attempts=group_attempts,
                    search_workers=search_workers,
                    audit=store.audit,
                )
                backend.sleep()
                awake = False
                prepared = prepare_update(groups, labels)
                pending = capture_fn(trainer, prepared, artifact, bundle)
                agreement = probability_agreement(pending)
                store.audit(
                    {
                        "kind": "rollout_captured",
                        "step": progress.step,
                        "round": progress.round_index,
                        "seconds": time.perf_counter() - started,
                        "tokens": prepared.token_count,
                        **agreement,
                    }
                )
                # Fail before the first update if the real backends disagree too far.
                if (
                    agreement["mean_absolute_logprob_difference"]
                    > mean_logprob_tolerance
                    or agreement["max_absolute_logprob_difference"]
                    > max_logprob_tolerance
                ):
                    raise RuntimeError(
                        "Sampling/replay probability agreement exceeded configured tolerances"
                    )
                progress = replace(
                    progress,
                    cursor=(progress.cursor + config.questions_per_update)
                    % len(train.questions),
                    round_index=progress.round_index + 1,
                )
            started = time.perf_counter()
            loss = backward_fn(trainer, pending)
            grad_norm = step_fn(trainer, config)
            progress = replace(
                progress,
                step=trainer.state.global_step,
                iteration=progress.iteration + 1,
            )
            signals = pending.prepared.signals
            metrics = {
                "kind": "optimizer_step",
                "step": progress.step,
                "round": progress.round_index - 1,
                "iteration": progress.iteration,
                "loss": loss,
                "gradient_norm": grad_norm,
                "mean_reward": sum(s.reward for s in signals) / len(signals),
                "zero_advantage_episodes": sum(s.advantage == 0 for s in signals),
                "tokens": pending.prepared.token_count,
                "seconds": time.perf_counter() - started,
            }
            artifact = export_fn(bundle, store.root / "transfers", progress.step)
            if progress.iteration == config.num_iterations:
                pending = None
                progress = replace(progress, iteration=0)
            checkpoint = store.save(trainer, artifact, progress, pending)
            store.audit(metrics | {"checkpoint": checkpoint.name})
            emit(
                f"Step {progress.step}/{config.max_steps} | reward={metrics['mean_reward']:.3f} "
                f"loss={loss:.5f} grad_norm={grad_norm:.4f} | checkpoint={checkpoint.name}",
                flush=True,
            )
        return {
            "status": "completed",
            "progress": asdict(progress),
            "run_directory": str(store.root),
        }
    except Exception as exc:
        trainer.model.zero_grad(set_to_none=True)
        store.audit(
            {
                "kind": "run_failed",
                "last_completed_step_in_memory": trainer.state.global_step,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        raise  # Never checkpoint a partially mutated/failed optimizer state.
    finally:
        if awake:
            backend.sleep()
