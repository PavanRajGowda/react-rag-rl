"""Hosted GRPO: same reward engine and rollouts, no local model or GPU trainer."""

import math
import time

from baseline_agent.persistence import write_json
from tinker_training.advantages import (
    filter_constant_groups,
    minibatch_update,
    prepare_update,
)
from tinker_training.backend import TinkerBackend
from tinker_training.collection import collect_groups
from tinker_training.loss import loss_metrics, pack_rows
from tinker_training.storage import Store


def run(
    config,
    train,
    dev,
    root,
    service,
    search,
    *,
    types,
    identity,
    validation_count=0,
    validation_interval=20,
):
    """Run or resume training while holding a nonblocking single-writer filesystem lock.

    Args:
        config: Validated experiment configuration.
        train: PreparedSplit of training questions with aligned private labels.
        dev: Retained interface argument; training performs no held-out evaluation.
        root: Local run directory for checkpoint pointers and compact statistics.
        service: Tinker ServiceClient, or a compatible CPU-test fake.
        search: Retriever callable with an optional batched-search method.
        types: SDK type namespace used to pack Datum and Adam parameters.
        identity: Frozen provider, dataset and source identity metadata.
        validation_count: Must be zero; held-out evaluation is a separate command.
        validation_interval: Retained compatibility argument; unused.

    Returns:
        The final committed progress dictionary, including the remote checkpoint path.

    Creates local files and makes paid hosted sampling/training/checkpoint calls.
    A second writer fails immediately instead of modifying the same run.
    """
    from pathlib import Path

    from filelock import FileLock

    Path(root).mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm

    with (
        FileLock(str(Path(root) / ".writer.lock"), timeout=0),
        tqdm(total=config.max_steps, desc="Minibatch slots", unit="batch") as progress,
    ):
        return _run(
            config,
            train,
            dev,
            root,
            service,
            search,
            types=types,
            identity=identity,
            validation_count=validation_count,
            validation_interval=validation_interval,
            progress=progress,
        )


def _run(
    config,
    train,
    dev,
    root,
    service,
    search,
    *,
    types,
    identity,
    validation_count=0,
    validation_interval=20,
    progress=None,
):
    """Execute the locked collection, replay, optimizer and checkpoint state machine.

    Restores model/optimizer state from the last checkpoint. Rollouts live only in
    memory; resume discards any unfinished pool and starts the next scheduled pool.
    Local durable progress advances only after a successful remote checkpoint.
    Unknown outcomes of mutating calls are not retried as additional updates. Failures
    are saved to the failure journal and re-raised, including cancellation signals.
    """
    if config.temperature != 1 or config.top_p != 1:
        raise ValueError("Initial migration requires T=1, top_p=1 for aligned logprobs")
    if config.dynamic_sampling:
        raise ValueError("Tinker study retains all groups; disable dynamic sampling")
    if validation_count:
        raise ValueError(
            "Training validation is disabled; use standalone baseline evaluation"
        )
    store = Store(root, config, identity)
    # This local pointer is the commit boundary. Model/optimizer tensors live at
    # its remote checkpoint; the JSON stores only control state and artifact paths.
    state_path = store.root / "progress.json"
    state = store.read(state_path)
    try:
        store.phase("restoring" if state else "initializing")
        # Restore optimizer moments as well as adapter weights. Restoring weights
        # alone would change the next update even with identical rollout data.
        if state:
            client = service.create_training_client_from_state_with_optimizer(
                state["checkpoint"]
            )
            store.restore_checkpoint_index(state["step"])
            # Recover metric publication if interruption followed pointer commit.
            if state.get("metrics"):
                store.commit_metrics(state["metrics"])
        else:
            client = service.create_lora_training_client(
                base_model=config.model_id,
                rank=config.lora_rank,
                seed=config.seed,
                train_mlp=True,
                train_attn=True,
                train_unembed=False,
            )
            initial = client.save_state("initial").result()
            state = {
                "step": 0,
                "cursor": 0,
                "round": 0,
                "iteration": 0,
                "pending": None,
                "checkpoint": initial.path,
            }
            write_json(state_path, state)
        tokenizer = client.get_tokenizer()
        reference = (
            service.create_sampling_client(base_model=config.model_id)
            if config.beta
            else None
        )
        info = client.get_info()
        model_info = info.model_dump(mode="json")
        write_json(store.root / "tinker-model-info.json", model_info)
        write_json(store.root / "model-history" / f"{time.time_ns()}.json", model_info)

        def backend():
            """Export current client weights and create a sampler bound to the committed
            policy.
            """
            sampler = client.save_weights_and_get_sampling_client()
            return TinkerBackend(
                sampler,
                tokenizer,
                config,
                checkpoint=state["checkpoint"],
                step=state["step"],
                types=types,
            )

        memory_pool = None
        sampler_dirty = False
        checkpoint_step = state["step"]
        label_map = {label.id: label for label in train.labels}

        def checkpoint():
            """Save weights/optimizer; resume starts a fresh pool after this checkpoint."""
            nonlocal checkpoint_step
            if state["step"] != checkpoint_step:
                saved = client.save_state(f"step-{state['step']:08d}").result()
                state["checkpoint"] = saved.path
                checkpoint_step = state["step"]
            durable = dict(state)
            if state["pending"]:
                durable.update(pending=None, iteration=0, round=state["round"] + 1)
                durable["discarded_pool_on_resume"] = True
            write_json(state_path, durable)
            store.mark_checkpoint(state["step"])

        # Sampling must use an explicitly exported policy snapshot, not an older
        # service sampler that happens to exist from a previous optimizer step.
        if state["step"] + state.get("skipped_minibatches", 0) >= config.max_steps:
            store.phase("completed")
            return state
        engine = backend()
        empty = 0
        if progress is not None:
            progress.update(state["step"] + state.get("skipped_minibatches", 0))
        while state["step"] + state.get("skipped_minibatches", 0) < config.max_steps:
            # A pending pool is reused across num_iterations updates. Its sampled
            # log probabilities stay fixed while current replay probabilities change.
            if memory_pool is not None:
                prepared, groups, labels, kl_metrics = memory_pool
            else:
                if sampler_dirty:
                    engine = backend()
                    sampler_dirty = False
                store.phase("generation")
                # Consume the seeded prepared order using a cursor, wrapping only
                # after exhaustion. This is not a new random question sample per step.
                questions = tuple(
                    train.questions[(state["cursor"] + i) % len(train.questions)]
                    for i in range(
                        min(config.questions_per_update, len(train.questions))
                    )
                )
                if len({q.id for q in questions}) != len(questions):
                    raise ValueError("Duplicate training questions")
                groups = collect_groups(
                    questions,
                    config,
                    engine.policy,
                    engine,
                    tokenizer,
                    search,
                    state["round"],
                    audit=store.audit,
                    allow_partial=True,
                )
                state["cursor"] = (state["cursor"] + len(questions)) % len(
                    train.questions
                )
                # A fully failed collection consumes no optimizer step. Bound the
                # number of such rounds so a broken retriever cannot loop forever.
                if not groups:
                    empty += 1
                    state["round"] += 1
                    state["pending"] = None
                    if empty >= config.max_consecutive_empty_batches:
                        raise RuntimeError("Repeated entirely unusable rollout batches")
                    continue
                empty = 0
                state["pending"] = True
                labels = tuple(label_map[g.episodes[0].question.id] for g in groups)
                prepared = prepare_update(groups, labels, tokenizer=tokenizer)
                store.rollout(prepared, labels, state["round"])
                kl_metrics = {}
                if reference is not None:
                    from dataclasses import replace

                    from tinker_training.kl import apply_reference_kl

                    eligible, _ = filter_constant_groups(prepared)
                    if eligible is not None:
                        store.phase("reference_scoring")
                        eligible, kl_metrics = apply_reference_kl(
                            eligible, reference, types
                        )
                        weights = {
                            (r["episode_id"], r["turn"]): r["token_advantages"]
                            for r in eligible.rows()
                        }
                        prepared = replace(
                            prepared,
                            token_weights=tuple(
                                weights.get(
                                    (r["episode_id"], r["turn"]),
                                    (0.0,) * len(r["target_ids"]),
                                )
                                for r in prepared.rows()
                            ),
                        )
                        kl_metrics["centering_population"] = (
                            "surviving completion tokens across the rollout pool"
                        )
                memory_pool = prepared, groups, labels, kl_metrics
            rollout_id = f"tinker-round-{state['round']:08d}"
            # Packing defines which token predictions contribute to the objective;
            # Tinker owns the actual differentiable forward/backward execution.
            pool_groups = len(groups)
            prepared, minibatch_count = minibatch_update(prepared, state["iteration"])
            prepared, filtering = filter_constant_groups(prepared)
            filtering.update(
                round=state["round"],
                minibatch=state["iteration"] % minibatch_count + 1,
                replay_epoch=state["iteration"] // minibatch_count + 1,
            )
            if prepared is None:
                event = filtering | {
                    "kind": "minibatch_skipped",
                    "reason": "zero_local_task_coefficients" if config.reward == "local_process_advantage_v1" else "constant_training_rewards",
                    "optimizer_step": state["step"],
                }
                store.audit(event)
                state["iteration"] += 1
                state["skipped_minibatches"] = state.get("skipped_minibatches", 0) + 1
                if state["iteration"] >= config.num_iterations * minibatch_count:
                    state.update(pending=None, iteration=0, round=state["round"] + 1)
                    memory_pool = None
                if progress is not None:
                    progress.update(1)
                continue
            data = pack_rows(prepared, types)
            started = time.perf_counter()
            store.phase("backward")
            result = client.forward_backward(
                data,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 1 - config.epsilon_low,
                    "clip_high_threshold": 1 + config.epsilon_high,
                },
            ).result()
            policy_metrics = loss_metrics(prepared, result)
            backward_seconds = time.perf_counter() - started
            store.phase("optimizer")
            # No application retry around mutating operations: restore on ambiguity.
            warmup = math.ceil(config.max_steps * config.warmup_ratio)
            learning_rate = config.learning_rate * min(
                1.0, (state["step"] + 1) / max(warmup, 1)
            )
            optimizer = client.optim_step(
                types.AdamParams(
                    learning_rate=learning_rate,
                    grad_clip_norm=config.max_grad_norm,
                )
            ).result()
            step = state["step"] + 1
            optimizer_metrics = optimizer.metrics or {}
            metrics = {
                "kind": "optimizer_step",
                "backend": "tinker",
                "step": step,
                "round": state["round"],
                "iteration": state["iteration"] + 1,
                "rollout_id": rollout_id,
                "loss": policy_metrics["loss"],
                "loss_definition": "token-mean clipped importance-weighted GRPO surrogate",
                "gradient_norm": optimizer_metrics.get(
                    "grad_norm",
                    optimizer_metrics.get(
                        "gradient_norm", optimizer_metrics.get("unclipped_grad_l2:mean")
                    ),
                ),
                "optimizer_metrics": optimizer_metrics,
                "service_loss_metrics": result.metrics,
                "reference_kl": kl_metrics,
                "minibatch_filtering": filtering,
                "mean_reward": sum(s.reward for s in prepared.signals)
                / len(prepared.signals),
                "zero_advantage_episodes": sum(
                    s.advantage == 0 for s in prepared.signals
                ),
                "groups": len(prepared.groups),
                "rollout_groups": pool_groups,
                "minibatch": state["iteration"] % minibatch_count + 1,
                "minibatches_per_pool": minibatch_count,
                "replay_epoch": state["iteration"] // minibatch_count + 1,
                "episodes": len(prepared.signals),
                "tokens": prepared.token_count,
                "seconds": time.perf_counter() - started,
                "backward_seconds": backward_seconds,
                "learning_rate_used": learning_rate,
                "policy_metrics": policy_metrics,
            }
            state.update(step=step, iteration=state["iteration"] + 1)
            sampler_dirty = True
            if state["iteration"] >= config.num_iterations * minibatch_count:
                state.update(pending=None, iteration=0, round=state["round"] + 1)
                memory_pool = None
            state["metrics"] = metrics
            if step % config.checkpoint_interval == 0:
                store.phase("checkpoint")
                checkpoint()
            store.commit_metrics(metrics)
            store.audit(metrics)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{metrics['loss']:.5f}",
                    reward=f"{metrics['mean_reward']:.3f}",
                    refresh=False,
                )
        checkpoint()
        store.phase("completed")
        return state
    except BaseException as exc:
        store.fail(exc)
        raise
