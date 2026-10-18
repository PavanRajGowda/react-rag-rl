"""Explicit paid, read-only replay diagnostics; never performs an optimizer step."""

import json
from pathlib import Path

import modal

from grpo_training.modal_app import cache, image, runs

image = image.add_local_python_source("experiments")

app = modal.App("finance-rag-replay-diagnostic")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/cache": cache, "/runs": runs},
    timeout=2400,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def investigate(payload, stage="small"):
    import gc
    import hashlib
    import inspect
    import time

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    # Gradient comparisons are many small CPU reductions, not CPU training.
    # Avoid oversized host thread pools stalling the diagnostic between GPU calls.
    torch.set_num_threads(1)

    from grpo_training.config import TrainingConfig
    from grpo_training.data import Question
    from grpo_training.model import load_training_model
    from grpo_training.trainer import collate_segments, create_replay_trainer

    config = TrainingConfig.from_dict(payload["config"])
    bundle = load_training_model(config, cache_dir="/cache/hub")
    root = Path("/runs/grpo-dapo-008-20groups-3turn-512")
    checkpoint = root / json.loads((root / "latest.json").read_text())["checkpoint"]
    path = checkpoint / "adapter/adapter_model.safetensors"
    adapter_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert adapter_hash == manifest["files"]["adapter/adapter_model.safetensors"]
    state = load_file(str(path), device="cpu")
    loaded = set_peft_model_state_dict(bundle.model, state)
    assert not loaded.unexpected_keys
    del state
    trainer = create_replay_trainer(
        bundle,
        [
            Question(
                "diagnostic", "Replay diagnostic", "train", 0, config.dataset_revision
            )
        ],
        "/tmp/replay-diagnostic-trainer",
    )
    model = trainer.model
    if stage == "fp32_representative":
        model.float()
        trainer.accelerator.native_amp = False
    original_weights = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    rows = [
        {
            "input_ids": tuple(r["prompt_ids"] + r["output_ids"]),
            "target_ids": tuple(r["output_ids"]),
        }
        for r in payload["rows"]
    ]
    device = trainer.accelerator.device
    pad = bundle.tokenizer.pad_token_id
    result = {
        "stage": stage,
        "adapter_sha256": adapter_hash,
        "checkpoint": checkpoint.name,
        "rows": payload["rows"],
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "trainable_dtypes": sorted(
            {str(p.dtype) for p in model.parameters() if p.requires_grad}
        ),
        "forward": [],
        "gradients": [],
    }
    if stage in ("representative", "fp32_representative"):
        trainer.create_optimizer()
        optimizer_state = torch.load(
            checkpoint / "training.pt", map_location="cpu", weights_only=False
        )["optimizer"]
        names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
        optimizer_groups = [
            {
                **{k: v for k, v in group.items() if k != "params"},
                "names": [names[id(p)] for p in group["params"]],
            }
            for group in trainer.optimizer.param_groups
        ]

        def scratch_update(grads):
            import copy

            parameters = {
                n: torch.nn.Parameter(v.clone()) for n, v in original_weights.items()
            }
            groups = [
                {
                    **{k: v for k, v in group.items() if k != "names"},
                    "params": [parameters[n] for n in group["names"]],
                }
                for group in optimizer_groups
            ]
            optimizer = torch.optim.AdamW(groups)
            optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            for n, p in parameters.items():
                p.grad = grads[n].clone()
            torch.nn.utils.clip_grad_norm_(
                list(parameters.values()), config.max_grad_norm
            )
            optimizer.step()
            return {n: p.detach() - original_weights[n] for n, p in parameters.items()}

        result["scratch_optimizer"] = (
            "CPU clones with saved step-20 AdamW moments and global gradient clipping; original weights unchanged"
        )

    def logps(selected, explicit_positions=False, extra_left_pad=0):
        inputs = collate_segments(selected, pad, device)
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        if extra_left_pad:
            ids = torch.nn.functional.pad(ids, (extra_left_pad, 0), value=pad)
            mask = torch.nn.functional.pad(mask, (extra_left_pad, 0), value=0)

        def positions(module, args, kwargs):
            attention = kwargs["attention_mask"]
            kwargs["position_ids"] = (attention.long().cumsum(-1) - 1).masked_fill(
                attention == 0, 0
            )
            return args, kwargs

        handle = (
            model.register_forward_pre_hook(positions, with_kwargs=True)
            if explicit_positions
            else None
        )
        try:
            values, _, _ = trainer._get_per_token_logps_and_entropies(
                model, ids, mask, inputs["completion_ids"].shape[1]
            )
            return [
                v[: len(r["target_ids"])].detach().float().cpu()
                for r, v in zip(selected, values)
            ]
        finally:
            if handle:
                handle.remove()

    def compare_values(name, reference, actual):
        deltas = torch.cat([(a - b).abs() for a, b in zip(reference, actual)])
        entry = {
            "name": name,
            "mean_abs_logprob_delta": deltas.mean().item(),
            "max_abs_logprob_delta": deltas.max().item(),
        }
        result["forward"].append(entry)
        print(json.dumps(entry), flush=True)

    model.eval()
    with torch.no_grad():
        baseline = [logps([r])[0] for r in rows]
        compare_values("same_shape_repeat", baseline, [logps([r])[0] for r in rows])
        compare_values("heterogeneous_batch", baseline, logps(rows))
        compare_values(
            "heterogeneous_batch_explicit_positions", baseline, logps(rows, True)
        )
        compare_values(
            "duplicate_same_shape",
            [baseline[0], baseline[0]],
            logps([rows[0], rows[0]]),
        )
        compare_values(
            "single_left_padding_32", [baseline[0]], logps([rows[0]], extra_left_pad=32)
        )
        compare_values(
            "duplicate_left_padding_32",
            [baseline[0], baseline[0]],
            logps([rows[0], rows[0]], extra_left_pad=32),
        )

    def gradient(selected, old, microbatch, training=True):
        model.zero_grad(set_to_none=True)
        model.train(training)
        torch.manual_seed(42)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        total = 0.0
        count = sum(len(r["target_ids"]) for r in selected)
        for start in range(0, len(selected), microbatch):
            part = selected[start : start + microbatch]
            inputs = collate_segments(part, pad, device)
            shape = inputs["completion_ids"].shape
            olds = torch.zeros(shape, device=device)
            advantages = torch.zeros_like(olds)
            for i, row in enumerate(part):
                length = len(row["target_ids"])
                olds[i, :length] = old[start + i].to(device)
                advantages[i, :length] = row.get(
                    "advantage", 1.0 if start + i == 0 else -0.5
                )
            inputs.update(
                old_per_token_logps=olds,
                advantages=advantages,
                num_items_in_batch=torch.tensor(count, device=device),
            )
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(model, inputs)
            trainer.accelerator.backward(loss)
            total += loss.detach().float().item()
        torch.cuda.synchronize()
        grads = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        return grads, {
            "loss": total,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }

    def compare_gradient(name, ref, actual, timing):
        dot = sum((ref[n].double() * actual[n].double()).sum().item() for n in ref)
        norm = sum(ref[n].double().square().sum().item() for n in ref)
        other = sum(actual[n].double().square().sum().item() for n in ref)
        diff = sum(
            (ref[n].double() - actual[n].double()).square().sum().item() for n in ref
        )
        entry = {
            "name": name,
            **timing,
            "gradient_cosine": dot / (norm * other) ** 0.5,
            "relative_l2_gradient_error": (diff / norm) ** 0.5,
            "gradient_norm": other**0.5,
        }
        result["gradients"].append(entry)
        print(json.dumps(entry), flush=True)

    if stage == "cross_precision":
        from experiments.grpo.replay_cross_precision import cross_precision

        cross_precision(model, trainer, rows, logps, gradient, compare_gradient, result)
        result["weights_unchanged"] = all(
            torch.equal(
                original_weights[n], p.detach().cpu().to(original_weights[n].dtype)
            )
            for n, p in model.named_parameters()
            if n in original_weights
        )
        assert result["weights_unchanged"]
        return result

    reference, timing = gradient(rows, baseline, 1)
    compare_gradient("sequential_reference", reference, reference, timing)
    if stage in ("representative", "fp32_representative"):
        update_reference = scratch_update(reference)
    cases = [("same_shape_repeat", 1), ("heterogeneous_batch", 2)]
    if stage in ("representative", "fp32_representative"):
        cases += [("heterogeneous_batch_4", 4), ("heterogeneous_batch_4_repeat", 4)]
    for name, micro in cases:
        actual, timing = gradient(rows, baseline, micro)
        compare_gradient(name, reference, actual, timing)
        if stage in ("representative", "fp32_representative"):
            updates = scratch_update(actual)
            norm = sum(
                x.double().square().sum().item() for x in update_reference.values()
            )
            diff = sum(
                (updates[n].double() - update_reference[n].double())
                .square()
                .sum()
                .item()
                for n in updates
            )
            dot = sum(
                (updates[n].double() * update_reference[n].double()).sum().item()
                for n in updates
            )
            other = sum(x.double().square().sum().item() for x in updates.values())
            result["gradients"][-1].update(
                scratch_adam_relative_update_error=(diff / norm) ** 0.5,
                scratch_adam_update_cosine=dot / (norm * other) ** 0.5,
            )
        del actual
    model.gradient_checkpointing_disable()
    for name, micro in (
        []
        if stage != "small"
        else [
            ("checkpointing_off_sequential", 1),
            ("checkpointing_off_batch", 2),
        ]
    ):
        actual, timing = gradient(rows, baseline, micro)
        compare_gradient(name, reference, actual, timing)
        del actual
    # Higher-precision control uses the same two small sequences and saved weights.
    del reference
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    if stage == "small":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

        for module in model.modules():
            if isinstance(module, qwen.Qwen3_5GatedDeltaNet):
                module.causal_conv1d_fn = None
                module.causal_conv1d_update = qwen.torch_causal_conv1d_update
                module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
                module.recurrent_gated_delta_rule = (
                    qwen.torch_recurrent_gated_delta_rule
                )
                norm = qwen.Qwen3_5RMSNormGated(
                    module.head_v_dim, eps=module.layer_norm_epsilon
                ).to(module.norm.weight.device)
                norm.weight.data.copy_(module.norm.weight.data)
                norm.requires_grad_(False)
                module.norm = norm
        model.float()
        trainer.accelerator.native_amp = False
        model.eval()
        with torch.no_grad():
            fp = [logps([r])[0] for r in rows]
            compare_values("fp32_heterogeneous_batch", fp, logps(rows))
            compare_values("fp32_batch_explicit_positions", fp, logps(rows, True))
        ref, timing = gradient(rows, fp, 1, training=False)
        compare_gradient("fp32_sequential", ref, ref, timing)
        actual, timing = gradient(rows, fp, 2, training=False)
        compare_gradient("fp32_batch", ref, actual, timing)
    result["weights_unchanged"] = all(
        torch.equal(original_weights[n], p.detach().cpu().to(original_weights[n].dtype))
        for n, p in model.named_parameters()
        if n in original_weights
    )
    result["source"] = inspect.getsource(trainer._get_per_token_logps_and_entropies)
    assert result["weights_unchanged"]
    return result


@app.local_entrypoint()
def main(output: str = "data/grpo/replay-investigation-001", stage: str = "small"):
    if stage not in ("small", "representative", "fp32_representative", "cross_precision"):
        raise ValueError("Unknown replay diagnostic stage")
    source = Path(
        "data/grpo/grpo-dapo-008-20groups-3turn-512/observability/validation/00000020"
    )
    identity = json.loads((source / "identity.json").read_text(encoding="utf-8"))
    names = (
        ["finqa_dev_159.json"]
        if stage == "small"
        else ["finqa_dev_159.json", "finqa_dev_173.json"]
    )
    records = [
        r
        for name in names
        for r in json.loads((source / name).read_text(encoding="utf-8"))["episode"][
            "generations"
        ][:2]
    ]
    rows = []
    for i, record in enumerate(records):
        prompt = record["request"]["prompt_ids"]
        if stage == "small":
            prompt = prompt[-(96 if i == 0 else 160) :]
        rows.append(
            {
                "prompt_ids": prompt,
                "output_ids": record["completion_ids"][: 24 if i == 0 else 32]
                if stage == "small"
                else record["completion_ids"],
            }
        )
    payload = {"config": identity["config"], "rows": rows}
    result = investigate.remote(payload, stage)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{stage}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "source")}, indent=2
        )
    )
