"""Same-process precision/kernel controls. Only called by the paid diagnostic."""


def cross_precision(model, trainer, rows, logps, gradient, compare_gradient, result):
    import gc
    import statistics
    from importlib.metadata import version

    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    assert qwen.is_fast_path_available, "Optimized Qwen kernels were not selected"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result["precision_control"] = (
        "TF32 disabled for matmul and convolution; FP32 fallback reference"
    )
    result["kernel_versions"] = {
        name: version(name)
        for name in ("flash-linear-attention", "fla-core", "causal-conv1d")
    }
    linear = [m for m in model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]
    assert linear and all(m.causal_conv1d_fn is not None for m in linear)
    original = [
        (
            m,
            m.norm,
            m.causal_conv1d_fn,
            m.causal_conv1d_update,
            m.chunk_gated_delta_rule,
            m.recurrent_gated_delta_rule,
        )
        for m in linear
    ]
    dtypes = {n: p.dtype for n, p in model.named_parameters()}
    buffer_dtypes = {n: p.dtype for n, p in model.named_buffers()}
    amp = trainer.accelerator.native_amp
    result["selected_kernels"] = {
        "linear_layers": len(linear),
        "convolution": str(linear[0].causal_conv1d_fn),
        "linear_attention": str(linear[0].chunk_gated_delta_rule),
        "normalization": str(type(linear[0].norm)),
    }

    def mode(fp32, fast):
        for m, norm, conv, update, chunk, recurrent in original:
            m.norm = norm
            m.causal_conv1d_fn = conv
            m.causal_conv1d_update = update
            m.chunk_gated_delta_rule = chunk
            m.recurrent_gated_delta_rule = recurrent
        if fp32:
            model.float()
        else:
            for name, parameter in model.named_parameters():
                parameter.data = parameter.data.to(dtypes[name])
            for name, buffer in model.named_buffers():
                buffer.data = buffer.data.to(buffer_dtypes[name])
        trainer.accelerator.native_amp = False if fp32 else amp
        if not fast:
            for m, norm, *_ in original:
                replacement = qwen.Qwen3_5RMSNormGated(
                    m.head_v_dim, eps=m.layer_norm_epsilon
                ).to(device=norm.weight.device, dtype=norm.weight.dtype)
                replacement.weight.data.copy_(norm.weight.data)
                replacement.requires_grad_(False)
                m.norm = replacement
                m.causal_conv1d_fn = None
                m.causal_conv1d_update = qwen.torch_causal_conv1d_update
                m.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
                m.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    # Preserve each sequence's advantage even when length bucketing changes order.
    for i, row in enumerate(rows):
        row["advantage"] = 1.0 if i == 0 else -0.5
    mode(True, False)
    model.eval()
    with torch.no_grad():
        old = [logps([r])[0] for r in rows]
    reference, timing = gradient(rows, old, 1)
    compare_gradient("fp32_fallback_sequential_reference", reference, reference, timing)
    result["old_probability_convention"] = (
        "One common FP32 sequential reference for every loss, precision and batching variant"
    )
    result["cross_precision_forward"] = []
    result["repeated_timings"] = []
    for fp32, fast, name in [
        (True, False, "fp32_fallback"),
        (False, False, "bf16_fallback"),
        (False, True, "bf16_optimized"),
        (False, True, "bf16_optimized_head_fp32"),
    ]:
        mode(fp32, fast)
        from trl.trainer.utils import add_fused_lm_head

        add_fused_lm_head(
            model.get_base_model(),
            temperature=trainer.temperature,
            cast_lm_head_to_fp32=name.endswith("head_fp32"),
            outputs=("log_probs", "entropy"),
        )
        model.eval()
        with torch.no_grad():
            for batching, actual in [
                ("sequential", [logps([r])[0] for r in rows]),
                ("batched", logps(rows)),
            ]:
                ref = torch.cat(old)
                differences = (torch.cat(actual) - ref).abs()
                entry = {
                    "backend": name,
                    "batching": batching,
                    "mean_abs": differences.mean().item(),
                    "max_abs": differences.max().item(),
                    "quantiles": {
                        str(q): torch.quantile(differences, q).item()
                        for q in (0.5, 0.9, 0.95, 0.99)
                    },
                    "by_reference_logprob": {},
                }
                for lower, upper in [
                    (-float("inf"), -10),
                    (-10, -5),
                    (-5, -1),
                    (-1, 0.00001),
                ]:
                    mask = (ref >= lower) & (ref < upper)
                    values = differences[mask]
                    entry["by_reference_logprob"][f"[{lower},{upper})"] = {
                        "tokens": len(values),
                        "mean_abs": values.mean().item() if len(values) else None,
                        "max_abs": values.max().item() if len(values) else None,
                    }
                result["cross_precision_forward"].append(entry)
                print(entry, flush=True)
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
        variants = [
            ("sequential", rows, old, 1),
            ("batch2", rows, old, 2),
            ("batch4", rows, old, 4),
            ("bucket2", [rows[i] for i in order], [old[i] for i in order], 2),
        ]
        backend_reference = None
        for variant, selected, values, microbatch in variants:
            warm, _ = gradient(selected, values, microbatch)
            del warm
            timings = []
            for repeat in range(3):
                actual, timing = gradient(selected, values, microbatch)
                timings.append(timing["seconds"])
                compare_gradient(
                    f"{name}_{variant}_repeat{repeat}", reference, actual, timing
                )
                if backend_reference is None:
                    backend_reference = actual
                norm = sum(
                    x.double().square().sum().item() for x in backend_reference.values()
                )
                other = sum(x.double().square().sum().item() for x in actual.values())
                dot = sum(
                    (actual[n].double() * backend_reference[n].double()).sum().item()
                    for n in actual
                )
                diff = sum(
                    (actual[n].double() - backend_reference[n].double())
                    .square()
                    .sum()
                    .item()
                    for n in actual
                )
                result["gradients"][-1].update(
                    same_backend_sequential_cosine=dot / (norm * other) ** 0.5,
                    same_backend_relative_l2_error=(diff / norm) ** 0.5,
                )
                del actual
            result["repeated_timings"].append(
                {
                    "backend": name,
                    "variant": variant,
                    "seconds": timings,
                    "median_seconds": statistics.median(timings),
                }
            )
    mode(False, True)
