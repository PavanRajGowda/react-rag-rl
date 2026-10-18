"""Explicit H100 execution-mode pilot. No optimizer steps or retrieval calls."""

import json
import re
import statistics
import time
import traceback
from pathlib import Path

import modal

from grpo_training.modal_app import cache, image, runs

image = image.add_local_python_source("experiments")

app = modal.App("finance-rag-vllm-execution-pilot")


def difference(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Expected aligned nonempty token probabilities")
    values = sorted(abs(a - b) for a, b in zip(left, right, strict=True))
    return {
        "tokens": len(values),
        "mean_abs": statistics.mean(values),
        "max_abs": values[-1],
        "p95_abs": values[int(0.95 * (len(values) - 1))],
    }


def summarize_results(directory):
    """Read-only summary; token alignment and wall-clock populations explicit."""
    directory = Path(directory)
    results = {
        m: json.loads((directory / f"{m}.json").read_text())
        for m in ("eager", "graphs", "compiled")
    }
    sampled = json.loads((directory / "sampled_replay.json").read_text())
    log_path = directory / "modal.log"
    log_lines = log_path.read_text(encoding="utf-8-sig").splitlines()
    containers = {}
    for line in log_lines:
        match = re.search(r'^(ta-[A-Z0-9]+).*"mode": "(eager|graphs|compiled)"', line)
        if match:
            containers[match[1]] = match[2]
    graph_dispatch = {m: [] for m in results}
    for line in log_lines:
        fields = re.search(
            r"^(ta-[A-Z0-9]+).*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(FULL|NONE)\s*\|\s*(\d+)\s*\|",
            line,
        )
        if fields and fields[1] in containers:
            graph_dispatch[containers[fields[1]]].append(
                {
                    "unpadded_tokens": int(fields[2]),
                    "padded_tokens": int(fields[3]),
                    "runtime_mode": fields[5],
                    "count": int(fields[6]),
                }
            )
    baseline = results["eager"]
    rows = []
    for mode, result in results.items():
        if result.get("failed"):
            rows.append({"mode": mode, "failed": True, "error": result["error"]})
            continue
        timing = []
        for item, reference in zip(result["timings"], baseline["timings"], strict=True):
            assert (item["requests"], item["tokens_per_request"]) == (
                reference["requests"],
                reference["tokens_per_request"],
            )
            timing.append(
                {
                    **item,
                    "speedup_vs_eager": reference["median_seconds"]
                    / item["median_seconds"],
                }
            )
        greedy = []
        for reference, item in zip(baseline["greedy"], result["greedy"], strict=True):
            prefix = 0
            for a, b in zip(reference["ids"], item["ids"], strict=True):
                if a != b:
                    break
                prefix += 1
            greedy.append(
                {
                    "identical": reference["ids"] == item["ids"],
                    "common_prefix_tokens": prefix,
                    "total_tokens": len(item["ids"]),
                }
            )
        original = [x for row in result["fixed_target_engine_logprobs"] for x in row]
        rows.append(
            {
                "mode": mode,
                "timings": timing,
                "engine_initialization_seconds": result[
                    "engine_initialization_seconds"
                ],
                "greedy_vs_eager": greedy,
                "fixed_target_vs_trainer": result[
                    "fixed_target_sampler_trainer_difference"
                ],
                "actual_sampled_decode_vs_trainer": sampled[mode]["difference"],
                "adapter_identity_checks": sampled[mode].get("adapter_identity_checks"),
                "cycles": [
                    {
                        **{
                            k: c[k]
                            for k in (
                                "cycle",
                                "adapter_control",
                                "sleep_seconds",
                                "wake_seconds",
                                "sync_seconds",
                                "first_post_wake_generation_seconds",
                            )
                        },
                        "fixed_target_change_vs_initial": difference(
                            original,
                            [x for row in c["fixed_target_logprobs"] for x in row],
                        ),
                    }
                    for c in result["cycles"]
                ],
                "graph_metrics": [
                    m for m in result["metrics"] if "cudagraph" in m["name"]
                ],
                "graph_dispatch_log_tables": graph_dispatch[mode],
                "cache_config": result["resolved_cache"],
                "adapter_restored": result["adapter_restored"],
            }
        )
    return {
        "source": str(directory),
        "graph_dispatch_log_source": str(log_path),
        "scope": "Qwen3.5-4B BF16, vLLM 0.22.0, one shared H100, fixed saved prompts, max_num_seqs=16; one warmup then three repetitions; no retrieval or optimizer updates.",
        "limitations": "Forced-length decoding ignores EOS. One-token timings are a prefill/first-token proxy, not isolated profiling. Memory is sampled whole-GPU usage, not an instrumented peak. Four prompts, three sleep/wake/sync cycles, and no training/generalization test. Initializations reuse FlashInfer cache. This does not benchmark the production 80-trajectory batch.",
        "results": rows,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    volumes={"/cache": cache, "/runs": runs},
    timeout=2400,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def evaluate(payload, mode):
    import hashlib
    import shutil
    import subprocess
    from dataclasses import asdict

    kernel_cache = Path("/cache/execution-flashinfer")
    if kernel_cache.exists():
        shutil.copytree(kernel_cache, "/root/.cache/flashinfer", dirs_exist_ok=True)

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from vllm import SamplingParams

    from grpo_training.config import TrainingConfig
    from grpo_training.model import export_adapter, load_training_model
    from grpo_training.vllm_backend import create_vllm_backend

    torch.set_num_threads(1)
    config = TrainingConfig.from_dict(payload["config"])
    bundle = load_training_model(config, cache_dir="/cache/hub")
    checkpoint = Path("/runs/grpo-dapo-008-20groups-3turn-512") / payload["checkpoint"]
    path = checkpoint / "adapter/adapter_model.safetensors"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["adapter_sha256"]
    loaded = set_peft_model_state_dict(bundle.model, load_file(str(path)))
    assert not loaded.unexpected_keys
    original = {
        n: p.detach().clone()
        for n, p in bundle.model.named_parameters()
        if p.requires_grad
    }
    artifact = export_adapter(bundle, "/tmp/execution-pilot", 0)
    result = {
        "mode": mode,
        "vllm_version": "0.22.0",
        "precision": "bfloat16",
        "shared_trainer_gpu": True,
        "prefix_caching": False,
        "max_num_seqs": 16,
        "flashinfer_cache_restored": kernel_cache.exists(),
        "adapter_sha256": payload["adapter_sha256"],
        "timings": [],
        "cycles": [],
        "training_updates": 0,
    }

    def memory():
        # nvidia-smi includes both processes; torch counters alone miss vLLM.
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

    started = time.perf_counter()
    backend = create_vllm_backend(bundle, execution_mode=mode, diagnostic_metrics=True)
    result["engine_initialization_seconds"] = time.perf_counter() - started
    backend.synchronize(artifact)
    engine = backend.engine
    resolved = engine.llm_engine.vllm_config
    result["resolved_compilation"] = str(resolved.compilation_config)
    result["resolved_cache"] = str(resolved.cache_config)
    result["model_inspection"] = repr(engine)
    prompts = [r["prompt_ids"] for r in payload["rows"]]

    def generate(count, tokens, temperature=1.0):
        params = [
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                seed=1729 + i,
                max_tokens=tokens,
                min_tokens=tokens,
                ignore_eos=True,
                logprobs=0,
            )
            for i in range(count)
        ]
        start = time.perf_counter()
        outputs = engine.generate(
            [{"prompt_token_ids": prompts[i % len(prompts)]} for i in range(count)],
            params,
            lora_request=backend._lora,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        saved = []
        for item in outputs:
            output = item.outputs[0]
            ids = list(output.token_ids)
            saved.append(
                {
                    "ids": ids,
                    "logprobs": [
                        float(p[t].logprob)
                        for t, p in zip(ids, output.logprobs, strict=True)
                    ],
                    "text": output.text,
                }
            )
        return elapsed, saved

    for count in (4, 16):
        for tokens in (1, 128):
            generate(count, tokens)  # Warm the exact shape, excluded below.
            times = []
            for repeat in range(3):
                elapsed, outputs = generate(count, tokens)
                times.append(elapsed)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "requests": count,
                            "tokens": tokens,
                            "repeat": repeat,
                            "seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
            result["timings"].append(
                {
                    "requests": count,
                    "tokens_per_request": tokens,
                    "seconds": times,
                    "median_seconds": statistics.median(times),
                    "output_tokens_per_second": count
                    * tokens
                    / statistics.median(times),
                    "gpu_memory_used_total_mib": memory(),
                }
            )
    _, result["greedy"] = generate(4, 64, temperature=0)

    # Identical fixed targets in every mode, independent of sampled divergence.
    def fixed_scores():
        outputs = engine.generate(
            [
                {"prompt_token_ids": r["prompt_ids"] + r["output_ids"][:32]}
                for r in payload["rows"]
            ],
            SamplingParams(
                temperature=1.0, top_p=1.0, max_tokens=1, seed=42, prompt_logprobs=0
            ),
            lora_request=backend._lora,
            use_tqdm=False,
        )
        return [
            [
                float(item.prompt_logprobs[len(row["prompt_ids"]) + i][token].logprob)
                for i, token in enumerate(row["output_ids"][:32])
            ]
            for item, row in zip(outputs, payload["rows"], strict=True)
        ]

    result["fixed_target_engine_logprobs"] = fixed_scores()
    _, sampled = generate(4, 64)
    result["sampled"] = sampled
    # Three full level-1 sleep/wake + production adapter replacement cycles.
    # Zeroing LoRA B is a temporary diagnostic control, not a trained policy.
    for cycle in range(3):
        start = time.perf_counter()
        backend.sleep()
        sleep_seconds = time.perf_counter() - start
        sleeping_memory = memory()
        bundle.model.eval()
        with torch.no_grad():
            for name, p in bundle.model.named_parameters():
                if name in original:
                    p.copy_(original[name])
                    if cycle == 0 and ".lora_B." in name:
                        p.zero_()
            replay = []
            for row in payload["rows"]:
                target = row["output_ids"][:32]
                ids = torch.tensor([row["prompt_ids"] + target], device="cuda")
                logits = (
                    bundle.model(input_ids=ids, logits_to_keep=len(target) + 1)
                    .logits[:, :-1]
                    .float()
                )
                replay.append(
                    logits.log_softmax(-1)
                    .gather(-1, torch.tensor([target], device="cuda").unsqueeze(-1))
                    .flatten()
                    .tolist()
                )
        if cycle == 1:
            result["fixed_target_trainer_logprobs"] = replay
            result["fixed_target_sampler_trainer_difference"] = difference(
                [x for row in result["fixed_target_engine_logprobs"] for x in row],
                [x for row in replay for x in row],
            )
        updated = export_adapter(bundle, "/tmp/execution-pilot", cycle + 1)
        start = time.perf_counter()
        backend.wake_up()
        wake_seconds = time.perf_counter() - start
        start = time.perf_counter()
        backend.synchronize(updated)
        sync_seconds = time.perf_counter() - start
        seconds, outputs = generate(4, 64, temperature=0)
        scores = fixed_scores()
        result["cycles"].append(
            {
                "cycle": cycle,
                "adapter_control": "zero_B" if cycle == 0 else "original",
                "policy": asdict(updated.policy),
                "sleep_seconds": sleep_seconds,
                "wake_seconds": wake_seconds,
                "sync_seconds": sync_seconds,
                "first_post_wake_generation_seconds": seconds,
                "sleeping_gpu_memory_used_total_mib": sleeping_memory,
                "greedy": outputs,
                "fixed_target_logprobs": scores,
                "sampler_trainer_difference": difference(
                    [x for row in scores for x in row],
                    [x for row in replay for x in row],
                ),
            }
        )
        print(json.dumps({"mode": mode, "cycle_complete": cycle}), flush=True)
    result["metrics"] = [
        {"type": type(m).__name__, **vars(m)} for m in engine.get_metrics()
    ]
    result["adapter_restored"] = all(
        torch.equal(p, original[n])
        for n, p in bundle.model.named_parameters()
        if n in original
    )
    assert result["adapter_restored"]
    backend.sleep()
    shutil.copytree("/root/.cache/flashinfer", kernel_cache, dirs_exist_ok=True)
    cache.commit()
    return json.loads(json.dumps(result, default=str))


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/cache": cache, "/runs": runs},
    timeout=600,
    retries=0,
    max_containers=1,
    single_use_containers=True,
)
def score_samples(payload, results):
    """Same sampled tokens: actual decode logprobs versus singleton HF replay."""
    import hashlib

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from grpo_training.config import TrainingConfig
    from grpo_training.model import load_training_model

    torch.set_num_threads(1)
    bundle = load_training_model(
        TrainingConfig.from_dict(payload["config"]), cache_dir="/cache/hub"
    )
    path = (
        Path("/runs/grpo-dapo-008-20groups-3turn-512")
        / payload["checkpoint"]
        / "adapter/adapter_model.safetensors"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["adapter_sha256"]
    assert not set_peft_model_state_dict(
        bundle.model, load_file(str(path))
    ).unexpected_keys
    bundle.model.eval()
    output = {}
    for mode, result in results.items():
        engine, replay = [], []
        with torch.no_grad():
            for row, sampled in zip(payload["rows"], result["sampled"], strict=True):
                target = sampled["ids"]
                ids = torch.tensor([row["prompt_ids"] + target], device="cuda")
                logits = (
                    bundle.model(input_ids=ids, logits_to_keep=len(target) + 1)
                    .logits[:, :-1]
                    .float()
                )
                replay.extend(
                    logits.log_softmax(-1)
                    .gather(-1, torch.tensor([target], device="cuda").unsqueeze(-1))
                    .flatten()
                    .tolist()
                )
                engine.extend(sampled["logprobs"])
        output[mode] = {
            "difference": difference(engine, replay),
            "sampler_logprobs": engine,
            "trainer_logprobs": replay,
        }
    original_b = {
        n: p.detach().clone()
        for n, p in bundle.model.named_parameters()
        if ".lora_B." in n
    }
    fixed = {}
    with torch.no_grad():
        for control in ("original", "zero_B"):
            if control == "zero_B":
                for name, p in bundle.model.named_parameters():
                    if name in original_b:
                        p.zero_()
            values = []
            for row in payload["rows"]:
                target = row["output_ids"][:32]
                ids = torch.tensor([row["prompt_ids"] + target], device="cuda")
                logits = (
                    bundle.model(input_ids=ids, logits_to_keep=len(target) + 1)
                    .logits[:, :-1]
                    .float()
                )
                values.extend(
                    logits.log_softmax(-1)
                    .gather(-1, torch.tensor([target], device="cuda").unsqueeze(-1))
                    .flatten()
                    .tolist()
                )
            fixed[control] = values
        for name, p in bundle.model.named_parameters():
            if name in original_b:
                p.copy_(original_b[name])
    for mode, result in results.items():
        output[mode]["adapter_identity_checks"] = [
            {
                "cycle": c["cycle"],
                "control": c["adapter_control"],
                "against_original_trainer": difference(
                    [x for row in c["fixed_target_logprobs"] for x in row],
                    fixed["original"],
                ),
                "against_zero_B_trainer": difference(
                    [x for row in c["fixed_target_logprobs"] for x in row],
                    fixed["zero_B"],
                ),
            }
            for c in result["cycles"]
        ]
    return output


@app.local_entrypoint()
def main(
    output: str = "data/grpo/execution-pilot-001",
    modes: str = "eager,graphs,compiled",
    score_only: bool = False,
):
    selected = modes.split(",")
    if not selected or any(m not in ("eager", "graphs", "compiled") for m in selected):
        raise ValueError("Expected eager,graphs,compiled")
    payload = json.loads(
        Path(
            "data/grpo/replay-investigation-optimized-001/cross_precision.json"
        ).read_text()
    )
    identity = json.loads(
        Path(
            "data/grpo/grpo-dapo-008-20groups-3turn-512/observability/validation/00000020/identity.json"
        ).read_text()
    )
    payload["config"] = identity["config"]
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    if score_only:
        results = {m: json.loads((target / f"{m}.json").read_text()) for m in selected}
        results = {m: r for m, r in results.items() if not r.get("failed")}
        scores = score_samples.remote(payload, results)
        (target / "sampled_replay.json").write_text(
            json.dumps(scores, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({m: r["difference"] for m, r in scores.items()}))
        return
    for mode in selected:
        try:
            result = evaluate.remote(payload, mode)
        except Exception:  # noqa: BLE001 - Preserve failure and test the next isolated mode.
            result = {"mode": mode, "failed": True, "error": traceback.format_exc()}
        (target / f"{mode}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "failed": result.get("failed", False),
                    "saved": str(target / f"{mode}.json"),
                }
            )
        )
