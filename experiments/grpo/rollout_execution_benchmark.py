"""80-trajectory execution comparison and four-step compiled GRPO smoke."""

import json
import shutil
import statistics
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import modal

from experiments.grpo.cache_benchmark import MeasuredBackend
from grpo_training.modal_app import (
    ROOT,
    cache,
    image,
    runs,
    search_transport,
)

image = image.add_local_python_source("experiments")

app = modal.App("finance-rag-vllm-80-pilot")


@app.function(image=image, timeout=120, retries=0)
def preflight(config_values):
    from grpo_training.config import TrainingConfig
    from grpo_training.environment import checked_response

    config = TrainingConfig.from_dict(config_values)
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    response = search_transport(service)(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    checked_response(response, config)
    if response.get("status") != "ok" or not response["results"]:
        raise RuntimeError("Search preflight failed before GPU allocation")


def kernel_cache(restore):
    for name in ("flashinfer", "vllm"):
        local, shared = Path(f"/root/.cache/{name}"), Path(f"/cache/execution-{name}")
        source, target = (shared, local) if restore else (local, shared)
        if source.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)


GPU = {
    "image": image,
    "gpu": "H100",
    "cpu": 8,
    "memory": 65536,
    "volumes": {"/cache": cache, "/runs": runs},
    "timeout": 3600,
    "retries": 0,
    "min_containers": 0,
    "max_containers": 1,
    "single_use_containers": True,
}


@app.function(**GPU)
def benchmark(payload, mode):
    import hashlib
    import subprocess

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from vllm import SamplingParams

    from grpo_training.collection import collect_groups
    from grpo_training.config import TrainingConfig
    from grpo_training.data import Question
    from grpo_training.model import export_adapter, load_training_model
    from grpo_training.vllm_backend import create_vllm_backend

    torch.set_num_threads(1)
    kernel_cache(True)
    config = TrainingConfig.from_dict(payload["config"])
    questions = tuple(Question(**q) for q in payload["questions"])
    bundle = load_training_model(config, cache_dir="/cache/hub")
    checkpoint = Path("/runs/grpo-dapo-008-20groups-3turn-512") / payload["checkpoint"]
    path = checkpoint / "adapter/adapter_model.safetensors"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["adapter_sha256"]
    assert not set_peft_model_state_dict(
        bundle.model, load_file(str(path))
    ).unexpected_keys
    artifact = export_adapter(bundle, "/tmp/rollout-execution-pilot", 0)
    started = time.perf_counter()
    backend = create_vllm_backend(
        bundle, execution_mode=mode, diagnostic_metrics=True, max_num_seqs=80
    )
    initialization = time.perf_counter() - started
    backend.synchronize(artifact)
    first_requests = []

    class Measured(MeasuredBackend):
        warmup_seconds = 0.0

        def generate(self, requests):
            if not first_requests:
                first_requests.extend(requests)
                start = time.perf_counter()
                backend.engine.generate(
                    [{"prompt_token_ids": list(r.prompt_ids)} for r in requests],
                    [
                        SamplingParams(
                            max_tokens=8, temperature=1, seed=r.seed, ignore_eos=True
                        )
                        for r in requests
                    ],
                    lora_request=backend._lora,
                    use_tqdm=False,
                )
                self.warmup_seconds = time.perf_counter() - start
            return super().generate(requests)

    measured = Measured(backend)
    events = []
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    started = time.perf_counter()
    groups = collect_groups(
        questions,
        config,
        artifact.policy,
        measured,
        bundle.tokenizer,
        search_transport(service),
        0,
        search_workers=4,
        audit=events.append,
    )
    rollout_seconds = time.perf_counter() - started - measured.warmup_seconds
    assert len(first_requests) == 80
    result = {
        "mode": mode,
        "config": config.to_dict(),
        "max_num_seqs": 80,
        "initialization_seconds": initialization,
        "warmup_seconds": measured.warmup_seconds,
        "rollout_wall_seconds_excluding_warmup": rollout_seconds,
        "generation_waves": measured.waves,
        "groups": [[e.to_dict() for e in g.episodes] for g in groups],
        "audit": events,
        "resolved_compilation": str(
            backend.engine.llm_engine.vllm_config.compilation_config
        ),
        "resolved_cache": str(backend.engine.llm_engine.vllm_config.cache_config),
        "fixed_length": [],
    }
    # Exact same request membership, prompts, seeds and output length in both arms.
    for repeat in range(4):
        started = time.perf_counter()
        outputs = backend.engine.generate(
            [{"prompt_token_ids": list(r.prompt_ids)} for r in first_requests],
            [
                SamplingParams(
                    temperature=1,
                    top_p=1,
                    seed=r.seed,
                    max_tokens=512,
                    min_tokens=512,
                    ignore_eos=True,
                    logprobs=0,
                )
                for r in first_requests
            ],
            lora_request=backend._lora,
            use_tqdm=False,
        )
        seconds = time.perf_counter() - started
        count = sum(len(o.outputs[0].token_ids) for o in outputs)
        assert count == 80 * 512
        item = {
            "repeat": repeat,
            "warmup": repeat == 0,
            "seconds": seconds,
            "output_tokens": count,
        }
        result["fixed_length"].append(item)
        print(json.dumps({"mode": mode, "fixed_length": item}), flush=True)
    result["fixed_length_median_seconds"] = statistics.median(
        i["seconds"] for i in result["fixed_length"] if not i["warmup"]
    )
    result["sampled_gpu_memory_mib"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    result["metrics"] = [
        {"type": type(m).__name__, **vars(m)} for m in backend.engine.get_metrics()
    ]
    backend.sleep()
    kernel_cache(False)
    cache.commit()
    return json.loads(json.dumps(result, default=str))


@app.function(**GPU)
def smoke(config_values, train_values, run_id):
    import torch

    from grpo_training.checkpoints import RunStore
    from grpo_training.config import TrainingConfig
    from grpo_training.data import PreparedSplit, Question, RewardLabel
    from grpo_training.model import load_training_model
    from grpo_training.runner import initialize_optimizer, run_identity, run_training
    from grpo_training.trainer import create_replay_trainer
    from grpo_training.vllm_backend import create_vllm_backend

    torch.set_num_threads(1)
    kernel_cache(True)
    config = TrainingConfig.from_dict(config_values)
    train = PreparedSplit(
        "train",
        tuple(Question(**q) for q in train_values["questions"]),
        tuple(RewardLabel(**v) for v in train_values["labels"]),
        tuple(train_values["skipped_blank_ids"]),
        train_values["selected_source_count"],
    )
    store = RunStore("/runs", run_id, run_identity(config, train), durable=runs.commit)
    (store.root / "execution.json").write_text(
        json.dumps(
            {
                "mode": "compiled",
                "max_num_seqs": 80,
                "validation": None,
                "intent": "Four-step execution smoke, fresh pretrained model and full training pool; not an effectiveness experiment",
            },
            indent=2,
        )
    )
    try:
        bundle = load_training_model(config, cache_dir="/cache/hub")
        trainer = create_replay_trainer(bundle, train.questions, store.root / "trainer")
        initialize_optimizer(trainer, config)
        backend = create_vllm_backend(
            bundle, execution_mode="compiled", diagnostic_metrics=True, max_num_seqs=80
        )
        service = modal.Cls.from_name("finance-rag-search", "SearchService")()
        result = run_training(
            trainer, bundle, backend, train, search_transport(service), store
        )
        result["execution_mode"] = "compiled"
        return result
    finally:
        kernel_cache(False)
        cache.commit()
        runs.commit()


@app.local_entrypoint()
def main(output: str = "data/grpo/execution-pilot-80-001", stage: str = "benchmark"):
    from grpo_training.config import TrainingConfig
    from grpo_training.data import load_local_splits, sample_training_questions
    from grpo_training.rewards import episode_reward

    if stage not in ("benchmark", "smoke"):
        raise ValueError("Expected benchmark or smoke")
    config = TrainingConfig.from_dict(
        json.loads(
            (
                ROOT / "grpo_training/configs/dapo-20-steps-20groups-3turn-512.json"
            ).read_text()
        )
    )
    config = replace(config, max_steps=4)
    train, _, _ = load_local_splits(ROOT / "data/finqa", config)
    train = sample_training_questions(train, len(train.questions), 42)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "intent.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "training_pool": len(train.questions),
                "stage": stage,
            },
            indent=2,
        )
    )
    preflight.remote(config.to_dict())
    if stage == "smoke":
        result = smoke.remote(
            config.to_dict(), asdict(train), "grpo-compiled-80-smoke-001"
        )
        (target / "smoke.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    previous = json.loads(
        Path(
            "data/grpo/replay-investigation-optimized-001/cross_precision.json"
        ).read_text()
    )
    payload = {
        "config": config.to_dict(),
        "questions": [asdict(q) for q in train.questions[:20]],
        "checkpoint": previous["checkpoint"],
        "adapter_sha256": previous["adapter_sha256"],
    }
    labels = {v.id: v for v in train.labels}
    for mode in ("eager", "compiled"):
        try:
            result = benchmark.remote(payload, mode)
            episodes = [e for g in result["groups"] for e in g]
            rewards = [
                episode_reward(
                    e["question"]["id"],
                    e["status"],
                    e["answer"],
                    labels[e["question"]["id"]],
                ).reward
                for e in episodes
            ]
            result["number_match"] = (
                sum(rewards) / 80
                if len(rewards) == 80 and None not in rewards
                else None
            )
            result["completed_episodes"] = len(episodes)
        except Exception:  # noqa: BLE001 - Preserve isolated arm failures for review.
            result = {"mode": mode, "failed": True, "error": traceback.format_exc()}
        (target / f"{mode}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in (
                        "mode",
                        "failed",
                        "error",
                        "fixed_length_median_seconds",
                        "number_match",
                        "rollout_wall_seconds_excluding_warmup",
                    )
                }
            )
        )
