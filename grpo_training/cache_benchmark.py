"""Prefix-cache pilot using actual GRPO groups, without optimizer or replay."""

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import modal

from baseline_agent.batch import retry_call, write_json
from grpo_training.config import TrainingConfig
from grpo_training.modal_app import cache, image, search_transport

app = modal.App("finance-rag-grpo-cache-pilot")
logger = logging.getLogger(__name__)


def summarize_cache(waves):
    """Token-weighted engine request counts; missing counters remain unknown."""
    requests = [item for wave in waves for item in wave["cache_stats"]]
    missing = sum(item["cached_tokens"] is None for item in requests)
    prompts = sum(item["prompt_tokens"] for item in requests)
    complete = bool(requests) and not missing
    hits = sum(item["cached_tokens"] or 0 for item in requests)
    return {
        "engine_requests": len(requests),
        "requests_missing_counter": missing,
        "prompt_tokens": prompts,
        "cached_tokens": hits if complete else None,
        "token_hit_fraction": hits / prompts if complete and prompts else None,
        "requests_with_cache_hit": sum(bool(item["cached_tokens"]) for item in requests)
        if complete
        else None,
    }


class MeasuredBackend:
    def __init__(self, backend):
        self.backend, self.waves = backend, []

    def generate(self, requests):
        peers = {}
        for request in requests:
            if request.turn == 1:
                peers.setdefault(request.question_id, []).append(request)
        sharing = []
        for question_id, group in peers.items():
            if len({r.prompt_ids for r in group}) != 1 or len(
                {r.seed for r in group}
            ) != len(group):
                raise ValueError(
                    "GRPO peers must share a prompt and use independent seeds"
                )
            sharing.append(
                {
                    "question_id": question_id,
                    "trajectories": len(group),
                    "identical_prompt_tokens": len(group[0].prompt_ids),
                    "unique_seeds": len({r.seed for r in group}),
                }
            )
        batch = self.backend.generate(requests)
        self.waves.append(
            {
                "requests": len(requests),
                "seconds": batch.seconds,
                "generated_tokens": sum(
                    len(getattr(g, "completion_ids", ()))
                    + getattr(g, "raw_token_count", 0)
                    for g in batch.generations
                ),
                "cache_stats": list(batch.cache_stats),
                "first_turn_groups": sharing,
            }
        )
        return batch


@app.function(image=image, volumes={"/cache": cache}, timeout=1800)
def prepare(config_values):
    from huggingface_hub import snapshot_download

    from grpo_training.environment import checked_response

    config = TrainingConfig.from_dict(config_values)
    snapshot_download(
        config.model_id,
        revision=config.model_revision,
        cache_dir="/cache/hub",
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.safetensors",
            "tokenizer*",
            "vocab*",
            "merges*",
        ],
    )
    cache.commit()
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    response = search_transport(service)(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    checked_response(response, config)
    if response.get("status") != "ok" or not response["results"]:
        raise RuntimeError("Search preflight failed")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/cache": cache},
    timeout=3600,
    single_use_containers=True,
)
def evaluate(questions, config_values, caching):
    import torch
    from vllm import SamplingParams

    from grpo_training.collection import collect_groups
    from grpo_training.data import Question
    from grpo_training.model import export_adapter, load_training_model
    from grpo_training.vllm_backend import create_vllm_backend

    config = TrainingConfig.from_dict(config_values)
    questions = tuple(Question(**q) for q in questions)  # No gold labels on GPU.
    started = time.perf_counter()
    bundle = load_training_model(config, cache_dir="/cache/hub")
    lora_b = [p for name, p in bundle.model.named_parameters() if ".lora_B." in name]
    if not lora_b or any(torch.count_nonzero(p).item() for p in lora_b):
        raise RuntimeError("Pilot requires a fresh zero-output adapter")
    artifact = export_adapter(bundle, "/tmp/grpo-cache-pilot", 0)
    backend = create_vllm_backend(
        bundle, quantization="fp8", enable_prefix_caching=caching
    )
    backend.synchronize(artifact)
    resolved = backend.engine.llm_engine.vllm_config.cache_config
    cache_config = {
        name: getattr(resolved, name)
        for name in (
            "enable_prefix_caching",
            "mamba_cache_mode",
            "block_size",
            "mamba_block_size",
            "hash_block_size",
        )
    }
    if cache_config["enable_prefix_caching"] is not caching:
        raise RuntimeError("Engine did not apply the requested prefix-cache setting")
    initialization_seconds = time.perf_counter() - started
    backend.engine.generate(
        [{"prompt_token_ids": bundle.tokenizer.encode("Write a short greeting.")}],
        SamplingParams(max_tokens=8, temperature=0, seed=42),
        lora_request=backend._lora,
        use_tqdm=False,
    )
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    measured = MeasuredBackend(backend)
    for offset in range(0, len(questions), config.questions_per_update):
        # Training changes/sleeps the policy between batches. Do not credit this
        # pilot with cross-batch cache reuse that training cannot retain.
        if caching and backend.engine.reset_prefix_cache() is not True:
            raise RuntimeError("Could not clear prefix cache between rollout batches")
        events, wave_start = [], len(measured.waves)
        started = time.perf_counter()
        groups = collect_groups(
            questions[offset : offset + config.questions_per_update],
            config,
            artifact.policy,
            measured,
            bundle.tokenizer,
            search_transport(service),
            offset // config.questions_per_update,
            search_workers=4,
            audit=events.append,
        )
        yield {
            "batch": offset // config.questions_per_update,
            "enable_prefix_caching": caching,
            "precision": "fp8",
            "adapter_no_op_verified": True,
            "resolved_cache_config": cache_config,
            "policy": asdict(artifact.policy),
            "config": config.to_dict(),
            "initialization_seconds": initialization_seconds,
            "rollout_wall_seconds": time.perf_counter() - started,
            "generation_waves": measured.waves[wave_start:],
            "groups": [
                {
                    "question_id": g.episodes[0].question.id,
                    "episodes": [e.to_dict() for e in g.episodes],
                }
                for g in groups
            ],
            "audit": events,
        }


@app.local_entrypoint()
def main(questions: int = 4, group_size: int = 4, questions_per_batch: int = 2):
    from tqdm import tqdm

    from grpo_training.data import load_pinned_split, sample_training_questions
    from grpo_training.rewards import episode_reward

    if (
        not 1 <= questions <= 8
        or not 2 <= group_size <= 8
        or not 1 <= questions_per_batch <= 4
    ):
        raise ValueError("Pilot bounded to eight questions, G=2–8 and Q=1–4")
    config = TrainingConfig(
        questions_per_update=questions_per_batch,
        num_generations=group_size,
        temperature=1.0,
        top_p=1.0,
        num_iterations=2,
        dynamic_sampling=False,
    )
    selected = sample_training_questions(
        load_pinned_split("train", config), questions, 42
    )
    labels = {label.id: label for label in selected.labels}
    directory = Path("data/grpo/cache-pilots") / uuid4().hex[:12]
    write_json(
        directory / "run.json",
        {
            "config": config.to_dict(),
            "question_ids": [q.id for q in selected.questions],
            "selection_seed": 42,
            "training_updates": 0,
            "cache_reset_between_batches": True,
            "precision": "fp8",
        },
    )
    print(f"Results: {directory.resolve()}", flush=True)
    prepare.remote(config.to_dict())
    summaries = {}
    for caching in (False, True):
        name = "cache-on" if caching else "cache-off"
        arm, batches, rows = directory / name, [], []
        try:
            for batch in tqdm(
                evaluate.remote_gen(
                    [asdict(q) for q in selected.questions], config.to_dict(), caching
                ),
                total=(questions + questions_per_batch - 1) // questions_per_batch,
                desc=name,
            ):
                retry_call(
                    lambda batch=batch, arm=arm: write_json(
                        arm / f"batch-{batch['batch']:04d}.json", batch
                    )
                )
                batches.append(batch)
                for group in batch["groups"]:
                    for episode in group["episodes"]:
                        qid = episode["question"]["id"]
                        label = labels[qid]
                        score = episode_reward(
                            qid, episode["status"], episode["answer"], label
                        )
                        rows.append(
                            {
                                "question_id": qid,
                                "episode_id": episode["episode_id"],
                                "sample_index": episode["sample_index"],
                                "seed": episode["seed"],
                                "status": episode["status"],
                                "answer": episode["answer"],
                                "gold_answer": label.gold_answer,
                                "reward": score.reward,
                                "generation_turns": len(episode["generations"]),
                                "generated_tokens": sum(
                                    len(g["completion_ids"])
                                    for g in episode["generations"]
                                ),
                            }
                        )
                with (arm / "episodes.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as stream:
                    writer = (
                        csv.DictWriter(stream, fieldnames=list(rows[0]))
                        if rows
                        else None
                    )
                    if writer:
                        writer.writeheader()
                        writer.writerows(rows)
        except Exception as exc:
            logger.exception("GRPO cache pilot arm failed")
            write_json(arm / "error.json", {"error": f"{type(exc).__name__}: {exc}"})
        waves = [wave for batch in batches for wave in batch["generation_waves"]]
        wall = sum(b["rollout_wall_seconds"] for b in batches)
        generation = sum(w["seconds"] for w in waves)
        summaries[name] = {
            "complete": len(rows) == questions * group_size,
            "expected_trajectories": questions * group_size,
            "trajectories": len(rows),
            "number_match": sum(r["reward"] or 0 for r in rows) / len(rows)
            if len(rows) == questions * group_size
            else None,
            "rollout_wall_seconds": wall,
            "trajectories_per_hour": len(rows) * 3600 / wall if wall else None,
            "generation_tokens_per_second": sum(w["generated_tokens"] for w in waves)
            / generation
            if generation
            else None,
            "generation_wave_seconds": generation,
            "cache": summarize_cache(waves),
        }
        retry_call(lambda: write_json(directory / "comparison.json", summaries))
    print(json.dumps(summaries, indent=2), flush=True)
