"""Paired BF16/FP8 inference-only pilot using the training vLLM bridge."""

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from uuid import uuid4

import modal

from baseline_agent.agent import AgentConfig, run_agent
from baseline_agent.batch import failure_result, retry_call, write_json
from baseline_agent.inference import QwenGenerator
from grpo_training.modal_app import cache, image
from grpo_training.records import GenerationRequest, SamplingParameters, generation_seed

logger = logging.getLogger(__name__)
app = modal.App("finance-rag-vllm-fp8-pilot")


class QueuedGenerator(QwenGenerator):
    """Each agent owns its history; only the coordinator touches the GPU."""

    def __init__(self, tokenizer, config, policy, question, queue):
        self.tokenizer, self.config, self.policy = tokenizer, config, policy
        self.question, self.queue, self.turn = question, queue, 0

    def generate(self, prompt):
        self.turn += 1
        ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=False))
        if len(ids) + self.config.max_new_tokens > self.config.max_context_tokens:
            raise ValueError("Prompt plus generation exceeds the context budget")
        request = GenerationRequest(
            self.question["id"],
            self.question["id"],
            self.question["id"],
            0,
            self.turn,
            generation_seed(
                self.config.seed + self.question["example_index"], self.turn
            ),
            self.policy,
            SamplingParameters(self.config.temperature, 0.95, 20, 1.05),
            prompt,
            ids,
        )
        future = Future()
        self.queue.put(("generation", (request, future)))
        return future.result(timeout=3300)


def batched_agents(questions, backend, tokenizer, policy, config, search, concurrency):
    """Barrier per wave; searches overlap in CPU threads, generation is batched.

    Each active agent emits exactly one generation request or terminal result per
    wave. Per-agent times include concurrent work and must not be summed as GPU time.
    """
    agent_config = AgentConfig(
        top_k=config.top_k,
        max_searches=config.max_searches,
        max_turns=config.max_turns,
        max_new_tokens=config.max_new_tokens,
        max_context_tokens=config.max_context_tokens,
        context_mode=config.context_mode,
        temperature=config.temperature,
        seed=config.seed,
    )
    measurements = []
    for offset in range(0, len(questions), concurrency):
        chunk, queue = questions[offset : offset + concurrency], Queue()

        def work(question, queue=queue):
            started = time.perf_counter()
            try:
                generator = QueuedGenerator(
                    tokenizer, agent_config, policy, question, queue
                )
                result = run_agent(
                    question, generator, search, agent_config, emit=lambda *a, **k: None
                )
            except Exception as exc:
                logger.exception("vLLM pilot agent failed")
                result = failure_result(question, agent_config, exc)
            queue.put(
                ("done", result | {"inference_seconds": time.perf_counter() - started})
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            workers = [pool.submit(work, question) for question in chunk]
            active = len(chunk)
            while active:
                events = [queue.get(timeout=3300) for _ in range(active)]
                ready = []
                for kind, payload in events:
                    if kind == "done":
                        active -= 1
                        yield payload, measurements
                    else:
                        ready.append(payload)
                if ready:
                    try:
                        batch = backend.generate(tuple(request for request, _ in ready))
                        if tuple(g.request for g in batch.generations) != tuple(
                            r for r, _ in ready
                        ):
                            raise ValueError("Generation batch membership changed")
                        measurements.append(
                            {
                                "requests": len(ready),
                                "seconds": batch.seconds,
                                "generated_tokens": sum(
                                    len(getattr(g, "completion_ids", ()))
                                    for g in batch.generations
                                ),
                            }
                        )
                        for (request, future), generation in zip(
                            ready, batch.generations, strict=True
                        ):
                            if hasattr(generation, "error"):
                                future.set_exception(generation.error)
                            else:
                                text = request.assistant_prefill + tokenizer.decode(
                                    list(generation.completion_ids),
                                    skip_special_tokens=True,
                                )
                                future.set_result(
                                    (
                                        text,
                                        {
                                            "input_tokens": len(request.prompt_ids),
                                            "generated_tokens": len(
                                                generation.completion_ids
                                            ),
                                            "generation_seconds": batch.seconds,
                                            "timing_definition": "shared generation-wave wall time; not additive across episodes",
                                            "sampling_logprobs": list(
                                                generation.sampling_logprobs
                                            ),
                                            "completion_ids": list(
                                                generation.completion_ids
                                            ),
                                        },
                                    )
                                )
                    except Exception as exc:
                        logger.exception("vLLM pilot generation wave failed")
                        for _, future in ready:
                            if not future.done():
                                future.set_exception(exc)
            for worker in workers:
                worker.result()


@app.function(image=image, volumes={"/cache": cache}, timeout=1800)
def prepare(config_values):
    from huggingface_hub import snapshot_download

    from grpo_training.config import TrainingConfig

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


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/cache": cache},
    timeout=3600,
    single_use_containers=True,
)
def evaluate(questions, config_values, precision, concurrency):
    import torch
    from vllm import SamplingParams

    from grpo_training.config import TrainingConfig
    from grpo_training.model import export_adapter, load_training_model
    from grpo_training.vllm_backend import create_vllm_backend

    config = TrainingConfig.from_dict(config_values)
    started = time.perf_counter()
    bundle = load_training_model(config, cache_dir="/cache/hub")
    artifact = export_adapter(bundle, "/tmp/fp8-pilot-adapters", 0)
    backend = create_vllm_backend(
        bundle, quantization="fp8" if precision == "fp8" else None
    )
    backend.synchronize(artifact)
    initialization_seconds = time.perf_counter() - started
    backend.engine.generate(
        [{"prompt_token_ids": bundle.tokenizer.encode("Write a short greeting.")}],
        SamplingParams(max_tokens=8, temperature=0, seed=42),
        lora_request=backend._lora,
        use_tqdm=False,
    )
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    started = time.perf_counter()
    for result, measurements in batched_agents(
        questions,
        backend,
        bundle.tokenizer,
        artifact.policy,
        config,
        lambda payload: retry_call(lambda: service.search.remote(payload)),
        concurrency,
    ):
        yield result | {
            "precision": precision,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "evaluation_mode": "rag",
            "gpu": "H100",
            "dtype": "bfloat16",
            "backend": "vllm-0.22.0",
            "concurrency": concurrency,
            "initialization_seconds": initialization_seconds,
            "elapsed_arm_seconds": time.perf_counter() - started,
            "generation_waves": list(measurements),
            "policy": asdict(artifact.policy),
            "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        }


@app.local_entrypoint()
def main(num_examples: int = 20, concurrency: int = 8):
    from tqdm import tqdm

    from baseline_agent.data import load_evaluation_examples
    from baseline_agent.evaluation import export_rows, record_result
    from grpo_training.config import TrainingConfig

    if not 1 <= num_examples <= 20 or not 1 <= concurrency <= 20:
        raise ValueError("Pilot supports at most 20 questions and concurrent episodes")
    examples = load_evaluation_examples(0, num_examples)
    config = TrainingConfig(temperature=0.2, top_p=0.95)
    directory = Path("data/baseline/vllm-fp8") / uuid4().hex[:12]
    write_json(
        directory / "run.json",
        {
            "config": config.to_dict(),
            "concurrency": concurrency,
            "question_ids": [e["question"]["id"] for e in examples],
            "precisions": ["bf16", "fp8"],
            "training_updates": 0,
        },
    )
    print(f"Results: {directory.resolve()}", flush=True)
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    service.search.remote(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    prepare.remote(config.to_dict())
    by_id, summaries = {e["question"]["id"]: e for e in examples}, {}
    for precision in ("bf16", "fp8"):
        rows, last = [], None
        arm = directory / precision
        try:
            for result in tqdm(
                evaluate.remote_gen(
                    [e["question"] for e in examples],
                    config.to_dict(),
                    precision,
                    concurrency,
                ),
                total=num_examples,
                desc=precision,
            ):
                label = by_id[result["question"]["id"]]
                row = retry_call(
                    lambda result=result, label=label, arm=arm: record_result(
                        result,
                        label["gold_answer"],
                        arm / "raw",
                        label["gold_context_id"],
                    )
                )
                rows.append(row)
                last = result
                retry_call(lambda rows=rows, arm=arm: export_rows(rows, arm))
        except Exception as exc:
            logger.exception("vLLM precision arm failed")
            write_json(arm / "error.json", {"error": f"{type(exc).__name__}: {exc}"})
        waves = last["generation_waves"] if last else []
        seconds = last["elapsed_arm_seconds"] if last else None
        generation_seconds = sum(w["seconds"] for w in waves)
        summaries[precision] = {
            "complete": len(rows) == num_examples,
            "questions": len(rows),
            "number_match": sum(r["score"] or 0 for r in rows) / num_examples
            if len(rows) == num_examples
            else None,
            "wall_seconds": seconds,
            "questions_per_hour": len(rows) * 3600 / seconds if seconds else None,
            "generation_tokens_per_second": sum(w["generated_tokens"] for w in waves)
            / generation_seconds
            if generation_seconds
            else None,
            "generation_wave_seconds": generation_seconds,
            "initialization_seconds": last["initialization_seconds"] if last else None,
            "timing_definition": "arm wall time excludes model/engine initialization and warmup; wave times counted once",
        }
        write_json(directory / "comparison.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)
