"""Paired, bounded quantization pilot using the existing sequential RAG agent.

This isolates weight precision, not vLLM concurrency. Gold labels stay local.
"""

import csv
import json
import logging
import statistics
import time
from pathlib import Path
from uuid import uuid4

import modal

from baseline_agent.modal_app import MODEL_ID, MODEL_REVISION, cache

app = modal.App("finance-rag-quantization-pilot")
logger = logging.getLogger(__name__)
pilot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(
        str(Path(__file__).parent / "requirements.gpu.lock.txt")
    )
    .pip_install("bitsandbytes==0.50.2")
    .add_local_python_source("baseline_agent")
)


@app.function(image=pilot_image, volumes={"/model-cache": cache}, timeout=1800)
def prepare_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=f"/model-cache/{MODEL_REVISION}",
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
    image=pilot_image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/model-cache": cache},
    timeout=3600,
    max_containers=1,
    scaledown_window=2,
)
def evaluate(questions, precision):
    from dataclasses import replace

    import torch
    from transformers import set_seed

    from baseline_agent.agent import AgentConfig, run_agent
    from baseline_agent.batch import failure_result, retry_call
    from baseline_agent.inference import QwenGenerator

    config = AgentConfig()
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    started = time.perf_counter()
    generator = QwenGenerator(f"/model-cache/{MODEL_REVISION}", config, precision)
    generator.verbose = False
    load_seconds = time.perf_counter() - started
    # Same short synthetic warmup in every arm; excluded from question timings.
    warmup = generator.tokenizer("Warmup", return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generator.model.generate(**warmup, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()
    for question in questions:
        config = replace(config, seed=42 + question["example_index"])
        generator.config = config
        set_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fatal = False
        try:
            result = run_agent(
                question,
                generator,
                lambda payload: retry_call(lambda: service.search.remote(payload)),
                config,
                emit=lambda *args, **kwargs: None,
            )
        except Exception as exc:
            logger.exception("Quantization pilot question failed")
            result = failure_result(question, replace(config, seed=42), exc)
            fatal = isinstance(exc, torch.cuda.OutOfMemoryError)
        torch.cuda.synchronize()
        yield result | {
            "precision": precision,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "evaluation_mode": "rag",
            "gpu": "H100",
            "dtype": "float16",
            "backend": "transformers",
            "model_load_seconds": load_seconds,
            "inference_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "model_footprint_gib": generator.model.get_memory_footprint() / 2**30,
        }
        if fatal:
            return


def summarize(rows, expected):
    """Failures remain in the denominator; never report partial-arm accuracy."""
    completed = len(rows) == expected
    tokens = sum(row["generated_tokens"] for row in rows)
    seconds = sum(row["generation_seconds"] for row in rows)
    return {
        "expected_questions": expected,
        "received_questions": len(rows),
        "complete": completed,
        "number_match": sum(row["score"] or 0 for row in rows) / expected
        if completed
        else None,
        "exact_answer_match": sum(row["exact_answer_match"] for row in rows) / expected
        if completed
        else None,
        "mean_inference_seconds": statistics.mean(
            row["inference_seconds"] for row in rows
        )
        if rows
        else None,
        "generation_tokens_per_second": tokens / seconds if seconds else None,
        "generated_tokens": tokens,
        "peak_allocated_gib": max(
            (row["peak_allocated_gib"] for row in rows), default=None
        ),
        "infrastructure_errors": sum(
            row["status"] == "infrastructure_error" for row in rows
        ),
    }


@app.local_entrypoint()
def main(num_examples: int = 20, start: int = 0, resume: str = ""):
    from tqdm import tqdm

    from baseline_agent.batch import retry_call, write_json
    from baseline_agent.data import DATASET_REVISION, load_evaluation_examples
    from baseline_agent.evaluation import export_rows, record_result

    if not 1 <= num_examples <= 20:
        raise ValueError("Pilot is bounded to 1–20 questions")
    examples = load_evaluation_examples(start, num_examples)
    if any(example["skip_reason"] for example in examples):
        raise ValueError("Choose a selection without blank questions")
    root = Path("data/baseline/quantization").resolve()
    directory = Path(resume).resolve() if resume else root / uuid4().hex[:12]
    if resume and (not directory.is_relative_to(root) or not directory.is_dir()):
        raise ValueError("Resume must refer to an existing quantization pilot")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "question_ids": [e["question"]["id"] for e in examples],
        "precisions": ["fp16", "int8", "nf4"],
        "gpu": "H100",
        "backend": "transformers",
        "concurrent_episodes": 1,
        "bitsandbytes": "0.50.2",
        "status": "started",
    }
    if resume:
        saved = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        for key in manifest.keys() - {"status"}:
            if saved.get(key) != manifest[key]:
                raise ValueError(f"Resume mismatch: {key}")
    retry_call(lambda: write_json(directory / "run.json", manifest))
    print(f"Results: {directory.resolve()}", flush=True)
    # Readiness probe before allocating any paid GPU.
    modal.Cls.from_name("finance-rag-search", "SearchService")().search.remote(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    prepare_model.remote()
    all_rows, summaries = [], {}
    by_id = {e["question"]["id"]: e for e in examples}
    for precision in ("fp16", "int8", "nf4"):
        rows = []
        arm = directory / precision
        arm.mkdir(exist_ok=True)

        def save(result, precision=precision, arm=arm, rows=rows):
            example = by_id[result["question"]["id"]]
            if (
                result.get("precision") != precision
                or result.get("model_revision") != MODEL_REVISION
            ):
                raise ValueError("Saved result precision/revision mismatch")
            row = retry_call(
                lambda: record_result(
                    result,
                    example["gold_answer"],
                    arm / "raw",
                    example["gold_context_id"],
                )
            )
            row.update(
                precision=precision,
                exact_answer_match=str(result["answer"]).strip().casefold()
                == str(example["gold_answer"]).strip().casefold(),
                peak_allocated_gib=result["peak_allocated_gib"],
                model_footprint_gib=result["model_footprint_gib"],
            )
            rows.append(row)
            retry_call(lambda: export_rows(rows, arm))

        try:
            if resume:
                for example in examples:
                    path = arm / "raw" / (example["question"]["id"] + ".json")
                    if path.exists():
                        save(json.loads(path.read_text(encoding="utf-8")))
            completed = {row["id"] for row in rows}
            pending = [
                e["question"] for e in examples if e["question"]["id"] not in completed
            ]
            for result in tqdm(
                evaluate.remote_gen(pending, precision) if pending else (),
                total=len(pending),
                desc=precision,
            ):
                save(result)
        except Exception as exc:
            logger.exception("Quantization pilot arm failed")
            write_json(arm / "error.json", {"error": f"{type(exc).__name__}: {exc}"})
            print(f"{precision} failed: {exc}", flush=True)
        summaries[precision] = summarize(rows, num_examples)
        all_rows.extend(
            {k: v for k, v in row.items() if not k.startswith("_")} for row in rows
        )
        retry_call(lambda: write_json(directory / "comparison.json", summaries))
        if all_rows:
            with (directory / "comparison.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(all_rows)
    manifest["status"] = (
        "completed" if all(s["complete"] for s in summaries.values()) else "incomplete"
    )
    retry_call(lambda: write_json(directory / "run.json", manifest))
    print(json.dumps(summaries, indent=2), flush=True)
