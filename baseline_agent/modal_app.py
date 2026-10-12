"""Download on CPU, then run a bounded test batch on a temporary H100."""

import logging
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GPU = "H100"
logger = logging.getLogger(__name__)
app = modal.App("finance-rag-baseline")
cache = modal.Volume.from_name("finance-rag-qwen-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(
        str(Path(__file__).parent / "requirements.gpu.lock.txt")
    )
    .add_local_python_source("baseline_agent")
)


@app.function(
    image=image, volumes={"/model-cache": cache}, cpu=2, memory=4096, timeout=900
)
def download_model(revision):
    from huggingface_hub import snapshot_download

    path = f"/model-cache/{revision}"
    snapshot_download(
        MODEL_ID,
        revision=revision,
        local_dir=path,
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
    return path


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,
    memory=16384,
    volumes={"/model-cache": cache},
    timeout=3600,
    max_containers=1,
)
def run_examples(
    questions, config_values, revision, verbose=False, evaluation_mode="rag"
):
    import gc
    from dataclasses import replace

    import torch
    from transformers import set_seed

    from baseline_agent.agent import AgentConfig, run_agent
    from baseline_agent.batch import failure_result, retry_call
    from baseline_agent.inference import QwenGenerator
    from baseline_agent.question_only import QuestionOnlyGenerator, run_question_only

    service = (
        modal.Cls.from_name("finance-rag-search", "SearchService")()
        if evaluation_mode == "rag"
        else None
    )
    config = AgentConfig(**config_values)
    loaded = time.perf_counter()
    generator_class = (
        QwenGenerator if evaluation_mode == "rag" else QuestionOnlyGenerator
    )
    generator = generator_class(f"/model-cache/{revision}", config)
    generator.verbose = verbose
    model_load_seconds = time.perf_counter() - loaded
    try:
        for question in questions:
            started = time.perf_counter()
            restart = False
            try:
                question_config = replace(
                    config, seed=config.seed + question["example_index"]
                )
                generator.config = question_config
                set_seed(question_config.seed)
                if verbose:
                    print(
                        f"\n=== {question['id']}: {question['question']} ===",
                        flush=True,
                    )
                result = (
                    run_question_only(question, generator, question_config)
                    if evaluation_mode == "pretrained_only"
                    else run_agent(
                        question,
                        generator,
                        lambda payload: retry_call(
                            lambda: service.search.remote(payload)
                        ),
                        question_config,
                        emit=print if verbose else lambda *args, **kwargs: None,
                    )
                )
            except Exception as exc:
                logger.exception("Baseline example %s failed", question["id"])
                result = failure_result(question, config, exc)
                restart = isinstance(exc, torch.cuda.OutOfMemoryError)
            yield result | {
                "model_id": MODEL_ID,
                "model_revision": revision,
                "gpu": GPU,
                "dtype": "float16",
                "inference_seconds": time.perf_counter() - started,
                "model_load_seconds": model_load_seconds,
                "evaluation_mode": evaluation_mode,
            }
            if restart:
                return  # Remaining questions are retried with a fresh model invocation.
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main(
    example_index: int = 0,
    context_mode: str = "replace",
    max_searches: int = 6,
    top_k: int = 2,
    num_examples: int = 50,
    full_test: bool = False,
    batch_size: int = 10,
    attempts: int = 3,
    dashboard: bool = False,
    resume: str = "",
    retry_errors: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
    pretrained_only: bool = False,
):
    from baseline_agent.runner import run_evaluation

    try:
        run_evaluation(
            lambda questions, config, revision: run_examples.remote_gen(
                questions,
                config,
                revision,
                verbose,
                "pretrained_only" if pretrained_only else "rag",
            ),
            download_model.remote,
            lambda: modal.Cls.from_name(
                "finance-rag-search", "SearchService"
            )().search.remote(
                {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
            ),
            MODEL_ID,
            MODEL_REVISION,
            GPU,
            start=example_index,
            count=None if full_test else num_examples,
            context_mode=context_mode,
            max_searches=max_searches,
            top_k=top_k,
            batch_size=batch_size,
            attempts=attempts,
            dashboard=dashboard,
            resume=resume,
            retry_errors=retry_errors,
            dry_run=dry_run,
            evaluation_mode="pretrained_only" if pretrained_only else "rag",
        )
    except KeyboardInterrupt:
        print("Interrupted. Saved results are resumable.")
    except Exception as exc:
        logger.exception("Pilot/full evaluation stopped")
        raise SystemExit(1) from exc
