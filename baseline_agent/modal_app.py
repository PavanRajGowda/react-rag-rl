"""Download on CPU, then run exactly one test question on a temporary T4."""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import modal

MODEL_ID = "Qwen/Qwen3.5-2B"
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
    gpu="T4",
    cpu=2,
    memory=8192,
    volumes={"/model-cache": cache},
    timeout=1800,
    max_containers=1,
)
def run_examples(questions, config_values, revision):
    from baseline_agent.agent import AgentConfig, run_agent
    from baseline_agent.inference import QwenGenerator

    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    config = AgentConfig(**config_values)
    loaded = time.perf_counter()
    generator = QwenGenerator(f"/model-cache/{revision}", config)
    model_load_seconds = time.perf_counter() - loaded
    from dataclasses import asdict, replace

    from transformers import set_seed

    for question in questions:
        question_config = replace(config, seed=config.seed + question["example_index"])
        generator.config = question_config
        set_seed(question_config.seed)
        print(f"\n=== {question['id']}: {question['question']} ===", flush=True)
        started = time.perf_counter()
        try:
            result = run_agent(
                question, generator, service.search.remote, question_config
            )
        except Exception as exc:
            logger.exception("Baseline example %s failed", question["id"])
            result = {
                "question": question,
                "config": asdict(question_config),
                "status": "infrastructure_error",
                "answer": None,
                "turns": [],
                "searches": [],
                "events": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        yield result | {
            "model_id": MODEL_ID,
            "model_revision": revision,
            "gpu": "T4",
            "dtype": "float16",
            "inference_seconds": time.perf_counter() - started,
            "model_load_seconds": model_load_seconds,
        }


@app.local_entrypoint()
def main(
    example_index: int = 0,
    context_mode: str = "replace",
    max_searches: int = 6,
    top_k: int = 2,
    num_examples: int = 1,
    dashboard: bool = False,
    resume: str = "",
):
    from dataclasses import asdict

    from huggingface_hub import HfApi

    from baseline_agent.agent import AgentConfig
    from baseline_agent.data import load_evaluation_examples
    from baseline_agent.evaluation import export_rows, record_result

    examples = load_evaluation_examples(example_index, num_examples)
    config = AgentConfig(
        context_mode=context_mode, max_searches=max_searches, top_k=top_k
    )
    print(
        f"Selected {len(examples)} test questions starting at index {example_index}",
        flush=True,
    )
    # Warm the independent CPU service before allocating a T4. This readiness
    # probe is unrelated to the test question and never enters the model prompt.
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    probe = service.search.remote(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    if probe.get("status") != "ok":
        raise RuntimeError(
            "Deploy the working search service before running the baseline"
        )
    revision = HfApi().model_info(MODEL_ID).sha
    experiment_id = uuid4().hex[:12] if not resume else Path(resume).name
    directory = (
        Path("data/baseline/experiments") / experiment_id
        if not resume
        else Path(resume).resolve()
    )
    if resume and (
        not directory.is_relative_to(Path("data/baseline/experiments").resolve())
        or not directory.is_dir()
    ):
        raise ValueError("Resume directory must be an existing baseline experiment")
    labels = {x["question"]["id"]: x["gold_answer"] for x in examples}
    contexts = {x["question"]["id"]: x["gold_context_id"] for x in examples}
    rows = []
    completed = set()
    for example in examples:
        saved = directory / (example["question"]["id"] + ".json")
        if resume and saved.exists():
            previous = json.loads(saved.read_text(encoding="utf-8"))
            if (
                previous["model_revision"] != revision
                or previous["question"] != example["question"]
                or previous["config"]
                != asdict(config)
                | {"seed": config.seed + example["question"]["example_index"]}
            ):
                raise ValueError(
                    "Saved example has incompatible model, dataset or configuration"
                )
            rows.append(
                record_result(
                    previous,
                    example["gold_answer"],
                    directory,
                    example["gold_context_id"],
                )
            )
            completed.add(example["question"]["id"])
    if dashboard and rows:
        from baseline_agent.observability import write_dashboard

        write_dashboard(rows, directory / "tensorboard-resumed")
    pending = [x["question"] for x in examples if x["question"]["id"] not in completed]
    if pending:
        download_model.remote(revision)
    summary = export_rows(rows, directory) if rows else {}
    # Only whitelisted questions cross this call boundary. Scoring remains
    # local after predictions return, even when five share one loaded model.
    generated_results = (
        run_examples.remote_gen(pending, asdict(config), revision)
        if pending
        else iter(())
    )
    for result in generated_results:
        result["run_id"] = experiment_id
        rows.append(
            record_result(
                result,
                labels[result["question"]["id"]],
                directory,
                contexts[result["question"]["id"]],
            )
        )
        summary = export_rows(rows, directory)
        if dashboard:
            from baseline_agent.observability import write_dashboard

            try:
                write_dashboard(
                    rows[-1:],
                    directory / "tensorboard-resumed"
                    if resume
                    else directory / "tensorboard",
                    start_step=len(rows) - 1,
                )
            except Exception:
                logger.exception(
                    "Dashboard export failed; continuing with CSV and JSON"
                )
                dashboard = False
        print(
            json.dumps(
                {
                    "id": rows[-1]["id"],
                    "correct_answer": rows[-1]["correct_answer"],
                    "our_answer": rows[-1]["our_answer"],
                    "score": rows[-1]["score"],
                    "inference_seconds": rows[-1]["inference_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
    print(
        json.dumps(
            summary
            | {
                "experiment": str(directory.resolve()),
                "csv": str((directory / "results.csv").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
