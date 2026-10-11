"""Download on CPU, then run exactly one test question on a temporary T4."""

import json
from pathlib import Path
from uuid import uuid4

import modal

MODEL_ID = "Qwen/Qwen3.5-2B"
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
    timeout=900,
    max_containers=1,
)
def run_example(question, config_values, revision):
    from baseline_agent.agent import AgentConfig, run_agent
    from baseline_agent.inference import QwenGenerator

    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    config = AgentConfig(**config_values)
    generator = QwenGenerator(f"/model-cache/{revision}", config)
    result = run_agent(question, generator, service.search.remote, config)
    return result | {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "gpu": "T4",
        "dtype": "float16",
    }


@app.local_entrypoint()
def main(
    example_index: int = 0,
    context_mode: str = "replace",
    max_searches: int = 6,
    top_k: int = 5,
):
    from dataclasses import asdict

    from huggingface_hub import HfApi

    from baseline_agent.agent import AgentConfig
    from baseline_agent.data import load_question

    question = load_question(example_index)
    config = AgentConfig(
        context_mode=context_mode, max_searches=max_searches, top_k=top_k
    )
    print(json.dumps(question, indent=2), flush=True)
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
    download_model.remote(revision)
    result = run_example.remote(question, asdict(config), revision)
    result["run_id"] = uuid4().hex[:12]
    output = Path("data/baseline") / f"{question['id']}-{context_mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    transcript = output.with_suffix(".txt")
    transcript.write_text(
        "Question: "
        + question["question"]
        + "\n\n"
        + "\n\n".join(x["text"] for x in result["events"]),
        encoding="utf-8",
    )
    history = output.parent / "runs"
    history.mkdir(exist_ok=True)
    stem = f"{question['id']}-{context_mode}-{result['run_id']}"
    (history / f"{stem}.json").write_text(
        output.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (history / f"{stem}.txt").write_text(
        transcript.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "answer": result["answer"],
                "searches": len(result["searches"]),
                "numeric_validation": "not_performed",
                "trace": str(output.resolve()),
                "transcript": str(transcript.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
