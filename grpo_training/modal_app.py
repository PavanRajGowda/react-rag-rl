"""Reviewed Modal wiring. Import/CPU tests never launch a paid GPU job."""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
app = modal.App("finance-rag-grpo")
cache = modal.Volume.from_name("finance-rag-grpo-cache", create_if_missing=True)
runs = modal.Volume.from_name("finance-rag-grpo-runs", create_if_missing=True)
writers = modal.Dict.from_name("finance-rag-grpo-writers", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.11")
    .apt_install("libnuma1", "libgomp1", "git")
    .uv_sync(
        uv_project_dir=str(ROOT / "grpo_training"),
        frozen=True,
        uv_version="0.13.0",
        extra_options="--no-dev",
    )
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("grpo_training", "baseline_agent")
    .add_local_file(
        ROOT / "grpo_training/pyproject.toml", "/root/grpo_training/pyproject.toml"
    )
    .add_local_file(ROOT / "grpo_training/uv.lock", "/root/grpo_training/uv.lock")
)


@app.function(
    image=image, cpu=2, memory=4096, volumes={"/cache": cache}, timeout=1800, retries=0
)
def warm_model(config_values):
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
    from grpo_training.environment import checked_response

    # Fail on unavailable/mismatched retrieval before allocating a training GPU.
    service = modal.Cls.from_name("finance-rag-search", "SearchService")()
    response = search_transport(service)(
        {"ticker": "AAPL", "year": 2002, "query": "net sales", "top_k": 1}
    )
    checked_response(response, config)
    if response.get("status") != "ok" or not response["results"]:
        raise RuntimeError(
            "Search preflight did not retrieve its known diagnostic report"
        )


def search_transport(service, *, timeout=45.0):
    """Controller owns retries. A timeout does not kill the shared search container."""

    def search(payload):
        async def invoke():
            call = None

            async def request():
                nonlocal call
                call = await service.search.spawn.aio(payload)
                return await call.get.aio()

            try:
                return await asyncio.wait_for(request(), timeout=timeout)
            except (TimeoutError, modal.exception.TimeoutError):
                if call is not None:
                    try:
                        await asyncio.wait_for(
                            call.cancel.aio(terminate_containers=False), timeout=2.0
                        )
                    except Exception:
                        logger.exception(
                            "Best-effort search cancellation failed after its deadline"
                        )
                raise TimeoutError(f"Search exceeded {timeout} seconds") from None

        return asyncio.run(invoke())

    return search


def acquire_writer(registry, run_id):
    token = uuid.uuid4().hex
    if not registry.put(run_id, token, skip_if_exists=True):
        raise RuntimeError(
            "Run already has a writer lock. Verify any old job stopped before clearing a stale lock."
        )
    return token


def release_writer(registry, run_id, token):
    if registry.get(run_id) == token:
        registry.pop(run_id)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    volumes={"/cache": cache, "/runs": runs},
    timeout=24 * 3600,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def train_run(run_id, config_values, train_values, identity, resume=False):
    from grpo_training.checkpoints import Progress, RunStore
    from grpo_training.config import TrainingConfig
    from grpo_training.data import PreparedSplit, Question, RewardLabel
    from grpo_training.model import load_training_model
    from grpo_training.runner import initialize_optimizer, run_identity, run_training
    from grpo_training.trainer import create_replay_trainer
    from grpo_training.vllm_backend import create_vllm_backend

    token = acquire_writer(writers, run_id)
    try:
        config = TrainingConfig.from_dict(config_values)
        train = PreparedSplit(
            "train",
            tuple(Question(**q) for q in train_values["questions"]),
            tuple(RewardLabel(**label) for label in train_values["labels"]),
            tuple(train_values["skipped_blank_ids"]),
            train_values["selected_source_count"],
        )
        if run_identity(config, train) != identity:
            raise ValueError("Remote source/data identity differs from local preflight")
        store = RunStore("/runs", run_id, identity, resume=resume, durable=runs.commit)
        cache.reload()
        bundle = load_training_model(config, cache_dir="/cache/hub")
        trainer = create_replay_trainer(bundle, train.questions, store.root / "trainer")
        initialize_optimizer(trainer, config)
        progress, pending, rng = Progress(), None, None
        if resume:
            progress, pending, rng = store.restore(trainer, bundle, train.labels)
        else:
            # Preserve an initial recovery point even if vLLM startup fails.
            from grpo_training.model import export_adapter

            artifact = export_adapter(bundle, store.root / "transfers", 0)
            store.save(trainer, artifact, progress)
        backend = create_vllm_backend(bundle)
        service = modal.Cls.from_name("finance-rag-search", "SearchService")()
        return run_training(
            trainer,
            bundle,
            backend,
            train,
            search_transport(service),
            store,
            progress=progress,
            pending=pending,
            rng=rng,
        )
    finally:
        try:
            runs.commit()
        finally:
            release_writer(writers, run_id, token)


@app.local_entrypoint()
def main(
    run_id: str,
    config_path: str = "grpo_training/configs/smoke.json",
    train_count: int = 16,
    resume: bool = False,
):
    """Explicit paid execution command, to be run only after GPU smoke approval."""
    from grpo_training.config import TrainingConfig
    from grpo_training.data import load_local_splits
    from grpo_training.runner import run_identity

    config = TrainingConfig.from_dict(
        json.loads(Path(config_path).read_text(encoding="utf-8"))
    )
    train, _, _ = load_local_splits(
        ROOT / "data/finqa", config, train_count=train_count, dev_count=1
    )
    identity = run_identity(config, train)
    warm_model.remote(config.to_dict())
    result = train_run.remote(run_id, config.to_dict(), asdict(train), identity, resume)
    print(json.dumps(result, indent=2))
