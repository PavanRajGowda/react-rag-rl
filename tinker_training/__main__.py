"""Primary hosted training and baseline evaluation entry point."""

import argparse
import hashlib
import json
from pathlib import Path

from baseline_agent.persistence import write_json
from tinker_training.config import TrainingConfig


def search_client(trace_sink=None, endpoint=None):
    """Return async map retrieval; rollout pools own fresh sessions and cleanup."""
    from retrieval_service.async_client import AsyncSearchClient

    return AsyncSearchClient(trace_sink=trace_sink, endpoint=endpoint)


def main(argv=None):
    """Parse preflight, baseline or training commands and configure hosted Tinker
    execution. Preflight validates locally without creating a service client;
    baseline/train may incur provider and CPU retrieval costs. Closes the service with
    success/error status on exit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "baseline", "preflight"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("data/tinker/run-001"))
    parser.add_argument("--corpus", type=Path, default=Path("data/hotpotqa-documents"))
    parser.add_argument(
        "--validation-count",
        type=int,
        default=0,
        help="Training validation disabled; use baseline for held-out evaluation",
    )
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Baseline validation questions only; training uses full train split",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint",
        help="Full Tinker training-state path for separate post-training evaluation",
    )
    parser.add_argument(
        "--search-app", help="Optional isolated Modal search application"
    )
    parser.add_argument("--search-function", default="dynamic_search")
    parser.add_argument(
        "--search-function-id", help="Optional ephemeral Modal endpoint ID"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Baseline question offset; skip already-benchmarked questions",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--stream-attempts",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Baseline result-stream attempts; 1 disables application stream retries",
    )
    parser.add_argument(
        "--trace-search",
        action="store_true",
        help="Buffer correlated search timing and event-loop diagnostics; export on exit",
    )
    args = parser.parse_args(argv)
    if args.mode == "train" and (args.validation_count or args.checkpoint):
        parser.error(
            "Training performs no validation; --checkpoint is for baseline evaluation only"
        )
    if (
        args.validation_count < 0
        or args.validation_interval < 1
        or args.count < 1
        or args.batch_size < 1
        or args.start < 0
    ):
        parser.error("Invalid evaluation/batch limits")
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    # This identifies a service-model contract, not an alleged HF weight checksum.
    model = values.get("model_id", "Qwen/Qwen3.5-4B")
    values.update(
        temperature=1.0,
        top_p=1.0,
        loss_type="tinker_ppo",
        model_revision=hashlib.sha256(("tinker-0.34:" + model).encode()).hexdigest()[
            :40
        ],
    )
    config = TrainingConfig.from_dict(values)
    identity = {
        "rollout_execution": "async-tinker-map-search-action-stops-v2",
        "provider": "tinker",
        "sdk": "0.34.0",
        "model": model,
        "model_revision_semantics": "service contract identifier; HF bytes unverified",
        "lora": "service attention+MLP, no unembedding; alpha/target modules service-owned",
        "validation_count": args.validation_count,
        "validation_interval": args.validation_interval,
    }
    if args.mode == "preflight":
        # Safe without API credentials: no service/client creation.
        print(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "identity": identity,
                    "live_verified": False,
                },
                indent=2,
            )
        )
        return
    import tinker
    from tinker import types

    service = tinker.ServiceClient(display_name=args.run_dir.name)
    from retrieval_service.timing import SearchTrace

    trace = (
        SearchTrace(args.run_dir / "search-performance") if args.trace_search else None
    )
    status = "success"
    try:
        if args.search_app or args.search_function_id:
            import modal

            search = search_client(
                trace_sink=trace,
                endpoint=modal.Function.from_id(args.search_function_id)
                if args.search_function_id
                else modal.Function.from_name(args.search_app, args.search_function),
            )
        else:
            search = (
                search_client(trace_sink=trace)
                if trace is not None
                else search_client()
            )
        # Verify async retrieval before creating hosted model resources.
        import asyncio

        from tinker_training.search_contract import checked_response

        async def readiness():
            async with search.open_session() as session:
                checked_response(
                    await session(
                        {"query": "Arthur's Magazine", "top_k": config.top_k}
                    ),
                    config,
                )

        asyncio.run(readiness())
        if trace is not None:
            from retrieval_service.timing import validate_timing

            validate_timing(trace.requests[-1])
        if args.mode == "train":
            from tinker_training.data import load_local_splits, training_schedule
            from tinker_training.runner import run

            train, dev, hashes = load_local_splits(args.corpus, config)
            train = training_schedule(train, config)
            identity["training_schedule"] = {
                "mix": config.training_question_mix,
                "questions": len(train.questions),
                "ordered_ids_sha256": hashlib.sha256(
                    json.dumps([q.id for q in train.questions]).encode()
                ).hexdigest(),
            }
            identity["dataset"] = hashes
            from tinker_training.provenance import training_source_hashes

            identity["sources"] = training_source_hashes()
            run(
                config,
                train,
                dev,
                args.run_dir,
                service,
                search,
                types=types,
                identity=identity,
                validation_count=args.validation_count,
                validation_interval=args.validation_interval,
            )
        else:
            from baseline_agent.data import load_evaluation_examples
            from baseline_agent.phases import run_phase
            from baseline_agent.sampling import repeat_examples
            from batched_inference.baseline import batched_agents
            from tinker_training.backend import TinkerBackend
            from tinker_training.records import SamplingParameters

            examples = repeat_examples(
                load_evaluation_examples(start=args.start, count=args.count),
                args.repetitions,
                config.seed,
            )
            manifest = {
                "config": config.to_dict(),
                "identity": identity,
                "count": args.count,
                "start": args.start,
                "repetitions": args.repetitions,
                "stream_attempts": args.stream_attempts,
                "checkpoint": args.checkpoint,
            }
            path = args.run_dir / "tinker-baseline.json"
            if (
                path.exists()
                and json.loads(path.read_text(encoding="utf-8")) != manifest
            ):
                raise ValueError("Baseline identity changed; use a new run directory")
            write_json(path, manifest)
            if args.checkpoint:
                trained = service.create_training_client_from_state_with_optimizer(
                    args.checkpoint
                )
                sampler = trained.save_weights_and_get_sampling_client()
            else:
                sampler = service.create_sampling_client(base_model=model)
            tokenizer = sampler.get_tokenizer()
            backend = TinkerBackend(
                sampler,
                tokenizer,
                config,
                types=types,
                checkpoint=args.checkpoint or "base",
            )

            def invoke(questions, config_values, concurrency):
                """Adapt evaluation questions and config values into the shared
                batched_agents iterator using the current hosted policy and tokenizer.
                """
                import time

                started = time.perf_counter()
                rollout_started_utc_ns = time.time_ns()
                completed = 0
                try:
                    for result, measurements in batched_agents(
                        questions,
                        backend,
                        tokenizer,
                        backend.policy,
                        config,
                        search,
                        concurrency,
                        sampling=SamplingParameters(1.0, 1.0),
                        trajectory_attempts=config.max_trajectory_attempts,
                    ):
                        completed += 1
                        yield result | {
                            "backend": "tinker",
                            "model_id": model,
                            "model_revision": config.model_revision,
                            "generation_requests": measurements,
                        }
                finally:
                    elapsed = time.perf_counter() - started
                    write_json(
                        args.run_dir / "inference-performance.json",
                        {
                            "rollout_started_utc_ns": rollout_started_utc_ns,
                            "rollout_finished_utc_ns": time.time_ns(),
                            "completed_trajectories": completed,
                            "rollout_stream_seconds": elapsed,
                            "trajectories_per_second": completed / elapsed
                            if elapsed
                            else None,
                            "timing_definition": "Client rollout stream wall time, including per-result persistence; excludes initial search/model readiness and final aggregate exports.",
                        },
                    )

            run_phase(
                "validation",
                examples,
                args.run_dir,
                config,
                args.batch_size,
                invoke,
                stream_attempts=args.stream_attempts,
            )
    except BaseException:
        status = "errored"
        raise
    finally:
        try:
            if trace is not None:
                trace.save()
        finally:
            service.close(status=status).result()


if __name__ == "__main__":
    main()
