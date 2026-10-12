"""Local orchestration: progress, provenance, checkpointing and resumable chunks."""

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from baseline_agent.agent import AgentConfig
from baseline_agent.batch import (
    PersistenceError,
    failure_result,
    retry_call,
    run_batches,
    write_json,
)
from baseline_agent.data import DATASET_REVISION, load_evaluation_examples
from baseline_agent.evaluation import export_rows, record_result
from finance_rag.search import SEARCH_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def behavior_hash():
    package = Path(__file__).parent
    return hashlib.sha256(
        b"".join(
            (package / name).read_bytes()
            for name in ("prompt.py", "protocol.py", "agent.py", "inference.py")
        )
    ).hexdigest()


def validate_resume(saved, expected):
    for key in expected:
        if saved.get(key) != expected[key]:
            raise ValueError(
                f"Resume mismatch for {key}; use a new experiment for changed model, GPU, instructions or configuration"
            )


def run_evaluation(
    invoke,
    download,
    probe,
    model_id,
    revision,
    gpu,
    *,
    start=0,
    count=50,
    context_mode="replace",
    max_searches=6,
    top_k=2,
    batch_size=10,
    attempts=3,
    dashboard=False,
    resume="",
    retry_errors=False,
    dry_run=False,
):
    started = time.perf_counter()
    if not 1 <= batch_size <= 50 or not 1 <= attempts <= 5:
        raise ValueError("batch_size must be 1–50; attempts must be 1–5")
    config = AgentConfig(
        context_mode=context_mode, max_searches=max_searches, top_k=top_k
    )
    root = Path("data/baseline/experiments").resolve()
    directory = Path(resume).resolve() if resume else root / uuid4().hex[:12]
    if resume and (not directory.is_relative_to(root) or not directory.is_dir()):
        raise ValueError("Resume directory must be an existing baseline experiment")
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {directory}\nResume with --resume {directory}", flush=True)
    expected = {
        "model_id": model_id,
        "model_revision": revision,
        "gpu": gpu,
        "dtype": "float16",
        "dataset_revision": DATASET_REVISION,
        "base_config": asdict(config),
        "behavior_sha256": behavior_hash(),
        "search_schema_version": SEARCH_SCHEMA_VERSION,
        "start": start,
    }
    manifest_path = directory / "run.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if resume else {}
    if resume:
        validate_resume(previous, expected)
    manifest = (
        previous
        | expected
        | {
            "status": "preparing",
            "batch_size": batch_size,
            "attempts": attempts,
            "updated_at_unix": time.time(),
        }
    )
    write_json(manifest_path, manifest)

    def log(error):
        with (directory / "worker-errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(error) + "\n")
        tqdm.write(f"Worker retry: {error['error']}")

    def metadata(result):
        return result | {
            "model_id": model_id,
            "model_revision": revision,
            "gpu": gpu,
            "dtype": "float16",
            "inference_seconds": result.get("inference_seconds", 0),
            "model_load_seconds": result.get("model_load_seconds", 0),
            "run_id": directory.name,
        }

    try:
        examples = retry_call(lambda: load_evaluation_examples(start, count), attempts)
        if resume and len(examples) < previous.get("selected_count", 0):
            raise ValueError(
                "A resumed selection may expand to the full test set, but cannot shrink"
            )
        manifest.update(
            selected_count=len(examples),
            blank_questions=sum(bool(x["skip_reason"]) for x in examples),
        )
        print(
            f"Selected {len(examples)} rows; {manifest['blank_questions']} blank rows will be skipped. GPU: {gpu}; chunk size: {batch_size}.",
            flush=True,
        )
        if dry_run:
            manifest["status"] = "prepared"
            write_json(manifest_path, manifest)
            print("Dry run complete: no model download, search call or GPU invocation.")
            return directory
        lookup = {x["question"]["id"]: x for x in examples}
        rows, completed = {}, set()
        for example in examples:
            question = example["question"]
            path = directory / (question["id"] + ".json")
            raw_path = directory / "raw" / (question["id"] + ".json")
            saved = None
            if resume:
                for candidate in (path, raw_path):
                    if candidate.exists():
                        try:
                            saved = json.loads(candidate.read_text(encoding="utf-8"))
                            break
                        except (json.JSONDecodeError, UnicodeError):
                            logger.warning(
                                "Corrupt checkpoint %s; trying raw result or rerunning",
                                candidate,
                            )
            if saved is None or (
                retry_errors
                and saved["status"] in ("infrastructure_error", "postprocessing_error")
            ):
                continue
            wanted_config = asdict(config) | {
                "seed": config.seed + question["example_index"]
            }
            validate_resume(
                saved,
                {
                    "question": question,
                    "config": wanted_config,
                    "model_revision": revision,
                    "model_id": model_id,
                    "gpu": gpu,
                },
            )
            rows[question["id"]] = record_result(
                saved, example["gold_answer"], directory, example["gold_context_id"]
            )
            completed.add(question["id"])
        dashboard_active = dashboard
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        with tqdm(
            total=len(examples),
            initial=len(completed),
            desc=f"FinQA {gpu}",
            unit="question",
            dynamic_ncols=True,
        ) as progress:

            def persist(result):
                nonlocal dashboard_active
                key = result["question"]["id"]
                example = lookup[key]
                try:
                    write_json(directory / "raw" / (key + ".json"), result)
                    try:
                        row = record_result(
                            result,
                            example["gold_answer"],
                            directory,
                            example["gold_context_id"],
                        )
                    except (OSError, PermissionError):
                        raise
                    except Exception as exc:
                        logger.exception(
                            "Local scoring/trace failure for %s; raw result preserved",
                            key,
                        )
                        row = record_result(
                            metadata(
                                failure_result(
                                    example["question"],
                                    config,
                                    exc,
                                    "postprocessing_error",
                                )
                            ),
                            example["gold_answer"],
                            directory,
                            example["gold_context_id"],
                        )
                    rows[key] = row
                    completed.add(key)
                    ordered = sorted(rows.values(), key=lambda x: x["example_index"])
                    summary = export_rows(ordered, directory)
                except Exception as exc:
                    raise PersistenceError(
                        f"Cannot safely persist {key}. Raw results and prior checkpoints are in {directory}"
                    ) from exc
                if dashboard_active:
                    try:
                        from baseline_agent.observability import write_dashboard

                        write_dashboard(
                            [row],
                            directory / "tensorboard",
                            start_step=row["example_index"],
                        )
                    except Exception:
                        logger.exception("Dashboard export failed; CSV/JSON continue")
                        dashboard_active = False
                progress.update(1)
                accuracy = summary["answer_match_accuracy"]
                progress.set_postfix(
                    accuracy=f"{accuracy:.1%}" if accuracy is not None else "n/a",
                    errors=summary["infrastructure_errors"],
                    last=row["status"],
                )

            for example in examples:
                if (
                    example["skip_reason"]
                    and example["question"]["id"] not in completed
                ):
                    persist(
                        metadata(
                            failure_result(
                                example["question"],
                                config,
                                ValueError(example["skip_reason"]),
                                "skipped_blank_question",
                            )
                        )
                    )
            pending = [
                x["question"] for x in examples if x["question"]["id"] not in completed
            ]
            if pending:
                response = retry_call(probe, attempts)
                if (
                    response.get("status") != "ok"
                    or response.get("search_schema_version") != SEARCH_SCHEMA_VERSION
                ):
                    raise RuntimeError(
                        "Search preflight failed or deployed schema is stale. Redeploy modal_app.py before allocating the GPU"
                    )
                expected_corpus = json.loads(
                    Path("data/finqa/manifest.json").read_text(encoding="utf-8")
                )["corpus_version"]
                if response.get("corpus_version") != expected_corpus:
                    raise RuntimeError(
                        "Deployed corpus version differs from local corpus"
                    )
                retry_call(lambda: download(revision), attempts)
                run_batches(
                    pending,
                    lambda questions: invoke(questions, asdict(config), revision),
                    persist,
                    lambda question, error: metadata(
                        failure_result(question, config, error)
                    ),
                    log,
                    batch_size=batch_size,
                    attempts=attempts,
                )
        summary = export_rows(
            sorted(rows.values(), key=lambda x: x["example_index"]), directory
        )
        manifest.update(
            status="completed_with_errors"
            if summary["infrastructure_errors"]
            else "completed",
            completed_count=len(completed),
            invocation_wall_seconds=time.perf_counter() - started,
        )
        write_json(manifest_path, manifest)
        print(
            json.dumps(
                summary
                | {"experiment": str(directory), "csv": str(directory / "results.csv")},
                indent=2,
            )
        )
        return directory
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            last_error=f"{type(exc).__name__}: {exc}",
        )
        write_json(manifest_path, manifest)
        print(
            f"Stopped safely. Existing results remain in {directory}. Resume using the same selection/configuration.",
            flush=True,
        )
        raise
