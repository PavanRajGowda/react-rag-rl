"""Backend-independent resumable evaluation outputs and scoring."""

import json
import math
import time
from pathlib import Path

from baseline_agent.persistence import write_json


def run_phase(
    name,
    examples,
    directory,
    config,
    concurrency,
    invoke,
    *,
    stream_attempts=3,
    export_every=None,
    export_interval_seconds=None,
):
    """Run a resumable scored evaluation phase and persist each trajectory as it arrives.
    examples contains public questions plus private labels; invoke supplies a batched
    result iterator. stream_attempts bounds retries of unsaved questions, while valid
    existing raw results are re-scored and reused. Returns True only when every example
    is saved without technical errors; publishes explicit completion/error status.

    Raw trajectory files are written immediately for resume. Scoring stays in memory;
    duplicate scored JSON/text files and per-result live status writes are omitted.
    Aggregate exports run only at completion by default. Optional export_every
    and export_interval_seconds enable periodic refreshes when explicitly set.
    This avoids quadratic per-result rewriting.
    """
    from tqdm import tqdm

    from baseline_agent.evaluation import export_rows, record_result

    if type(stream_attempts) is not int or not 1 <= stream_attempts <= 3:
        raise ValueError("stream_attempts must be 1..3")
    if export_every is not None and (type(export_every) is not int or export_every < 1):
        raise ValueError("export_every must be a positive integer")
    if export_interval_seconds is not None and (
        not math.isfinite(export_interval_seconds) or export_interval_seconds <= 0
    ):
        raise ValueError("export_interval_seconds must be positive and finite")
    directory = Path(directory) / name
    lookup = {e["question"]["id"]: e for e in examples}
    rows, pending = {}, []
    for example in examples:
        key = example["question"]["id"]
        path = directory / "raw" / (key + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if stream_attempts == 1 or saved["status"] not in (
                "infrastructure_error",
                "postprocessing_error",
            ):
                rows[key] = record_result(
                    saved,
                    example["gold_answer"],
                    directory,
                    example["gold_document_ids"],
                    write_artifacts=False,
                )
                continue
        pending.append(example["question"])
    # Keep progress accuracy incremental; it must not require rebuilding CSV/JSON.
    scored_count = sum(row["score"] is not None for row in rows.values())
    score_sum = sum(row["score"] for row in rows.values() if row["score"] is not None)
    last_export_count, last_export_at = len(rows), time.monotonic()
    write_json(
        directory / "status.json",
        {"status": "running", "total": len(examples), "saved": len(rows)},
    )
    with tqdm(
        total=len(examples), initial=len(rows), desc=name, unit="trajectory"
    ) as progress:
        # A broken remote stream retries only unsaved questions, never discards saved trajectories.
        for attempt in range(stream_attempts):
            if not pending:
                break
            try:
                for result in invoke(pending, config.to_dict(), concurrency):
                    key = result["question"]["id"]
                    if key not in lookup or key in rows:
                        raise ValueError("Unexpected/duplicate result ID")
                    label = lookup[key]
                    write_json(directory / "raw" / (key + ".json"), result)
                    rows[key] = record_result(
                        result,
                        label["gold_answer"],
                        directory,
                        label["gold_document_ids"],
                        write_artifacts=False,
                    )
                    score = rows[key]["score"]
                    if score is not None:
                        scored_count += 1
                        score_sum += score
                    now = time.monotonic()
                    if (
                        export_every is not None
                        and len(rows) - last_export_count >= export_every
                    ) or (
                        export_interval_seconds is not None
                        and now - last_export_at >= export_interval_seconds
                    ):
                        export_rows(list(rows.values()), directory)
                        last_export_count, last_export_at = len(rows), time.monotonic()
                    progress.update(1)
                    progress.set_postfix(
                        refresh=False,
                        accuracy=f"{score_sum / scored_count:.1%}"
                        if scored_count
                        else "N/A",
                        last=result["status"],
                    )
            except Exception as exc:
                write_json(
                    directory / f"attempt-{attempt}-error.json", {"error": repr(exc)}
                )
                if attempt == stream_attempts - 1:
                    raise
            pending = [
                e["question"] for e in examples if e["question"]["id"] not in rows
            ]
    if pending:
        raise RuntimeError("Remote evaluation ended with missing questions")
    ordered = sorted(rows.values(), key=lambda r: r["example_index"])
    summary = export_rows(ordered, directory)
    technical = [
        r
        for r in ordered
        if r["status"] in ("infrastructure_error", "postprocessing_error")
    ]
    status = "completed_with_errors" if technical else "completed"
    write_json(
        directory / "status.json",
        {"status": status, "total": len(examples), "saved": len(rows)},
    )
    print(json.dumps({"phase": name, "summary": summary}, indent=2), flush=True)
    return not technical and len(rows) == len(examples)
