import json

import pytest

from baseline_agent.agent import AgentConfig
from baseline_agent.batch import PersistenceError, run_batches
from baseline_agent.data import load_evaluation_examples
from baseline_agent.runner import run_evaluation, validate_resume


def question(index):
    return {"id": f"q{index}", "question": "Revenue?", "example_index": index}


def test_stream_retry_preserves_partial_results_and_skips_duplicates():
    calls, saved = [], []

    def invoke(questions):
        calls.append([q["id"] for q in questions])
        for q in questions:
            yield {"question": q}
            if len(calls) == 1:
                yield {"question": q}  # Server replay: do not count twice.
                raise ConnectionError("stream dropped")

    run_batches(
        [question(i) for i in range(3)],
        invoke,
        lambda r: saved.append(r["question"]["id"]),
        lambda q, e: {"question": q},
        lambda error: None,
        sleep=lambda seconds: None,
    )
    assert calls == [["q0", "q1", "q2"], ["q1", "q2"]]
    assert saved == ["q0", "q1", "q2"]


def test_exhausted_global_failure_is_recorded_and_breaks_circuit():
    calls, saved, logs = [], [], []

    def unavailable(questions):
        calls.append(questions)
        raise TimeoutError("worker unavailable")

    run_batches(
        [question(i) for i in range(5)],
        unavailable,
        saved.append,
        lambda q, e: {"question": q, "status": "infrastructure_error"},
        logs.append,
        batch_size=2,
        attempts=2,
        sleep=lambda seconds: None,
    )
    assert len(calls) == 2
    assert len(saved) == 5
    assert all(r["status"] == "infrastructure_error" for r in saved)
    assert "Circuit breaker" in logs[-1]["error"]


def test_storage_failure_is_not_misreported_as_model_failure():
    def cannot_save(result):
        raise PersistenceError("disk full")

    with pytest.raises(PersistenceError, match="disk full"):
        run_batches(
            [question(0)],
            lambda qs: iter([{"question": qs[0]}]),
            cannot_save,
            lambda q, e: None,
            lambda e: None,
        )


def test_full_selection_handles_blank_and_removes_old_100_row_cap(monkeypatch):
    import datasets

    rows = [
        {
            "id": f"q{i}",
            "question": "" if i == 49 else "Revenue?",
            "program_answer": "1",
            "context_id": f"c{i}",
        }
        for i in range(120)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: rows)
    examples = load_evaluation_examples(0, None)
    assert len(examples) == 120
    assert examples[49]["skip_reason"] == "blank_question"
    assert examples[49]["question"]["question"] == ""
    assert "gold_context_id" not in examples[0]["question"]
    assert len(load_evaluation_examples(0, 120)) == 120


def local_setup(monkeypatch, tmp_path, count=5, blank=False):
    from baseline_agent import runner

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/finqa/manifest.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"corpus_version": "corpus"}))
    examples = [
        {
            "question": question(i),
            "gold_answer": str(i),
            "gold_context_id": f"ctx{i}",
            "skip_reason": "blank_question" if blank else None,
        }
        for i in range(count)
    ]
    monkeypatch.setattr(
        runner,
        "load_evaluation_examples",
        lambda start, selected: (
            examples[start:] if selected is None else examples[start : start + selected]
        ),
    )
    return examples


def result(q, config, revision, status="answered"):
    return {
        "question": q,
        "model_id": "model",
        "model_revision": revision,
        "gpu": "H100",
        "config": config | {"seed": config["seed"] + q["example_index"]},
        "status": status,
        "answer": str(q["example_index"]),
        "turns": [],
        "searches": [],
        "events": [],
        "inference_seconds": 1.0,
    }


def test_pilot_can_resume_into_full_set_without_rerunning_saved_rows(
    monkeypatch, tmp_path, capsys
):
    local_setup(monkeypatch, tmp_path)
    calls = []

    def invoke(questions, config, revision):
        assert all(
            "gold_answer" not in q and "gold_context_id" not in q for q in questions
        )
        calls.extend(q["id"] for q in questions)
        return iter(result(q, config, revision) for q in questions)

    options = {
        "invoke": invoke,
        "download": lambda revision: None,
        "probe": lambda: {
            "status": "ok",
            "corpus_version": "corpus",
            "search_schema_version": 4,
        },
        "model_id": "model",
        "revision": "revision",
        "gpu": "H100",
    }
    directory = run_evaluation(**options, count=2)
    run_evaluation(**options, count=None, resume=str(directory))
    assert calls == ["q0", "q1", "q2", "q3", "q4"]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["examples_completed"] == 5
    assert summary["infrastructure_errors"] == 0
    assert json.loads((directory / "run.json").read_text())["status"] == "completed"
    assert "FinQA H100" in capsys.readouterr().err


def test_all_blank_or_dry_run_never_calls_search_download_or_gpu(monkeypatch, tmp_path):
    local_setup(monkeypatch, tmp_path, count=1, blank=True)

    def unexpected(*args):
        raise AssertionError("should not allocate resources")

    options = {
        "invoke": unexpected,
        "download": unexpected,
        "probe": unexpected,
        "model_id": "model",
        "revision": "revision",
        "gpu": "H100",
        "count": 1,
    }
    run_evaluation(**options, dry_run=True)
    directory = run_evaluation(**options)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["examples_skipped"] == 1
    assert summary["examples_scored"] == 0


def test_preflight_stale_schema_fails_before_gpu_allocation(monkeypatch, tmp_path):
    local_setup(monkeypatch, tmp_path, count=1)
    with pytest.raises(RuntimeError, match="schema is stale"):
        run_evaluation(
            lambda *args: pytest.fail("GPU invoked"),
            lambda revision: pytest.fail("download invoked"),
            lambda: {"status": "ok", "search_schema_version": 3},
            "model",
            "revision",
            "H100",
            count=1,
        )


def test_resume_rejects_changed_instructions_or_model():
    with pytest.raises(ValueError, match="Resume mismatch"):
        validate_resume({"model_revision": "old"}, {"model_revision": "new"})


def test_worker_failure_records_error_and_next_question_still_finishes(
    monkeypatch, tmp_path
):
    local_setup(monkeypatch, tmp_path, count=2)
    from baseline_agent.batch import failure_result

    def invoke(questions, config, revision):
        for q in questions:
            if q["example_index"] == 0:
                yield result(q, config, revision) | failure_result(
                    q, AgentConfig(**config), RuntimeError("search down")
                )
            else:
                yield result(q, config, revision)

    directory = run_evaluation(
        invoke,
        lambda revision: None,
        lambda: {
            "status": "ok",
            "corpus_version": "corpus",
            "search_schema_version": 4,
        },
        "model",
        "revision",
        "H100",
        count=2,
    )
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["examples_completed"] == 2
    assert summary["infrastructure_errors"] == 1


def test_corrupt_scored_checkpoint_recovers_from_raw_without_gpu(monkeypatch, tmp_path):
    local_setup(monkeypatch, tmp_path, count=1)
    probe = lambda: {
        "status": "ok",
        "corpus_version": "corpus",
        "search_schema_version": 4,
    }
    directory = run_evaluation(
        lambda qs, c, r: iter(result(q, c, r) for q in qs),
        lambda r: None,
        probe,
        "model",
        "revision",
        "H100",
        count=1,
    )
    (directory / "q0.json").write_text("{broken")
    run_evaluation(
        lambda *args: pytest.fail("GPU rerun"),
        lambda r: pytest.fail("download rerun"),
        lambda: pytest.fail("search called"),
        "model",
        "revision",
        "H100",
        count=1,
        resume=str(directory),
    )
    assert json.loads((directory / "q0.json").read_text())["answer"] == "0"


def test_retry_errors_resumes_only_failed_examples(monkeypatch, tmp_path):
    local_setup(monkeypatch, tmp_path, count=2)
    probe = lambda: {
        "status": "ok",
        "corpus_version": "corpus",
        "search_schema_version": 4,
    }
    directory = run_evaluation(
        lambda qs, c, r: iter(
            result(
                q,
                c,
                r,
                "infrastructure_error" if q["example_index"] == 0 else "answered",
            )
            for q in qs
        ),
        lambda r: None,
        probe,
        "model",
        "revision",
        "H100",
        count=2,
    )
    repeated = []

    def invoke(questions, config, revision):
        repeated.extend(q["id"] for q in questions)
        return iter(result(q, config, revision) for q in questions)

    run_evaluation(
        invoke,
        lambda r: None,
        probe,
        "model",
        "revision",
        "H100",
        count=2,
        resume=str(directory),
        retry_errors=True,
    )
    assert repeated == ["q0"]
    assert (
        json.loads((directory / "summary.json").read_text())["infrastructure_errors"]
        == 0
    )
