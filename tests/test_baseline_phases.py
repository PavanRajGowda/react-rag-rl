from tinker_training.config import TrainingConfig


def test_phase_persists_each_result_and_resumes_without_resampling(tmp_path):
    """Verify phase persists each result and resumes without resampling."""
    from dataclasses import asdict

    from baseline_agent.agent import AgentConfig
    from baseline_agent.phases import run_phase

    config = TrainingConfig()
    question = {"id": "q", "question": "Who?", "example_index": 0}
    example = {"question": question, "gold_answer": "Aster", "gold_document_ids": []}
    calls = []

    def invoke(questions, values, concurrency):
        """Yield deterministic trajectory results and record which questions the phase
        actually requested.
        """
        calls.append(questions)
        yield {
            "question": question,
            "status": "answered",
            "answer": "Aster",
            "turns": [],
            "searches": [],
            "events": [],
            "inference_seconds": 1,
            "model_revision": config.model_revision,
            "config": asdict(AgentConfig()),
        }

    assert run_phase("debug", [example], tmp_path, config, 1, invoke)
    assert run_phase("debug", [example], tmp_path, config, 1, invoke)
    assert len(calls) == 1
    assert (tmp_path / "debug/results.csv").exists()
    assert (tmp_path / "debug/raw/q.json").exists()


def test_aggregate_exports_are_batched_but_raw_results_are_immediate(
    tmp_path, monkeypatch
):
    """Persist each trajectory before advancing; export at thresholds and finalization."""
    from dataclasses import asdict

    from baseline_agent import evaluation
    from baseline_agent.agent import AgentConfig
    from baseline_agent.phases import run_phase

    config = TrainingConfig()
    examples = [
        {
            "question": {"id": f"q{i}", "question": "Who?", "example_index": i},
            "gold_answer": "Aster",
            "gold_document_ids": [],
        }
        for i in range(5)
    ]
    exports = []
    original = evaluation.export_rows

    def export(rows, directory):
        """Record aggregate export sizes while exercising the actual file exporter."""
        exports.append(len(rows))
        return original(rows, directory)

    def invoke(questions, values, concurrency):
        """Assert the prior raw result is durable before yielding the next one."""
        for i, question in enumerate(questions):
            if i:
                assert (tmp_path / "debug/raw" / f"q{i - 1}.json").exists()
            yield {
                "question": question,
                "status": "answered",
                "answer": "Aster",
                "turns": [],
                "searches": [],
                "events": [],
                "inference_seconds": 1,
                "model_revision": config.model_revision,
                "config": asdict(AgentConfig()),
            }

    monkeypatch.setattr(evaluation, "export_rows", export)
    assert run_phase(
        "debug",
        examples,
        tmp_path,
        config,
        5,
        invoke,
        export_every=2,
        export_interval_seconds=3600,
    )
    assert exports == [2, 4, 5]


def test_progress_only_keeps_raw_results_without_per_result_dashboard_writes(
    tmp_path, monkeypatch
):
    """Raw resume artifacts remain immediate; live/duplicate presentation files do not."""
    from dataclasses import asdict

    from baseline_agent import phases
    from baseline_agent.agent import AgentConfig

    questions = [
        {"id": f"q{i}", "question": "Who?", "example_index": i} for i in range(3)
    ]
    examples = [
        {"question": q, "gold_answer": "Aster", "gold_document_ids": []}
        for q in questions
    ]
    writes = []
    original = phases.write_json

    def write(path, value):
        writes.append(path.name)
        original(path, value)

    monkeypatch.setattr(phases, "write_json", write)

    def invoke(pending, config, concurrency):
        for i, q in enumerate(pending):
            if i:
                assert (tmp_path / "validation/raw" / f"q{i - 1}.json").exists()
            yield {
                "question": q,
                "status": "answered",
                "answer": "Aster",
                "turns": [],
                "searches": [],
                "events": [],
                "inference_seconds": 1,
                "model_revision": TrainingConfig().model_revision,
                "config": asdict(AgentConfig()),
            }

    assert phases.run_phase(
        "validation", examples, tmp_path, TrainingConfig(), 3, invoke
    )
    assert writes.count("status.json") == 2
    assert len(list((tmp_path / "validation/raw").glob("*.json"))) == 3
    assert not (tmp_path / "validation/q0.json").exists()
    assert not (tmp_path / "validation/q0.txt").exists()
    assert (tmp_path / "validation/results.csv").exists()
