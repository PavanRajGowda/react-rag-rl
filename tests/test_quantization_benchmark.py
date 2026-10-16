from baseline_agent.quantization_benchmark import summarize


def test_incomplete_arm_does_not_report_accuracy():
    assert summarize([], 20)["number_match"] is None
    assert summarize([], 20)["complete"] is False


def test_failure_counts_in_accuracy_denominator():
    row = {
        "score": 1,
        "exact_answer_match": True,
        "generated_tokens": 100,
        "generation_seconds": 2,
        "inference_seconds": 3,
        "peak_allocated_gib": 4,
        "status": "answered",
    }
    failed = row | {
        "score": 0, "exact_answer_match": False, "status": "infrastructure_error"
    }
    summary = summarize([row, failed], 2)
    assert summary["number_match"] == 0.5
    assert summary["generation_tokens_per_second"] == 50
    assert summary["infrastructure_errors"] == 1
