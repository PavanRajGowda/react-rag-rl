"""Bounded reward invariants and audited group integration; no model calls."""

import json
from dataclasses import replace

import pytest

from tests.test_grpo_advantages import make_group
from tinker_training.advantages import group_advantages, prepare_update
from tinker_training.config import TrainingConfig
from tinker_training.data import RewardLabel
from tinker_training.replay import QuestionGroup
from tinker_training.rewards import (
    DocumentOutcome,
    bounded_process_reward,
    episode_reward,
    group_outcome,
)


def score(answer="42", status="answered"):
    """Create a deterministic terminal score with an independent private label."""
    return episode_reward("q", status, answer, RewardLabel("q", "42", ("d",)))


def test_coverage_counts_documents_not_searches():
    """One search retrieving multiple gold IDs can earn full coverage."""
    cfg = TrainingConfig(reward="bounded_process_trajectory_v1")
    turns = [{"kind": "search", "Q": 0, "I": 0}, {"kind": "answer", "Q": 0, "I": 0}]
    result = bounded_process_reward(score(), DocumentOutcome(2, 2, True), turns, cfg)
    assert result["coverage"] == 1
    assert result["reward"] == pytest.approx(1 + 0.25 / 3)
    assert (
        bounded_process_reward(score(), DocumentOutcome(0, None, None), turns, cfg)[
            "coverage"
        ]
        == 0
    )


def test_penalty_rates_and_empty_episodes():
    """Count OR search flags once; protocol denominator includes final answers."""
    cfg = TrainingConfig()
    turns = [
        {"kind": "search", "Q": 1, "I": 0},
        {"kind": "search", "Q": 0, "I": 0},
        {"kind": "invalid", "Q": 0, "I": 1},
        {"kind": "answer", "Q": 0, "I": 0},
    ]
    result = bounded_process_reward(
        score("9"), DocumentOutcome(2, 1, False), turns, cfg
    )
    assert result["penalized_search_rate"] == 0.5
    assert result["protocol_violation_rate"] == 0.25
    assert result["reward"] == pytest.approx(-0.25 / 12)
    assert (
        bounded_process_reward(
            score(None, "turn_limit"), DocumentOutcome(0, None, None), [], cfg
        )["reward"]
        == 0
    )
    assert (
        bounded_process_reward(
            score(None, "infrastructure_error"), DocumentOutcome(1, None, None), [], cfg
        )["reward"]
        is None
    )
    with pytest.raises(ValueError, match="audited"):
        bounded_process_reward(score(), DocumentOutcome(1, None, None), [], cfg)


@pytest.mark.parametrize(
    "weights", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1 / 3, 1 / 3, 1 / 3)]
)
def test_bounds_and_correctness_priority(weights):
    """All admissible component extremes preserve correct > incorrect ordering."""
    cfg = TrainingConfig(
        process_bonus_scale=0.49,
        process_coverage_weight=weights[0],
        process_redundancy_weight=weights[1],
        process_invalid_weight=weights[2],
    )
    correct = []
    incorrect = []
    for coverage in [0, 1]:
        for q in [0, 1]:
            for invalid in [0, 1]:
                turns = [{"kind": "search", "Q": q, "I": invalid}]
                doc = DocumentOutcome(1, coverage, bool(coverage))
                for terminal, values in [(score(), correct), (score("9"), incorrect)]:
                    result = bounded_process_reward(terminal, doc, turns, cfg)
                    assert -1 <= result["process_quality"] <= 1
                    values.append(result["reward"])
    assert min(correct) > max(incorrect)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"process_bonus_scale": 0.5},
        {"process_bonus_scale": float("nan")},
        {"process_coverage_weight": -0.1},
        {"process_coverage_weight": 1},
        {"process_invalid_weight": True},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    """Reject configurations that invalidate the mathematical bounds."""
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_group_uses_real_audit_and_broadcast_advantage():
    """Search audit yields bounded score; group-relative credit reaches every turn."""
    cfg = TrainingConfig(
        num_generations=2,
        questions_per_update=1,
        reward="bounded_process_trajectory_v1",
    )
    group = make_group(cfg, lengths=((2, 3), (2, 3)))
    episodes = []
    for episode in group.episodes:
        audit = {
            "question": {"id": episode.question.id},
            "turns": [
                {
                    "turn": 1,
                    "model_output": "<reasoning>Find.</reasoning><search>fact</search>",
                },
                {
                    "turn": 2,
                    "model_output": f"<reasoning>Done.</reasoning><answer>{episode.answer}</answer>",
                },
            ],
            "searches": [
                {
                    "turn": 1,
                    "response": {
                        "status": "ok",
                        "results": [{"document_id": "d", "text": "Fact"}],
                    },
                }
            ],
        }
        episodes.append(replace(episode, controller_audit_json=json.dumps(audit)))
    group = QuestionGroup(tuple(episodes))
    label = RewardLabel(episodes[0].question.id, "42", ("d",))
    outcome = group_outcome(group, label)
    assert outcome.rewards == pytest.approx((1 + 0.25 / 3, 0.25 / 3))
    prepared = prepare_update((group,), (label,))
    by_id = {s.episode_id: s.advantage for s in prepared.signals}
    for row in prepared.rows():
        assert row["advantage"] == by_id[row["episode_id"]]
        assert "token_advantages" not in row


def test_combined_centering_equals_component_centering():
    """Use real reward computation with differing outcomes and process rates."""
    cfg = TrainingConfig(reward="bounded_process_trajectory_v1")
    components = [(1, 1.0, 0.25, 0.0), (1, 0.5, 0.5, 0.2),
                  (0, 1.0, 0.0, 0.0), (0, 0.0, 1.0, 0.2)]
    results = []
    for correct, coverage, q, invalid in components:
        turns = [{"kind": "search", "Q": int(k < 4 * q), "I": 0}
                 for k in range(4)]
        turns.append({"kind": "invalid" if invalid else "answer", "Q": 0,
                      "I": int(bool(invalid))})
        results.append(bounded_process_reward(
            score("42" if correct else "9"),
            DocumentOutcome(2, int(2 * coverage), coverage == 1), turns, cfg))
    combined = group_advantages([r["reward"] for r in results])
    outcome = group_advantages([c[0] for c in components])
    coverage = group_advantages([r["coverage"] for r in results])
    search = group_advantages([r["penalized_search_rate"] for r in results])
    protocol = group_advantages([r["protocol_violation_rate"] for r in results])
    separately = [o + cfg.process_bonus_scale * (
        cfg.process_coverage_weight * c - cfg.process_redundancy_weight * q
        - cfg.process_invalid_weight * i)
        for o, c, q, i in zip(outcome, coverage, search, protocol, strict=True)]
    assert combined == pytest.approx(separately)
    assert sum(combined) == pytest.approx(0)
    assert results[0]["reward"] == pytest.approx(1.0625)
