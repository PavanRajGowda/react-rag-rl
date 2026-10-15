"""Episode-level grouping/advantages without importing any GPU/ML packages."""

import math
import subprocess
import sys
from dataclasses import replace

import pytest

from grpo_training.advantages import prepare_update
from grpo_training.config import TrainingConfig
from grpo_training.data import Question, RewardLabel
from grpo_training.records import (
    EpisodeRecord,
    GenerationRecord,
    GenerationRequest,
    PolicyVersion,
    SamplingParameters,
    generation_seed,
)
from grpo_training.replay import GroupRequiresRetry, QuestionGroup


def make_group(
    config=None,
    *,
    number=0,
    answers=("42", "9"),
    lengths=((2, 3), (1,)),
    status=None,
    policy=None,
):
    config = config or TrainingConfig(num_generations=2, questions_per_update=1)
    policy = policy or PolicyVersion(
        config.model_revision, config.model_revision, "a" * 64, 0, "b" * 64
    )
    question = Question(
        f"finqa_train_{number}", "Revenue?", "train", number, config.dataset_revision
    )
    sampling = SamplingParameters(config.temperature, config.top_p)
    episodes = []
    for index, (answer, turns) in enumerate(zip(answers, lengths, strict=True)):
        episode_id, group_id, seed = f"e-{number}-{index}", f"g-{number}", 42 + index
        records = []
        for turn, size in enumerate(turns, 1):
            request = GenerationRequest(
                episode_id,
                group_id,
                question.id,
                index,
                turn,
                generation_seed(seed, turn),
                policy,
                sampling,
                "<reasoning>",
                (1,) * (turn + 1),
            )
            records.append(
                GenerationRecord(
                    request,
                    tuple((i % 3) + 2 for i in range(size)),
                    (-2.0,) * size,
                    "dummy",
                    "action",
                )
            )
        episodes.append(
            EpisodeRecord(
                episode_id,
                group_id,
                index,
                seed,
                question,
                config,
                policy,
                sampling,
                status or ("answered" if turns else "context_limit"),
                answer if turns else None,
                tuple(records),
            )
        )
    return QuestionGroup(tuple(episodes))


def label(group):
    return RewardLabel(group.episodes[0].question.id, "42", "diagnostic-only")


def test_one_advantage_per_episode_not_per_turn():
    group = make_group()
    prepared = prepare_update([group], [label(group)])
    expected = 0.5 / (math.sqrt(0.5) + 1e-4)
    assert [s.reward for s in prepared.signals] == [1.0, 0.0]
    assert [s.advantage for s in prepared.signals] == pytest.approx(
        [expected, -expected]
    )
    assert [r["advantage"] for r in prepared.rows()] == pytest.approx(
        [expected, expected, -expected]
    )
    assert prepared.token_count == 6
    assert all("gold" not in key for row in prepared.rows() for key in row)


@pytest.mark.parametrize("mode", ["group", "batch", "none"])
def test_center_each_group_and_choose_explicit_scale(mode):
    config = TrainingConfig(
        num_generations=2, questions_per_update=2, scale_rewards=mode
    )
    a = make_group(config)
    b = make_group(config, number=1, answers=("42", "42"))
    prepared = prepare_update([a, b], [label(a), label(b)])
    denominator = {"group": math.sqrt(0.5) + 1e-4, "batch": 0.5 + 1e-4, "none": 1}[mode]
    assert [s.advantage for s in prepared.signals] == pytest.approx(
        [0.5 / denominator, -0.5 / denominator, 0, 0]
    )


@pytest.mark.parametrize("answers", [("42", "42"), ("9", "9")])
def test_zero_variance_groups_remain_present(answers):
    group = make_group(answers=answers)
    prepared = prepare_update([group], [label(group)])
    assert len(prepared.signals) == 2 and len(prepared.rows()) == 3
    assert all(s.advantage == 0 for s in prepared.signals)


def test_zero_token_context_failure_still_contributes_to_baseline():
    group = make_group(lengths=((1,), ()))
    prepared = prepare_update([group], [label(group)])
    assert [s.reward for s in prepared.signals] == [1.0, 0.0]
    assert len(prepared.rows()) == 1 and prepared.signals[0].advantage > 0


def test_technical_failure_rejects_update():
    group = make_group(status="infrastructure_error")
    with pytest.raises(GroupRequiresRetry):
        prepare_update([group], [label(group)])


def test_no_tokens_is_not_a_fake_training_batch():
    group = make_group(lengths=((), ()))
    with pytest.raises(ValueError, match="no sampled tokens"):
        prepare_update([group], [label(group)])


def test_label_and_batch_membership_validation():
    group = make_group()
    with pytest.raises(ValueError, match="Labels"):
        prepare_update([group], [])
    with pytest.raises(ValueError, match="unique"):
        prepare_update([group], [label(group), label(group)])
    with pytest.raises(ValueError, match="questions_per_update"):
        prepare_update([group, group], [label(group)])
    dev = QuestionGroup(
        tuple(
            replace(e, question=replace(e.question, split="dev"))
            for e in group.episodes
        )
    )
    with pytest.raises(ValueError, match="train-only"):
        prepare_update([dev], [label(dev)])


def test_mixed_policy_and_duplicate_questions_rejected():
    config = TrainingConfig(num_generations=2, questions_per_update=2)
    first = make_group(config)
    second = make_group(
        config, number=1, policy=replace(first.episodes[0].policy, optimizer_step=1)
    )
    with pytest.raises(ValueError, match="one config/policy"):
        prepare_update([first, second], [label(first), label(second)])
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_update([first, first], [label(first)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon_low": 1},
        {"epsilon_low": 0},
        {"epsilon_high": -1},
        {"epsilon_high": float("nan")},
        {"vllm_importance_sampling_cap": 0.5},
    ],
)
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_imports_still_do_not_load_torch_or_trl():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import grpo_training.advantages; import grpo_training.trainer; assert not {'torch','trl','transformers','vllm'} & sys.modules.keys()",
        ],
        check=True,
    )
