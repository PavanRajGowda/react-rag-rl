"""Outcome reward, dynamic eligibility and replay advantages; CPU only."""

import math
import statistics
from dataclasses import replace

import pytest

from grpo_training.advantages import prepare_update
from grpo_training.checkpoints import capture_dict, restore_capture
from grpo_training.config import TrainingConfig
from grpo_training.metrics import rollout_metrics
from grpo_training.replay import GroupRequiresRetry
from grpo_training.rewards import group_outcome
from grpo_training.runner import run_training
from grpo_training.sampling import collect_dynamic, has_learning_signal
from grpo_training.trainer import CapturedUpdate
from tests.test_grpo_advantages import label, make_group
from tests.test_grpo_runner import inputs, setup_run


def config(**kwargs):
    return TrainingConfig(
        reward="number_match_turn_efficiency",
        max_turns=3,
        num_generations=4,
        questions_per_update=1,
        **kwargs,
    )


def group(answers=("42", "42", "42", "9"), turns=(2, 2, 3, 1), **kwargs):
    return make_group(
        config(**kwargs), answers=answers, lengths=tuple((1,) * n for n in turns)
    )


def test_mixed_group_exact_rewards_and_signed_normalized_advantages():
    peers = group()
    outcome = group_outcome(peers, label(peers))
    assert outcome.rewards == pytest.approx((1, 1, 0.6, 0))
    assert [s.reward for s in outcome.scores] == [1, 1, 1, 0]
    assert outcome.min_successful_turns == 2
    update = prepare_update([peers], [label(peers)])
    expected = [
        (v - 0.65) / (statistics.stdev([1, 1, 0.6, 0]) + 1e-4) for v in [1, 1, 0.6, 0]
    ]
    assert [s.advantage for s in update.signals] == pytest.approx(expected)
    assert update.signals[2].advantage < 0  # Signed in mixed groups, by design.
    assert update.signals[3].advantage < update.signals[2].advantage
    assert [row["advantage"] for row in update.rows()] == pytest.approx(
        [
            a
            for a, turns in zip(expected, (2, 2, 3, 1), strict=True)
            for _ in range(turns)
        ]
    )


def test_all_correct_clip_after_normalization_without_alpha_or_recentering():
    peers = group(answers=("42",) * 4, turns=(2, 2, 2, 3))
    update = prepare_update([peers], [label(peers)])
    expected = 0.1 / (0.2 + 1e-4)
    assert [s.advantage for s in update.signals] == pytest.approx([expected] * 3 + [0])
    assert has_learning_signal(peers, {label(peers).id: label(peers)})
    metrics, traces = rollout_metrics(update, [label(peers)])
    row = metrics["episodes"][-1]
    assert row["training_reward"] == pytest.approx(0.6)
    assert row["advantage_raw"] == pytest.approx(-0.3)
    assert row["advantage_normalized"] < 0 and row["advantage"] == 0
    assert traces[peers.episodes[-1].episode_id]["training_signal"]["advantage"] == 0
    assert metrics["groups"][0]["all_correct"]
    assert metrics["groups"][0]["learning_signal"]
    assert not metrics["groups"][0]["answer_learning_signal"]
    assert metrics["rates"]["nm"]["value"] == 1  # Correctness is not shaped reward.


@pytest.mark.parametrize(
    "answers,turns,expected",
    [
        (("9",) * 4, (1, 2, 3, 2), (0,) * 4),
        (("42",) * 4, (3,) * 4, (1,) * 4),
    ],
)
def test_uniform_groups_do_not_create_signal(answers, turns, expected):
    peers = group(answers=answers, turns=turns)
    outcome = group_outcome(peers, label(peers))
    assert outcome.rewards == expected and not outcome.learning_signal
    assert all(
        s.advantage == 0 for s in prepare_update([peers], [label(peers)]).signals
    )


def test_lambda_zero_and_numeric_type_preserve_original_behavior():
    peers = group(turn_penalty_lambda=0)
    numeric = replace(peers.episodes[0].config, reward="number_match_0_1")
    baseline = make_group(
        numeric,
        answers=("42", "42", "42", "9"),
        lengths=((1, 1), (1, 1), (1, 1, 1), (1,)),
    )
    assert (
        prepare_update([peers], [label(peers)]).signals
        == prepare_update([baseline], [label(baseline)]).signals
    )
    numeric = replace(numeric, turn_penalty_lambda=100)
    assert (
        numeric.reward == "number_match_0_1"
    )  # Unused coefficient has no shaping effect.


@pytest.mark.parametrize("value", [-1, True, math.inf, math.nan])
def test_invalid_lambda(value):
    with pytest.raises(ValueError, match="turn_penalty_lambda"):
        config(turn_penalty_lambda=value)


def test_success_ordering_guard_and_config_roundtrip():
    with pytest.raises(ValueError, match="above incorrect"):
        config(turn_penalty_lambda=0.5)
    c = config()
    assert TrainingConfig.from_dict(c.to_dict()) == c
    assert TrainingConfig().reward == "number_match_0_1"


def test_lambda_controls_only_excess_successful_turn_penalty():
    peers = group(turn_penalty_lambda=0.1)
    assert group_outcome(peers, label(peers)).rewards == pytest.approx((1, 1, 0.9, 0))


@pytest.mark.parametrize("mode", ["group", "batch", "none"])
def test_all_correct_clip_uses_selected_scale_and_shared_batch_statistics(mode):
    c = replace(config(), questions_per_update=2, scale_rewards=mode)
    a = make_group(
        c, answers=("42", "42", "42", "9"), lengths=((1, 1), (1, 1), (1, 1, 1), (1,))
    )
    b = make_group(
        c, number=1, answers=("42",) * 4, lengths=((1, 1), (1, 1), (1, 1), (1, 1, 1))
    )
    update = prepare_update([a, b], [label(a), label(b)])
    denominator = {
        "group": 0.2 + 1e-4,
        "batch": statistics.stdev([1, 1, 0.6, 0, 1, 1, 1, 0.6]) + 1e-4,
        "none": 1,
    }[mode]
    assert [s.advantage for s in update.signals[4:]] == pytest.approx(
        [0.1 / denominator] * 3 + [0]
    )


def test_unfinished_trajectory_cannot_earn_turn_reward_from_a_stale_correct_answer():
    peers = make_group(
        config(),
        answers=("42",) * 4,
        lengths=((1,), (1, 1), (1, 1, 1), (1,)),
        status="turn_limit",
    )
    outcome = group_outcome(peers, label(peers))
    assert outcome.rewards == (0,) * 4 and outcome.min_successful_turns is None


def test_technical_errors_never_supply_variance_or_fake_negatives():
    peers = make_group(
        config(),
        answers=("42",) * 4,
        lengths=((1,),) * 4,
        status="infrastructure_error",
    )
    outcome = group_outcome(peers, label(peers))
    assert outcome.rewards == (None,) * 4 and not outcome.learning_signal
    with pytest.raises(GroupRequiresRetry):
        prepare_update([peers], [label(peers)])


def test_checkpoint_recomputes_shaping_and_preserves_clipped_advantages():
    peers = group(answers=("42",) * 4, turns=(2, 2, 2, 3))
    prepared = prepare_update([peers], [label(peers)])
    old = tuple((-2.0,) * len(r["target_ids"]) for r in prepared.rows())
    correction = tuple((1.0,) * len(values) for values in old)
    captured = CapturedUpdate(prepared, old, correction)
    restored = restore_capture(capture_dict(captured), [label(peers)])
    assert restored == captured


def test_dynamic_sampling_accepts_all_correct_with_efficiency_signal():
    c = config(dynamic_sampling=True)
    train = inputs(c)
    events = []

    def collect(questions, *args, **kwargs):
        return tuple(
            make_group(
                c,
                number=q.source_index,
                answers=("42",) * 4,
                lengths=((1, 1), (1, 1), (1, 1), (1, 1, 1)),
            )
            for q in questions
        )

    groups, _, consumed = collect_dynamic(
        train, 0, c, None, None, None, None, 0, collect=collect, audit=events.append
    )
    assert len(groups) == consumed == 1
    event = next(e for e in events if e["kind"] == "dynamic_group")
    assert event["accepted"] and event["correct"] == 4
    assert event["training_rewards"] == pytest.approx((1, 1, 1, 0.6))


def test_runner_keeps_all_correct_signal_after_probability_recovery(tmp_path):
    c = config(dynamic_sampling=True, max_steps=2, num_iterations=2)
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(
        tmp_path, config=c
    )

    def collect(questions, config, policy, *args, **kwargs):
        return tuple(
            make_group(
                config,
                number=q.source_index,
                policy=policy,
                answers=("42",) * 4,
                lengths=((1, 1), (1, 1), (1, 1), (1, 1, 1)),
            )
            for q in questions
        )

    result = run_training(
        trainer,
        bundle,
        backend,
        train,
        None,
        store,
        **(kwargs | {"collect_fn": collect}),
    )
    assert result["status"] == "completed" and trainer.state.global_step == 2
    assert trace.count("capture") == 1 and trace.count("backward") == 2
