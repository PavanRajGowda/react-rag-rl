"""Reference scoring, token masks and frozen minibatch reuse without hosted calls."""

from types import SimpleNamespace

import pytest

from tests.test_grpo_advantages import label, make_group
from tests.test_tinker_migration import TYPES, Future
from tinker_training.advantages import minibatch_update, prepare_update
from tinker_training.config import TrainingConfig
from tinker_training.kl import apply_reference_kl
from tinker_training.loss import pack_rows


def test_reference_kl_alignment_mask_and_resume(tmp_path):
    cfg = TrainingConfig(
        beta=0.01, questions_per_update=2, minibatch_questions=1, num_generations=2
    )
    groups = tuple(make_group(cfg, number=i) for i in range(2))
    prepared = prepare_update(groups, tuple(label(g) for g in groups))
    calls = []

    def score(prompt, prompt_logprobs_last_n):
        calls.append(prompt)
        n = prompt_logprobs_last_n
        return Future([None] * (len(prompt) - n) + [-2.0 - i for i in range(n)])

    reference = SimpleNamespace(compute_logprobs=score)
    path = tmp_path / "reference.json"
    adjusted, metrics = apply_reference_kl(prepared, reference, TYPES, path)
    assert len(calls) == len(prepared.rows())
    for original, row in zip(prepared.rows(), adjusted.rows(), strict=True):
        assert row["input_ids"] == original["input_ids"]
        assert row["sampling_logprobs"] == original["sampling_logprobs"]
        assert len(row["token_advantages"]) == len(row["target_ids"])
    delta = [w - r["advantage"] for r in adjusted.rows() for w in r["token_advantages"]]
    assert sum(delta) == pytest.approx(0, abs=1e-12)
    assert max(delta) > 0 > min(delta)
    mini, _ = minibatch_update(adjusted, 0)
    packed = pack_rows(mini, TYPES)
    for row, datum in zip(mini.rows(), packed, strict=True):
        n = len(row["target_ids"])
        values = datum.loss_fn_inputs["advantages"].data
        assert values[:-n] == [0] * (len(values) - n)
        assert values[-n:] == pytest.approx(
            [v / mini.token_count for v in row["token_advantages"]]
        )
    resumed, _ = apply_reference_kl(prepared, SimpleNamespace(), TYPES, path)
    assert resumed.token_weights == adjusted.token_weights
    assert metrics["beta"] == 0.01
    with pytest.raises(ValueError, match="identity"):
        other = prepare_update(
            (make_group(cfg, number=9),), (label(make_group(cfg, number=9)),)
        )
        apply_reference_kl(other, reference, TYPES, path)


def test_missing_reference_completion_logprob_rejected(tmp_path):
    group = make_group(TrainingConfig(beta=0.01, num_generations=2))
    prepared = prepare_update((group,), (label(group),))
    ref = SimpleNamespace(
        compute_logprobs=lambda prompt, **_: Future([None] * len(prompt))
    )
    with pytest.raises(ValueError, match="reference completion"):
        apply_reference_kl(prepared, ref, TYPES, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_runner_uses_frozen_base_reference_across_updates(tmp_path):
    from tests.rollout_fixtures import inputs
    from tests.test_tinker_migration import Service, search
    from tinker_training.runner import run

    cfg = TrainingConfig(
        beta=0.01,
        num_generations=2,
        questions_per_update=1,
        num_iterations=2,
        max_steps=2,
    )
    service = Service()
    scored = []

    def create(base_model):
        assert base_model == cfg.model_id

        def score(prompt, prompt_logprobs_last_n):
            scored.append(1)
            n = prompt_logprobs_last_n
            return Future(
                [None] * (len(prompt) - n) + [-1.5 - i * 0.001 for i in range(n)]
            )

        return SimpleNamespace(compute_logprobs=score)

    service.create_sampling_client = create
    train = inputs(cfg)
    state = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": "reference"},
        validation_count=0,
    )
    assert state["step"] == 2
    assert len(scored) == 4  # Two turns per sibling, scored only once across reuse.
    assert state["metrics"]["reference_kl"]["beta"] == 0.01
