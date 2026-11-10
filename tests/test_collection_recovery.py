"""Shared collection preserves healthy peers during bounded trajectory replacement."""

from types import SimpleNamespace as NS

import pytest

from tests.agent_fixtures import ToyTokenizer
from tinker_training.config import TrainingConfig
from tinker_training.records import PolicyVersion
from tinker_training.policy import template_digest


def test_collector_drops_only_exhausted_group_and_preserves_other_question():
    """Verify collector drops only exhausted group and preserves other question."""
    from tests.rollout_fixtures import ScriptedBackend, inputs
    from tests.agent_fixtures import search
    from tinker_training.backend import Batch as GenerationBatch
    from tinker_training.backend import Failure as BackendFailure
    from tinker_training.collection import collect_groups

    config = TrainingConfig(num_generations=2, questions_per_update=2)
    train = inputs(config)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )

    class Backend(ScriptedBackend):
        def generate(self, requests):
            """Inject the selected trajectory or batch failure while returning healthy
            scripted peers where applicable.
            """
            batch = super().generate(requests)
            return GenerationBatch(
                tuple(
                    BackendFailure(
                        g.request, TimeoutError("temporary generation error")
                    )
                    if g.request.question_id == train.questions[0].id
                    and g.request.sample_index == 0
                    else g
                    for g in batch.generations
                ),
                batch.seconds,
            )

    backend = Backend(tokenizer)
    events = []
    groups = collect_groups(
        train.questions[:2],
        config,
        policy,
        backend,
        tokenizer,
        search,
        0,
        audit=events.append,
    )
    assert len(groups) == 1 and groups[0].episodes[0].question == train.questions[1]
    assert sum(backend.batch_sizes) == 8
    assert set(backend.batch_sizes) == {1}
    assert all("retry-0" in e.episode_id for e in groups[0].episodes)
    assert len([e for e in events if e["kind"] == "group_dropped"]) == 1
    assert len([e for e in events if e["kind"] == "trajectory_failed"]) == 2


def test_completed_trajectories_are_audited_before_later_engine_failure():
    """Verify completed trajectories are audited before later engine failure."""
    from tests.rollout_fixtures import inputs
    from tests.agent_fixtures import ANSWER, SEARCH, search
    from tinker_training.backend import Batch as GenerationBatch
    from tinker_training.backend import Generation as BackendGeneration
    from tinker_training.collection import collect_groups

    config = TrainingConfig(num_generations=2, questions_per_update=2)
    train = inputs(config)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    calls = []
    events = []

    def generate(requests):
        """Inject the selected trajectory or batch failure while returning healthy scripted
        peers where applicable.
        """
        calls.append(1)
        if requests[0].question_id == train.questions[0].id and requests[0].turn == 4:
            raise RuntimeError("shared engine OOM")
        items = []
        for request in requests:
            text = (
                SEARCH
                if request.turn == 1 or request.question_id == train.questions[0].id
                else ANSWER
            )
            ids = tuple(tokenizer.encode(text, add_special_tokens=False))
            items.append(
                BackendGeneration(request, ids, (-2.0,) * len(ids), "action", 1)
            )
        return GenerationBatch(tuple(items), 0.01)

    with pytest.raises(RuntimeError, match="engine OOM"):
        collect_groups(
            train.questions[:2],
            config,
            policy,
            NS(generate=generate),
            tokenizer,
            search,
            0,
            audit=events.append,
        )
    completed = [e for e in events if e["kind"] == "trajectory_completed"]
    assert len(completed) == 2 and all(
        e["question_id"] == train.questions[1].id for e in completed
    )


def test_collection_audits_never_duplicate_full_trajectories():
    """Journal events carry counters and IDs, not large token/search payloads."""
    from tests.rollout_fixtures import ScriptedBackend, inputs
    from tests.agent_fixtures import search
    from tinker_training.collection import collect_groups

    config = TrainingConfig(num_generations=2, questions_per_update=1)
    train = inputs(config)
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    audit = []
    groups = collect_groups(
        train.questions[:1],
        config,
        policy,
        ScriptedBackend(tokenizer),
        tokenizer,
        search,
        0,
        audit=audit.append,
    )
    assert len(groups) == 1
    assert all("episode" not in e and "episodes" not in e for e in audit)
