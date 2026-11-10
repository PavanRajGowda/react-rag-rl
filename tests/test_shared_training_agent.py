"""Baseline/training controller parity and compact replay recording."""

import json
from dataclasses import replace

from batched_inference.baseline import batched_agents
from tests.rollout_fixtures import ScriptedBackend, inputs
from tests.agent_fixtures import ToyTokenizer, search
from tinker_training.config import TrainingConfig
from tinker_training.records import PolicyVersion, SamplingParameters
from tinker_training.policy import template_digest
from tinker_training.trajectory import generate_episodes


def test_baseline_and_training_have_identical_prompts_tokens_and_searches():
    """Shared controller yields exactly matching histories for the same seeded inputs."""
    config = TrainingConfig(num_generations=2, questions_per_update=1)
    question = inputs(config).questions[0]
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    spec = {
        "episode_id": "e",
        "group_id": "g",
        "sample_index": 1,
        "seed": 42,
        "question": question,
    }
    episode = next(
        generate_episodes(
            [spec], config, policy, ScriptedBackend(tokenizer), tokenizer, search
        )
    )
    q = {
        "id": "e",
        "group_id": "g",
        "training_question_id": question.id,
        "sample_index": 1,
        "sampling_seed": 42,
        "question": question.text,
        "example_index": question.source_index,
    }
    result, _ = next(
        batched_agents(
            [q],
            ScriptedBackend(tokenizer),
            tokenizer,
            policy,
            config,
            search,
            1,
            sampling=SamplingParameters(1, 1),
        )
    )
    assert episode.status == result["status"] == "answered"
    assert episode.answer == result["answer"]
    audit = json.loads(episode.controller_audit_json)
    assert [s["request"] for s in audit["searches"]] == [
        s["request"] for s in result["searches"]
    ]
    for record, turn in zip(episode.generations, result["turns"], strict=True):
        assert record.request.prompt_ids == tuple(turn["prompt_ids"])
        assert record.request.prompt_text == turn["model_input"]
        assert record.completion_ids == tuple(turn["completion_ids"])
        assert record.sampling_logprobs == tuple(turn["sampling_logprobs"])
    assert all("prompt_ids" not in t and "model_input" not in t for t in audit["turns"])


def test_training_validation_uses_shared_agent(tmp_path):
    """The validation entry point uses shared collection and records accepted answers."""
    from tinker_training.validation import evaluate

    cfg = TrainingConfig()
    prepared = inputs(cfg)
    dev = replace(
        prepared,
        split="dev",
        questions=tuple(replace(q, split="dev") for q in prepared.questions),
    )
    tokenizer = ToyTokenizer()
    policy = PolicyVersion(
        cfg.model_revision, cfg.model_revision, "base", 0, template_digest(tokenizer)
    )
    result = evaluate(
        dev, cfg, policy, ScriptedBackend(tokenizer), tokenizer, search, tmp_path
    )
    assert result["questions"] == 3 and result["correct"] == 3
