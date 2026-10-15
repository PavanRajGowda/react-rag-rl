"""Immutable rollout data contracts. No generation, reward or loss computation."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

from grpo_training.config import TrainingConfig
from grpo_training.data import Question
from grpo_training.rewards import MODEL_TERMINATIONS, TECHNICAL_TERMINATIONS


def token_ids(values, *, allow_empty=False):
    values = tuple(values)
    if (not values and not allow_empty) or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError(
            "Expected nonnegative integer token IDs, with a nonempty sequence"
        )
    return values


def nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def generation_seed(episode_seed, turn):
    return int(hashlib.sha256(f"{episode_seed}:{turn}".encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class PolicyVersion:
    model_revision: str
    tokenizer_revision: str
    adapter_sha256: str  # SHA of adapter state; 'base' means no trained adapter.
    optimizer_step: int
    template_sha256: str  # Chat-template + system-prompt + prefill identity.

    def __post_init__(self):
        for name in ("model_revision", "tokenizer_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{40}", getattr(self, name)
            ):
                raise ValueError(f"{name} must be a pinned revision")
        for name in ("adapter_sha256", "template_sha256"):
            value = getattr(self, name)
            if name == "adapter_sha256" and value == "base":
                continue
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a SHA256 digest")
        nonnegative_integer(self.optimizer_step, "optimizer_step")

    @property
    def id(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SamplingParameters:
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0  # Generation top-k, distinct from retrieval K.
    repetition_penalty: float = 1.0

    def __post_init__(self):
        for value in (self.temperature, self.top_p, self.repetition_penalty):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Sampling parameters must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be <= 1")
        nonnegative_integer(self.top_k, "generation top_k")


@dataclass(frozen=True)
class GenerationRequest:
    episode_id: str
    group_id: str
    question_id: str
    sample_index: int
    turn: int
    seed: int
    policy: PolicyVersion
    sampling: SamplingParameters
    prompt_text: str
    prompt_ids: tuple[int, ...]
    assistant_prefill: str = "<reasoning>"

    def __post_init__(self):
        for name in (
            "episode_id",
            "group_id",
            "question_id",
            "prompt_text",
            "assistant_prefill",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty text")
        nonnegative_integer(self.sample_index, "sample_index")
        nonnegative_integer(self.seed, "seed")
        if self.seed >= 2**32 or type(self.turn) is not int or self.turn < 1:
            raise ValueError("Invalid seed/turn")
        if (
            type(self.policy) is not PolicyVersion
            or type(self.sampling) is not SamplingParameters
        ):
            raise TypeError("Expected explicit policy and sampling records")
        if not self.prompt_text.endswith(self.assistant_prefill):
            raise ValueError("Prompt must end in its recorded controller prefill")
        object.__setattr__(self, "prompt_ids", token_ids(self.prompt_ids))


@dataclass(frozen=True)
class GenerationRecord:
    request: GenerationRequest
    completion_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    completion_text: str  # Decoded sampled continuation, without controller prefill.
    finish_reason: str  # action / eos / length, as reported by the adapter.
    generation_seconds: float | None = None

    def __post_init__(self):
        if type(self.request) is not GenerationRequest:
            raise TypeError("Expected GenerationRequest")
        object.__setattr__(self, "completion_ids", token_ids(self.completion_ids))
        values = tuple(self.sampling_logprobs)
        if len(values) != len(self.completion_ids) or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or v > 0
            for v in values
        ):
            raise ValueError(
                "One finite nonpositive sampling logprob is required per sampled token"
            )
        object.__setattr__(self, "sampling_logprobs", values)
        if not isinstance(self.completion_text, str) or self.finish_reason not in (
            "action",
            "eos",
            "length",
        ):
            raise ValueError("Invalid decoded text/finish reason")
        value = self.generation_seconds
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("generation_seconds must be finite/nonnegative or None")

    @property
    def controller_output(self):
        return self.request.assistant_prefill + self.completion_text


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    group_id: str
    sample_index: int
    seed: int
    question: Question
    config: TrainingConfig
    policy: PolicyVersion
    sampling: SamplingParameters
    status: str
    answer: str | None
    generations: tuple[GenerationRecord, ...]
    controller_audit_json: str | None = None  # Full history, never a replay input.

    def __post_init__(self):
        if (
            type(self.question) is not Question
            or type(self.config) is not TrainingConfig
            or type(self.policy) is not PolicyVersion
            or type(self.sampling) is not SamplingParameters
        ):
            raise TypeError("Expected typed question, config, policy and sampling")
        if self.status not in MODEL_TERMINATIONS | TECHNICAL_TERMINATIONS:
            raise ValueError("Episode must have a known terminal status")
        nonnegative_integer(self.sample_index, "sample_index")
        nonnegative_integer(self.seed, "seed")
        if (
            any(
                not isinstance(value, str) or not value
                for value in (self.episode_id, self.group_id)
            )
            or self.seed >= 2**32
        ):
            raise ValueError("Missing episode/group ID or invalid seed")
        if self.answer is not None and not isinstance(self.answer, str):
            raise TypeError("Answer must be text or None")
        if self.controller_audit_json is not None:
            if not isinstance(self.controller_audit_json, str):
                raise TypeError("Controller audit must be serialized JSON or None")
            if not isinstance(json.loads(self.controller_audit_json), dict):
                raise ValueError("Controller audit must contain an object")
        if (
            self.question.dataset_revision != self.config.dataset_revision
            or self.question.split not in ("train", "dev")
            or self.policy.model_revision != self.config.model_revision
        ):
            raise ValueError("Episode revisions/split disagree with configuration")
        if (self.sampling.temperature, self.sampling.top_p) != (
            self.config.temperature,
            self.config.top_p,
        ):
            raise ValueError("Sampling and experiment configuration disagree")
        object.__setattr__(self, "generations", tuple(self.generations))
        if self.status == "answered" and (
            not self.generations
            or not isinstance(self.answer, str)
            or not self.answer.strip()
        ):
            raise ValueError(
                "Answered episodes need a sampled generation and accepted answer"
            )
        if not self.generations and self.status not in {
            "context_limit",
            *TECHNICAL_TERMINATIONS,
        }:
            raise ValueError(
                "Only context/technical failures may have no sampled generations"
            )
        if len(self.generations) > self.config.max_turns:
            raise ValueError("Episode exceeds turn limit")
        for index, record in enumerate(self.generations, 1):
            if type(record) is not GenerationRecord:
                raise TypeError("Expected GenerationRecord")
            request = record.request
            if (
                request.episode_id,
                request.group_id,
                request.sample_index,
                request.seed,
                request.question_id,
                request.turn,
                request.policy,
                request.sampling,
            ) != (
                self.episode_id,
                self.group_id,
                self.sample_index,
                generation_seed(self.seed, index),
                self.question.id,
                index,
                self.policy,
                self.sampling,
            ):
                raise ValueError(
                    "Generation identity/order/policy disagrees with episode"
                )
            if (
                len(record.completion_ids) > self.config.max_new_tokens
                or len(request.prompt_ids) + self.config.max_new_tokens
                > self.config.max_context_tokens
                or len(request.prompt_ids) + len(record.completion_ids)
                > self.config.max_context_tokens
            ):
                raise ValueError("Generation exceeds configured token limits")

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        version = data.pop("schema_version", None)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported episode-record schema version")
        data["question"] = Question(**data["question"])
        data["config"] = TrainingConfig.from_dict(data["config"])
        data["policy"] = PolicyVersion(**data["policy"])
        data["sampling"] = SamplingParameters(**data["sampling"])
        generations = []
        for raw in data["generations"]:
            raw = dict(raw)
            request = dict(raw["request"])
            request["policy"] = PolicyVersion(**request["policy"])
            request["sampling"] = SamplingParameters(**request["sampling"])
            raw["request"] = GenerationRequest(**request)
            generations.append(GenerationRecord(**raw))
        data["generations"] = tuple(generations)
        return cls(**data)
