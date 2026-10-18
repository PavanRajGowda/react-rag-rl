"""Explicit experiment intent, independent of Torch/TRL and GPU imports."""

import math
import re
from dataclasses import asdict, dataclass, fields

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET_REVISION = "adf7fe1541ac37351ce1142544d8e3b43010ed92"


@dataclass(frozen=True)
class TrainingConfig:
    # This describes our experiment, not kwargs passed directly to GRPOConfig.
    model_id: str = "Qwen/Qwen3.5-4B"
    model_revision: str = MODEL_REVISION
    dataset_id: str = "G4KMU/t2-ragbench"
    dataset_config: str = "FinQA"
    dataset_revision: str = DATASET_REVISION
    corpus_version: str = "finqa-301676584646"
    search_schema_version: int = 4
    gpu: str = "H100"
    dtype: str = "bfloat16"
    seed: int = 42
    top_k: int = 2
    max_searches: int = 6
    max_turns: int = 8
    max_new_tokens: int = 1024  # Per generation, not per episode.
    max_context_tokens: int = 8192  # Actual prompt + newly generated tokens.
    context_mode: str = "replace"
    num_generations: int = 8  # Complete episodes per question.
    questions_per_update: int = 4
    replay_microbatch_segments: int = 1  # Turns are NOT independent GRPO samples.
    temperature: float = 0.7
    top_p: float = 1.0
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int = 3  # Review-stage smoke intent; nothing executes this yet.
    optimizer: str = "adamw"
    scheduler: str = "constant_with_warmup"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    beta: float = 0.0
    num_iterations: int = 1
    reward: str = "number_match_0_1"
    turn_penalty_lambda: float = 0.4  # Used only by number_match_turn_efficiency.
    vllm_mode: str = "colocate"
    vllm_importance_sampling_correction: bool = True
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    vllm_importance_sampling_cap: float = 2.0
    max_trajectory_attempts: int = 2
    max_consecutive_empty_batches: int = 3
    reference_probe_questions: int = 4
    dynamic_sampling: bool = False
    dynamic_sampling_multiplier: int = 4

    def __post_init__(self):
        positive_ints = (
            "search_schema_version",
            "max_searches",
            "max_turns",
            "max_new_tokens",
            "max_context_tokens",
            "questions_per_update",
            "replay_microbatch_segments",
            "max_steps",
            "lora_rank",
            "lora_alpha",
            "num_iterations",
            "num_generations",
            "reference_probe_questions",
            "dynamic_sampling_multiplier",
            "max_trajectory_attempts",
            "max_consecutive_empty_batches",
            "top_k",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dynamic_sampling_multiplier > 10:
            raise ValueError("Dynamic sampling budget multiplier cannot exceed ten")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in (
            "gradient_checkpointing",
            "vllm_importance_sampling_correction",
            "dynamic_sampling",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "temperature",
            "top_p",
            "learning_rate",
            "warmup_ratio",
            "max_grad_norm",
            "lora_dropout",
            "beta",
            "epsilon_low",
            "epsilon_high",
            "vllm_importance_sampling_cap",
            "turn_penalty_lambda",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        for name in ("temperature", "learning_rate", "max_grad_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not 0 < self.top_p <= 1
            or not 0 <= self.warmup_ratio < 1
            or not 0 <= self.lora_dropout < 1
            or self.beta < 0
        ):
            raise ValueError("Invalid sampling, warmup, dropout or KL range")
        if self.max_trajectory_attempts > 3 or self.max_consecutive_empty_batches > 10:
            raise ValueError("Recovery attempt/circuit-breaker limits are too large")
        if self.turn_penalty_lambda < 0:
            raise ValueError("turn_penalty_lambda must be nonnegative")
        if (
            self.reward == "number_match_turn_efficiency"
            and self.turn_penalty_lambda * (self.max_turns - 1) >= 1
        ):
            raise ValueError(
                "Choose turn_penalty_lambda * (max_turns - 1) < 1 so every correct reward stays above incorrect reward 0"
            )
        if self.num_generations < 2:
            raise ValueError("GRPO needs at least two episodes per question")
        if (
            not 0 < self.epsilon_low < 1
            or self.epsilon_high <= 0
            or self.vllm_importance_sampling_cap < 1
        ):
            raise ValueError("Invalid PPO clipping or sampling correction bounds")
        if self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("Context window must leave room for the prompt")
        if self.top_k > 10 or self.max_searches > 10 or self.max_turns > 20:
            raise ValueError("Search/turn limits exceed existing controller bounds")
        if self.replay_microbatch_segments > self.episodes_per_update:
            raise ValueError("Replay microbatch exceeds logical episode batch")
        for name in ("model_revision", "dataset_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{40}", getattr(self, name)
            ):
                raise ValueError(f"{name} must be a pinned 40-character commit SHA")
        for name in ("model_id", "dataset_id", "dataset_config", "corpus_version"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must be a nonempty string")
        supported = {
            "gpu": {"H100"},
            "dtype": {"bfloat16"},
            "context_mode": {"replace"},
            "optimizer": {"adamw"},
            "scheduler": {"constant_with_warmup"},
            "loss_type": {"dapo"},
            "scale_rewards": {"group", "batch", "none"},
            "reward": {"number_match_0_1", "number_match_turn_efficiency"},
            "vllm_mode": {"colocate"},
        }
        for name, values in supported.items():
            if (
                not isinstance(getattr(self, name), str)
                or getattr(self, name) not in values
            ):
                raise ValueError(f"Unsupported {name}: {getattr(self, name)!r}")

    @property
    def episodes_per_update(self):
        return self.questions_per_update * self.num_generations

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)
