"""Explicit experiment intent, independent of Torch/TRL and GPU imports."""

import math
import re
from dataclasses import asdict, dataclass, fields

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
from retrieval_service.dataset import CONFIG, CORPUS_VERSION, DATASET
from retrieval_service.dataset import REVISION as DATASET_REVISION


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen experiment specification validated before execution. Includes controller,
    reward, credit and optimization settings; retired local-GPU fields are historical
    metadata, not Tinker controls.
    """

    # Dataset/model identity is checked before resume; changing it starts a new run.
    model_id: str = "Qwen/Qwen3.5-4B"
    model_revision: str = MODEL_REVISION
    dataset_id: str = DATASET
    dataset_config: str = CONFIG
    dataset_revision: str = DATASET_REVISION
    corpus_version: str = CORPUS_VERSION
    search_schema_version: int = 6
    seed: int = 42
    # Controller limits: a turn is one generated reasoning + search/answer action.
    # Retrieval top_k is documents per search, not sampler vocabulary top-k.
    top_k: int = 2
    max_searches: int = 7
    max_turns: int = 8
    max_new_tokens: int = 1024  # Per generation, not per episode.
    max_context_tokens: int = 8192  # Actual prompt + newly generated tokens.
    context_mode: str = "append"
    max_evidence_passages: int = 12
    max_evidence_chars: int = 12000
    # Fresh rollout size is questions_per_update * num_generations. A replay row
    # represents one turn, so row count is generally larger than trajectory count.
    num_generations: int = 8  # Complete episodes per question.
    questions_per_update: int = 4
    # Bound whole active trajectories independently of the logical rollout pool.
    rollout_concurrency: int = 512
    training_question_mix: str = "all"
    checkpoint_interval: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    # Optimizer steps can reuse a sampled pool: num_iterations controls reuse,
    # while max_steps is the total number of committed parameter updates.
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int = 3  # Default optimizer-step target; presets/CLI may override it.
    optimizer: str = "adamw"
    scheduler: str = "constant_with_warmup"
    lora_rank: int = 16
    # Mean-center trajectory rewards; no standard-deviation normalization.
    loss_type: str = "tinker_ppo"
    scale_rewards: str = "none"
    beta: float = 0.0
    minibatch_questions: int = (
        0  # Zero uses the entire pool; otherwise keep sibling groups together.
    )
    # Bounded process quality weights form a convex combination; correctness stays 1.
    process_bonus_scale: float = 0.25
    process_coverage_weight: float = 1 / 3
    process_redundancy_weight: float = 1 / 3
    process_invalid_weight: float = 1 / 3
    # Direct local advantage adjustments, independent of trajectory reward weights.
    local_gold_weight: float = 0.33
    local_search_penalty_weight: float = 0.33
    local_protocol_penalty_weight: float = 0.33
    num_iterations: int = 1
    reward: str = "exact_match_0_1"
    epsilon_low: float = 0.2
    epsilon_high: float = 0.28
    # Technical failures get bounded retries; incorrect answers do not. The empty
    # batch breaker prevents unbounded generation when all trajectories fail.
    max_trajectory_attempts: int = 2
    max_consecutive_empty_batches: int = 3
    dynamic_sampling: bool = False

    def __post_init__(self):
        """Validate configuration types, ranges and supported experiment combinations.

        Called automatically by the frozen dataclass constructor. Raises ValueError
        for invalid limits, revisions, rewards or optimization settings before any
        remote execution. Retired hardware fields are compatibility metadata only.
        """
        positive_ints = (
            "max_evidence_passages",
            "max_evidence_chars",
            "search_schema_version",
            "max_searches",
            "max_turns",
            "max_new_tokens",
            "max_context_tokens",
            "questions_per_update",
            "checkpoint_interval",
            "rollout_concurrency",
            "max_steps",
            "lora_rank",
            "num_iterations",
            "num_generations",
            "max_trajectory_attempts",
            "max_consecutive_empty_batches",
            "top_k",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.minibatch_questions) is not int
            or not 0 <= self.minibatch_questions <= self.questions_per_update
        ):
            raise ValueError("minibatch_questions must be 0..questions_per_update")
        for name in (
            "process_bonus_scale",
            "process_coverage_weight",
            "process_redundancy_weight",
            "process_invalid_weight",
            "local_gold_weight",
            "local_search_penalty_weight",
            "local_protocol_penalty_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.process_bonus_scale < 0.5:
            raise ValueError(
                "process_bonus_scale must be in [0, 0.5) to preserve correctness ordering"
            )
        if not math.isclose(
            self.process_coverage_weight
            + self.process_redundancy_weight
            + self.process_invalid_weight,
            1.0,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Bounded process weights must sum to one")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("dynamic_sampling",):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "temperature",
            "top_p",
            "learning_rate",
            "warmup_ratio",
            "max_grad_norm",
            "beta",
            "epsilon_low",
            "epsilon_high",
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
        if not 0 < self.top_p <= 1 or not 0 <= self.warmup_ratio < 1 or self.beta < 0:
            raise ValueError("Invalid sampling, warmup, dropout or KL range")
        if self.max_trajectory_attempts > 3 or self.max_consecutive_empty_batches > 10:
            raise ValueError("Recovery attempt/circuit-breaker limits are too large")
        if self.num_generations < 2:
            raise ValueError("GRPO needs at least two episodes per question")
        if not 0 < self.epsilon_low < 1 or self.epsilon_high <= 0:
            raise ValueError("Invalid PPO clipping or sampling correction bounds")
        if self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("Context window must leave room for the prompt")
        if self.top_k > 10 or self.max_searches > 10 or self.max_turns > 20:
            raise ValueError("Search/turn limits exceed existing controller bounds")
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
            "context_mode": {"append", "replace", "bounded"},
            "optimizer": {"adamw"},
            "scheduler": {"constant_with_warmup"},
            "loss_type": {"tinker_ppo"},
            "scale_rewards": {"none"},
            "training_question_mix": {"all", "hard_medium_50_50"},
            "reward": {
                "bounded_process_trajectory_v1",
                "local_process_advantage_v1",
                "exact_match_0_1",
            },
        }
        for name, values in supported.items():
            if (
                not isinstance(getattr(self, name), str)
                or getattr(self, name) not in values
            ):
                raise ValueError(f"Unsupported {name}: {getattr(self, name)!r}")

    @property
    def episodes_per_update(self):
        """Return the requested number of trajectories in a fresh logical batch.

        This is questions_per_update multiplied by num_generations; technical failures
        may reduce the actual surviving count reported by collection.
        """
        return self.questions_per_update * self.num_generations

    def to_dict(self):
        """Return a serializable dictionary of all experiment configuration fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        """Construct TrainingConfig from a mapping, rejecting unknown field names.

        Dataclass validation runs after construction; the input mapping is not mutated.
        """
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)
