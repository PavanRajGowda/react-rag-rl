"""Record injected generations against the exact per-turn controller prompt."""

import hashlib
import json

from baseline_agent.prompt import SYSTEM_PROMPT
from grpo_training.environment import CONTINUE, SearchEpisode
from grpo_training.records import (
    EpisodeRecord,
    GenerationRecord,
    GenerationRequest,
    PolicyVersion,
    SamplingParameters,
    generation_seed,
    token_ids,
)

ASSISTANT_PREFILL = "<reasoning>"


def template_digest(tokenizer):
    """Template identity only; actual tokenizer/revision verification is Step 5."""
    template = tokenizer.chat_template
    if not isinstance(template, (str, dict)) or not template:
        raise ValueError("An explicit chat template is required")
    settings = {
        "chat_template": template,
        "system_prompt": SYSTEM_PROMPT,
        "continue": CONTINUE,
        "prefill": ASSISTANT_PREFILL,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "add_special_tokens_on_encode": False,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


class EpisodeRecorder:
    """No generation or reward is performed. One outstanding request per episode.

    Tokenizer is injected: apply_chat_template, encode and decode must follow
    the Hugging Face tokenizer interface. Tests use a deterministic toy codec.
    """

    def __init__(
        self, episode, *, episode_id, group_id, sample_index, seed, policy, sampling
    ):
        if (
            type(episode) is not SearchEpisode
            or type(policy) is not PolicyVersion
            or type(sampling) is not SamplingParameters
        ):
            raise TypeError("Expected episode, policy version and sampling parameters")
        if episode.snapshot()["turns"] or episode.done:
            raise ValueError("Recorder requires a fresh episode")
        self.episode = episode
        self.episode_id, self.group_id = episode_id, group_id
        self.sample_index, self.seed = sample_index, seed
        self.policy, self.sampling = policy, sampling
        if any(
            not isinstance(value, str) or not value for value in (episode_id, group_id)
        ):
            raise ValueError("Episode and group IDs must be nonempty text")
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or type(sample_index) is not int
            or not 0 <= sample_index < episode.config.num_generations
        ):
            raise ValueError("Invalid episode seed/sample index")
        if policy.model_revision != episode.config.model_revision or (
            sampling.temperature,
            sampling.top_p,
        ) != (episode.config.temperature, episode.config.top_p):
            raise ValueError("Policy/sampling and experiment configuration disagree")
        self._pending, self._generations = None, []

    def _check_history(self):
        observed = sum(
            "model_output" in turn for turn in self.episode.snapshot()["turns"]
        )
        if observed != len(self._generations):
            raise ValueError(
                "Controller advanced outside its recorder; generation history is incomplete"
            )

    def prepare(self, tokenizer):
        self._check_history()
        if self.episode.done:
            raise RuntimeError("Episode has terminated")
        if self._pending is not None:
            raise RuntimeError(
                "An outstanding generation must be recorded or failed first"
            )
        if template_digest(tokenizer) != self.policy.template_sha256:
            raise ValueError(
                "Tokenizer template/prompt identity disagrees with rollout policy"
            )
        prompt = (
            tokenizer.apply_chat_template(
                self.episode.messages(),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ASSISTANT_PREFILL
        )
        ids = token_ids(tokenizer.encode(prompt, add_special_tokens=False))
        if not self.episode.check_context_budget(len(ids)):
            return None  # No sampled tokens are fabricated for a context-limit turn.
        self._pending = GenerationRequest(
            self.episode_id,
            self.group_id,
            self.episode.question.id,
            self.sample_index,
            len(self._generations) + 1,
            generation_seed(self.seed, len(self._generations) + 1),
            self.policy,
            self.sampling,
            prompt,
            ids,
        )
        return self._pending

    def record(
        self,
        request,
        *,
        prompt_ids,
        policy_id,
        sampling,
        completion_ids,
        sampling_logprobs,
        tokenizer,
        finish_reason,
        generation_seconds=None,
    ):
        self._check_history()
        if self.episode.done or self._pending is None or request != self._pending:
            raise ValueError("Generation does not match the outstanding request")
        if template_digest(tokenizer) != self.policy.template_sha256:
            raise ValueError("Tokenizer template changed during generation")
        if (
            token_ids(prompt_ids) != request.prompt_ids
            or policy_id != request.policy.id
            or sampling != request.sampling
        ):
            raise ValueError(
                "Backend prompt/policy/sampling disagrees with requested generation"
            )
        ids = token_ids(completion_ids)
        record = GenerationRecord(
            request,
            ids,
            tuple(sampling_logprobs),
            tokenizer.decode(list(ids), skip_special_tokens=True),
            finish_reason,
            generation_seconds,
        )
        if (
            len(ids) > self.episode.config.max_new_tokens
            or len(request.prompt_ids) + len(ids)
            > self.episode.config.max_context_tokens
        ):
            raise ValueError("Sampled generation exceeds configured token limits")
        # Never re-encode decoded output: BPE decoding/re-encoding need not invert.
        self.episode.accept_output(record.controller_output)
        self._generations.append(record)
        self._pending = None
        return record

    def fail_generation(self, request, error):
        self._check_history()
        if self._pending is None or request != self._pending:
            raise ValueError("Failure does not match the outstanding request")
        self.episode.fail_generation(error)
        self._pending = None

    def finish(self):
        self._check_history()
        if not self.episode.done or self._pending is not None:
            raise RuntimeError(
                "Only terminal episodes with no pending generation can be finalized"
            )
        return EpisodeRecord(
            self.episode_id,
            self.group_id,
            self.sample_index,
            self.seed,
            self.episode.question,
            self.episode.config,
            self.policy,
            self.sampling,
            self.episode.status,
            self.episode.answer,
            tuple(self._generations),
            json.dumps(self.episode.snapshot(), ensure_ascii=False, allow_nan=False),
        )
