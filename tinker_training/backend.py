"""Token-in/token-out Tinker generation compatible with the shared scheduler."""

import hashlib
import math
import time
from dataclasses import dataclass

from baseline_agent.protocol import ACTION_STOP_SEQUENCES, action_complete
from tinker_training.decoding import ActionDecoder
from tinker_training.policy import template_digest
from tinker_training.records import PolicyVersion


@dataclass(frozen=True)
class Failure:
    """One request-specific technical sampling error returned in its original batch slot.
    Carries no fabricated completion or semantic reward.
    """

    request: object
    error: Exception
    raw_token_count: int = 0

    def record_into(self, recorder, tokenizer):
        """Tell the recorder that this request failed, terminating its technical attempt.

        recorder owns episode state. tokenizer is accepted for adapter compatibility
        but is unused; a failed request contributes no invented completion tokens.
        """
        recorder.fail_generation(self.request, self.error)


@dataclass(frozen=True)
class Generation:
    """Successful hosted sample with exact completion IDs and their frozen sampling log
    probabilities. request preserves routing and rollout-policy identity.
    """

    request: object
    completion_ids: tuple
    sampling_logprobs: tuple
    finish_reason: str
    engine_segments: int = 1


@dataclass(frozen=True)
class Batch:
    """Ordered sampling-wave results plus one shared wall time and optional provider
    accounting. The time is not additive across individual trajectories.
    """

    generations: tuple
    seconds: float
    cache_stats: tuple = ()


class TinkerBackend:
    """Submit all independent requests before waiting; Tinker schedules the GPUs.

    config.model_revision is a local provenance field, not a claim that Tinker
    hosts those exact HF bytes. The service model identifier is logged separately.
    """

    def __init__(
        self, sampler, tokenizer, config, *, checkpoint="base", step=0, types=None
    ):
        """Bind a hosted sampling client, tokenizer and experiment configuration.

        checkpoint and step identify the policy used by every subsequent request.
        The injected types namespace permits CPU fake-service tests without the SDK.
        Construction creates local identity records but does not sample any tokens.
        """
        if types is None:
            from tinker import types
        self.types, self.sampler, self.tokenizer, self.config = (
            types,
            sampler,
            tokenizer,
            config,
        )
        self.action_decoder = ActionDecoder(tokenizer)
        self.policy = PolicyVersion(
            config.model_revision,
            config.model_revision,
            checkpoint
            if checkpoint == "base"
            else hashlib.sha256(checkpoint.encode()).hexdigest(),
            step,
            template_digest(tokenizer),
        )

    def generate(self, requests):
        """Submit a generation wave and return ordered successes or per-request failures.

        Args:
            requests: Nonempty iterable of GenerationRequest objects for this policy.

        Returns:
            Batch with one Generation or Failure per request in the original order,
            shared wave duration, and client-side token accounting. Each success keeps
            exact sampled token IDs and their sampling log probabilities.

        Raises:
            ValueError: Requests duplicate a trajectory turn, use a different policy,
                exceed the context limit or request unsupported sampling settings.

        All sample futures are submitted before waiting for results. Submission,
        decoding
        and individual response errors become Failure records; shared request-contract
        errors raise. Sampling itself is performed by the remote Tinker service.
        """
        requests = tuple(requests)
        if not requests or len({(r.episode_id, r.turn) for r in requests}) != len(
            requests
        ):
            raise ValueError("Expected nonempty distinct generation requests")
        started, pending, output = time.perf_counter(), [], []
        for request in requests:
            if request.policy != self.policy:
                raise ValueError("Stale/mixed policy")
            if (
                (request.sampling.temperature, request.sampling.top_p) != (1.0, 1.0)
                or request.sampling.top_k != 0
                or request.sampling.repetition_penalty != 1.0
            ):
                raise ValueError(
                    "Tinker migration uses unmodified T=1, top_p=1 sampling"
                )
            if (
                len(request.prompt_ids) + self.config.max_new_tokens
                > self.config.max_context_tokens
            ):
                raise ValueError("Generation exceeds context budget")
        # Submit every request before calling result(): the provider can schedule
        # them together even though this client coordinator is synchronous.
        for request in requests:
            try:
                future = self.sampler.sample(
                    prompt=self.types.ModelInput.from_ints(list(request.prompt_ids)),
                    num_samples=1,
                    sampling_params=self.types.SamplingParams(
                        max_tokens=self.config.max_new_tokens,
                        seed=request.seed,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        stop=list(ACTION_STOP_SEQUENCES),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- isolate failed submissions
                future = exc
            pending.append((request, future))
        raw_tokens = 0
        # Results are consumed in input order for deterministic routing. This does
        # not imply the remote service executed one trajectory at a time.
        for request, future in pending:
            try:
                if isinstance(future, Exception):
                    raise future
                response = future.result()
                generation, count = self._decode_response(request, response)
                raw_tokens += count
                output.append(generation)
            except Exception as exc:  # noqa: BLE001 -- technical error, never synthetic reward
                output.append(Failure(request, exc))
        return Batch(
            tuple(output),
            time.perf_counter() - started,
            (
                {
                    "provider": "tinker",
                    "submitted_prompt_tokens": sum(len(r.prompt_ids) for r in requests),
                    "returned_sample_tokens": raw_tokens,
                    "billing_note": "client counts; authoritative billing is service usage",
                },
            ),
        )

    def _decode_response(self, request, response):
        """Validate returned tokens and retain the exact sampled action prefix."""
        if len(response.sequences) != 1:
            raise ValueError("Expected one sampled sequence")
        sequence = response.sequences[0]
        ids, logps = tuple(sequence.tokens), tuple(sequence.logprobs or ())
        if (
            not ids
            or len(ids) != len(logps)
            or len(ids) > self.config.max_new_tokens
            or any(type(i) is not int or i < 0 for i in ids)
            or any(not math.isfinite(p) or p > 0 for p in logps)
        ):
            raise ValueError("Invalid sampled tokens/log probabilities")
        # Record all returned tokens before trimming. Extra sampled suffix
        # tokens may be billable even though they are not replayed.
        raw_tokens = len(ids)
        reason = "length"
        end = self.action_decoder.first_stop_end(ids, request.assistant_prefill)
        if end is not None:
            ids, logps = ids[:end], logps[:end]
            text = request.assistant_prefill + self.tokenizer.decode(
                list(ids), skip_special_tokens=True
            )
            reason = "action" if action_complete(text) else "protocol_stop"
        elif ids[-1] == self.tokenizer.eos_token_id:
            reason = "eos"
        return Generation(request, ids, logps, reason), raw_tokens

    async def generate_async(self, request):
        """Await one hosted sample without blocking other trajectories.

        Tinker owns GPU scheduling. Returns (Generation or Failure, Batch) with
        this request's client-observed duration and pre-trimming token count.
        Contract errors raise before submission; technical request errors are local.
        """
        if request.policy != self.policy:
            raise ValueError("Stale/mixed policy")
        if (
            (request.sampling.temperature, request.sampling.top_p) != (1.0, 1.0)
            or request.sampling.top_k != 0
            or request.sampling.repetition_penalty != 1.0
        ):
            raise ValueError("Tinker migration uses unmodified T=1, top_p=1 sampling")
        if (
            len(request.prompt_ids) + self.config.max_new_tokens
            > self.config.max_context_tokens
        ):
            raise ValueError("Generation exceeds context budget")
        started = time.perf_counter()
        raw_tokens = 0
        try:
            response = await self.sampler.sample_async(
                prompt=self.types.ModelInput.from_ints(list(request.prompt_ids)),
                num_samples=1,
                sampling_params=self.types.SamplingParams(
                    max_tokens=self.config.max_new_tokens,
                    seed=request.seed,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    stop=list(ACTION_STOP_SEQUENCES),
                ),
            )
            generation, raw_tokens = self._decode_response(request, response)
        except Exception as exc:  # noqa: BLE001 -- isolate hosted sampling failures
            generation = Failure(request, exc)
        batch = Batch(
            (generation,),
            time.perf_counter() - started,
            (
                {
                    "provider": "tinker",
                    "submitted_prompt_tokens": len(request.prompt_ids),
                    "returned_sample_tokens": raw_tokens,
                    "billing_note": "client counts; authoritative billing is service usage",
                },
            ),
        )
        return generation, batch
