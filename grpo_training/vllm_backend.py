"""Batched vLLM sampling + explicit LoRA synchronization; lazy GPU imports."""

import json
import math
import os
import threading
import time
from dataclasses import dataclass

from baseline_agent.protocol import action_complete
from grpo_training.model import AdapterArtifact, check_runtime
from grpo_training.records import GenerationRecord, generation_seed, token_ids
from grpo_training.rollouts import template_digest


class TrajectoryGenerationError(ValueError):
    """One response is unusable; unrelated requests remain valid."""


@dataclass(frozen=True)
class BackendFailure:
    request: object
    error: Exception
    raw_token_count: int = 0

    def record_into(self, recorder, tokenizer):
        recorder.fail_generation(self.request, self.error)


@dataclass(frozen=True)
class BackendGeneration:
    request: object
    completion_ids: tuple[int, ...]
    sampling_logprobs: tuple[float, ...]
    finish_reason: str
    engine_segments: int

    def record_into(self, recorder, tokenizer):
        return recorder.record(
            self.request,
            prompt_ids=self.request.prompt_ids,
            policy_id=self.request.policy.id,
            sampling=self.request.sampling,
            completion_ids=self.completion_ids,
            sampling_logprobs=self.sampling_logprobs,
            tokenizer=tokenizer,
            finish_reason=self.finish_reason,
        )


@dataclass(frozen=True)
class GenerationBatch:
    generations: tuple[BackendGeneration | BackendFailure, ...]
    seconds: float  # Whole batch wall time, not fabricated per-sample timing.
    cache_stats: tuple[dict, ...] = ()  # Diagnostic engine-request counts only.


class VLLMBackend:
    """Serial update/generation phases, with parallel requests inside generate()."""

    def __init__(
        self,
        engine,
        tokenizer,
        config,
        eos_token_ids,
        *,
        sampling_factory=None,
        lora_factory=None,
    ):
        self.engine, self.tokenizer, self.config = engine, tokenizer, config
        self.eos_token_ids = frozenset(token_ids(eos_token_ids))
        self.sampling_factory, self.lora_factory = sampling_factory, lora_factory
        self._artifact = self._lora = None
        self._next_id, self._sleeping = 1, False
        self._lock = threading.Lock()

    @property
    def policy(self):
        return self._artifact.policy if self._artifact is not None else None

    def synchronize(self, artifact):
        if type(artifact) is not AdapterArtifact:
            raise TypeError("Expected an exported AdapterArtifact")
        with self._lock:
            if self._sleeping:
                raise RuntimeError("Wake vLLM before loading an adapter")
            artifact.verify()
            manifest = json.loads(
                (artifact.path / "transfer.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("config") != self.config.to_dict()
                or artifact.policy.model_revision != self.config.model_revision
                or artifact.policy.template_sha256 != template_digest(self.tokenizer)
            ):
                raise ValueError("Adapter does not belong to this experiment/template")
            if self.policy is not None:
                if artifact.policy.id == self.policy.id:
                    if (
                        self._lora.lora_int_id
                        not in self.engine.llm_engine.list_loras()
                    ):
                        self._artifact = self._lora = None
                        raise RuntimeError("Active adapter disappeared from vLLM")
                    return self.policy
                if artifact.policy.optimizer_step <= self.policy.optimizer_step:
                    raise ValueError("New weights need a strictly newer optimizer step")
            if self.lora_factory is None:
                from vllm.lora.request import LoRARequest

                factory = LoRARequest
            else:
                factory = self.lora_factory
            lora = factory(
                lora_name=artifact.policy.id,
                lora_int_id=self._next_id,
                lora_path=str(artifact.path.resolve()),
                base_model_name=self.config.model_id,
            )
            self._next_id += 1
            old = self._lora
            self._artifact = self._lora = None  # Fail closed if transfer/ack fails.
            if (
                self.engine.llm_engine.add_lora(lora) is not True
                or lora.lora_int_id not in self.engine.llm_engine.list_loras()
            ):
                raise RuntimeError("vLLM did not acknowledge the new adapter")
            if (
                old is not None
                and old.lora_int_id in self.engine.llm_engine.list_loras()
                and self.engine.llm_engine.remove_lora(old.lora_int_id) is not True
            ):
                raise RuntimeError("vLLM did not retire the previous adapter")
            self.engine.llm_engine.reset_prefix_cache()
            self._artifact, self._lora = artifact, lora
            return self.policy

    def sleep(self):
        with self._lock:
            self.engine.sleep(level=1)
            self._sleeping = True

    def wake_up(self):
        with self._lock:
            self.engine.wake_up()
            self._sleeping = False

    def _params(self, request, remaining, segment):
        if self.sampling_factory is None:
            from vllm import SamplingParams

            factory = SamplingParams
        else:
            factory = self.sampling_factory
        sampling = request.sampling
        return factory(
            n=1,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
            seed=request.seed
            if segment == 0
            else generation_seed(request.seed, segment),
            max_tokens=remaining,
            logprobs=0,
            stop=["</search>", "</answer>"],
            stop_token_ids=sorted(self.eos_token_ids),
            include_stop_str_in_output=True,
            skip_special_tokens=True,
            ignore_eos=False,
        )

    def _extract(self, result, prompt, limit):
        if tuple(result.prompt_token_ids or ()) != tuple(prompt):
            raise ValueError("vLLM returned a changed prompt")
        if not result.finished or not result.outputs or len(result.outputs) != 1:
            raise TrajectoryGenerationError(
                "vLLM returned incomplete/multiple outputs for one request"
            )
        lora = result.lora_request
        if lora is None or (lora.lora_int_id, lora.lora_name, lora.lora_path) != (
            self._lora.lora_int_id,
            self._lora.lora_name,
            self._lora.lora_path,
        ):
            raise ValueError("vLLM output does not acknowledge the loaded adapter")
        output = result.outputs[0]
        try:
            try:
                ids = token_ids(output.token_ids)
            except (ValueError, TypeError) as exc:
                raise TrajectoryGenerationError(str(exc)) from exc
            if (
                len(ids) > limit
                or output.logprobs is None
                or len(output.logprobs) != len(ids)
            ):
                raise TrajectoryGenerationError(
                    "Missing/oversized sampled tokens or probability vector"
                )
            probabilities = []
            for token, values in zip(ids, output.logprobs, strict=True):
                if token not in values:
                    raise TrajectoryGenerationError(
                        "Sampled-token logprob is missing; never invent one"
                    )
                value = values[token].logprob
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value > 0
                ):
                    raise TrajectoryGenerationError("Invalid sampled-token logprob")
                probabilities.append(float(value))
            return ids, tuple(probabilities), output
        except (TypeError, AttributeError, KeyError, IndexError) as exc:
            raise TrajectoryGenerationError(
                f"Malformed per-request completion: {exc}"
            ) from exc

    def generate(self, requests):
        requests = tuple(requests)
        if not requests:
            raise ValueError("Generation batch is empty")
        with self._lock:
            if self.policy is None or self._sleeping:
                raise RuntimeError("Generation requires an awake, synchronized vLLM")
            self._artifact.verify()
            if self._lora.lora_int_id not in self.engine.llm_engine.list_loras():
                raise RuntimeError("Loaded adapter is no longer registered")
            keys = [(r.episode_id, r.turn) for r in requests]
            if len(set(keys)) != len(keys) or any(
                r.policy != self.policy for r in requests
            ):
                raise ValueError("Duplicate requests or stale/mixed rollout policies")
            for request in requests:
                if (request.sampling.temperature, request.sampling.top_p) != (
                    self.config.temperature,
                    self.config.top_p,
                ) or len(
                    request.prompt_ids
                ) + self.config.max_new_tokens > self.config.max_context_tokens:
                    raise ValueError("Request sampling/context differs from experiment")
            states = [
                {"ids": (), "probs": (), "segments": 0, "done": None} for _ in requests
            ]
            started = time.perf_counter()
            cache_stats = []
            pending = list(range(len(requests)))
            while pending:
                prompts, parameters = [], []
                for index in pending:
                    request, state = requests[index], states[index]
                    prompts.append(
                        {"prompt_token_ids": list(request.prompt_ids + state["ids"])}
                    )
                    parameters.append(
                        self._params(
                            request,
                            self.config.max_new_tokens - len(state["ids"]),
                            state["segments"],
                        )
                    )
                results = self.engine.generate(
                    prompts,
                    sampling_params=parameters,
                    lora_request=self._lora,
                    use_tqdm=False,
                )
                if len(results) != len(pending):
                    raise ValueError("vLLM changed generation batch membership")
                again = []
                for index, result, prompt in zip(
                    pending, results, prompts, strict=True
                ):
                    cached = getattr(result, "num_cached_tokens", None)
                    size = len(prompt["prompt_token_ids"])
                    cache_stats.append(
                        {
                            "episode_id": requests[index].episode_id,
                            "turn": requests[index].turn,
                            "prompt_tokens": size,
                            "cached_tokens": cached
                            if type(cached) is int and 0 <= cached <= size
                            else None,
                        }
                    )
                    state, request = states[index], requests[index]
                    before = len(state["ids"])
                    try:
                        ids, probabilities, output = self._extract(
                            result,
                            prompt["prompt_token_ids"],
                            self.config.max_new_tokens - before,
                        )
                    except TrajectoryGenerationError as exc:
                        state["error"] = exc
                        raw = getattr(result, "outputs", ()) or ()
                        state["failed_token_count"] = len(state["ids"]) + sum(
                            len(out.token_ids)
                            for out in raw
                            if isinstance(
                                getattr(out, "token_ids", None), (list, tuple)
                            )
                        )
                        continue
                    state["ids"] += ids
                    state["probs"] += probabilities
                    state["segments"] += 1
                    # String stopping can fire on quoted closing tags in reasoning.
                    # Preserve baseline action_complete semantics; resume if needed.
                    for end in range(before + 1, len(state["ids"]) + 1):
                        text = request.assistant_prefill + self.tokenizer.decode(
                            list(state["ids"][:end]), skip_special_tokens=True
                        )
                        if action_complete(text):
                            state["ids"], state["probs"], state["done"] = (
                                state["ids"][:end],
                                state["probs"][:end],
                                "action",
                            )
                            break
                    if state["done"] is None:
                        if state["ids"][-1] in self.eos_token_ids:
                            state["done"] = "eos"
                        elif (
                            output.finish_reason == "length"
                            or len(state["ids"]) == self.config.max_new_tokens
                        ):
                            state["done"] = "length"
                        elif output.finish_reason == "stop" and output.stop_reason in (
                            "</search>",
                            "</answer>",
                        ):
                            again.append(index)
                        else:
                            details = {
                                "episode_id": request.episode_id,
                                "turn": request.turn,
                                "finish_reason": output.finish_reason,
                                "stop_reason": output.stop_reason,
                                "tail_token_ids": list(state["ids"][-8:]),
                                "expected_eos_token_ids": sorted(self.eos_token_ids),
                                "generated_tokens": len(state["ids"]),
                                "engine_segments": state["segments"],
                            }
                            state["error"] = TrajectoryGenerationError(
                                "Unrecognized stop or omitted sampled EOS token: "
                                + json.dumps(details, sort_keys=True)
                            )
                pending = again
            generations = []
            for request, state in zip(requests, states, strict=True):
                if state.get("error") is not None:
                    generations.append(
                        BackendFailure(
                            request,
                            state["error"],
                            state.get("failed_token_count", len(state["ids"])),
                        )
                    )
                    continue
                # Reuse record-level checks without re-encoding any sampled text.
                GenerationRecord(
                    request,
                    state["ids"],
                    state["probs"],
                    self.tokenizer.decode(list(state["ids"]), skip_special_tokens=True),
                    state["done"],
                )
                generations.append(
                    BackendGeneration(
                        request,
                        state["ids"],
                        state["probs"],
                        state["done"],
                        state["segments"],
                    )
                )
            return GenerationBatch(
                tuple(generations), time.perf_counter() - started, tuple(cache_stats)
            )


def create_vllm_backend(
    bundle,
    gpu_memory_utilization=0.3,
    *,
    quantization=None,
    enable_prefix_caching=False,
    execution_mode="eager",
    diagnostic_metrics=False,
    max_num_seqs=None,
):
    """Explicit GPU operation. No server, distributed setup or automatic launch."""
    check_runtime()
    if execution_mode not in ("eager", "graphs", "compiled"):
        raise ValueError("Unknown vLLM execution mode")
    if type(diagnostic_metrics) is not bool:
        raise ValueError("diagnostic_metrics must be a boolean")
    if max_num_seqs is not None and (
        type(max_num_seqs) is not int or max_num_seqs <= 0
    ):
        raise ValueError("max_num_seqs must be a positive integer or None")
    if quantization not in (None, "fp8"):
        raise ValueError("Only BF16 or experimental FP8 inference is supported")
    if type(enable_prefix_caching) is not bool:
        raise ValueError("enable_prefix_caching must be a boolean")
    if (
        isinstance(gpu_memory_utilization, bool)
        or not isinstance(gpu_memory_utilization, (int, float))
        or not 0 < gpu_memory_utilization < 1
    ):
        raise ValueError("vLLM memory utilization must be between 0 and 1")
    if bundle.config.lora_rank not in (8, 16, 32, 64):
        raise ValueError("Initial vLLM bridge supports LoRA ranks 8/16/32/64")
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn") != "spawn":
        raise RuntimeError("Colocated CUDA worker must use spawn, not fork")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM

    execution = {}
    capture_limit = max_num_seqs or 16
    capture_sizes = [n for n in (1, 2, 4, 8, 16, 32, 48, 64) if n < capture_limit] + [
        capture_limit
    ]
    if max_num_seqs is not None:
        execution["max_num_seqs"] = max_num_seqs
    if diagnostic_metrics:
        execution["disable_log_stats"] = False
        execution["max_num_seqs"] = capture_limit
    if execution_mode != "eager":
        # Bounded diagnostic capture shapes; production continues to use eager.
        execution.update(
            {
                "compilation_config": {
                    "mode": 0 if execution_mode == "graphs" else 3,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": capture_sizes,
                    "max_cudagraph_capture_size": capture_limit,
                },
                "cudagraph_metrics": True,
                "max_num_seqs": capture_limit,
            }
        )
    engine = LLM(
        model=str(bundle.snapshot),
        tokenizer=str(bundle.snapshot),
        dtype="bfloat16",
        quantization=quantization,
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=bundle.config.max_context_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=bundle.config.lora_rank,
        max_loras=1,
        max_cpu_loras=2,
        enable_sleep_mode=True,
        enforce_eager=execution_mode == "eager",
        enable_prefix_caching=enable_prefix_caching,
        generation_config="vllm",
        logprobs_mode="processed_logprobs",
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=bundle.config.seed,
        **execution,
    )
    actual = engine.get_tokenizer()
    if (
        actual.get_vocab() != bundle.tokenizer.get_vocab()
        or actual.eos_token_id != bundle.tokenizer.eos_token_id
    ):
        raise RuntimeError("Training/vLLM tokenizers disagree")
    eos = {bundle.tokenizer.eos_token_id}
    metadata = bundle.snapshot / "generation_config.json"
    if metadata.is_file():
        extra = json.loads(metadata.read_text()).get("eos_token_id")
        if extra is not None:
            eos.update(token_ids(extra if isinstance(extra, list) else [extra]))
    return VLLMBackend(engine, bundle.tokenizer, bundle.config, sorted(eos))
