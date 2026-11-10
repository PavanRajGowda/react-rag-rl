"""Shared asynchronous hosted generation and search trajectories."""

import asyncio
import inspect
import time
from contextlib import AsyncExitStack
from dataclasses import asdict

from baseline_agent.agent import AgentConfig, agent_steps
from baseline_agent.inference import PromptGenerator
from baseline_agent.persistence import failure_result
from retrieval_service.timing import search_context
from tinker_training.records import (
    GenerationRequest,
    SamplingParameters,
    generation_seed,
)


class GenerationContractError(Exception):
    """Internal marker for shared backend errors that must abort collection."""


class BatchGenerator(PromptGenerator):
    """Per-trajectory prompt/token state; the round coordinator owns inference."""

    def __init__(self, tokenizer, config, policy, question, sampling=None):
        """Bind per-trajectory tokenizer, agent settings, policy and sampling parameters.
        Initializes the local turn counter; does not generate tokens.
        """
        self.tokenizer, self.config, self.policy = tokenizer, config, policy
        self.question, self.turn = question, 0
        self.sampling = sampling or SamplingParameters(
            config.temperature, 0.95, 20, 1.05
        )

    def request(self, prompt):
        """Convert a rendered next-turn prompt into a seeded GenerationRequest."""
        self.turn += 1
        ids = self.encode_prompt(prompt)
        if len(ids) + self.config.max_new_tokens > self.config.max_context_tokens:
            raise ValueError("Prompt plus generation exceeds the context budget")
        return GenerationRequest(
            self.question["id"],
            self.question.get("group_id", self.question["id"]),
            self.question.get("training_question_id", self.question["id"]),
            self.question.get("sample_index", 0),
            self.turn,
            generation_seed(
                self.question.get(
                    "sampling_seed", self.config.seed + self.question["example_index"]
                ),
                self.turn,
            ),
            self.policy,
            self.sampling,
            prompt,
            ids,
        )


async def async_agents(
    questions,
    backend,
    tokenizer,
    policy,
    config,
    search,
    concurrency,
    *,
    evaluation_mode="rag",
    sampling=None,
    trajectory_attempts=1,
    on_sample=None,
    on_event=None,
    generation_errors="deliver",
):
    """Yield completed trajectories without generation or search wave barriers.

    Each active trajectory awaits its own hosted generation and retrieval. The
    concurrency limit counts entire active trajectories; completed slots refill
    immediately. Search session ownership covers all tasks, including cancellation.
    Synchronous injected CPU fakes are accepted for local tests only; production
    backends expose generate_async and retrieval returns awaitables.
    """
    if evaluation_mode != "rag" or not callable(search):
        raise ValueError("RAG inference requires a retriever")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Concurrency must be a positive integer")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate trajectory IDs")
    if type(trajectory_attempts) is not int or not 1 <= trajectory_attempts <= 3:
        raise ValueError("Trajectory attempts must be 1..3")
    settings = AgentConfig(
        **{
            k: v
            for k, v in asdict(config).items()
            if k in AgentConfig.__dataclass_fields__
        }
    )
    limit = asyncio.Semaphore(concurrency)

    async def once(question, retriever):
        """Await IO around the unchanged resumable search/reason/answer controller."""
        started = time.perf_counter()
        generator = BatchGenerator(tokenizer, settings, policy, question, sampling)
        steps = agent_steps(question, generator, settings, emit=lambda *a, **k: None)
        measurements = []
        try:
            operation, payload = next(steps)
            while True:
                if operation == "generation":
                    try:
                        request = generator.request(payload)
                    except ValueError as exc:
                        operation, payload = steps.throw(exc)
                        continue
                    try:
                        if hasattr(backend, "generate_async"):
                            generation, batch = await backend.generate_async(request)
                        else:
                            # CPU-only scripted test backends, never hosted IO.
                            await asyncio.sleep(0)
                            batch = backend.generate([request])
                            generation = batch.generations[0]
                    except Exception as exc:
                        if generation_errors == "raise":
                            raise GenerationContractError() from exc
                        raise
                    error = getattr(generation, "error", None)
                    if error is not None:
                        raise error
                    if on_sample is not None:
                        on_sample(request, generation)
                    measurements.append(
                        {
                            "requests": 1,
                            "seconds": batch.seconds,
                            "cache_stats": list(getattr(batch, "cache_stats", ())),
                            "generated_tokens": len(generation.completion_ids),
                        }
                    )
                    if on_event is not None:
                        on_event("generation_request", [request], batch)
                    value = _decode_generation(
                        request, generation, tokenizer, batch.seconds
                    )
                else:
                    with search_context(question["id"], generator.turn):
                        value = retriever(payload)
                        if inspect.isawaitable(value):
                            value = await value
                    if isinstance(value, Exception):
                        raise value
                    if isinstance(value, dict) and value.get("status") == "error":
                        raise RuntimeError(
                            f"Search failed: {value.get('code', 'unknown')}"
                        )
                operation, payload = steps.send(value)
        except StopIteration as done:
            result = done.value
        except GenerationContractError as exc:
            raise exc.__cause__
        except Exception as exc:  # noqa: BLE001 -- isolate technical trajectory failures
            result = failure_result(question, settings, exc)
        finally:
            steps.close()
        result.update(
            evaluation_mode=evaluation_mode,
            inference_seconds=time.perf_counter() - started,
        )
        return result, measurements

    async def task(question, retriever):
        """Keep seeds and bounded technical retries local to each trajectory."""
        async with limit:
            for attempt in range(trajectory_attempts):
                trial = question | {"trajectory_attempt": attempt + 1}
                if attempt:
                    base_seed = question.get(
                        "sampling_seed", config.seed + question["example_index"]
                    )
                    trial["sampling_seed"] = generation_seed(base_seed, attempt)
                result, measurements = await once(trial, retriever)
                if result["status"] not in (
                    "infrastructure_error",
                    "postprocessing_error",
                ):
                    break
            return result, measurements

    async with AsyncExitStack() as stack:
        retriever = (
            await stack.enter_async_context(search.open_session())
            if hasattr(search, "open_session")
            else search
        )
        tasks = [asyncio.create_task(task(q, retriever)) for q in questions]
        try:
            for ready in asyncio.as_completed(tasks):
                yield await ready
        finally:
            for item in tasks:
                if not item.done():
                    item.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def batched_agents(*args, **kwargs):
    """Synchronous iterator boundary for existing evaluation/training consumers.

    One event loop owns the asynchronous rollout pool. Network IO runs concurrently;
    no worker threads or synchronous generation/search waves are introduced here.
    Synchronous result persistence runs between yielded completions.
    """
    stream = async_agents(*args, **kwargs)
    with asyncio.Runner() as runner:
        try:
            while True:
                try:
                    yield runner.run(anext(stream))
                except StopAsyncIteration:
                    break
        finally:
            runner.run(stream.aclose())


def _decode_generation(request, generation, tokenizer, seconds):
    """Convert one backend output into the synchronous generator return format.

    Args:
        request: GenerationRequest with prompt IDs and assistant prefill.
        generation: Backend output with completion_ids and sampling_logprobs.
        tokenizer: Tokenizer used to decode sampled IDs.
        seconds: Client-observed latency for this individual generation request.

    Returns:
        (assistant_text, metadata), including prompt/completion IDs and token counts."""
    text = request.assistant_prefill + tokenizer.decode(
        list(generation.completion_ids), skip_special_tokens=True
    )
    return text, {
        "input_tokens": len(request.prompt_ids),
        "prompt_ids": list(request.prompt_ids),
        "finish_reason": getattr(generation, "finish_reason", None),
        "generated_tokens": len(generation.completion_ids),
        "generation_seconds": seconds,
        "timing_definition": "individual client generation request latency; includes provider waits",
        "sampling_logprobs": list(generation.sampling_logprobs),
        "completion_ids": list(generation.completion_ids),
    }
