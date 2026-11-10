"""Training records attached to the same batched agent used for baseline inference."""

import inspect
import json
from contextlib import asynccontextmanager
from dataclasses import asdict

from batched_inference.baseline import batched_agents
from tinker_training.policy import template_digest
from tinker_training.records import EpisodeRecord, GenerationRecord, SamplingParameters
from tinker_training.search_contract import checked_response


def generate_episodes(
    specs, config, policy, backend, tokenizer, search, *, on_event=None
):
    """Yield replay-ready records from shared baseline control and batching.

    specs contain episode/group IDs, seed, sibling index and a typed Question.
    Gold labels never enter this function. The callback retains exact sampled IDs
    and old log probabilities once; controller audits omit repeated prompt/token data.
    No per-completion full-trajectory serialization or disk export occurs here.
    """
    specs = tuple(specs)
    if not specs:
        return
    if template_digest(tokenizer) != policy.template_sha256:
        raise ValueError(
            "Tokenizer template/prompt identity disagrees with rollout policy"
        )
    by_id = {s["episode_id"]: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError("Duplicate episode identity")
    records = {key: [] for key in by_id}
    sampling = SamplingParameters(config.temperature, config.top_p)

    def record(request, generation):
        """Keep exact backend tokens; never reconstruct replay IDs from decoded text."""
        if request.policy != policy or request.sampling != sampling:
            raise ValueError("Sample policy/sampling mismatch")
        records[request.episode_id].append(
            GenerationRecord(
                request,
                generation.completion_ids,
                generation.sampling_logprobs,
                tokenizer.decode(
                    list(generation.completion_ids), skip_special_tokens=True
                ),
                generation.finish_reason,
            )
        )

    async def checked(payload):
        """Await the supplied retriever before applying corpus/schema validation."""
        value = search(payload)
        if inspect.isawaitable(value):
            value = await value
        return checked_response(value, config)[0]

    if hasattr(search, "open_session"):

        @asynccontextmanager
        async def open_session():
            """One validated map session per rollout pool, closed on every exit."""
            async with search.open_session() as session:

                async def retrieve(payload):
                    return checked_response(await session(payload), config)[0]

                yield retrieve

        checked.open_session = open_session
    questions = [
        {
            "id": s["episode_id"],
            "group_id": s["group_id"],
            "training_question_id": s["question"].id,
            "sample_index": s["sample_index"],
            "sampling_seed": s["seed"],
            "question": s["question"].text,
            "example_index": s["question"].source_index,
        }
        for s in specs
    ]
    for result, _ in batched_agents(
        questions,
        backend,
        tokenizer,
        policy,
        config,
        checked,
        min(len(questions), config.rollout_concurrency),
        sampling=sampling,
        on_sample=record,
        on_event=on_event,
        generation_errors="raise",
    ):
        spec = by_id[result["question"]["id"]]
        audit = {
            "error": result.get("error"),
            "question": asdict(spec["question"]),
            "status": result["status"],
            "answer": result["answer"],
            "turns": [
                {
                    k: t[k]
                    for k in ("turn", "model_output", "visible_evidence")
                    if k in t
                }
                for t in result["turns"]
            ],
            "searches": result["searches"],
            "events": result["events"],
        }
        yield EpisodeRecord(
            spec["episode_id"],
            spec["group_id"],
            spec["sample_index"],
            spec["seed"],
            spec["question"],
            config,
            policy,
            sampling,
            result["status"],
            result["answer"],
            tuple(records.pop(spec["episode_id"])),
            json.dumps(audit, ensure_ascii=False, allow_nan=False),
        )
