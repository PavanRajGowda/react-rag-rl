"""Bounded generation/search loop with injectable backends for testing."""

import time
from dataclasses import asdict, dataclass

from baseline_agent.evidence import EvidenceStore, information_event, visible_evidence
from baseline_agent.protocol import (
    ProtocolError,
    format_information,
    parse_action,
    render_history,
)
from baseline_agent.scoring import normalize_text


@dataclass
class AgentConfig:
    """Controller limits and sampling settings for baseline evaluation. This is not a
    hosted training client or a request to allocate hardware.
    """
    top_k: int = 2
    max_searches: int = 7
    max_turns: int = 8
    max_new_tokens: int = 1024
    max_context_tokens: int = 8192
    context_mode: str = "append"
    max_evidence_passages: int = 12
    max_evidence_chars: int = 12000
    seed: int = 42
    temperature: float = 0.2

    def __post_init__(self):
        """Validate this configuration’s search, turn, context and sampling limits.

        Raises ValueError for invalid settings; returns None."""
        if (
            self.max_evidence_passages < 1
            or self.max_evidence_chars < 1
            or not 1 <= self.top_k <= 10
            or not 1 <= self.max_searches <= 10
            or not 1 <= self.max_turns <= 20
        ):
            raise ValueError("Invalid search/turn limits")
        if (
            self.context_mode not in ("append", "replace", "bounded")
            or self.max_new_tokens < 1
            or self.max_context_tokens <= self.max_new_tokens
            or self.temperature <= 0
        ):
            raise ValueError("Invalid context/generation configuration")


def run_agent(question, generator, search, config, emit=print):
    """Generate one complete trajectory for a single question, synchronously.

    Args:
        question: Question dictionary containing at least ``question`` and ``id``.
        generator: Per-trajectory generator exposing make_prefix() and generate().
            generate(prompt) returns (assistant_text, token/timing metadata).
            Batch callers use agent_steps() to coordinate the same controller.
        search: Callable accepting a query/top_k dictionary and returning a search
            response with status and results. Infrastructure exceptions propagate.
        config: AgentConfig controlling retrieval, history and generation limits.
        emit: Print-compatible logging callable; use a quiet callable to suppress it.

    Returns:
        Dictionary with question, config, status, answer, turns, searches and events.
        Each turn records one model inference; a trajectory can contain many turns.
        Timing for the whole trajectory and evaluation_mode are added by the caller."""
    steps = agent_steps(question, generator, config, emit)
    try:
        kind, payload = next(steps)
        while True:
            try:
                value = (
                    generator.generate(payload)
                    if kind == "generation"
                    else search(payload)
                )
            except Exception as exc:  # noqa: BLE001 - route failure to its trajectory
                kind, payload = steps.throw(exc)
            else:
                kind, payload = steps.send(value)
    except StopIteration as done:
        return done.value
    finally:
        steps.close()


def agent_steps(question, generator, config, emit=print):
    """Synchronous resumable controller shared by single and batch evaluation.

    Yields ("generation", prompt_text) or ("search", request_dict). The caller
    sends the corresponding generation tuple or search response back with send().
    Returns the completed result dictionary through StopIteration.value.
    No threads, asyncio, model calls or search calls are created here.
    """
    prefix = generator.make_prefix(question["question"])
    events, turns, searches = [], [], []
    store = EvidenceStore(config.max_evidence_passages, config.max_evidence_chars)
    successful_evidence = False
    status, answer = "turn_limit", None
    for number in range(1, config.max_turns + 1):
        prompt = (
            generator.render_prompt(events, config.context_mode)
            if hasattr(generator, "render_prompt")
            else render_history(prefix, events, config.context_mode)
        )
        emit(f"\n--- Model generation {number} ---", flush=True)
        try:
            segment, generation = yield "generation", prompt
        except ValueError as exc:
            status = "context_limit"
            turns.append({"turn": number, "error": str(exc)})
            break
        if hasattr(generator, "record_generation"):
            generator.record_generation(generation["completion_ids"], segment)
        emit("\n", flush=True)
        turns.append(
            {
                "turn": number,
                "model_input": prompt,
                "visible_evidence": visible_evidence(events, config.context_mode),
                "model_output": segment,
                **generation,
            }
        )
        events.append({"kind": "model", "text": segment})
        try:
            kind, action = parse_action(segment)
        except ProtocolError as exc:
            observation = format_information(
                {"status": "error", "code": "invalid_protocol", "message": str(exc)}
            )
            events.append(
                {
                    "kind": "feedback",
                    "text": "Controller formatting feedback (not a search result):\n"
                    + observation,
                }
            )
            emit(observation, flush=True)
            continue
        # This gate asks whether retrieval supplied evidence, not whether the
        # answer is correct or every gold document was found. The controller never
        # sees gold labels; retrospective evaluation handles those questions.
        if kind == "answer":
            if successful_evidence or (
                normalize_text(action) == "insufficient information"
                and any(s["response"].get("status") == "ok" for s in searches)
            ):
                status, answer = "answered", action
                break
            observation = format_information(
                {
                    "status": "error",
                    "code": "evidence_required",
                    "message": "Search and retrieve evidence before answering.",
                }
            )
            events.append({"kind": "feedback", "text": observation})
            emit(observation, flush=True)
            continue
        if len(searches) >= config.max_searches:
            status = "search_limit"
            break
        request = action | {"top_k": config.top_k}
        emit(f"\nSearch request: {request}", flush=True)
        started = time.perf_counter()
        # Infrastructure errors abort this debugging run; they do not become
        # evidence, model mistakes, or fabricated empty retrieval results.
        result = yield "search", request
        elapsed = time.perf_counter() - started
        searches.append(
            {"turn": number, "request": request, "response": result, "seconds": elapsed}
        )
        successful_evidence |= result.get("status") == "ok" and bool(
            result.get("results")
        )
        event = information_event(result, store, config.context_mode)
        event["search_turn"] = number
        observation = event["text"]
        events.append(event)
        emit(observation, flush=True)
    return {
        "question": question,
        "config": asdict(config),
        "status": status,
        "answer": answer,
        "turns": turns,
        "searches": searches,
        "events": events,
        "answer_validation": "not_performed",
    }
