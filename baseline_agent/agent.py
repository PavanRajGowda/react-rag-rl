"""Bounded generation/search loop with injectable backends for testing."""

import time
from dataclasses import asdict, dataclass

from baseline_agent.protocol import (
    ProtocolError,
    format_information,
    parse_action,
    render_history,
)


@dataclass
class AgentConfig:
    top_k: int = 2
    max_searches: int = 6
    max_turns: int = 8
    max_new_tokens: int = 1024
    max_context_tokens: int = 8192
    context_mode: str = "replace"
    seed: int = 42
    temperature: float = 0.2

    def __post_init__(self):
        if (
            not 1 <= self.top_k <= 10
            or not 1 <= self.max_searches <= 10
            or not 1 <= self.max_turns <= 20
        ):
            raise ValueError("Invalid search/turn limits")
        if (
            self.context_mode not in ("append", "replace")
            or self.max_new_tokens < 1
            or self.max_context_tokens <= self.max_new_tokens
            or self.temperature <= 0
        ):
            raise ValueError("Invalid context/generation configuration")


def run_agent(question, generator, search, config, emit=print):
    prefix = generator.make_prefix(question["question"])
    events, turns, searches = [], [], []
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
            segment, generation = generator.generate(prompt)
        except ValueError as exc:
            status = "context_limit"
            turns.append({"turn": number, "error": str(exc)})
            break
        emit("\n", flush=True)
        turns.append(
            {
                "turn": number,
                "model_input": prompt,
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
        if kind == "answer":
            if successful_evidence:
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
        result = search(request)
        elapsed = time.perf_counter() - started
        searches.append(
            {"turn": number, "request": request, "response": result, "seconds": elapsed}
        )
        successful_evidence |= result.get("status") == "ok" and bool(
            result.get("results")
        )
        observation = format_information(result)
        events.append({"kind": "information", "text": observation})
        emit(observation, flush=True)
    return {
        "question": question,
        "config": asdict(config),
        "status": status,
        "answer": answer,
        "turns": turns,
        "searches": searches,
        "events": events,
        "numeric_validation": "not_performed",
    }
