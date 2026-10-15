"""Independent text-level search episodes; no model, reward or cloud imports."""

import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass

from baseline_agent.protocol import (
    ProtocolError,
    format_information,
    parse_action,
    select_history,
)
from grpo_training.config import TrainingConfig
from grpo_training.data import Question, policy_messages

# Same observation suffix as QwenGenerator.render_prompt; parity tested directly.
CONTINUE = "\nContinue the original question. If the evidence is sufficient, calculate and emit the final numerical answer in answer tags. Otherwise request evidence using search with company, year and query child tags. Never use search to say that no search is needed. Replace all example placeholders with actual values."
DOMAIN_ERRORS = frozenset(
    {
        "unknown_company",
        "ambiguous_company",
        "company_mismatch",
        "report_year_unavailable",
        "invalid_request",
    }
)


class SearchContractError(RuntimeError):
    """Malformed/mismatched backend output must not become model feedback."""


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_seconds: float = 1.0

    def __post_init__(self):
        if type(self.attempts) is not int or not 1 <= self.attempts <= 5:
            raise ValueError("attempts must be an integer from 1 to 5")
        if (
            isinstance(self.backoff_seconds, bool)
            or not isinstance(self.backoff_seconds, (int, float))
            or not math.isfinite(self.backoff_seconds)
            or not 0 <= self.backoff_seconds <= 10
        ):
            raise ValueError("backoff_seconds must be finite and between 0 and 10")


def checked_response(response, config):
    """Fail closed on backend failures; known resolution errors are observations."""
    if not isinstance(response, dict):
        raise SearchContractError("Search response must be an object")
    if response.get("status") == "error":
        if response.get("code") not in DOMAIN_ERRORS or not isinstance(
            response.get("message"), str
        ):
            raise SearchContractError("Unknown or malformed search error")
    elif response.get("status") == "ok":
        if (
            response.get("corpus_version") != config.corpus_version
            or type(response.get("search_schema_version")) is not int
            or response.get("search_schema_version") != config.search_schema_version
        ):
            raise SearchContractError("Search corpus/schema disagrees with experiment")
        docs = response.get("results")
        if not isinstance(docs, list) or len(docs) > config.top_k:
            raise SearchContractError(
                "Search results must be a list within requested K"
            )
        for doc in docs:
            if not isinstance(doc, dict) or any(
                not isinstance(doc.get(k), str) or not doc[k].strip()
                for k in ("context_id", "company_name", "text")
            ):
                raise SearchContractError("Search document lacks required text fields")
            if any(type(doc.get(k)) is not int for k in ("report_year", "page_number")):
                raise SearchContractError("Search document lacks year/page metadata")
    else:
        raise SearchContractError("Unrecognized search response status")
    # Validate rendering before recording successful evidence or feeding the model.
    observation = format_information(response)
    return deepcopy(response), observation


class SearchEpisode:
    """One independent sample of a question; feed outputs without generating them.

    `search` is an injected synchronous API callable. It MUST enforce its own
    per-call deadline; retries bound attempts, not a hung blocking call. It must
    not wrap this controller in another external retry layer. Server-internal
    retries remain the transport's responsibility. Feed one output at a time per
    episode; batching should interleave separate episodes, not mutate one from
    multiple threads. No transport is instantiated in this commit.
    """

    def __init__(
        self,
        question,
        config,
        search,
        retry=None,
        *,
        sleep=time.sleep,
        clock=time.perf_counter,
    ):
        if type(question) is not Question or type(config) is not TrainingConfig:
            raise TypeError("Expected Question and TrainingConfig, without gold labels")
        if (
            question.split not in ("train", "dev")
            or question.dataset_revision != config.dataset_revision
        ):
            raise ValueError(
                "Episode needs a train/dev question from the configured revision"
            )
        if not isinstance(question.text, str) or not question.text.strip():
            raise ValueError("Episode question must be nonblank text")
        if not callable(search):
            raise TypeError("search must be an API callable")
        self.question, self.config = question, config
        self._search, self._sleep, self._clock = search, sleep, clock
        self.retry = retry if retry is not None else RetryPolicy()
        if type(self.retry) is not RetryPolicy:
            raise TypeError("Expected RetryPolicy")
        self._events, self._turns, self._searches = [], [], []
        self._status, self._answer, self._error = "running", None, None
        self._successful_evidence = False

    @property
    def status(self):
        return self._status

    @property
    def answer(self):
        return self._answer

    @property
    def done(self):
        return self._status != "running"

    def _require_running(self):
        if self.done:
            raise RuntimeError(f"Episode already terminated: {self.status}")

    def events(self, *, active_only=False):
        events = (
            select_history(self._events, self.config.context_mode)
            if active_only
            else self._events
        )
        return deepcopy(events)

    def messages(self):
        conversation = policy_messages(self.question)
        for event in self.events(active_only=True):
            conversation.append(
                {
                    "role": "assistant" if event["kind"] == "model" else "user",
                    "content": event["text"]
                    if event["kind"] == "model"
                    else event["text"] + CONTINUE,
                }
            )
        return conversation

    def check_context_budget(self, input_tokens):
        """Called by the future tokenizer adapter BEFORE each generation."""
        self._require_running()
        if type(input_tokens) is not int or input_tokens < 1:
            raise ValueError("input_tokens must be a positive integer")
        if input_tokens + self.config.max_new_tokens <= self.config.max_context_tokens:
            return True
        self._status = "context_limit"
        self._error = f"Prompt {input_tokens} + generation reserve {self.config.max_new_tokens} exceeds context budget {self.config.max_context_tokens}; nothing was truncated"
        self._turns.append({"turn": len(self._turns) + 1, "error": self._error})
        return False

    def fail_generation(self, error):
        """Backend failure, not a model mistake. Group handling is deferred."""
        self._require_running()
        if not isinstance(error, Exception):
            raise TypeError("Expected a generation exception")
        self._status, self._error = (
            "infrastructure_error",
            f"{type(error).__name__}: {error}",
        )
        self._turns.append({"turn": len(self._turns) + 1, "error": self._error})

    def _feedback(self, code, message, prefix=""):
        self._events.append(
            {
                "kind": "feedback",
                "text": prefix
                + format_information(
                    {"status": "error", "code": code, "message": message}
                ),
            }
        )

    def _execute_search(self, request, turn):
        attempts, result, observation = [], None, None
        started = self._clock()
        error = None
        for attempt in range(1, self.retry.attempts + 1):
            before = self._clock()
            try:
                result, observation = checked_response(
                    self._search(deepcopy(request)), self.config
                )
            except Exception as exc:  # noqa: BLE001 -- backend errors are audited and terminate unscorable, never hidden as evidence.
                error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {
                        "attempt": attempt,
                        "seconds": self._clock() - before,
                        "error": error,
                    }
                )
                # Contract/integration errors do not improve with a retry.
                if (
                    isinstance(
                        exc, (SearchContractError, TypeError, KeyError, ValueError)
                    )
                    or attempt == self.retry.attempts
                ):
                    break
                self._sleep(self.retry.backoff_seconds * 2 ** (attempt - 1))
            else:
                attempts.append({"attempt": attempt, "seconds": self._clock() - before})
                error = None
                break
        self._searches.append(
            {
                "turn": turn,
                "request": deepcopy(request),
                "response": result,
                "seconds": self._clock() - started,
                "attempts": attempts,
            }
        )
        if error is not None:
            self._status, self._error = "infrastructure_error", error
            return
        self._successful_evidence |= result["status"] == "ok" and bool(
            result["results"]
        )
        self._events.append({"kind": "information", "text": observation})

    def accept_output(self, output):
        """Process one complete generation (including its assistant prefill)."""
        self._require_running()
        if not isinstance(output, str):
            raise TypeError("Model output must be text")
        turn = len(self._turns) + 1
        self._turns.append({"turn": turn, "model_output": output})
        self._events.append({"kind": "model", "text": output})
        try:
            kind, action = parse_action(output)
        except ProtocolError as exc:
            self._feedback(
                "invalid_protocol",
                str(exc),
                "Controller formatting feedback (not a search result):\n",
            )
        else:
            if kind == "answer":
                # Deliberately the baseline's EVER-seen evidence gate, not an
                # active-context or gold-hit requirement. Do not change the agent.
                if self._successful_evidence:
                    self._status, self._answer = "answered", action
                else:
                    self._feedback(
                        "evidence_required",
                        "Search and retrieve evidence before answering.",
                    )
            elif len(self._searches) >= self.config.max_searches:
                self._status = "search_limit"
            else:
                self._execute_search(action | {"top_k": self.config.top_k}, turn)
        if not self.done and turn >= self.config.max_turns:
            self._status = "turn_limit"

    def snapshot(self):
        """Text-level audit only; token/log-probability records arrive in Step 4."""
        return deepcopy(
            {
                "question": {
                    "id": self.question.id,
                    "question": self.question.text,
                    "split": self.question.split,
                    "example_index": self.question.source_index,
                    "dataset_revision": self.question.dataset_revision,
                },
                "config": asdict(self.config),
                "status": self.status,
                "answer": self.answer,
                "turns": self._turns,
                "searches": self._searches,
                "events": self._events,
                "error": self._error,
                "numeric_validation": "not_performed",
            }
        )
