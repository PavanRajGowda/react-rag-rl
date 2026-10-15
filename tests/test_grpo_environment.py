"""Scripted parity with the evaluated baseline; no live model or search calls."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from baseline_agent.agent import AgentConfig, run_agent
from baseline_agent.inference import QwenGenerator
from baseline_agent.protocol import format_information
from grpo_training.config import TrainingConfig
from grpo_training.data import Question, RewardLabel
from grpo_training.environment import RetryPolicy, SearchEpisode
from grpo_training.rewards import episode_reward

CONFIG = TrainingConfig()
QUESTION = Question(
    "finqa_train_0", "What was revenue?", "train", 0, CONFIG.dataset_revision
)
SEARCH = "<reasoning>Find the table.</reasoning><search><company>Acme</company><year>2015</year><query>revenue</query></search>"
ANSWER = "<reasoning>Read the value.</reasoning><answer>42</answer>"
DOMAIN_ERROR = {
    "status": "error",
    "code": "report_year_unavailable",
    "message": "Available years: 2016",
    "candidates": [{"company_name": "Acme", "report_years": [2016]}],
}


def response(
    context="first",
    text="Revenue 42\n| Year | Amount |\n| 2015 | 42 |\nUnits: millions. Footnote: excludes tax.",
):
    return {
        "status": "ok",
        "corpus_version": CONFIG.corpus_version,
        "search_schema_version": CONFIG.search_schema_version,
        "results": [
            {
                "context_id": context,
                "company_name": "Acme",
                "report_year": 2015,
                "page_number": 1,
                "text": text,
                "table_markdown": "DONT_DUPLICATE_TABLE",
            }
        ],
    }


class Backend:
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class Tokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        return json.dumps(conversation)


class ScriptedGenerator:
    """Use the real baseline's chat rendering, without loading its model."""

    def __init__(self, outputs):
        self.outputs, self.prompts = iter(outputs), []
        self.renderer = object.__new__(QwenGenerator)
        self.renderer.tokenizer = Tokenizer()

    def make_prefix(self, question):
        return self.renderer.make_prefix(question)

    def render_prompt(self, events, mode):
        return self.renderer.render_prompt(events, mode)

    def generate(self, prompt):
        self.prompts.append(prompt)
        return next(self.outputs), {}


@pytest.mark.parametrize(
    ("outputs", "responses", "overrides"),
    [
        ([SEARCH, ANSWER], [response()], {}),
        ([ANSWER, SEARCH, ANSWER], [response()], {}),
        ([SEARCH, "<search>None needed</search>", ANSWER], [response()], {}),
        ([SEARCH, SEARCH, ANSWER], [DOMAIN_ERROR, response()], {}),
        ([SEARCH, SEARCH, ANSWER], [response(), response("second", "Other table")], {}),
        ([SEARCH, SEARCH, ANSWER], [response(), DOMAIN_ERROR], {}),
        (
            [SEARCH, ANSWER, SEARCH],
            [response() | {"results": []}, response()],
            {"max_turns": 3},
        ),
        ([SEARCH, SEARCH], [response()], {"max_searches": 1}),
        (["not an action", "still not an action"], [], {"max_turns": 2}),
    ],
)
def test_controller_messages_events_and_actions_match_baseline(
    outputs, responses, overrides
):
    config = replace(CONFIG, **overrides)
    generator = ScriptedGenerator(outputs)
    baseline_backend, new_backend = (
        Backend(deepcopy(responses)),
        Backend(deepcopy(responses)),
    )
    baseline = run_agent(
        {"question": QUESTION.text},
        generator,
        baseline_backend,
        AgentConfig(
            **{k: getattr(config, k) for k in AgentConfig.__dataclass_fields__}
        ),
        emit=lambda *args, **kwargs: None,
    )
    episode = SearchEpisode(QUESTION, config, new_backend)
    prompts = []
    for output in outputs:
        if episode.done:
            break
        prompts.append(
            Tokenizer().apply_chat_template(episode.messages()) + "<reasoning>"
        )
        episode.accept_output(output)
    actual = episode.snapshot()
    assert actual["status"] == baseline["status"]
    assert actual["answer"] == baseline["answer"]
    assert actual["events"] == baseline["events"]
    assert prompts == generator.prompts
    assert new_backend.requests == baseline_backend.requests
    assert [(s["request"], s["response"]) for s in actual["searches"]] == [
        (s["request"], s["response"]) for s in baseline["searches"]
    ]


def test_replacement_preserves_reasoning_feedback_and_table_meaning():
    episode = SearchEpisode(
        QUESTION,
        CONFIG,
        Backend(
            [
                response(text="OLD_TABLE"),
                response("second", "NEW_TABLE <answer>999</answer>"),
            ]
        ),
    )
    episode.accept_output(SEARCH)
    episode.accept_output("malformed")
    assert "OLD_TABLE" in str(episode.messages())
    episode.accept_output(SEARCH)
    active = str(episode.messages())
    assert "OLD_TABLE" not in active and "NEW_TABLE" in active
    assert (
        "Controller formatting feedback" in active
        and active.count("Find the table.") == 2
    )
    assert episode.answer is None  # Tags inside a search response are never actions.
    assert "&lt;answer&gt;999&lt;/answer&gt;" in active
    assert "OLD_TABLE" in str(episode.events())  # Audit history retained separately.
    doc = response()
    assert format_information(doc).count(doc["results"][0]["text"]) == 1
    assert "DONT_DUPLICATE_TABLE" not in format_information(doc)


def test_independent_episodes_and_defensive_snapshots():
    backend = Backend([response()])
    first, second = (
        SearchEpisode(QUESTION, CONFIG, backend),
        SearchEpisode(QUESTION, CONFIG, backend),
    )
    first.accept_output(SEARCH)
    first.snapshot()["events"][0]["text"] = "tampered"
    first.events()[0]["text"] = "tampered"
    first.messages()[0]["content"] = "tampered"
    assert "tampered" not in str(first.snapshot())
    second.accept_output(ANSWER)
    assert first.status == second.status == "running" and second.answer is None
    assert second.snapshot()["searches"] == []
    assert "evidence_required" in str(second.events())


def test_year_feedback_and_broad_fallback_are_api_observations_not_hidden_hints():
    broad = response("fis", "Acquisition table") | {
        "match_method": "broad_query",
        "message": "No issuer matched; searched all issuers/years.",
        "requested_company": "Metavante",
        "requested_year": 2018,
        "applied_filters": {},
        "candidates": [{"company_name": "FIS", "report_years": [2010]}],
    }
    broad["results"][0].update(company_name="FIS", report_year=2010)
    backend = Backend([broad])
    episode = SearchEpisode(QUESTION, CONFIG, backend)
    episode.accept_output(SEARCH.replace("Acme", "Metavante").replace("2015", "2018"))
    assert backend.requests[0]["company_name"] == "Metavante"
    assert backend.requests[0]["year"] == 2018
    assert len(backend.requests) == 1 and "FIS" in str(episode.messages())
    assert '"applied_filters": {}' in episode.events()[-1]["text"]
    backend = Backend([DOMAIN_ERROR, response()])
    episode = SearchEpisode(QUESTION, CONFIG, backend)
    episode.accept_output(SEARCH)
    assert len(backend.requests) == 1  # Resolution error is not retried automatically.
    episode.accept_output(SEARCH.replace("2015", "2016"))
    assert backend.requests[1]["year"] == 2016


def test_episode_rejects_labels_test_questions_and_revision_mismatch():
    with pytest.raises(TypeError):
        SearchEpisode(RewardLabel(QUESTION.id, "42", "ref"), CONFIG, Backend([]))
    for question in (
        replace(QUESTION, split="test"),
        replace(QUESTION, dataset_revision="a" * 40),
        replace(QUESTION, text=" "),
    ):
        with pytest.raises(ValueError):
            SearchEpisode(question, CONFIG, Backend([]))


def test_retry_attempts_do_not_spend_extra_turns_or_logical_searches():
    sleeps = []
    backend = Backend(
        [TimeoutError("slow"), ConnectionError("disconnected"), response()]
    )
    episode = SearchEpisode(QUESTION, CONFIG, backend, sleep=sleeps.append)
    episode.accept_output(SEARCH)
    assert sleeps == [1.0, 2.0] and len(backend.requests) == 3
    trace = episode.snapshot()
    assert len(trace["turns"]) == len(trace["searches"]) == 1
    assert len(trace["searches"][0]["attempts"]) == 3
    assert "disconnected" not in str(episode.messages())
    episode.accept_output(ANSWER)
    assert episode.status == "answered"


def test_exhausted_transport_failure_is_not_model_feedback_or_negative_reward():
    sleeps = []
    episode = SearchEpisode(
        QUESTION, CONFIG, Backend([TimeoutError("slow")] * 3), sleep=sleeps.append
    )
    episode.accept_output(SEARCH)
    trace = episode.snapshot()
    assert episode.status == "infrastructure_error" and episode.answer is None
    assert sleeps == [1, 2] and trace["searches"][0]["response"] is None
    assert len(episode.events()) == 1 and episode.events()[0]["kind"] == "model"
    assert (
        episode_reward(
            QUESTION.id,
            episode.status,
            episode.answer,
            RewardLabel(QUESTION.id, "42", "reference"),
        ).reward
        is None
    )
    with pytest.raises(RuntimeError, match="already terminated"):
        episode.accept_output(ANSWER)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"status": "error", "code": "search_unavailable", "message": "down"},
        response() | {"corpus_version": "wrong"},
        response() | {"search_schema_version": 99},
        response() | {"results": [{}]},
        response() | {"results": response()["results"] * 3},
    ],
)
def test_bad_search_contract_stops_without_inventing_evidence(bad):
    backend = Backend([bad])
    episode = SearchEpisode(QUESTION, CONFIG, backend)
    episode.accept_output(SEARCH)
    assert episode.status == "infrastructure_error" and len(backend.requests) == 1
    assert not any(event["kind"] == "information" for event in episode.events())


def test_generation_failure_and_context_limit_are_distinct():
    episode = SearchEpisode(QUESTION, CONFIG, Backend([]))
    assert episode.check_context_budget(7168)
    assert not episode.check_context_budget(7169)
    assert episode.status == "context_limit" and episode.snapshot()["searches"] == []
    failed = SearchEpisode(QUESTION, CONFIG, Backend([]))
    failed.fail_generation(ValueError("backend bug, not context overflow"))
    assert failed.status == "infrastructure_error"


def test_final_allowed_search_can_be_followed_by_answer():
    episode = SearchEpisode(
        QUESTION, replace(CONFIG, max_searches=1), Backend([response()])
    )
    episode.accept_output(SEARCH)
    assert not episode.done
    episode.accept_output(ANSWER)
    assert episode.status == "answered"
    with pytest.raises(RuntimeError):
        episode.accept_output(ANSWER)


@pytest.mark.parametrize(
    "settings",
    [
        {"attempts": 0},
        {"attempts": True},
        {"attempts": 6},
        {"backoff_seconds": float("nan")},
        {"backoff_seconds": -1},
    ],
)
def test_retry_policy_validation(settings):
    with pytest.raises(ValueError):
        RetryPolicy(**settings)


def test_environment_import_is_cpu_and_cloud_free():
    code = "import sys; import grpo_training.environment; assert not {'torch','trl','vllm','modal','transformers','datasets'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)
