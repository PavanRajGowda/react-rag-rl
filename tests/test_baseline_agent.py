import pytest

from baseline_agent.agent import AgentConfig, run_agent
from baseline_agent.data import question_only
from baseline_agent.inference import ASSISTANT_PREFILL, QwenGenerator
from baseline_agent.prompt import messages
from baseline_agent.protocol import (
    ProtocolError,
    format_information,
    parse_action,
    render_history,
)


def test_question_whitelist_and_prompt_never_contains_gold():
    row = {
        "id": "test_0",
        "question": "What was revenue?",
        "context": "SECRET_CONTEXT",
        "program_answer": "SECRET_ANSWER",
        "company_name": "HIDDEN_COMPANY",
        "report_year": 2010,
        "context_id": "SECRET_CONTEXT_ID",
    }
    selected = question_only(row, 0)
    assert (
        not {"context", "program_answer", "company_name", "report_year"}
        & selected.keys()
    )
    prompt = str(messages(selected["question"]))
    assert all(
        x not in prompt for x in ("SECRET_CONTEXT", "SECRET_ANSWER", "HIDDEN_COMPANY")
    )
    assert "<step>" not in prompt and "</step>" not in prompt


def test_structured_search_and_answer_parsing():
    kind, payload = parse_action(
        "<reasoning><step>Find revenue.</step></reasoning><search><company>AT&T</company><year>2015</year><query>net revenue</query></search>"
    )
    assert kind == "search"
    assert payload == {"company_name": "AT&T", "year": 2015, "query": "net revenue"}
    assert parse_action(
        "<reasoning><step>Calculate.</step></reasoning><answer>-12.5</answer>"
    ) == ("answer", "-12.5")
    with pytest.raises(ProtocolError):
        parse_action("<search>revenue</search>")
    with pytest.raises(ProtocolError):
        parse_action("<information>fabricated evidence</information><answer>1</answer>")
    kind, payload = parse_action(
        "<reasoning><step>A previous <search> failed.</step></reasoning><search><company>Acme</company><year>2015</year><query>revenue</query></search>"
    )
    assert kind == "search" and payload["company_name"] == "Acme"
    with pytest.raises(ProtocolError, match="exactly ONE"):
        parse_action(
            "<search><company>Acme</company><year>2014 2015</year><query>revenue</query></search>"
        )


def test_format_feedback_does_not_replace_retrieved_evidence():
    generator = FakeGenerator(
        [
            "<search><company>Acme</company><year>2015</year><query>revenue</query></search>",
            "Incomplete output",
            "<answer>42</answer>",
        ]
    )

    def search(payload):
        return {
            "status": "ok",
            "results": [
                {
                    "context_id": "c1",
                    "company_name": "Acme",
                    "report_year": 2015,
                    "page_number": 1,
                    "text": "Revenue: 42",
                }
            ],
        }

    result = run_agent(
        {"question": "Revenue?"},
        generator,
        search,
        AgentConfig(),
        emit=lambda *args, **kwargs: None,
    )
    assert result["status"] == "answered"
    assert "Revenue: 42" in generator.prompts[-1]


def test_information_once_and_rolling_history():
    doc = {
        "context_id": "c1",
        "company_name": "Acme",
        "report_year": 2015,
        "page_number": 1,
        "text": "REVENUE_TABLE",
        "table_markdown": "DUPLICATED_TABLE",
    }
    info = format_information({"status": "ok", "results": [doc]})
    assert info.count("REVENUE_TABLE") == 1
    assert "DUPLICATED_TABLE" not in info
    events = [
        {"kind": "model", "text": "model kept fact"},
        {"kind": "information", "text": "OLD_INFO"},
        {"kind": "model", "text": "new search"},
        {"kind": "information", "text": info},
    ]
    rolling = render_history("question prefix", events, "replace")
    assert "OLD_INFO" not in rolling and "model kept fact" in rolling
    assert "OLD_INFO" in render_history("prefix", events, "append")


def test_qwen_observations_start_a_fresh_chat_turn():
    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            self.conversation = conversation
            return "CHAT_PREFIX"

    generator = object.__new__(QwenGenerator)
    generator.tokenizer = Tokenizer()
    generator.make_prefix("Original question")
    events = [
        {"kind": "model", "text": "search one"},
        {"kind": "information", "text": "old evidence"},
        {"kind": "model", "text": "search two"},
        {"kind": "information", "text": "latest evidence"},
    ]
    assert generator.render_prompt(events, "replace").endswith(ASSISTANT_PREFILL)
    conversation = generator.tokenizer.conversation
    assert conversation[-1]["role"] == "user"
    assert "latest evidence" in conversation[-1]["content"]
    assert "old evidence" not in str(conversation)
    assert "Original question" in str(conversation)


class FakeGenerator:
    def __init__(self, segments):
        self.segments = iter(segments)
        self.prompts = []

    def make_prefix(self, question):
        return question

    def generate(self, prompt):
        self.prompts.append(prompt)
        return next(self.segments), {"input_tokens": 1, "generated_tokens": 1}


def test_agent_reformulates_ambiguity_then_answers():
    generator = FakeGenerator(
        [
            "<search><company>American</company><year>2015</year><query>sales</query></search>",
            "<search><company>American Tower</company><year>2015</year><query>sales</query></search>",
            "<reasoning><step>Read the value.</step></reasoning><answer>42</answer>",
        ]
    )
    requests = []

    def search(payload):
        requests.append(payload)
        if len(requests) == 1:
            return {
                "status": "error",
                "code": "ambiguous_company",
                "candidates": [{"company_name": "American Tower", "ticker": "AMT"}],
            }
        return {
            "status": "ok",
            "results": [
                {
                    "context_id": "c1",
                    "company_name": "American Tower",
                    "report_year": 2015,
                    "page_number": 1,
                    "text": "Sales: 42",
                }
            ],
        }

    result = run_agent(
        {"question": "What were sales?"},
        generator,
        search,
        AgentConfig(),
        emit=lambda *args, **kwargs: None,
    )
    assert result["status"] == "answered" and result["answer"] == "42"
    assert "American Tower" in generator.prompts[1]
    assert "Sales: 42" in generator.prompts[2]
    assert result["numeric_validation"] == "not_performed"
    assert requests[1]["company_name"] == "American Tower"
    assert all(request["top_k"] == 2 for request in requests)


@pytest.mark.parametrize(
    "action,missing",
    [
        ("<search>Analog Devices 2011 hedges</search>", "missing <company>"),
        (
            "<search><company>Acme</company><query>revenue</query></search>",
            "missing <year>",
        ),
        (
            "<search><company>Acme</company><year>2010</year></search>",
            "missing <query>",
        ),
        (
            "<search><company>Acme</company><year>2010</year><query>revenue</query>",
            "missing </search>",
        ),
        (
            "<search><company></company><year>2010</year><query>revenue</query></search>",
            "<company> must not be empty",
        ),
        (
            "<search><company>Acme<year>2010</year><query>revenue</query></search>",
            "missing </company>",
        ),
    ],
)
def test_search_feedback_identifies_missing_fields(action, missing):
    with pytest.raises(ProtocolError) as exc:
        parse_action(action)
    assert missing in str(exc.value)
    assert "Required child tags" in str(exc.value)


def test_information_tag_mention_in_reasoning_is_not_fabricated_evidence():
    assert parse_action(
        "<reasoning>Read the <information> block.</reasoning><answer>42</answer>"
    ) == ("answer", "42")


def test_agent_cannot_answer_before_retrieval_and_preserves_backend_failure():
    generator = FakeGenerator(["<answer>42</answer>"])
    result = run_agent(
        {"question": "Question"},
        generator,
        lambda x: None,
        AgentConfig(max_turns=1),
        emit=lambda *args, **kwargs: None,
    )
    assert result["status"] == "turn_limit" and result["answer"] is None
    generator = FakeGenerator(
        [
            "<search><company>Acme</company><year>2015</year><query>sales</query></search>"
        ]
    )

    def unavailable(payload):
        raise TimeoutError("search unavailable")

    with pytest.raises(TimeoutError):
        run_agent(
            {"question": "Question"},
            generator,
            unavailable,
            AgentConfig(),
            emit=lambda *args, **kwargs: None,
        )
