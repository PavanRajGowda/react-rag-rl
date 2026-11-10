"""Exact token-history tests without a model, GPU or network."""

import pytest

from baseline_agent.prompt import messages
from batched_inference.conversation import TokenConversation


class Tokenizer:
    chat_template = "toy-rewriting-chat-v1"
    eos_token, eos_token_id = "<END>", 0

    def __init__(self):
        """Initialize a toy tokenizer with controlled rendering and token-boundary
        behavior.
        """
        self.encoded = []

    def apply_chat_template(self, messages, **kwargs):
        # Historical assistants omit the native empty-think prefix deliberately.
        """Render toy role/content messages with deliberate history behavior for
        prefix-preservation tests.
        """
        return (
            "".join(f"<{m['role']}>{m['content']}<END>\n" for m in messages)
            + "<assistant><think></think>"
        )

    def encode(self, text, *, add_special_tokens):
        """Encode text into deterministic test IDs with the fixture's boundary behavior."""
        self.encoded.append(text)
        result = []
        for index, part in enumerate(text.split(self.eos_token)):
            if index:
                result.append(0)
            result.extend(ord(char) + 1 for char in part)
        return result

    def decode(self, ids, *, skip_special_tokens):
        """Decode toy IDs for readable controller output without changing recorded token
        history.
        """
        return "".join(
            ("" if skip_special_tokens else self.eos_token)
            if value == 0
            else "A"
            if value == 999999
            else chr(value - 1)
            for value in ids
        )


def observation(output, text="New evidence"):
    """Construct a new information event for incremental conversation preparation."""
    return [{"kind": "model", "text": output}, {"kind": "information", "text": text}]


def test_noncanonical_sampled_ids_survive_multiple_turns_without_reencoding():
    """Verify noncanonical sampled ids survive multiple turns without reencoding."""
    tok = Tokenizer()
    history = TokenConversation(tok, messages("PRIVATE ORIGINAL QUESTION"))
    events = []
    for turn in range(2):
        before = history.ids
        history.record((999999,), "<reasoning>A")
        events += observation("<reasoning>A", f"Evidence {turn}")
        _, after = history.prepare(events)
        assert after[: len(before) + 1] == before + (999999,)
    assert len(tok.encoded) == 3  # One initial encoding and one new bridge per turn.
    assert all("PRIVATE ORIGINAL QUESTION" not in text for text in tok.encoded[1:])
    assert tuple(tok.encode(history.text, add_special_tokens=False)) != history.ids
    assert history.prepare(events)[1] == history.ids
    events[0]["text"] = "rewritten history"
    with pytest.raises(ValueError, match="changed"):
        history.prepare(events)


def test_feedback_and_boundary_marker_text_are_safe_and_order_is_checked():
    """Verify feedback and boundary marker text are safe and order is checked."""
    tok = Tokenizer()
    history = TokenConversation(tok, messages("Question"))
    history.record((999999,), "<reasoning>A")
    with pytest.raises(RuntimeError, match="observation"):
        history.record((999999,), "<reasoning>A")
    with pytest.raises(RuntimeError, match="feedback"):
        history.prepare([])
    events = observation(
        "<reasoning>A", "__COMPLETED_ASSISTANT_BOUNDARY__ invalid protocol"
    )
    events[1]["kind"] = "feedback"
    text, ids = history.prepare(events)
    assert "invalid protocol" in text and 999999 in ids


def test_sampled_end_token_is_retained_without_duplicating_message_terminator():
    """Verify sampled end token is retained without duplicating message terminator."""
    tok = Tokenizer()
    history = TokenConversation(tok, messages("Question"))
    expected = history.ids + (999999, 0)
    history.record((999999, 0), "<reasoning>A")
    _, after = history.prepare(observation("<reasoning>A"))
    assert after[: len(expected)] == expected
    assert after[len(expected)] != 0
    assert "A<END>\n<user>" in history.text
