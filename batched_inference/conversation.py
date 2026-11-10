"""Append-only token history for Qwen's text-only agent conversations."""

from baseline_agent.prompt import CONTINUE

ASSISTANT_PREFILL = "<reasoning>"
TOKEN_HISTORY_VERSION = "append-sampled-ids-v1"


class TokenConversation:
    """Render the initial prompt once; encode only new controller observations."""

    def __init__(self, tokenizer, messages):
        """Initialize history from a tokenizer and initial role/content messages.

        Encodes the initial prompt once; stores diagnostic text and authoritative IDs."""
        self.tokenizer = tokenizer
        self.text = self._render(messages) + ASSISTANT_PREFILL
        self.ids = tuple(tokenizer.encode(self.text, add_special_tokens=False))
        self._event_count = 0
        self._accepted_events = ()
        self._last_output = None
        self._last_ids = ()

    def _render(self, messages):
        """Render role/content message dictionaries into text with an assistant header.

        Returns a string; does not tokenize or mutate the retained conversation."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def record(self, completion_ids, controller_output):
        """Append one sampled completion without re-encoding its token IDs.

        Args:
            completion_ids: Nonnegative integer IDs returned by the inference backend.
            controller_output: Decoded assistant text including the reasoning prefill.

        Returns:
            None; updates retained history. Raises if IDs/text or call order disagree."""
        if self._last_output is not None:
            raise RuntimeError("Previous completion has not received its observation")
        if not controller_output.startswith(ASSISTANT_PREFILL):
            raise ValueError("Completion is missing its controller prefill")
        ids = tuple(completion_ids)
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Expected nonnegative integer completion IDs")
        expected = ASSISTANT_PREFILL + self.tokenizer.decode(
            list(ids), skip_special_tokens=True
        )
        if controller_output != expected:
            raise ValueError("Sampled token IDs disagree with controller output")
        raw_text = self.tokenizer.decode(list(ids), skip_special_tokens=False)
        # Decoded text is an audit view. The sampled IDs are authoritative: BPE
        # decoding followed by encoding is not guaranteed to recover them exactly.
        self.ids += ids
        self.text += raw_text
        self._last_output, self._last_ids = controller_output, ids

    def prepare(self, events):
        """Append new controller observations and prepare the next inference prompt.

        Args:
            events: Complete ordered model/observation event history with kind and text.
                Previously consumed events must remain unchanged.

        Returns:
            (prompt_text, prompt_ids), where prompt_ids is the authoritative tuple.
            Existing sampled tokens are preserved; only new observation/boundary text
            is encoded. Raises for rewritten history or missing controller feedback."""
        signatures = tuple((event["kind"], event["text"]) for event in events)
        if signatures[: self._event_count] != self._accepted_events:
            raise ValueError(
                "Append-only conversation history was changed or shortened"
            )
        pending = events[self._event_count :]
        if not pending:
            if self._last_output is not None:
                raise RuntimeError("Completion is awaiting controller feedback")
            return self.text, self.ids
        if (
            self._last_output is None
            or pending[0]["kind"] != "model"
            or pending[0]["text"] != self._last_output
            or len(pending) < 2
            or any(event["kind"] == "model" for event in pending[1:])
        ):
            raise ValueError("Conversation events do not match recorded sampled tokens")
        observations = [event["text"] + CONTINUE for event in pending[1:]]
        marker = "__COMPLETED_ASSISTANT_BOUNDARY__"
        while any(marker in text for text in observations):
            marker += "_"
        # Only the suffix after the placeholder is used. Qwen may rewrite that
        # placeholder's prefix; it can never rewrite our retained history.
        scaffold = self._render(
            [
                {"role": "user", "content": "Previous question."},
                {"role": "assistant", "content": marker},
                *({"role": "user", "content": text} for text in observations),
            ]
        )
        if scaffold.count(marker) != 1:
            raise ValueError("Chat template did not preserve the assistant boundary")
        bridge = scaffold.split(marker, 1)[1] + ASSISTANT_PREFILL
        end = getattr(self.tokenizer, "eos_token", None)
        if (
            end
            and self._last_ids
            and self._last_ids[-1] == getattr(self.tokenizer, "eos_token_id", None)
            and bridge.startswith(end)
        ):
            bridge = bridge[
                len(end) :
            ]  # The sampled message terminator is already present.
        # Encode only the NEW observation/header bridge. Previously sampled
        # reasoning/actions and previously visible evidence stay untouched.
        self.ids += tuple(self.tokenizer.encode(bridge, add_special_tokens=False))
        self.text += bridge
        self._event_count = len(events)
        self._accepted_events = signatures
        self._last_output, self._last_ids = None, ()
        return self.text, self.ids
