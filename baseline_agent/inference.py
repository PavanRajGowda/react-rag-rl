"""Per-trajectory prompt rendering and exact token history; no model execution."""

from baseline_agent.prompt import CONTINUE, messages
from baseline_agent.protocol import select_history
from batched_inference.conversation import ASSISTANT_PREFILL, TokenConversation


class PromptGenerator:
    """Tokenizer-backed prompt/history utilities shared by baseline adapters. Despite its
    name this class never calls an LLM; the backend performs sampling.
    """
    def make_prefix(self, question):
        """Start a new trajectory for question text and return its initial prompt string.

        Resets per-trajectory messages and token history; does not run model inference."""
        self.base_messages = messages(question)
        self._token_history = None
        self._append_mode = False
        return self.render_prompt([], "replace")

    def render_prompt(self, events, context_mode):
        """Build the next prompt string from event dictionaries and a context mode.

        Append mode preserves exact sampled token history; bounded/replace select history.
        Call make_prefix() first. This method prepares input, not model output."""
        self._append_mode = context_mode == "append"
        if self._append_mode:
            if self._token_history is None:
                self._token_history = TokenConversation(
                    self.tokenizer, self.base_messages
                )
            return self._token_history.prepare(events)[0]
        conversation = list(self.base_messages)
        for event in select_history(events, context_mode):
            if event["kind"] == "model":
                conversation.append({"role": "assistant", "content": event["text"]})
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": event["text"] + CONTINUE,
                    }
                )
        # The requested checkpoint is an ordinary chat model, not a
        # Search-R1-trained continuation model. Start a fresh assistant chat
        # turn after each observation rather than inviting immediate EOS.
        return (
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ASSISTANT_PREFILL
        )

    def encode_prompt(self, prompt):
        """Return authoritative token IDs for the supplied rendered prompt string.

        Append mode reuses retained IDs and verifies text; other modes tokenize the text."""
        if getattr(self, "_append_mode", False):
            if prompt != self._token_history.text:
                raise ValueError(
                    "Generation prompt differs from retained token history"
                )
            return self._token_history.ids
        return tuple(self.tokenizer.encode(prompt, add_special_tokens=False))

    def record_generation(self, completion_ids, controller_output):
        """Record sampled completion IDs and decoded controller text in append mode.

        Returns None; other history modes require no update here."""
        if getattr(self, "_append_mode", False):
            self._token_history.record(completion_ids, controller_output)
