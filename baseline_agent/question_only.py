"""Pretrained-only setting: one question, one generation, no retrieval interface."""

from dataclasses import asdict

from baseline_agent.inference import ASSISTANT_PREFILL, QwenGenerator
from baseline_agent.protocol import ProtocolError, parse_action

SYSTEM_PROMPT = """Answer the financial question using only your existing knowledge
and information explicitly present in the question. No documents, retrieved
context, search service, or other tools are available.

Give concise reasoning inside <reasoning>...</reasoning>, then put only the
final numerical answer inside <answer>...</answer>. Close reasoning before
answering. Do not quote action tags inside reasoning. Never emit a search action.
Preserve negative signs and decimals. Do not include currency symbols, units or
percent signs in answer. Express percentage answers as decimal fractions;
do not multiply ratios by 100. For yes/no questions, answer Yes or No.
If you cannot determine the answer, use <answer>unknown</answer> rather than
inventing financial facts. There is one input and one output; no follow-up
information will be supplied.
"""


def messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Question: " + question},
    ]


class QuestionOnlyGenerator(QwenGenerator):
    def make_prefix(self, question):
        return (
            self.tokenizer.apply_chat_template(
                messages(question),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ASSISTANT_PREFILL
        )


def run_question_only(question, generator, config):
    prompt = generator.make_prefix(question["question"])
    turns, events = [], []
    answer, status, error = None, "invalid_protocol", None
    try:
        segment, generation = generator.generate(prompt)
        turns.append(
            {"turn": 1, "model_input": prompt, "model_output": segment, **generation}
        )
        events.append({"kind": "model", "text": segment})
        try:
            kind, value = parse_action(segment)
            if kind != "answer":
                raise ProtocolError(
                    "Search actions are unavailable in pretrained-only evaluation"
                )
            answer, status = (
                value,
                "abstained" if value.casefold() == "unknown" else "answered",
            )
        except ProtocolError as exc:
            error = str(exc)  # Logged locally; no repair turn or tool call.
    except ValueError as exc:
        status, error = "context_limit", str(exc)
        turns.append({"turn": 1, "error": error})
    return {
        "question": question,
        "config": asdict(config),
        "status": status,
        "answer": answer,
        "turns": turns,
        "events": events,
        "searches": [],
        "error": error,
        "evaluation_mode": "pretrained_only",
        "execution_limits": {"generations": 1, "search_calls": 0},
        "numeric_validation": "not_performed",
    }
