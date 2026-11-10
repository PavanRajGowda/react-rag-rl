"""CPU-only shared-agent fixtures; no retired recorder implementation."""

import json
from dataclasses import replace

from tinker_training.config import TrainingConfig
from tinker_training.data import Question
from tinker_training.policy import template_digest
from tinker_training.records import PolicyVersion


class ToyTokenizer:
    chat_template = "json-conversation-v1"

    def apply_chat_template(self, messages, **kwargs):
        """Render a deterministic toy chat prefix for exact-token recorder tests."""
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return json.dumps(messages) + "\nASSISTANT:"

    def encode(self, text, *, add_special_tokens):
        """Map fixture text to deterministic integer IDs without a model tokenizer."""
        assert add_special_tokens is False
        return [ord(char) + 1 for char in text]

    def decode(self, ids, *, skip_special_tokens):
        """Convert toy IDs back to text for controller parsing in CPU tests."""
        return "".join(chr(value - 1) for value in ids if value)


TOKENIZER = ToyTokenizer()
CONFIG = TrainingConfig(num_generations=2, context_mode="replace")
QUESTION = Question("multihop_train_0", "Revenue?", "train", 0, CONFIG.dataset_revision)
POLICY = PolicyVersion(
    CONFIG.model_revision, CONFIG.model_revision, "base", 0, template_digest(TOKENIZER)
)
SEARCH = "Find values.</reasoning><search>revenue</search>"
ANSWER = "Calculate.</reasoning><answer>42</answer>"


def search(request):
    """Return deterministic fixture evidence for recorded search turns."""
    context = "OLD_EVIDENCE" if request["query"] == "revenue" else "NEW_EVIDENCE"
    return {
        "status": "ok",
        "corpus_version": CONFIG.corpus_version,
        "search_schema_version": 6,
        "results": [
            {
                "context_id": context,
                "document_id": context,
                "title": "Acme news",
                "source": "Example",
                "published_at": "2023-10-01T00:00:00+00:00",
                "text": context,
            }
        ],
    }


class ScriptedTrajectory:
    """Accumulate scripted outputs, then execute the real shared-agent path."""

    def __init__(self, index, config, backend):
        self.index, self.config, self.backend = index, config, backend
        self.outputs = []

    def finish(self):
        """Return a typed episode via generate_episodes, without a legacy controller."""
        from types import SimpleNamespace

        from tinker_training.backend import Batch, Generation
        from tinker_training.trajectory import generate_episodes

        def generate(requests):
            items = []
            for request in requests:
                ids = tuple(
                    TOKENIZER.encode(
                        self.outputs[request.turn - 1], add_special_tokens=False
                    )
                )
                items.append(Generation(request, ids, (-2.0,) * len(ids), "action"))
            return Batch(tuple(items), 0.01)

        spec = {
            "question": QUESTION,
            "episode_id": f"episode-{self.index}",
            "group_id": "batch-0-question-0",
            "sample_index": self.index,
            "seed": 42 + self.index,
        }
        episode = next(
            generate_episodes(
                [spec],
                self.config,
                POLICY,
                SimpleNamespace(generate=generate),
                TOKENIZER,
                self.backend,
            )
        )
        return replace(
            episode,
            generations=tuple(
                replace(g, generation_seconds=0.1) for g in episode.generations
            ),
        )


def recorder(index=0, *, config=CONFIG, backend=search):
    """Return a scripted input builder, not an independent production recorder."""
    return ScriptedTrajectory(index, config, backend)


def submit(rec, text):
    """Append one scripted model continuation for later shared-agent execution."""
    rec.outputs.append(text)


def response(context="first", text="Revenue 42"):
    """Return a valid search result for hosted validation tests."""
    return {
        "status": "ok",
        "corpus_version": CONFIG.corpus_version,
        "search_schema_version": CONFIG.search_schema_version,
        "results": [{"document_id": context, "title": "Acme news", "text": text}],
    }
