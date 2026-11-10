"""CPU-only dataset and scripted generation fixtures, independent of any trainer."""

from tests.agent_fixtures import ANSWER, SEARCH
from tinker_training.backend import Batch as GenerationBatch
from tinker_training.backend import Generation as BackendGeneration
from tinker_training.data import PreparedSplit, Question, RewardLabel


def inputs(config):
    """Create aligned toy configuration, question, tokenizer and policy records for CPU
    rollout tests.
    """
    train = PreparedSplit(
        "train",
        tuple(
            Question(
                f"multihop_train_{i}", "Revenue?", "train", i, config.dataset_revision
            )
            for i in range(3)
        ),
        tuple(
            RewardLabel(f"multihop_train_{i}", "42", "diagnostic-only")
            for i in range(3)
        ),
        (),
        3,
    )
    return train


class ScriptedBackend:
    def __init__(self, tokenizer):
        """Store the tokenizer used by this deterministic, network-free sampling fake."""
        self.tokenizer, self.batch_sizes = tokenizer, []

    def generate(self, requests):
        """Return scripted search/answer generations for the supplied requests, retaining
        request order and exact fixture token IDs.
        """
        self.batch_sizes.append(len(requests))
        items = []
        for request in requests:
            text = (
                SEARCH.replace("revenue", request.episode_id)
                if request.turn == 1
                else ANSWER
            )
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            items.append(
                BackendGeneration(request, ids, (-2.0,) * len(ids), "action", 1)
            )
        return GenerationBatch(tuple(items), 0.01)
