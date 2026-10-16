from types import SimpleNamespace as NS

from baseline_agent.vllm_fp8_benchmark import batched_agents
from grpo_training.config import TrainingConfig
from grpo_training.records import PolicyVersion


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def apply_chat_template(self, conversation, **kwargs):
        return str(conversation)


def test_agents_batch_independent_histories_and_count_wave_time_once():
    config = TrainingConfig(temperature=0.2, top_p=0.95)
    policy = PolicyVersion(
        config.model_revision, config.model_revision, "base", 0, "a" * 64
    )
    calls = []

    class Backend:
        def generate(self, requests):
            calls.append(requests)
            items = []
            for request in requests:
                output = (
                    "</reasoning><search><company>Acme</company><year>2010</year><query>revenue</query></search>"
                    if request.turn == 1
                    else "</reasoning><answer>42</answer>"
                )
                ids = tuple(map(ord, output))
                items.append(
                    NS(
                        request=request,
                        completion_ids=ids,
                        sampling_logprobs=(0.0,) * len(ids),
                    )
                )
            return NS(generations=tuple(items), seconds=1.0)

    questions = [
        {
            "id": f"finqa_test_{i}",
            "example_index": i,
            "question": f"Revenue for example {i}?",
        }
        for i in range(3)
    ]

    def search(payload):
        return {
            "status": "ok",
            "results": [
                {
                    "context_id": "ctx",
                    "text": "Revenue was 42",
                    "company_name": "Acme",
                    "report_year": 2010,
                    "page_number": 1,
                }
            ],
        }

    results = list(
        batched_agents(questions, Backend(), Tokenizer(), policy, config, search, 2)
    )
    assert len(results) == 3
    assert all(result["answer"] == "42" for result, _ in results)
    assert [len(call) for call in calls] == [2, 2, 1, 1]
    assert len({r.seed for r in calls[0]}) == 2
    assert sum(w["seconds"] for w in results[-1][1]) == 4
    assert all(r.sampling.top_k == 20 for call in calls for r in call)
