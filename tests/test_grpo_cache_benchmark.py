from types import SimpleNamespace as NS

import pytest

from experiments.grpo.cache_benchmark import MeasuredBackend, summarize_cache


def test_cache_fraction_is_token_weighted_and_missing_is_unknown():
    waves = [
        {
            "cache_stats": [
                {"prompt_tokens": 100, "cached_tokens": 50},
                {"prompt_tokens": 300, "cached_tokens": 0},
            ]
        }
    ]
    result = summarize_cache(waves)
    assert result["token_hit_fraction"] == 0.125
    assert result["requests_with_cache_hit"] == 1
    waves[0]["cache_stats"][1]["cached_tokens"] = None
    assert summarize_cache(waves)["token_hit_fraction"] is None
    assert summarize_cache([])["cached_tokens"] is None


def test_real_group_measurement_requires_independent_seeds_and_equal_prompts():
    batch = NS(generations=[NS(completion_ids=(1, 2))], seconds=2, cache_stats=())
    measured = MeasuredBackend(NS(generate=lambda requests: batch))
    requests = [
        NS(question_id="q", turn=1, seed=i, prompt_ids=(1, 2, 3)) for i in range(4)
    ]
    assert measured.generate(requests) is batch
    assert measured.waves[0]["first_turn_groups"] == [
        {
            "question_id": "q",
            "trajectories": 4,
            "identical_prompt_tokens": 3,
            "unique_seeds": 4,
        }
    ]
    requests[1].seed = 0
    with pytest.raises(ValueError, match="independent"):
        measured.generate(requests)
    requests[1].seed = 1
    requests[1].prompt_ids = (4,)
    with pytest.raises(ValueError, match="share"):
        measured.generate(requests)
