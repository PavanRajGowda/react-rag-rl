"""CPU-only rollout/replay contracts with a toy tokenizer and probability model."""

import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from grpo_training.config import TrainingConfig
from grpo_training.data import Question
from grpo_training.environment import SearchEpisode
from grpo_training.records import (
    EpisodeRecord,
    GenerationRecord,
    PolicyVersion,
    SamplingParameters,
)
from grpo_training.replay import (
    GroupRequiresRetry,
    QuestionGroup,
    collate_replay,
    replay_rows,
)
from grpo_training.rollouts import EpisodeRecorder, template_digest


class ToyTokenizer:
    chat_template = "json-conversation-v1"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return json.dumps(messages) + "\nASSISTANT:"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) + 1 for char in text]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(value - 1) for value in ids if value)


TOKENIZER = ToyTokenizer()
CONFIG = TrainingConfig(num_generations=2)
QUESTION = Question("finqa_train_0", "Revenue?", "train", 0, CONFIG.dataset_revision)
POLICY = PolicyVersion(
    CONFIG.model_revision, CONFIG.model_revision, "base", 0, template_digest(TOKENIZER)
)
SAMPLING = SamplingParameters()
SEARCH = "Find values.</reasoning><search><company>Acme</company><year>2015</year><query>revenue</query></search>"
ANSWER = "Calculate.</reasoning><answer>42</answer>"


def search(request):
    context = "OLD_EVIDENCE" if request["query"] == "revenue" else "NEW_EVIDENCE"
    return {
        "status": "ok",
        "corpus_version": CONFIG.corpus_version,
        "search_schema_version": 4,
        "results": [
            {
                "context_id": context,
                "company_name": "Acme",
                "report_year": 2015,
                "page_number": 1,
                "text": context,
            }
        ],
    }


def recorder(index=0, *, config=CONFIG, policy=POLICY, backend=search):
    episode = SearchEpisode(QUESTION, config, backend)
    return EpisodeRecorder(
        episode,
        episode_id=f"episode-{index}",
        group_id="batch-0-question-0",
        sample_index=index,
        seed=42 + index,
        policy=policy,
        sampling=SAMPLING,
    )


def toy_logprobs(prefix, completion):
    """A normalized toy next-token distribution that depends on the full prefix."""
    context = sum(prefix)
    values = []
    for token in completion:
        center = context % 128
        normalizer = sum(math.exp(-abs(v - center) / 10) for v in range(128))
        values.append(-abs(token - center) / 10 - math.log(normalizer))
        context += token
    return tuple(values)


def submit(rec, text, *, finish_reason="action"):
    request = rec.prepare(TOKENIZER)
    ids = (
        (0,)
        if text == "EOS"
        else tuple(TOKENIZER.encode(text, add_special_tokens=False))
    )
    return rec.record(
        request,
        prompt_ids=request.prompt_ids,
        policy_id=request.policy.id,
        sampling=request.sampling,
        completion_ids=ids,
        sampling_logprobs=toy_logprobs(request.prompt_ids, ids),
        tokenizer=TOKENIZER,
        finish_reason=finish_reason,
        generation_seconds=0.1,
    )


def completed(index=0, *, extra_search=False):
    rec = recorder(index)
    submit(rec, SEARCH)
    if extra_search:
        submit(rec, SEARCH.replace("revenue", "cost"))
    submit(rec, ANSWER)
    return rec.finish()


def test_exact_replacement_prompts_and_one_loss_target_per_sampled_token():
    first, second = completed(0, extra_search=True), completed(1)
    assert "OLD_EVIDENCE" in first.generations[1].request.prompt_text
    assert "OLD_EVIDENCE" not in first.generations[2].request.prompt_text
    assert "NEW_EVIDENCE" in first.generations[2].request.prompt_text
    audit = json.loads(first.controller_audit_json)
    assert len(audit["searches"]) == 2 and "OLD_EVIDENCE" in str(audit)
    assert (
        "Find values." in first.generations[2].request.prompt_text
    )  # Earlier model reasoning stays.
    group = QuestionGroup((second, first))  # Sorting preserves G=2, not five turns.
    batch = collate_replay(group, pad_token_id=0)
    assert len(group.episodes) == 2 and len(batch["rows"]) == 5
    assert sum(sum(mask) for mask in batch["loss_mask"]) == sum(
        len(turn.completion_ids) for ep in group.episodes for turn in ep.generations
    )
    for index, row in enumerate(batch["rows"]):
        length = len(row["input_ids"]) - len(row["target_ids"])
        assert row["labels"][:length] == (-100,) * length
        assert row["input_ids"][-len(row["target_ids"]) :] == row["target_ids"]
        assert row["logit_positions"][0] == length - 1
        assert row["logit_positions"][-1] == len(row["input_ids"]) - 2
        # Teacher forcing must reproduce probabilities for the actual sampled
        # prompt, including replacement. We do not score a stitched transcript.
        expected = toy_logprobs(row["input_ids"][:length], row["target_ids"])
        assert row["sampling_logprobs"][length:] == expected
        assert sum(batch["attention_mask"][index]) == len(row["input_ids"])
        assert all(
            value == -100
            for value, active in zip(
                batch["labels"][index], batch["loss_mask"][index], strict=True
            )
            if not active
        )
    actual = first.generations[2]
    wrong_prefix = tuple(
        TOKENIZER.encode(
            actual.request.prompt_text + "OLD_EVIDENCE", add_special_tokens=False
        )
    )
    assert toy_logprobs(wrong_prefix, actual.completion_ids) != actual.sampling_logprobs


def test_record_roundtrip_freezes_arrays_and_preserves_logprobs_policy_and_seed():
    original = completed()
    serialized = json.loads(json.dumps(original.to_dict()))
    restored = EpisodeRecord.from_dict(serialized)
    assert restored == original
    serialized["generations"][0]["request"]["prompt_ids"][0] = 999
    assert restored == original
    assert original.generations[0].request.seed != original.generations[1].request.seed
    assert len(original.policy.id) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"sampling_logprobs": ()},
        {"sampling_logprobs": (float("nan"),)},
        {"sampling_logprobs": (float("-inf"),)},
        {"sampling_logprobs": (0.1,)},
        {"completion_ids": (-1,)},
        {"completion_ids": (True,)},
        {"completion_ids": ()},
        {"finish_reason": "unknown"},
        {"generation_seconds": -1},
    ],
)
def test_bad_sample_vectors_or_metadata_fail_before_controller_advance(change):
    rec = recorder()
    request = rec.prepare(TOKENIZER)
    arguments = {
        "completion_ids": (0,),
        "sampling_logprobs": (-0.5,),
        "completion_text": "",
        "finish_reason": "eos",
    }
    with pytest.raises(ValueError):
        GenerationRecord(request, **(arguments | change))
    assert rec.episode.snapshot()["turns"] == []


def test_backend_must_confirm_exact_prompt_policy_and_sampling():
    rec = recorder()
    request = rec.prepare(TOKENIZER)
    kwargs = {
        "prompt_ids": request.prompt_ids,
        "policy_id": request.policy.id,
        "sampling": request.sampling,
        "completion_ids": (0,),
        "sampling_logprobs": (-0.5,),
        "tokenizer": TOKENIZER,
        "finish_reason": "eos",
    }
    for changed in (
        {"policy_id": "stale"},
        {"prompt_ids": (1,)},
        {"sampling": replace(SAMPLING, temperature=1.0)},
    ):
        with pytest.raises(ValueError):
            rec.record(request, **(kwargs | changed))
    assert rec.episode.snapshot()["turns"] == []
    with pytest.raises(RuntimeError):
        rec.prepare(TOKENIZER)
    with pytest.raises(RuntimeError):
        rec.finish()
    rec.record(
        request, **kwargs
    )  # EOS is a sampled token even if decoded text is empty.
    assert len(rec.episode.snapshot()["turns"]) == 1


def test_context_limit_and_generation_failure_do_not_fabricate_tokens():
    config = replace(CONFIG, max_new_tokens=10, max_context_tokens=11)
    rec = recorder(config=config)
    assert rec.prepare(TOKENIZER) is None
    assert rec.finish().status == "context_limit" and not rec.finish().generations
    rec = recorder()
    request = rec.prepare(TOKENIZER)
    rec.fail_generation(request, TimeoutError("backend failed"))
    assert (
        rec.finish().status == "infrastructure_error" and not rec.finish().generations
    )


def test_token_limits_and_template_changes_fail_closed():
    rec = recorder()
    request = rec.prepare(TOKENIZER)
    with pytest.raises(ValueError):
        rec.record(
            request,
            prompt_ids=request.prompt_ids,
            policy_id=request.policy.id,
            sampling=SAMPLING,
            completion_ids=(1,) * (CONFIG.max_new_tokens + 1),
            sampling_logprobs=(-0.5,) * (CONFIG.max_new_tokens + 1),
            tokenizer=TOKENIZER,
            finish_reason="length",
        )
    changed = ToyTokenizer()
    changed.chat_template = "changed"
    with pytest.raises(ValueError, match="template"):
        recorder().prepare(changed)


def test_groups_reject_missing_duplicate_mixed_question_or_stale_policy_samples():
    first, second = completed(0), completed(1)
    with pytest.raises(ValueError):
        QuestionGroup((first,))
    with pytest.raises(ValueError):
        QuestionGroup((first, first))
    for changed in (
        replace(second, question=replace(QUESTION, text="Different?")),
        replace(
            second,
            group_id="other",
            generations=(),
            status="context_limit",
            answer=None,
        ),
        replace(
            second,
            policy=replace(POLICY, optimizer_step=1),
            generations=(),
            status="context_limit",
            answer=None,
        ),
        replace(
            second, seed=first.seed, generations=(), status="context_limit", answer=None
        ),
    ):
        with pytest.raises(ValueError):
            QuestionGroup((first, changed))
    failed = replace(second, status="infrastructure_error", answer=None)
    group = QuestionGroup((first, failed))
    assert not group.scorable
    with pytest.raises(GroupRequiresRetry):
        replay_rows(group)


def test_malformed_output_feedback_is_prompt_only_and_length_reason_is_preserved():
    rec = recorder(config=replace(CONFIG, max_turns=2))
    submit(rec, "Incomplete", finish_reason="length")
    submit(rec, "Again incomplete", finish_reason="length")
    episode = rec.finish()
    assert episode.status == "turn_limit"
    assert (
        "Controller formatting feedback" in episode.generations[1].request.prompt_text
    )
    assert episode.generations[0].finish_reason == "length"
    other = replace(
        episode,
        episode_id="episode-1",
        sample_index=1,
        seed=43,
        generations=(),
        status="context_limit",
    )
    rows = replay_rows(QuestionGroup((episode, other)))
    assert len(rows) == 2  # Context-only/empty peers remain group members, not rows.
    assert sum(sum(row["loss_mask"]) for row in rows) == sum(
        len(t.completion_ids) for t in episode.generations
    )


def test_sampled_eos_is_not_masked_even_when_pad_uses_the_same_id():
    config = replace(CONFIG, max_turns=2)
    first, second = recorder(0, config=config), recorder(1, config=config)
    submit(first, SEARCH)
    submit(first, "EOS", finish_reason="eos")
    submit(second, SEARCH)
    submit(second, ANSWER)
    group = QuestionGroup((first.finish(), second.finish()))
    batch = collate_replay(group, pad_token_id=0)
    eos_row = 1
    eos_position = len(group.episodes[0].generations[1].request.prompt_ids)
    assert batch["input_ids"][eos_row][eos_position] == 0
    assert batch["labels"][eos_row][eos_position] == 0
    assert (
        batch["loss_mask"][eos_row][eos_position]
        == batch["attention_mask"][eos_row][eos_position]
        == 1
    )
    assert batch["sampling_logprobs"][eos_row][eos_position] < 0
    assert all(value == 0 for value in batch["loss_mask"][eos_row][eos_position + 1 :])


def test_deserialization_rejects_bad_generation_order_and_false_answer_record():
    data = json.loads(json.dumps(completed().to_dict()))
    data["generations"][1]["request"]["turn"] = 1
    with pytest.raises(ValueError):
        EpisodeRecord.from_dict(data)
    data = json.loads(json.dumps(completed().to_dict()))
    data["generations"] = []
    with pytest.raises(ValueError):
        EpisodeRecord.from_dict(data)


def test_recording_imports_no_training_model_or_cloud_libraries():
    code = "import sys; import grpo_training.rollouts, grpo_training.replay; assert not {'torch','trl','vllm','transformers','modal','datasets'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_controller_cannot_be_advanced_behind_a_pending_recording_request():
    rec = recorder()
    request = rec.prepare(TOKENIZER)
    rec.episode.accept_output("Unrecorded external output")
    with pytest.raises(ValueError, match="outside its recorder"):
        rec.record(
            request,
            prompt_ids=request.prompt_ids,
            policy_id=request.policy.id,
            sampling=SAMPLING,
            completion_ids=(0,),
            sampling_logprobs=(-0.5,),
            tokenizer=TOKENIZER,
            finish_reason="eos",
        )
