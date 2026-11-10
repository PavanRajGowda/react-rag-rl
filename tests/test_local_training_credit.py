"""End-to-end local credit preparation, filtering, KL addition and replay packing."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_grpo_advantages import make_group
from tests.test_tinker_migration import Future, TYPES
from tinker_training.config import TrainingConfig
from tinker_training.data import RewardLabel
from tinker_training.replay import QuestionGroup
from tinker_training.advantages import prepare_update, filter_constant_groups, minibatch_update
from tinker_training.kl import apply_reference_kl
from tinker_training.loss import pack_rows


def fixture(*, answers=("42","9"), gold=True, invalid=False, number=0, **kwargs):
    """Controlled audit matches the same shared event parser used in production."""
    cfg=TrainingConfig(reward="local_process_advantage_v1", num_generations=2,
                       questions_per_update=2,minibatch_questions=1,**kwargs)
    group=make_group(cfg,number=number,answers=answers,lengths=((2,3),(2,3)))
    episodes=[]
    for e in group.episodes:
        audit={"question":{"id":e.question.id},"turns":[
            {"turn":1,"model_output":"broken" if invalid else "<reasoning>Find</reasoning><search>query</search>"},
            {"turn":2,"model_output":f"<reasoning>Answer</reasoning><answer>{e.answer}</answer>"}],
            "searches":[] if invalid else [{"turn":1,"response":{"status":"ok","results":[
                {"document_id":"d" if gold else "other","text":"fact"}]}}]}
        episodes.append(replace(e,controller_audit_json=json.dumps(audit)))
    return QuestionGroup(tuple(episodes)),RewardLabel(group.episodes[0].question.id,"42",("d",))


def test_local_turn_weights_reach_packed_tokens():
    g,label=fixture()
    p=prepare_update((g,),(label,))
    rows=p.rows()
    for r in rows:
        expected=r["advantage"]+(.33 if r["turn"]==1 else 0)
        assert r["token_advantages"]==pytest.approx([expected]*len(r["target_ids"]))
    kept,_=filter_constant_groups(p)
    mini,_=minibatch_update(kept,0)
    for r,datum in zip(mini.rows(),pack_rows(mini,TYPES),strict=True):
        n=len(r["target_ids"])
        a=datum.loss_fn_inputs["advantages"].data
        assert a[:-n]==[0]*(len(a)-n)
        assert a[-n:]==pytest.approx([v/mini.token_count for v in r["token_advantages"]])


def test_tied_outcomes_keep_process_signal_and_drop_only_task_zero():
    good,label=fixture(answers=("9","9"))
    zero,label2=fixture(answers=("9","9"),gold=False,number=1)
    p=prepare_update((good,zero),(label,label2))
    assert all(s.advantage==0 for s in p.signals)
    kept,metrics=filter_constant_groups(p)
    assert kept.groups==(good,)
    assert kept.token_count==p.token_count
    assert metrics["surviving_groups"]==1
    mini,_=minibatch_update(p,1)
    assert filter_constant_groups(mini)[0] is None


def test_protocol_whole_turn_fallback_is_local_and_requires_tokenizer():
    g,label=fixture(answers=("9","9"),invalid=True)
    with pytest.raises(ValueError,match="tokenizer"):
        prepare_update((g,),(label,))
    p=prepare_update((g,),(label,),tokenizer=SimpleNamespace(decode=lambda ids,**kw:"x"*len(ids)))
    for row in p.rows():
        expected=-.33 if row["turn"]==1 else 0
        assert row["token_advantages"]==pytest.approx([expected]*len(row["target_ids"]))


def test_kl_adds_to_local_weights_and_preserves_task_eligibility():
    g,label=fixture(answers=("9","9"),beta=.01)
    p=prepare_update((g,),(label,))
    def score(prompt,prompt_logprobs_last_n):
        n=prompt_logprobs_last_n
        return Future([None]*(len(prompt)-n)+[-2-i for i in range(n)])
    adjusted,_=apply_reference_kl(p,SimpleNamespace(compute_logprobs=score),TYPES)
    delta=[b-a for r,s in zip(p.rows(),adjusted.rows(),strict=True)
           for a,b in zip(r["token_advantages"],s["token_advantages"],strict=True)]
    assert sum(delta)==pytest.approx(0,abs=1e-12)
    assert max(abs(d) for d in delta)<.1
    assert adjusted.task_active_group_ids==p.task_active_group_ids
    assert filter_constant_groups(adjusted)[0] is not None


def test_rollout_metrics_count_tied_groups_with_local_signal():
    from tinker_training.metrics import rollout_metrics
    g,label=fixture(answers=("9","9"))
    p=prepare_update((g,),(label,))
    metrics,_=rollout_metrics(p,(label,),include_traces=False)
    assert metrics["rates"]["groups_with_signal"]["value"]==1
    assert metrics["credit"]["task_token_advantage"]["n"]==p.token_count


@pytest.mark.parametrize("value",[-1,float("nan"),True])
def test_invalid_local_weights_rejected(value):
    with pytest.raises(ValueError):
        TrainingConfig(local_gold_weight=value)
