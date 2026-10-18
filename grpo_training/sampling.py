"""Bounded DAPO question-group oversampling under one fixed rollout policy."""

from grpo_training.rewards import group_outcome


def has_learning_signal(group, labels):
    return group_outcome(group, labels[group.episodes[0].question.id]).learning_signal


def collect_dynamic(
    train,
    cursor,
    config,
    policy,
    backend,
    tokenizer,
    search,
    round_index,
    *,
    collect,
    audit,
    **kwargs,
):
    """Accepted peers remain intact; refill with new questions, never lucky retries.

    Cursor advances for every candidate question, including dropped groups. Budget
    is in questions, bounded by one pass through the selection per logical batch.
    """
    labels = {label.id: label for label in train.labels}
    accepted, consumed = [], 0
    budget = min(
        len(train.questions),
        config.questions_per_update * config.dynamic_sampling_multiplier,
    )
    while len(accepted) < config.questions_per_update and consumed < budget:
        count = min(config.questions_per_update - len(accepted), budget - consumed)
        questions = tuple(
            train.questions[(cursor + consumed + i) % len(train.questions)]
            for i in range(count)
        )
        groups = collect(
            questions,
            config,
            policy,
            backend,
            tokenizer,
            search,
            round_index * (budget + 1) + consumed,
            audit=audit,
            **kwargs,
        )
        consumed += count
        for group in groups:
            outcome = group_outcome(group, labels[group.episodes[0].question.id])
            keep = outcome.learning_signal
            audit(
                {
                    "kind": "dynamic_group",
                    "round": round_index,
                    "question_id": group.episodes[0].question.id,
                    "accepted": keep,
                    "episodes": len(group.episodes),
                    "correct": sum(score.reward == 1 for score in outcome.scores),
                    "reward_type": config.reward,
                    "training_rewards": outcome.rewards,
                    "turns": outcome.turns,
                    "min_successful_turns": outcome.min_successful_turns,
                    "generated_tokens": sum(
                        len(r.completion_ids)
                        for e in group.episodes
                        for r in e.generations
                    ),
                }
            )
            if keep:
                accepted.append(group)
    audit(
        {
            "kind": "dynamic_sampling",
            "round": round_index,
            "candidate_questions": consumed,
            "accepted_groups": len(accepted),
            "budget": budget,
            "budget_exhausted": consumed == budget,
        }
    )
    return (
        tuple(accepted),
        tuple(labels[g.episodes[0].question.id] for g in accepted),
        consumed,
    )
