"""Pure packing and metrics for clipped importance-weighted GRPO updates."""

import math

from tinker_training.metrics import distribution, pair


def pack_rows(prepared, types):
    """Pack replay rows into Tinker Datum objects with next-token-aligned loss inputs.

    Args:
        prepared: PreparedUpdate with exact input/target IDs and per-token credit.
        types: Tinker SDK namespace, injectable to support CPU-only tests.

    Returns:
        List of Datum objects whose model_input is full[:-1] and targets are full[1:].
        Prompt-prediction advantages are zero. Completion advantages are divided by
        one update-wide sampled-token denominator; old log probabilities remain frozen.

    Raises ValueError for an empty update, missing prompt or token alignment errors.
    Masking removes direct prompt loss; the prompt still conditions completion logits.
    """
    rows, data = prepared.rows(), []
    denominator = prepared.token_count
    if denominator < 1:
        raise ValueError("Update has no training tokens")
    for row in rows:
        output = tuple(row["target_ids"])
        full = tuple(row["input_ids"])
        prompt_length = len(full) - len(output)
        if prompt_length < 1 or not output:
            raise ValueError("Need a prompt and sampled completion")
        # Input position j predicts full[j + 1]. There are prompt_length - 1
        # prompt targets, then exactly len(output) sampled-completion targets.
        zeros = [0.0] * (prompt_length - 1)
        credit = list(row.get("token_advantages", [row["advantage"]] * len(output)))
        old = list(row["sampling_logprobs"][-len(output) :])
        if (
            len(credit) != len(output)
            or len(old) != len(output)
            or any(not math.isfinite(v) for v in credit + old)
        ):
            raise ValueError("Invalid token credit/logprob alignment")
        # Zero advantage removes direct prediction loss on prompt/tool targets.
        # Those tokens still condition completion predictions through the model.
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(list(full[:-1])),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=list(full[1:]), dtype="int64", shape=[len(full) - 1]
                    ),
                    "logprobs": types.TensorData(
                        data=zeros + old, dtype="float32", shape=[len(full) - 1]
                    ),
                    "advantages": types.TensorData(
                        data=zeros + [v / denominator for v in credit],
                        dtype="float32",
                        shape=[len(full) - 1],
                    ),
                },
            )
        )
    return data


def loss_metrics(prepared, result):
    """Reconstruct sampled-token clipped-policy loss and importance-ratio diagnostics.

    prepared supplies frozen old log probabilities, credit and clipping bounds; result
    supplies current replay log probabilities returned by the service. Checks exact
    row/target alignment and finite ratios. Returns local metric dictionaries, not a
    differentiable loss or a new gradient computation. Exact entropy is unavailable.
    """
    rows = prepared.rows()
    if len(result.loss_fn_outputs) != len(rows):
        raise ValueError("Training output row count changed")
    ratios, logs, clipped, active, total = [], [], 0, 0, 0
    loss = 0.0
    config = prepared.config
    for row, output in zip(rows, result.loss_fn_outputs, strict=True):
        tensor = output["logprobs"]
        values = tensor.tolist() if hasattr(tensor, "tolist") else tensor.data
        count = len(row["target_ids"])
        if len(values) != len(row["input_ids"]) - 1:
            raise ValueError("Training logprob alignment changed")
        current, old = values[-count:], row["sampling_logprobs"][-count:]
        weights = row.get("token_advantages", [row["advantage"]] * count)
        for new, previous, weight in zip(current, old, weights, strict=True):
            # exp(log pi_current - log pi_sampled) is the importance ratio.
            # "previous" belongs to the rollout policy, not the last replay update.
            delta = new - previous
            if not math.isfinite(delta) or abs(delta) > 80:
                raise ValueError("Nonfinite or overflowing importance ratio")
            ratio = math.exp(delta)
            bound = min(max(ratio, 1 - config.epsilon_low), 1 + config.epsilon_high)
            # The sign of weight determines which clipped branch limits the
            # objective. Keep the same min operation for positive/negative credit.
            loss -= min(ratio * weight, bound * weight) / prepared.token_count
            clipped += bool(
                (weight > 0 and ratio > 1 + config.epsilon_high)
                or (weight < 0 and ratio < 1 - config.epsilon_low)
            )
            active += weight != 0
            ratios.append(ratio)
            logs.append(delta)
            total += 1
    outside = sum(
        r < 1 - config.epsilon_low or r > 1 + config.epsilon_high for r in ratios
    )
    return {
        "loss": loss,
        "tokens": total,
        "current_over_old_ratio": distribution(ratios),
        "current_over_old_log_ratio": distribution(logs),
        "ratio_distribution_available": True,
        "ratio_outside_bounds": pair(outside, total),
        "objective_clipped_learning_tokens": pair(clipped, active),
        "objective_clipped_all_tokens": pair(clipped, total),
        "old_to_current_kl_sample_estimate": pair(-sum(logs), total),
        "entropy_nats": distribution([]),
        "entropy_definition": "unavailable; sampled surprisal is not full-vocabulary entropy",
    }
