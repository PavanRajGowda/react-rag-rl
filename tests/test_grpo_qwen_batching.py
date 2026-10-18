"""Actual hybrid Qwen model in tiny FP32 CPU form, not a probability stub."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from grpo_training.trainer import collate_segments


@pytest.fixture
def hybrid_model():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
    )
    return Qwen3_5ForCausalLM(config).eval()


ROWS = (
    {"input_ids": (1, 2, 3, 4, 5), "target_ids": (4, 5)},
    {"input_ids": (2, 3, 4, 5, 6, 7, 8, 9), "target_ids": (7, 8, 9)},
)


def scored(model, rows, pad_id=0):
    batch = collate_segments(rows, pad_id, "cpu")
    ids = torch.cat((batch["prompt_ids"], batch["completion_ids"]), 1)
    mask = torch.cat((batch["prompt_mask"], batch["completion_mask"]), 1)
    n = batch["completion_ids"].shape[1]
    logits = model(ids, attention_mask=mask).logits[:, -n - 1 : -1]
    values = (
        logits.log_softmax(-1)
        .gather(-1, batch["completion_ids"].unsqueeze(-1))
        .squeeze(-1)
    )
    return values, batch["completion_mask"]


def test_actual_hybrid_fp32_unequal_padding_and_pad_values(hybrid_model):
    with torch.no_grad():
        separate = [scored(hybrid_model, [row])[0][0] for row in ROWS]
        batched, _ = scored(hybrid_model, ROWS)
        changed_pad, _ = scored(hybrid_model, ROWS, pad_id=63)
        for i, row in enumerate(ROWS):
            n = len(row["target_ids"])
            torch.testing.assert_close(
                batched[i, :n], separate[i], atol=1e-6, rtol=1e-5
            )
            torch.testing.assert_close(
                batched[i, :n], changed_pad[i, :n], atol=1e-6, rtol=1e-5
            )


def test_actual_hybrid_fp32_accumulated_gradients(hybrid_model):
    denominator = sum(len(row["target_ids"]) for row in ROWS)
    for i, row in enumerate(ROWS):
        values, mask = scored(hybrid_model, [row])
        (-values * mask * (1.0 if i == 0 else -0.5)).sum().div(denominator).backward()
    reference = {
        n: p.grad.clone()
        for n, p in hybrid_model.named_parameters()
        if p.grad is not None
    }
    hybrid_model.zero_grad(set_to_none=True)
    values, mask = scored(hybrid_model, ROWS)
    advantages = torch.tensor([[1.0], [-0.5]])
    (-values * mask * advantages).sum().div(denominator).backward()
    for name, parameter in hybrid_model.named_parameters():
        if name in reference:
            torch.testing.assert_close(
                parameter.grad, reference[name], atol=2e-6, rtol=2e-4
            )
