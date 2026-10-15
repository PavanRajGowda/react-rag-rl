"""Fixed training-only initial-generation prefixes; exact vocabulary KL, no loss."""

import hashlib
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

from grpo_training.checkpoints import write_json
from grpo_training.data import policy_messages
from grpo_training.rollouts import ASSISTANT_PREFILL, template_digest


class ReferenceProbe:
    def __init__(self, trainer, bundle, questions, root):
        import torch
        from safetensors.torch import load_file, save_file

        self.model = trainer.accelerator.unwrap_model(trainer.model)
        self.device = next(self.model.parameters()).device
        self.temperature = bundle.config.temperature
        selected = tuple(questions[: bundle.config.reference_probe_questions])
        self.ids = tuple(
            tuple(
                bundle.tokenizer.encode(
                    bundle.tokenizer.apply_chat_template(
                        policy_messages(q),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    + ASSISTANT_PREFILL,
                    add_special_tokens=False,
                )
            )
            for q in selected
        )
        if not self.ids or any(
            not ids
            or len(ids) + bundle.config.max_new_tokens
            > bundle.config.max_context_tokens
            for ids in self.ids
        ):
            raise ValueError(
                "Reference prefixes must fit the same generation context budget"
            )
        metadata = {
            "schema_version": 1,
            "model_revision": bundle.config.model_revision,
            "template_sha256": template_digest(bundle.tokenizer),
            "temperature": self.temperature,
            "question_ids": [q.id for q in selected],
            "prefix_ids": [list(ids) for ids in self.ids],
            "definition": "KL(current || fixed pretrained reference), full vocabulary at each fixed initial-generation prefix; nats; temperature-scaled, no filtering",
        }
        directory = Path(root) / "reference_probe"
        if directory.exists():
            actual = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            expected_hash = actual.pop("distribution_sha256")
            if (
                actual != metadata
                or hashlib.sha256(
                    (directory / "reference.safetensors").read_bytes()
                ).hexdigest()
                != expected_hash
            ):
                raise ValueError("Reference probe identity/hash mismatch")
            self.reference = load_file(str(directory / "reference.safetensors"))[
                "logprobs"
            ]
        else:
            with self.inference(), self.model.disable_adapter():
                self.reference = torch.stack(
                    [self.logprobs(ids).cpu() for ids in self.ids]
                )
            staging = Path(root) / f".reference-{uuid.uuid4().hex}"
            staging.mkdir()
            save_file(
                {"logprobs": self.reference.contiguous()},
                str(staging / "reference.safetensors"),
            )
            metadata["distribution_sha256"] = hashlib.sha256(
                (staging / "reference.safetensors").read_bytes()
            ).hexdigest()
            write_json(staging / "metadata.json", metadata)
            staging.rename(directory)
        if (
            self.reference.ndim != 2
            or len(self.reference) != len(self.ids)
            or not torch.isfinite(self.reference).all()
        ):
            raise ValueError("Invalid fixed reference distribution")

    @contextmanager
    def inference(self):
        import torch

        was_training = self.model.training
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                yield
        finally:
            self.model.train(was_training)

    def logprobs(self, ids):
        import torch

        tokens = torch.tensor([ids], device=self.device)
        output = self.model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            logits_to_keep=1,
            use_cache=False,
            return_dict=True,
        )
        return (output.logits[0, -1].float() / self.temperature).log_softmax(-1)

    def measure(self):
        import torch

        values = []
        with self.inference():
            for ids, reference in zip(self.ids, self.reference, strict=True):
                current = self.logprobs(ids)
                if (
                    current.shape != reference.shape
                    or not torch.isfinite(current).all()
                ):
                    raise ValueError(
                        "Policy/reference vocabulary or probabilities disagree"
                    )
                value = (
                    (current.exp() * (current - reference.to(current.device)))
                    .sum()
                    .item()
                )
                if not (-1e-5 <= value < float("inf")):
                    raise ValueError("Invalid reference KL")
                values.append(
                    max(0.0, value)
                )  # Only remove tiny floating-point cancellation.
        return {
            "sum_nats": sum(values),
            "positions": len(values),
            "mean_nats": sum(values) / len(values),
            "max_nats": max(values),
            "temperature": self.temperature,
            "definition": "full-vocabulary KL(current || fixed pretrained reference) on fixed initial-generation prefixes",
        }
