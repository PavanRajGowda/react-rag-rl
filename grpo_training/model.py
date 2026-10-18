"""Lazy model/LoRA loading and immutable adapter transfers. No execution on import."""

import hashlib
import json
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from grpo_training.config import TrainingConfig
from grpo_training.records import PolicyVersion
from grpo_training.rollouts import template_digest

PINS = {
    "trl": "1.15.0",
    "vllm": "0.22.0",
    "torch": "2.11.0",
    "transformers": "5.5.4",
    "peft": "0.18.1",
    "accelerate": "1.12.0",
    "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2",
    "causal-conv1d": "1.7.0",
}
TARGET = re.compile(
    r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def text_lora_targets(module_names):
    targets = tuple(sorted(name for name in module_names if TARGET.fullmatch(name)))
    if (
        not targets
        or not any(".self_attn." in name for name in targets)
        or not any(".mlp." in name for name in targets)
    ):
        raise ValueError(
            "Expected both full-attention and MLP text projections in Qwen3.5"
        )
    return targets


def trainable_names(named_parameters):
    names = tuple(
        sorted(name for name, parameter in named_parameters if parameter.requires_grad)
    )
    if not names:
        raise ValueError("No trainable adapter parameters")
    for name in names:
        match = re.fullmatch(
            r"base_model\.model\.(.+)\.lora_[AB]\.default\.weight", name
        )
        if match is None or TARGET.fullmatch(match[1]) is None:
            raise ValueError(f"Unexpected trainable parameter: {name}")
    return names


def preserve_explicit_targets(model, targets):
    """Undo PEFT's large-list suffix compression after checking injected layers."""
    names = trainable_names(model.named_parameters())
    actual = {
        re.fullmatch(r"base_model\.model\.(.+)\.lora_[AB]\.default\.weight", name)[1]
        for name in names
    }
    if actual != set(targets) or text_lora_targets(targets) != targets:
        raise ValueError("Injected LoRA layers disagree with reviewed text targets")
    # PEFT may replace hundreds of exact names with short suffixes during
    # injection. Keep the saved transfer config explicit; do not broaden checks.
    model.peft_config["default"].target_modules = set(targets)


def check_runtime():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Training runtime requires Python 3.11")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Training model/vLLM runtime requires Linux x86_64")
    actual = {name: version(name).split("+", 1)[0] for name in PINS}
    if actual != PINS:
        raise RuntimeError(f"Runtime differs from reviewed dependency lock: {actual}")
    return actual


@dataclass
class TrainingModel:
    model: object
    tokenizer: object
    snapshot: Path
    config: TrainingConfig
    targets: tuple[str, ...]


def check_optimized_kernels(module=None):
    if module is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    if not module.is_fast_path_available:
        raise RuntimeError(
            "Pinned optimized linear-attention/convolution kernels are unavailable"
        )


def load_training_model(config, cache_dir=None):
    """Explicit GPU operation; call only in a separately approved smoke/job."""
    check_runtime()
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    check_optimized_kernels()
    snapshot = Path(
        snapshot_download(
            config.model_id,
            revision=config.model_revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.safetensors",
                "tokenizer*",
                "vocab*",
                "merges*",
            ],
        )
    )
    if snapshot.name != config.model_revision or snapshot.parent.name != "snapshots":
        raise RuntimeError("Hub did not resolve the pinned model snapshot")
    metadata = json.loads((snapshot / "config.json").read_text())
    if metadata.get("quantization_config") or metadata.get("model_type") != "qwen3_5":
        raise RuntimeError("Expected the unquantized dense Qwen3.5 checkpoint")
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    if type(tokenizer.eos_token_id) is not int:
        raise RuntimeError("Tokenizer has no EOS ID")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    targets = text_lora_targets(name for name, _ in model.named_modules())
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=list(targets),
        ),
    )
    preserve_explicit_targets(model, targets)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    trainable_names(model.named_parameters())
    model.train()
    return TrainingModel(model, tokenizer, snapshot, config, targets)


def adapter_digest(path):
    hashes = []
    for name in ADAPTER_FILES:
        file = Path(path) / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"Missing or symlinked adapter file: {name}")
        hashes.append((name, hashlib.sha256(file.read_bytes()).hexdigest()))
    digest = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, tuple(hashes)


@dataclass(frozen=True)
class AdapterArtifact:
    path: Path
    policy: PolicyVersion
    targets: tuple[str, ...]

    def verify(self):
        actual, hashes = adapter_digest(self.path)
        manifest = json.loads((self.path / "transfer.json").read_text(encoding="utf-8"))
        adapter = json.loads(
            (self.path / "adapter_config.json").read_text(encoding="utf-8")
        )
        if actual != self.policy.adapter_sha256 or manifest.get("policy") != asdict(
            self.policy
        ):
            raise ValueError("Adapter bytes or policy changed after export")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("files") != dict(hashes)
            or manifest.get("targets") != list(self.targets)
        ):
            raise ValueError("Invalid adapter transfer manifest")
        config = TrainingConfig(**manifest["config"])
        if (
            config.model_revision != self.policy.model_revision
            or self.policy.tokenizer_revision != config.model_revision
        ):
            raise ValueError("Adapter checkpoint revision mismatch")
        if (
            adapter.get("r"),
            adapter.get("lora_alpha"),
            adapter.get("lora_dropout"),
        ) != (config.lora_rank, config.lora_alpha, config.lora_dropout):
            raise ValueError("Adapter LoRA settings differ from experiment")
        if (
            set(adapter.get("target_modules", [])) != set(self.targets)
            or text_lora_targets(self.targets) != self.targets
        ):
            raise ValueError(
                "Adapter target modules disagree with reviewed text targets"
            )
        if (
            adapter.get("peft_type") != "LORA"
            or adapter.get("bias") != "none"
            or adapter.get("modules_to_save")
        ):
            raise ValueError("Only plain text LoRA weights are supported")
        return self


def export_adapter(bundle, output_root, optimizer_step):
    """Transfer adapter only, not an optimizer/resume checkpoint. Never overwrite."""
    trainable_names(bundle.model.named_parameters())
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory is confined to this explicitly chosen transfer root.
    with tempfile.TemporaryDirectory(prefix=".adapter-export-", dir=root) as temporary:
        staging = Path(temporary)
        bundle.model.save_pretrained(
            str(staging), safe_serialization=True, selected_adapters=["default"]
        )
        digest, hashes = adapter_digest(staging)
        policy = PolicyVersion(
            bundle.config.model_revision,
            bundle.config.model_revision,
            digest,
            optimizer_step,
            template_digest(bundle.tokenizer),
        )
        manifest = {
            "schema_version": 1,
            "policy": asdict(policy),
            "files": dict(hashes),
            "targets": list(bundle.targets),
            "config": bundle.config.to_dict(),
        }
        (staging / "transfer.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        artifact = AdapterArtifact(staging, policy, bundle.targets).verify()
        destination = root / policy.id
        if destination.exists():
            existing = AdapterArtifact(destination, policy, bundle.targets).verify()
            if (
                json.loads((destination / "transfer.json").read_text(encoding="utf-8"))
                != manifest
            ):
                raise ValueError(
                    "Existing transfer identity belongs to a different experiment"
                )
            return existing
        # Rename only files created here, never move/delete an existing directory.
        destination.mkdir()
        for name in (*ADAPTER_FILES, "transfer.json"):
            (staging / name).replace(destination / name)
        return AdapterArtifact(destination, artifact.policy, bundle.targets).verify()
