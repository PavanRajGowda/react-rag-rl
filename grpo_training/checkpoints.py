"""Complete, hash-verified checkpoints. Torch imports only during save/restore."""

import hashlib
import json
import os
import random
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from grpo_training.advantages import prepare_update
from grpo_training.model import ADAPTER_FILES, AdapterArtifact, export_adapter
from grpo_training.provenance import digest
from grpo_training.records import EpisodeRecord, PolicyVersion
from grpo_training.replay import QuestionGroup
from grpo_training.trainer import CapturedUpdate


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Progress:
    step: int = 0
    cursor: int = 0
    round_index: int = 0
    iteration: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("Progress must contain nonnegative integers")


def capture_dict(captured):
    if captured is None:
        return None
    return {
        "groups": [[e.to_dict() for e in g.episodes] for g in captured.prepared.groups],
        "signals": [asdict(s) for s in captured.prepared.signals],
        "old_logprobs": captured.old_logprobs,
        "corrections": captured.corrections,
    }


def restore_capture(value, labels):
    if value is None:
        return None
    groups = tuple(
        QuestionGroup(tuple(EpisodeRecord.from_dict(e) for e in group))
        for group in value["groups"]
    )
    ids = {g.episodes[0].question.id for g in groups}
    prepared = prepare_update(groups, [label for label in labels if label.id in ids])
    if value["signals"] != [asdict(s) for s in prepared.signals]:
        raise ValueError(
            "Checkpoint episode rewards/advantages disagree with current labels"
        )
    return CapturedUpdate(prepared, value["old_logprobs"], value["corrections"])


def rng_state():
    import numpy as np
    import torch

    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype="uint32"), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Checkpoint CUDA RNG topology mismatch")
        torch.cuda.set_rng_state_all(state["cuda"])


class RunStore:
    """One externally locked writer per run. Incomplete directories are never latest."""

    def __init__(self, root, run_id, identity, *, resume=False, durable=lambda: None):
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id
        ):
            raise ValueError("Run ID must be a short name without path separators")
        self.root = Path(root).resolve() / run_id
        self.identity, self.durable = identity, durable
        path = self.root / "run.json"
        if resume:
            if json.loads(path.read_text(encoding="utf-8")) != identity:
                raise ValueError(
                    "Run config/data/source identity changed; refusing resume"
                )
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            write_json(path, identity)
            self.durable()

    def audit(self, event):
        # Unique files preserve failed/replayed attempts without overwriting old logs.
        directory = self.root / "events"
        directory.mkdir(exist_ok=True)
        now = datetime.now(UTC)
        write_json(
            directory / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}.json",
            {"recorded_at_utc": now.isoformat(), **event},
        )

    def save(self, trainer, artifact, progress, pending=None):
        import torch

        artifact.verify()
        if (
            progress.step != trainer.state.global_step
            or artifact.policy.optimizer_step != progress.step
        ):
            raise ValueError("Checkpoint policy/optimizer step mismatch")
        if (pending is None) != (progress.iteration == 0):
            raise ValueError("Checkpoint pending-reuse state mismatch")
        staging = self.root / f".partial-{uuid.uuid4().hex}"
        staging.mkdir()
        adapter = staging / "adapter"
        adapter.mkdir()
        for name in (*ADAPTER_FILES, "transfer.json"):
            shutil.copyfile(artifact.path / name, adapter / name)
        state = {
            "schema_version": 1,
            "identity_sha256": digest(self.identity),
            "progress": asdict(progress),
            "policy": asdict(artifact.policy),
            "targets": artifact.targets,
            "pending": capture_dict(pending),
        }
        write_json(staging / "state.json", state)
        torch.save(
            {
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.lr_scheduler.state_dict(),
                "rng": rng_state(),
            },
            staging / "training.pt",
        )
        files = {
            p.relative_to(staging).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in staging.rglob("*")
            if p.is_file()
        }
        write_json(staging / "manifest.json", {"schema_version": 1, "files": files})
        destination = self.root / f"checkpoint-{progress.step:08d}-{uuid.uuid4().hex}"
        staging.rename(destination)
        pointer = self.root / f".latest-{uuid.uuid4().hex}.json"
        write_json(pointer, {"checkpoint": destination.name})
        pointer.replace(self.root / "latest.json")
        self.durable()  # All files closed before committing a Modal Volume.
        return destination

    def latest(self):
        name = json.loads((self.root / "latest.json").read_text(encoding="utf-8"))[
            "checkpoint"
        ]
        if not re.fullmatch(r"checkpoint-[0-9]{8}-[0-9a-f]{32}", name):
            raise ValueError("Invalid checkpoint pointer")
        directory = self.root / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "state.json",
            "training.pt",
            *(f"adapter/{n}" for n in (*ADAPTER_FILES, "transfer.json")),
        }
        if manifest.get("schema_version") != 1 or set(manifest["files"]) != expected:
            raise ValueError("Invalid checkpoint file manifest")
        for name, expected_hash in manifest["files"].items():
            path = directory / name
            if (
                path.is_symlink()
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
            ):
                raise ValueError(f"Checkpoint integrity failure: {name}")
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state["identity_sha256"] != digest(
            self.identity
        ):
            raise ValueError("Checkpoint belongs to another run")
        artifact = AdapterArtifact(
            directory / "adapter",
            PolicyVersion(**state["policy"]),
            tuple(state["targets"]),
        ).verify()
        return directory, state, artifact

    def restore(self, trainer, bundle, labels):
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        directory, state, artifact = self.latest()
        progress = Progress(**state["progress"])
        if artifact.policy.optimizer_step != progress.step:
            raise ValueError("Checkpoint progress/policy mismatch")
        if (
            json.loads((artifact.path / "transfer.json").read_text(encoding="utf-8"))[
                "config"
            ]
            != bundle.config.to_dict()
        ):
            raise ValueError("Checkpoint experiment mismatch")
        result = set_peft_model_state_dict(
            bundle.model,
            load_file(str(artifact.path / "adapter_model.safetensors")),
            adapter_name="default",
        )
        if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
            raise ValueError("Checkpoint adapter tensors do not match the live model")
        if (
            export_adapter(bundle, self.root / "transfers", progress.step).policy
            != artifact.policy
        ):
            raise ValueError("Restored adapter bytes disagree with checkpoint policy")
        training = torch.load(
            directory / "training.pt", map_location="cpu", weights_only=True
        )
        trainer.optimizer.load_state_dict(training["optimizer"])
        trainer.lr_scheduler.load_state_dict(training["scheduler"])
        trainer.state.global_step = progress.step
        pending = restore_capture(state["pending"], labels)
        if (pending is None) != (progress.iteration == 0):
            raise ValueError("Checkpoint pending-reuse state mismatch")
        if pending is not None and (
            pending.prepared.config != bundle.config
            or progress.step
            != pending.prepared.policy.optimizer_step + progress.iteration
            or not 0 < progress.iteration < bundle.config.num_iterations
        ):
            raise ValueError("Invalid checkpoint rollout-reuse position")
        # Restore RNG AFTER fresh model/trainer/vLLM construction, which can consume it.
        return progress, pending, training["rng"]
