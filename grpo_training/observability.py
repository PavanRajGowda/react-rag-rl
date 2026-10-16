"""Live JSON/TensorBoard metrics, with no extra policy forward for PPO diagnostics."""

import hashlib
import json
import math
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from grpo_training.checkpoints import write_json
from grpo_training.metrics import distribution, pair, rollout_metrics, sampling_metrics
from grpo_training.provenance import digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


class NullObserver:
    def start(self, *args):
        pass

    def phase(self, *args, **kwargs):
        pass

    def event(self, *args):
        pass

    def rollout(self, *args, **kwargs):
        pass

    def reference_step(self, *args):
        return None

    def step(self, *args):
        pass

    def close(self, *args):
        pass

    @contextmanager
    def loss(self, trainer):
        yield None


class LossMetrics:
    def __init__(self, config):
        self.config = config
        self.inputs = None
        self.log_ratios, self.entropies = [], []
        self.outside = self.clipped = self.learning_tokens = self.tokens = 0
        self.old_minus_current = 0.0

    def record(self, current, entropy):
        inputs = self.inputs
        mask = inputs["completion_mask"].bool()
        ratios = (
            (current.detach() - inputs["old_per_token_logps"])[mask]
            .float()
            .cpu()
            .tolist()
        )
        advantages = inputs["advantages"][mask].float().cpu().tolist()
        lower, upper = (
            math.log1p(-self.config.epsilon_low),
            math.log1p(self.config.epsilon_high),
        )
        for value, advantage in zip(ratios, advantages, strict=True):
            if not math.isfinite(value):
                raise ValueError("Nonfinite policy log-ratio")
            self.outside += value < lower or value > upper
            self.learning_tokens += advantage != 0
            self.clipped += (advantage < 0 and value < lower) or (
                advantage > 0 and value > upper
            )
        self.tokens += len(ratios)
        self.log_ratios.extend(ratios)
        self.old_minus_current -= sum(ratios)
        if entropy is not None:
            self.entropies.extend(entropy.detach()[mask].float().cpu().tolist())

    def summary(self):
        safe_ratios = all(abs(value) <= 80 for value in self.log_ratios)
        ratios = [math.exp(value) for value in self.log_ratios] if safe_ratios else []
        return {
            "tokens": self.tokens,
            "current_over_old_log_ratio": distribution(self.log_ratios),
            "current_over_old_ratio": distribution(ratios),
            "ratio_distribution_available": safe_ratios,
            "ratio_outside_bounds": pair(self.outside, self.tokens),
            "objective_clipped_learning_tokens": pair(
                self.clipped, self.learning_tokens
            ),
            "objective_clipped_all_tokens": pair(self.clipped, self.tokens),
            "old_to_current_kl_sample_estimate": pair(
                self.old_minus_current, self.tokens
            ),
            "entropy_nats": distribution(self.entropies),
        }


@contextmanager
def observe_loss(trainer, config):
    collector = LossMetrics(config)
    original = trainer._get_per_token_logps_and_entropies

    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        collector.record(output[0], output[1])
        return output  # Preserve original graph/tensors and TRL objective.

    trainer._get_per_token_logps_and_entropies = observed
    trainer._loss_metrics = collector
    try:
        yield collector
    finally:
        trainer._get_per_token_logps_and_entropies = original
        trainer._loss_metrics = None
        collector.inputs = None
        trainer._metrics[
            "train"
        ].clear()  # Native microbatch averages are not our aggregates.


def gpu_memory():
    import torch

    if not torch.cuda.is_available():
        return {"device_used_bytes": None, "torch_peak_allocated_bytes": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "device_used_bytes": total - free,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
    }


class RunObserver:
    def __init__(
        self,
        store,
        trainer,
        bundle,
        train,
        *,
        writer=None,
        reference=None,
        reference_catalog=None,
    ):
        from grpo_training.reference import ReferenceProbe

        self.store, self.trainer, self.config = store, trainer, bundle.config
        self.train, self.catalog = train, reference_catalog
        self.root = store.root / "observability"
        self.root.mkdir(exist_ok=True)
        catalog_path = self.root / "reference_metadata.json"
        catalog_record = {
            "sha256": digest(reference_catalog or {}),
            "metadata": reference_catalog or {},
        }
        if (
            catalog_path.exists()
            and json.loads(catalog_path.read_text(encoding="utf-8")) != catalog_record
        ):
            raise ValueError("Diagnostic reference metadata changed on resume")
        atomic_json(catalog_path, catalog_record)
        self.reference = reference or ReferenceProbe(
            trainer, bundle, train.questions, store.root
        )
        if writer is None:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(
                str(store.root / "tensorboard"),
                purge_step=trainer.state.global_step + 1,
            )
        self.writer, self.status = writer, {}
        previous_live = self.root / "live.json"
        self.recovery_counts = (
            json.loads(previous_live.read_text(encoding="utf-8")).get("recovery", {})
            if previous_live.exists()
            else {}
        )
        self.previous_sampling_totals = (
            json.loads(previous_live.read_text(encoding="utf-8")).get(
                "dynamic_totals", {}
            )
            if previous_live.exists()
            else {}
        )
        self.recovery_counts = (
            dict.fromkeys(
                (
                    "failed_trajectory_attempts",
                    "replacement_trajectories",
                    "dropped_groups",
                    "probability_recovery_attempts",
                    "skipped_empty_batches",
                ),
                0,
            )
            | self.recovery_counts
        )
        self.last_commit, self.rollout_id = 0.0, None
        self.closed = False
        self.start_step = trainer.state.global_step
        self.last_loss = None
        path = self.root / "index.json"
        self.index = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"steps": [], "references": [], "committed_step": self.start_step}
        )
        self.index["steps"] = [
            row for row in self.index["steps"] if row["step"] <= self.start_step
        ]
        self.index["references"] = [
            step for step in self.index["references"] if step <= self.start_step
        ]
        self.index["committed_step"] = self.start_step

    def publish(self, *, force=False):
        atomic_json(self.root / "index.json", self.index)
        atomic_json(
            self.root / "live.json",
            self.status
            | {
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "recovery": self.recovery_counts,
            },
        )
        if force or time.monotonic() - self.last_commit >= 5:
            self.writer.flush()
            self.store.durable()
            self.last_commit = time.monotonic()

    def histogram(self, name, values, step):
        import numpy as np

        if len(values):
            self.writer.add_histogram(name, np.asarray(values, dtype="float64"), step)

    def phase(self, name, **fields):
        if name == "backward":
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        self.status.update(
            phase=name, phase_started_at_utc=datetime.now(UTC).isoformat(), **fields
        )
        self.publish()

    def start(self, progress):
        self.status = {
            "step": progress.step,
            "committed_step": progress.step,
            "max_steps": self.config.max_steps,
            "training_config": self.config.to_dict(),
        }
        if self.previous_sampling_totals:
            self.status["dynamic_totals"] = dict(self.previous_sampling_totals)
        self.reference_step(progress.step)
        self.phase("ready")

    def event(self, event):
        if event["kind"] == "dynamic_sampling":
            self.status["dynamic_sampling"] = event
        if event["kind"] == "dynamic_group":
            totals = self.status.setdefault(
                "dynamic_totals",
                {
                    "groups": 0,
                    "accepted": 0,
                    "episodes": 0,
                    "correct": 0,
                    "generated_tokens": 0,
                },
            )
            totals["groups"] += 1
            totals["accepted"] += int(event["accepted"])
            for key in ("episodes", "correct", "generated_tokens"):
                totals[key] += event[key]

        kind = event["kind"]
        increments = {
            "trajectory_failed": ("failed_trajectory_attempts", 1),
            "trajectory_retry_started": (
                "replacement_trajectories",
                event.get("trajectories", 0),
            ),
            "group_dropped": ("dropped_groups", 1),
            "probability_recovery": ("probability_recovery_attempts", 1),
            "batch_skipped": ("skipped_empty_batches", 1),
        }
        if kind in increments:
            key, increment = increments[kind]
            self.recovery_counts[key] = self.recovery_counts.get(key, 0) + increment
            self.status["last_recovery_event"] = event
        if kind == "generation_batch":
            self.status["last_generation_batch"] = {
                k: v for k, v in event.items() if k != "kind"
            }
        elif kind in ("generation_started", "searches_started", "collection_progress"):
            self.status.update({k: v for k, v in event.items() if k != "kind"})
            self.status["phase"] = (
                "generation"
                if kind == "generation_started"
                else "search/record"
                if kind == "searches_started"
                else self.status.get("phase")
            )
        self.publish()

    def rollout(self, captured, *, fresh, round_index):
        prepared = captured.prepared
        identity = [
            {
                "episode": e.episode_id,
                "policy": e.policy.id,
                "status": e.status,
                "tokens": [g.completion_ids for g in e.generations],
            }
            for group in prepared.groups
            for e in group.episodes
        ]
        self.rollout_id = digest(identity)[:24]
        path = self.root / "rollouts" / f"{self.rollout_id}.json"
        if not path.exists():
            metrics, traces = rollout_metrics(prepared, self.train.labels, self.catalog)
            for row in metrics["episodes"]:
                key = hashlib.sha256(
                    (self.rollout_id + row["episode_id"]).encode()
                ).hexdigest()[:24]
                row["trace_key"] = key
                atomic_json(
                    self.root / "traces" / f"{key}.json", traces[row["episode_id"]]
                )
            metrics.update(
                rollout_id=self.rollout_id,
                round=round_index,
                policy_step=prepared.policy.optimizer_step,
                sampling=sampling_metrics(captured),
                sampling_summary=self.status.get("dynamic_sampling", {})
                if self.status.get("dynamic_sampling", {}).get("round") == round_index
                else {},
            )
            atomic_json(path, metrics)
        self.status.update(rollout_id=self.rollout_id, fresh_rollout=fresh)
        if fresh:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            step = prepared.policy.optimizer_step
            for name, rate in metrics["rates"].items():
                if rate["value"] is not None:
                    self.writer.add_scalar(f"rollout/{name}", rate["value"], step)
            self.histogram(
                "rollout/advantages", [s.advantage for s in prepared.signals], step
            )
            self.histogram(
                "rollout/rewards", [s.reward for s in prepared.signals], step
            )
            sampling = metrics["sampling"]
            self.writer.add_scalar(
                "sampling/raw_correction_above_cap",
                sampling["raw_correction_above_cap"]["value"],
                step,
            )
            self.writer.add_scalar(
                "sampling/mean_applied_correction",
                sampling["applied_correction"]["mean"],
                step,
            )
        self.publish()

    def reference_step(self, step):
        path = self.root / "reference" / f"{step:08d}.json"
        if path.exists() and step <= self.start_step:
            return json.loads(path.read_text(encoding="utf-8"))
        self.phase("reference_kl", step=step)
        started = time.perf_counter()
        result = self.reference.measure() | {
            "step": step,
            "seconds": time.perf_counter() - started,
            "timing": "post-update; step zero is initial",
        }
        atomic_json(path, result)
        if step not in self.index["references"]:
            self.index["references"].append(step)
        self.writer.add_scalar(
            "reference/kl_current_to_initial_nats", result["mean_nats"], step
        )
        self.writer.add_scalar(
            "performance/reference_kl_seconds", result["seconds"], step
        )
        return result

    @contextmanager
    def loss(self, trainer):
        with observe_loss(trainer, self.config) as collector:
            yield collector
            self.last_loss = collector

    def step(self, metrics):
        step = metrics["step"]
        record = metrics | {
            "rollout_id": self.rollout_id,
            "gpu": gpu_memory(),
            "loss_timing": "pre-update",
            "reference_timing": "post-update",
            "recovery_counts": dict(self.recovery_counts),
        }
        atomic_json(self.root / "steps" / f"{step:08d}.json", record)
        self.index["steps"] = [r for r in self.index["steps"] if r["step"] != step] + [
            {"step": step, "rollout_id": self.rollout_id}
        ]
        self.index["committed_step"] = step
        for key in (
            "loss",
            "gradient_norm",
            "learning_rate_used",
            "learning_rate_next",
            "seconds",
            "backward_seconds",
            "optimizer_seconds",
            "checkpoint_seconds",
        ):
            if record.get(key) is not None:
                self.writer.add_scalar(f"training/{key}", record[key], step)
        self.writer.add_scalar(
            "training/gradient_clipped",
            int(record["gradient_norm"] > self.config.max_grad_norm),
            step,
        )
        for key, count in self.recovery_counts.items():
            self.writer.add_scalar(f"recovery/{key}", count, step)
        loss = record.get("policy_metrics")
        if loss:
            for key in (
                "ratio_outside_bounds",
                "objective_clipped_learning_tokens",
                "objective_clipped_all_tokens",
                "old_to_current_kl_sample_estimate",
            ):
                if loss[key]["value"] is not None:
                    self.writer.add_scalar(f"policy/{key}", loss[key]["value"], step)
            if loss["entropy_nats"]["mean"] is not None:
                self.writer.add_scalar(
                    "policy/mean_entropy_nats", loss["entropy_nats"]["mean"], step
                )
        if self.last_loss:
            self.histogram(
                "policy/current_over_old_log_ratios", self.last_loss.log_ratios, step
            )
            self.last_loss = None
        self.status.update(
            step=step, committed_step=step, phase="checkpointed", last_step=record
        )
        self.publish(force=True)

    def close(self, status):
        if not self.closed:
            self.status.update(phase=status)
            self.publish(force=True)
            self.writer.close()
            self.closed = True
