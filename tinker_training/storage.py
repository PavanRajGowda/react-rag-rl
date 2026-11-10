"""Local experiment journal and saved Tinker metrics."""

import csv
import json
from pathlib import Path

from baseline_agent.persistence import write_json
from tinker_training.metrics import distribution, pair, rollout_metrics


class Store:
    """Local experiment journal and offline metric exports for hosted execution. Holds
    checkpoint pointers and raw/metric artifacts; remote model and optimizer tensors
    remain in Tinker.
    """

    def __init__(self, root, config, identity):
        """Create or validate a local run journal against frozen config and source
        identity.

        Existing run.json must match exactly. Loads the observability index but does not
        contact Tinker or restore model weights; mismatched identity raises ValueError.
        """
        self.root, self.config = Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "run.json"
        record = {"backend": "tinker", "config": config.to_dict(), "identity": identity}
        # Prevent mixing results from different inputs or source versions under
        # one run ID. A refactored experiment must use a new run directory.
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Tinker run identity changed; use a new run directory")
        write_json(path, record)
        self.obs = self.root / "observability"
        self.index = self.read(
            self.obs / "index.json",
            {"steps": [], "references": [], "validation": [], "committed_step": 0},
        )

    @staticmethod
    def read(path, default=None):
        """Load a JSON artifact, returning default only when the file is absent.

        Malformed JSON and filesystem errors propagate instead of being treated as
        missing.
        """
        return (
            json.loads(Path(path).read_text(encoding="utf-8"))
            if Path(path).exists()
            else default
        )

    def phase(self, name):
        """Track the current phase without writing live dashboard state."""
        self.current_phase = name

    def audit(self, event):
        """Append one JSON-serializable event to events.jsonl for later inspection."""
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def rollout(self, prepared, labels, round_index):
        """Write aggregate batch statistics only, without tokens or trajectory records.

        Returns the logical rollout ID derived from round_index. labels are used for
        diagnostics only; repeated updates may reference the same fresh rollout.
        """
        key = f"tinker-round-{round_index:08d}"
        metrics, _ = rollout_metrics(prepared, labels, include_traces=False)
        from collections import Counter

        metrics["episode_count"] = len(prepared.signals)
        metrics["group_count"] = len(prepared.groups)
        metrics["difficulty"] = dict(
            Counter(g.episodes[0].question.difficulty for g in prepared.groups)
        )
        metrics.pop("episodes", None)
        metrics.pop("groups", None)
        metrics.update(
            rollout_id=key,
            round=round_index,
            policy_step=prepared.policy.optimizer_step,
            sampling={
                "available": False,
                "training_minus_sampler_logprob": distribution([]),
                "applied_correction": distribution([]),
                "raw_correction_above_cap": pair(0, 0),
                "sampler_to_training_kl_sample_estimate": pair(0, 0),
            },
        )
        write_json(self.obs / "rollouts" / f"{key}.json", metrics)
        return key

    def commit_metrics(self, metrics):
        """Record a completed update and rebuild the flat steps.csv export.

        metrics must include step and rollout_id. Writes the per-step record before
        advancing the latest-step index; checkpoint durability is tracked separately.
        """
        step = metrics["step"]
        write_json(self.obs / "steps" / f"{step:08d}.json", metrics)
        self.index["steps"] = [s for s in self.index["steps"] if s["step"] != step] + [
            {"step": step, "rollout_id": metrics["rollout_id"]}
        ]
        # Saved progress follows committed optimizer checkpoints, not the
        # beginning of a possibly interrupted backward/optimizer operation.
        self.index["latest_step"] = step
        write_json(self.obs / "index.json", self.index)
        self._export_steps()

    def mark_checkpoint(self, step):
        """Mark only model/optimizer state actually stored remotely as recoverable."""
        self.index["committed_step"] = step
        write_json(self.obs / "index.json", self.index)

    def restore_checkpoint_index(self, step):
        """Exclude statistics from rolled-back updates; retain a rollback audit event."""
        abandoned = [s for s in self.index["steps"] if s["step"] > step]
        if abandoned:
            self.audit(
                {
                    "kind": "resume_rollback",
                    "checkpoint_step": step,
                    "discarded_metric_steps": [s["step"] for s in abandoned],
                }
            )
        self.index["steps"] = [s for s in self.index["steps"] if s["step"] <= step]
        self.index.update(committed_step=step, latest_step=step)
        write_json(self.obs / "index.json", self.index)
        self._export_steps()

    def _export_steps(self):
        """Export only active update records, excluding rolled-back attempts."""
        rows = [
            self.read(self.obs / "steps" / f"{s['step']:08d}.json")
            for s in self.index["steps"]
        ]
        fields = [
            "step",
            "round",
            "iteration",
            "loss",
            "gradient_norm",
            "mean_reward",
            "tokens",
            "groups",
            "episodes",
            "seconds",
            "learning_rate_used",
        ]
        with (self.root / "steps.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def fail(self, error):
        """Keep one compact exception record without publishing dashboard state."""
        import traceback

        write_json(
            self.root / "failure.json",
            {
                "phase": getattr(self, "current_phase", "unknown"),
                "error_type": type(error).__name__,
                "message": str(error),
                "checkpoint_step": self.index.get("committed_step", 0),
                "traceback": "".join(traceback.format_exception(error)),
            },
        )
