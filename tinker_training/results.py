"""Offline reader for saved training metrics; no server or live publication."""

import json
from pathlib import Path

from tinker_training.metrics import distribution, merge_rates, pair


def read(path, default=None):
    """Read a local JSON artifact or return default if absent; malformed files still raise."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


class Reader:
    """Read-only, modification-time-cached view of a run observability directory. Combines
    step, rollout and validation files without contacting the training provider.
    """

    def __init__(self, root):
        """Bind a run root for read-only dashboard snapshots; no training or remote
        synchronization starts here.
        """
        self.root, self.cache = Path(root), {}

    def file(self, name):
        """Read an artifact relative to this reader root, caching by file modification
        time. Returns None when absent. The caller supplies the observability directory;
        this method does not search alternate run roots.
        """
        path = self.root / name
        if not path.exists():
            return None
        stamp = path.stat().st_mtime_ns
        if name not in self.cache or self.cache[name][0] != stamp:
            self.cache[name] = stamp, read(path)
        return self.cache[name][1]

    def snapshot(self, window=20, episode_limit=500):
        """Build the dashboard payload from committed steps, unique fresh rollouts,
        validation and live phase records. window limits optimizer-step summaries;
        episode_limit bounds returned detail. Reused rollouts are deduplicated by
        identity.
        """
        index = self.file("index.json") or {
            "steps": [],
            "references": [],
            "committed_step": 0,
        }
        steps = [
            self.file(f"steps/{r['step']:08d}.json")
            for r in sorted(index["steps"], key=lambda r: r["step"])
            if r["step"] <= index["committed_step"]
        ]
        steps = [s for s in steps if s]
        selected = steps[-window:] if window else steps
        ids = list(dict.fromkeys(s["rollout_id"] for s in selected))
        rollouts = [self.file(f"rollouts/{key}.json") for key in ids]
        rollouts = [r for r in rollouts if r]
        rates = {
            name: merge_rates(
                [r["rates"][name] for r in rollouts if name in r["rates"]]
            )
            for name in {k for r in rollouts for k in r["rates"]}
        }
        tokens = sum(s["tokens"] for s in selected)
        reference = [
            self.file(f"reference/{step:08d}.json")
            for step in index["references"]
            if step <= index["committed_step"]
        ]
        reference = [r for r in reference if r]
        series = []
        cumulative_loss, cumulative_tokens = [0.0], [0]
        for i, step in enumerate(steps):
            cumulative_loss.append(cumulative_loss[-1] + step["loss"] * step["tokens"])
            cumulative_tokens.append(cumulative_tokens[-1] + step["tokens"])
            first = max(0, i - window + 1) if window else 0
            count = cumulative_tokens[i + 1] - cumulative_tokens[first]
            series.append(
                {
                    "step": step["step"],
                    "loss": step["loss"],
                    "gradient_norm": step["gradient_norm"],
                    "entropy": (step.get("policy_metrics") or {})
                    .get("entropy_nats", {})
                    .get("mean"),
                    "mean_reward": step.get("mean_reward"),
                    "iteration": step.get("iteration"),
                    "rolling_loss": (cumulative_loss[i + 1] - cumulative_loss[first])
                    / count
                    if count
                    else None,
                }
            )
        live = self.file("live.json") or {}
        config = live.get("training_config") or read(
            self.root.parent / "run.json", {}
        ).get("config", {})
        all_ids = list(dict.fromkeys(s["rollout_id"] for s in steps))
        all_rollouts = [
            r for key in all_ids if (r := self.file(f"rollouts/{key}.json"))
        ]
        batches = []
        for rollout in all_rollouts:
            associated = [s for s in steps if s["rollout_id"] == rollout["rollout_id"]]
            sampling = rollout.get("sampling_summary") or {}
            if (
                not sampling
                and live.get("dynamic_sampling")
                and live.get("dynamic_sampling", {}).get("round")
                == rollout.get("round")
            ):
                sampling = live["dynamic_sampling"]
            batches.append(
                {
                    "rollout_id": rollout["rollout_id"],
                    "round": rollout.get("round"),
                    "optimizer_steps": [s["step"] for s in associated],
                    "reuse_iterations": [s.get("iteration") for s in associated],
                    "question_ids": list(
                        dict.fromkeys(e["id"] for e in rollout["episodes"])
                    ),
                    "actual_groups": len(rollout["groups"]),
                    "actual_trajectories": len(rollout["episodes"]),
                    "candidate_questions": sampling.get("candidate_questions"),
                    "budget": sampling.get("budget"),
                }
            )
        policy = [s["policy_metrics"] for s in selected if s.get("policy_metrics")]
        episodes = [e for r in rollouts for e in r["episodes"]]
        hits = [e for e in episodes if e.get("first_hit_rank")]
        return {
            "live": live,
            "training_config": config,
            "batches": batches,
            "batch_totals": {
                "fresh_batches": len(batches),
                "accepted_groups": sum(b["actual_groups"] for b in batches),
                "accepted_trajectories": sum(b["actual_trajectories"] for b in batches),
            },
            "committed_step": index["committed_step"],
            "steps": selected,
            "series": series[-500:],
            "reference": reference,
            "validation": [
                read(p)
                | {
                    "efficiency": read(p).get("efficiency")
                    or read(self.root / "validation_efficiency" / p.name)
                }
                for p in sorted((self.root / "validation").glob("*.json"))
                if int(p.stem) <= index["committed_step"]
            ],
            "rates": rates,
            "rollouts": len(rollouts),
            "episodes": episodes[-episode_limit:] if episode_limit else episodes,
            "episode_count": len(episodes),
            "first_hit": {
                "searches": distribution(e["first_hit_search"] for e in hits),
                "generations": distribution(
                    e["first_hit_turn"] for e in hits if e["first_hit_turn"] is not None
                ),
                "reciprocal_rank": pair(
                    sum(1 / e["first_hit_rank"] for e in hits), len(hits)
                ),
                "first_search_hit": pair(
                    sum(e["first_hit_search"] == 1 for e in episodes), len(episodes)
                ),
            },
            "lengths": {
                k: distribution(e[k] for e in episodes if e.get(k) is not None)
                for k in (
                    "generations",
                    "searches",
                    "input_tokens",
                    "output_tokens",
                    "peak_input_tokens",
                    "seconds",
                    "search_seconds",
                )
            },
            "groups": [g for r in rollouts for g in r["groups"]],
            "weighted_loss": pair(
                sum(s["loss"] * s["tokens"] for s in selected), tokens
            ),
            "policy_rates": {
                name: merge_rates([p[name] for p in policy])
                for name in (
                    "ratio_outside_bounds",
                    "objective_clipped_learning_tokens",
                    "objective_clipped_all_tokens",
                    "old_to_current_kl_sample_estimate",
                )
            },
            "credit": [r["credit"] for r in rollouts if r.get("credit")],
            "sampling": [r["sampling"] for r in rollouts],
            "rollout_details": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("episodes", "groups", "credit")
                }
                for r in rollouts
            ],
            "definitions": {
                "window": "optimizer steps; fresh rollouts deduplicated by rollout_id",
                "loss": live.get(
                    "loss_definition",
                    (selected[-1].get("loss_definition") if selected else None)
                    or "token-mean clipped importance-weighted GRPO surrogate",
                ),
                "reference_kl": "post-update; fixed initial-generation train prefixes, full vocabulary, nats",
                "reference_warning": "A probe diagnostic, not trajectory-wide KL or a penalty; beta remains zero",
                "ratios": "current/old: optimizer drift; old/vLLM: backend mismatch; distinct from fixed-reference KL",
                "timings": "generation times are batch wall times; episode time includes batching/tool queue waits; unavailable stays null",
            },
        }
