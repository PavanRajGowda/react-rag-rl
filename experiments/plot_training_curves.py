"""Plot compact saved training statistics offline; no service or inference calls."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot(root):
    """Plot rollout behavior separately from filtered optimizer-minibatch metrics."""
    state = read(root / "progress.json")
    config = read(root / "run.json")["config"]
    process = config["reward"] == "bounded_process_trajectory_v1"
    local = config["reward"] == "local_process_advantage_v1"
    arm = "Local advantage-shaping" if local else "Process-reward" if process else "EM-only"
    if local and config["local_gold_weight"] == config["local_search_penalty_weight"] == config["local_protocol_penalty_weight"]:
        arm = f'Local shaping, weights {config["local_gold_weight"]:g}'
    index = read(root / "observability/index.json")
    steps = [
        read(root / "observability/steps" / f"{s['step']:08d}.json")
        for s in sorted(index["steps"], key=lambda s: s["step"])
        if s["step"] <= state["step"]
    ]
    rounds = sorted({s["round"] for s in steps})
    pools = [
        read(root / "observability/rollouts" / f"tinker-round-{n:08d}.json")
        for n in rounds
    ]
    x = [s["step"] for s in steps]
    xp = [p["policy_step"] for p in pools]
    out = root / "plots"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 170,
        }
    )

    def line(ax, xx, yy, title, ylabel, *, label=None, color="#176b9b"):
        ax.plot(
            xx, yy, marker="o", markersize=4, linewidth=1.7, color=color, label=label
        )
        ax.set(title=title, xlabel="Optimizer updates completed", ylabel=ylabel)
        ax.grid(alpha=0.18)
        ax.set_xticks(range(0, state["step"] + 1, 2))
        if label:
            ax.legend(frameon=False, fontsize=9)

    def save(fig, name, subtitle):
        fig.suptitle(
            f"{arm} training: {state['step']} optimizer updates",
            fontsize=17,
            fontweight="bold",
            y=0.995,
        )
        fig.text(0.5, 0.952, subtitle, ha="center", fontsize=10, color="#444444")
        fig.tight_layout(rect=(0, 0.035, 1, 0.93), h_pad=2.2, w_pad=2)
        fig.text(
            0.02,
            0.012,
            "Rollout points use different question pools; these are training diagnostics, not held-out learning curves.",
            fontsize=9,
            color="#555555",
        )
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ax = axes.ravel()
    em = [100 * p["rates"]["em"]["value"] for p in pools]
    line(
        ax[0],
        xp,
        em,
        "Full rollout exact match (before filtering)",
        "Correct trajectories (%)",
    )
    line(
        ax[1],
        x,
        [s["mean_reward"] for s in steps],
        "Filtered minibatch mean reward",
        "Combined reward" if process else "Mean EM reward",
    )
    line(
        ax[2],
        x,
        [s["loss"] for s in steps],
        "Token-averaged clipped policy loss",
        "Loss",
    )
    line(
        ax[3],
        x,
        [s["gradient_norm"] for s in steps],
        "Unclipped gradient L2 norm",
        "Gradient norm",
    )
    line(
        ax[4],
        xp,
        [p["lengths"]["searches"]["mean"] for p in pools],
        "Mean agent actions per trajectory",
        "Count",
        label="Searches",
    )
    line(
        ax[4],
        xp,
        [p["lengths"]["generations"]["mean"] for p in pools],
        "Mean agent actions per trajectory",
        "Count",
        label="Generation turns",
        color="#ce7627",
    )
    line(
        ax[5],
        xp,
        [100 * p["rates"]["ever"]["value"] for p in pools],
        "Full gold-document coverage",
        "Trajectories (%)",
    )
    save(
        fig,
        "training-curves",
        f"{len(pools)} fresh pools · 512 questions × 5 trajectories · 50/50 hard–medium · no validation",
    )

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ax = axes.ravel()
    line(
        ax[0],
        xp,
        [p["lengths"]["output_tokens"]["mean"] for p in pools],
        "Mean generated length",
        "Tokens / trajectory",
    )
    for level, color in [("hard", "#176b9b"), ("medium", "#ce7627")]:
        line(
            ax[1],
            x,
            [
                100
                * s["minibatch_filtering"]["difficulty_surviving"].get(level, 0)
                / s["minibatch_filtering"]["difficulty_original"][level]
                for s in steps
            ],
            "Group retention by difficulty",
            "Groups retained (%)",
            label=level.title(),
            color=color,
        )
    line(
        ax[2],
        xp,
        [
            next(s for s in steps if s["round"] == p["round"])["reference_kl"][
                "sampled_logratio_mean"
            ]
            for p in pools
        ],
        "Rollout vs frozen reference: sampled log-ratio",
        "Mean log-ratio (nats)",
    )
    line(
        ax[3],
        xp,
        [
            100
            * next(s for s in steps if s["round"] == p["round"])["reference_kl"][
                "adjustment_abs_mean"
            ]
            / next(s for s in steps if s["round"] == p["round"])["reference_kl"][
                "task_advantage_abs_token_mean"
            ]
            for p in pools
        ],
        "KL adjustment relative to task advantage",
        "Mean absolute magnitude ratio (%)",
    )
    line(
        ax[4],
        x,
        [s["policy_metrics"]["current_over_old_ratio"]["mean"] for s in steps],
        "Current / rollout probability ratio",
        "Mean ratio",
    )
    ax[4].axhline(1, color="#888888", linestyle="--", linewidth=1)
    ax[4].ticklabel_format(axis="y", style="plain", useOffset=False)
    line(
        ax[5],
        x,
        [
            100 * s["policy_metrics"]["objective_clipped_learning_tokens"]["value"]
            for s in steps
        ],
        "PPO clipping on replayed completion tokens",
        "Tokens clipped (%)",
    )
    save(
        fig,
        "training-diagnostics",
        "KL is a sampled diagnostic · ratios exclude masked prompt tokens · filtering is based on final training reward",
    )

    if local:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        task = [p["credit"]["task_token_advantage"] for p in pools]
        line(axes[0,0], xp, [d["mean"] for d in task],
             "Local task coefficient: token-weighted mean", "Mean coefficient")
        rms = [(d["mean"]**2 + (d["std"] or 0)**2 * (d["n"]-1)/d["n"])**.5 for d in task]
        line(axes[0,1], xp, rms, "Local task coefficient: RMS", "RMS (no scaling applied)")
        line(axes[1,0], xp, [100*p["rates"]["groups_with_signal"]["value"] for p in pools],
             "Groups with nonzero local task signal", "Groups (%)")
        line(axes[1,1], xp, [p["lengths"]["output_tokens"]["mean"] for p in pools],
             "Generated trajectory length", "Tokens / trajectory")
        save(fig, "local-credit-diagnostics", "Task coefficients before KL; same training schedule, fresh trajectories each pool")

    summary = {
        "optimizer_steps": state["step"],
        "training_method": arm,
        "rollout_pools": len(pools),
        "questions": sum(p["group_count"] for p in pools),
        "trajectories": sum(p["episode_count"] for p in pools),
        "full_pool_em_mean": float(np.mean(em)) / 100,
        "full_pool_em_first": em[0] / 100,
        "full_pool_em_last": em[-1] / 100,
        "full_pool_em_min": min(em) / 100,
        "full_pool_em_max": max(em) / 100,
        "loss_range": [min(s["loss"] for s in steps), max(s["loss"] for s in steps)],
        "gradient_range": [
            min(s["gradient_norm"] for s in steps),
            max(s["gradient_norm"] for s in steps),
        ],
        "retained_groups": sum(s["groups"] for s in steps),
        "original_groups": sum(
            s["minibatch_filtering"]["original_groups"] for s in steps
        ),
        "checkpoint": state["checkpoint"],
        "qualification": "Different training questions each pool; no held-out or final-policy rollout after the last update. No causal learning claim from endpoint differences.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(
        """# Outcome training curves

`training-curves.png/pdf`: full-pool EM, filtered reward, policy loss, gradients,
search/generation turns and complete gold-document coverage.

`training-diagnostics.png/pdf`: output length, group retention by difficulty,
reference sampled log-ratio, relative KL adjustment, importance ratios and clipping.

Full-pool points appear at the policy step used to generate that pool (0,2,...,18).
The final update has optimizer metrics but no post-training evaluation or new rollout.
Filtered mean rewards are selection-conditioned and not representative accuracy.
Loss is a clipped surrogate, not answer error; lower loss alone does not imply
better QA. Exact reference KL and full-vocabulary entropy are not measured.
Tinker service-wide PPO metrics can include masked positions; plots use local
completion-only ratios and clipping. The raw records remain unchanged.
No model/search calls were made to generate these figures.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
