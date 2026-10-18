"""Offline rollout latency plots. No Modal, model, or training execution."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_training(root):
    # Committed steps only, deduplicated: optimizer reuse must not duplicate samples.
    steps = [json.loads(p.read_text()) for p in (root / "steps").glob("*.json")]
    ids = {step["rollout_id"] for step in steps}
    records = [
        json.loads((root / "rollouts" / f"{key}.json").read_text()) for key in ids
    ]
    rows, batches = [], []
    for record in sorted(records, key=lambda r: r["round"]):
        key = record["rollout_id"]
        peers = record["episodes"]
        assert len({e["episode_id"] for e in peers}) == len(peers)
        for episode in peers:
            if episode["seconds"] is None:
                raise ValueError("Missing observed completion time")
            rows.append(
                {
                    "cohort": "BF16 training pilot",
                    "batch_id": key,
                    "episode_id": episode["episode_id"],
                    "question_id": episode["id"],
                    "turns": episode["generations"],
                    "completion_seconds": episode["seconds"],
                    "output_tokens": episode["output_tokens"],
                    "correct": int(bool(episode["nm"])),
                    "status": episode["status"],
                    "length_truncated_turns": episode["length_truncated_generations"],
                }
            )
        times = sorted(e["seconds"] for e in peers)
        half_time = times[(len(times) + 1) // 2 - 1]
        batches.append(
            {
                "cohort": "BF16 training pilot",
                "batch_id": key,
                "round": record["round"],
                "trajectories": len(peers),
                "max_turns": max(e["generations"] for e in peers),
                "last_completion_seconds": times[-1],
                "half_complete_seconds": half_time,
                "tail_after_half_seconds": times[-1] - half_time,
                "output_tokens": sum(e["output_tokens"] for e in peers),
            }
        )
    return rows, batches


def load_cache(root):
    rows, waves, batches, token_lengths = [], [], [], []
    for arm in ("cache-off", "cache-on"):
        scores = {
            r["episode_id"]: float(r["reward"])
            for r in csv.DictReader(
                (root / arm / "episodes.csv").open(encoding="utf-8-sig")
            )
        }
        for path in sorted((root / arm).glob("batch-*.json")):
            record = json.loads(path.read_text())
            key = f"{arm} / batch {record['batch'] + 1}"
            episodes = [e for group in record["groups"] for e in group["episodes"]]
            for episode in episodes:
                audit = json.loads(episode["controller_audit_json"])
                lengths = [len(g["completion_ids"]) for g in episode["generations"]]
                token_lengths.extend(lengths)
                rows.append(
                    {
                        "cohort": f"FP8 {arm}",
                        "batch_id": key,
                        "episode_id": episode["episode_id"],
                        "question_id": episode["question"]["id"],
                        "turns": len(lengths),
                        "completion_seconds": audit["collection_seconds"],
                        "output_tokens": sum(lengths),
                        "correct": int(scores[episode["episode_id"]] == 1),
                        "status": episode["status"],
                        "length_truncated_turns": sum(
                            g["finish_reason"] == "length"
                            for g in episode["generations"]
                        ),
                    }
                )
            current = []
            for index, wave in enumerate(record["generation_waves"], 1):
                turns = {r["turn"] for r in wave["cache_stats"]}
                assert turns == {index}, (
                    "Do not equate generation-wave index with turn without checking"
                )
                row = {
                    "batch_id": key,
                    "turn": index,
                    "active": wave["requests"],
                    "batch_size": len(episodes),
                    "seconds": wave["seconds"],
                    "generated_tokens": wave["generated_tokens"],
                }
                waves.append(row)
                current.append(row)
            generation = sum(w["seconds"] for w in current)
            batches.append(
                {
                    "batch_id": key,
                    "trajectories": len(episodes),
                    "rollout_wall_seconds": record["rollout_wall_seconds"],
                    "generation_wave_seconds": generation,
                    "time_after_turn_2_seconds": sum(
                        w["seconds"] for w in current if w["turn"] > 2
                    ),
                    "time_at_one_active_seconds": sum(
                        w["seconds"] for w in current if w["active"] == 1
                    ),
                    "time_at_half_or_less_seconds": sum(
                        w["seconds"]
                        for w in current
                        if w["active"] <= len(episodes) / 2
                    ),
                }
            )
    return rows, waves, batches, token_lengths


def ranks(values):
    ordered = sorted(set(values))
    return (
        [
            sum(i for i, v in enumerate(sorted(values)) if v == value)
            / values.count(value)
            for value in values
        ]
        if ordered
        else []
    )


def spearman(x, y):
    import numpy as np

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def plot(training, batches, waves, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "savefig.facecolor": "white",
        }
    )
    turns = np.array([r["turns"] for r in training])
    seconds = np.array([r["completion_seconds"] for r in training])
    tokens = np.array([r["output_tokens"] for r in training])
    jitter = np.array(
        [
            (
                int(hashlib.sha256(r["episode_id"].encode()).hexdigest()[:4], 16)
                / 65535
                - 0.5
            )
            * 0.2
            for r in training
        ]
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), layout="constrained")
    fig.suptitle(
        f"Long rollouts and batch tails\nBF16 GRPO pilot: {len(training)} trajectories, {len(batches)} fresh batches",
        fontsize=17,
        fontweight="bold",
    )
    ax = axes[0, 0]
    ax.scatter(
        turns + jitter, seconds, s=30, alpha=0.45, color="#3165a5", edgecolor="none"
    )
    unique = sorted(set(turns))
    med = [np.median(seconds[turns == t]) for t in unique]
    q1 = [np.quantile(seconds[turns == t], 0.25) for t in unique]
    q3 = [np.quantile(seconds[turns == t], 0.75) for t in unique]
    ax.errorbar(
        unique,
        med,
        yerr=[np.array(med) - q1, np.array(q3) - med],
        color="#ce6b16",
        marker="s",
        capsize=4,
        linewidth=2,
        label="Median and middle 50%",
    )
    ax.set(
        title="A. More turns, longer observed completion",
        xlabel="Generation turns (includes the final answer)",
        ylabel="Observed completion latency (s)",
        xticks=unique,
        ylim=(0, max(seconds) * 1.08),
    )
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[0, 1]
    bins = np.arange(0, max(seconds) + 25, 20)
    ax.hist(
        [
            seconds[turns == 2],
            seconds[(turns >= 3) & (turns <= 4)],
            seconds[turns >= 5],
        ],
        bins=bins,
        stacked=True,
        color=["#3165a5", "#56a89c", "#df8a42"],
        label=["2 turns", "3–4 turns", "5–8 turns"],
        edgecolor="white",
    )
    ax.set(
        title="B. Distribution of observed completion times",
        xlabel="Observed completion latency (s)",
        ylabel="Trajectories",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    bx = [b["max_turns"] for b in batches]
    by = [b["last_completion_seconds"] for b in batches]
    ax.scatter(bx, by, s=65, color="#ce6b16", edgecolor="white")
    ax.set(
        title="C. Longest trajectory versus batch completion",
        xlabel="Maximum generation turns in the batch",
        ylabel="Last observed completion in batch (s)",
        xticks=unique,
    )
    ax.text(
        0.03,
        0.95,
        f"Spearman ρ = {spearman(bx, by):.2f}; n = {len(batches)} batches",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    ax = axes[1, 1]
    ax.scatter(tokens, seconds, s=30, color="#258d79", alpha=0.5, edgecolor="none")
    ax.set(
        title="D. Generated token volume also matters",
        xlabel="Generated tokens across the trajectory",
        ylabel="Observed completion latency (s)",
    )
    ax.text(
        0.03,
        0.95,
        f"Spearman ρ = {spearman(list(tokens), list(seconds)):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    for ax in axes.flat:
        ax.grid(alpha=0.16)
        ax.set_axisbelow(True)
    fig.supxlabel(
        "Completion times include shared generation/search waits; these are not isolated per-request GPU timings.",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"turns_and_latency.{suffix}", dpi=180)
    plt.close(fig)

    keys = list(dict.fromkeys(w["batch_id"] for w in waves))
    colors = {8: "#3165a5", 5: "#319c94", 4: "#df9c37", 1: "#c85555"}
    fig, ax = plt.subplots(figsize=(13, 5.7), layout="constrained")
    for y, key in enumerate(keys):
        start = 0
        for wave in [w for w in waves if w["batch_id"] == key]:
            ax.barh(
                y,
                wave["seconds"],
                left=start,
                color=colors[wave["active"]],
                height=0.62,
                edgecolor="white",
            )
            ax.text(
                start + wave["seconds"] / 2,
                y,
                f"Turn {wave['turn']}\n{wave['active']} active\n{wave['seconds']:.1f} s",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
            start += wave["seconds"]
        ax.text(start + 1, y, f"{start:.1f} s", va="center", fontsize=10)
    ax.set(
        yticks=range(len(keys)),
        yticklabels=keys,
        xlabel="Cumulative generation-wave wall time (s); search waits excluded",
    )
    ax.invert_yaxis()
    ax.set_xlim(
        0,
        max(sum(w["seconds"] for w in waves if w["batch_id"] == key) for key in keys)
        + 10,
    )
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(color=colors[n], label=f"{n} active trajectories")
            for n in (8, 5, 4, 1)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "The batch tail: fewer active trajectories, but generation continues",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_title(
        "Actual GRPO collector · FP8 · Q=2, G=4 · four measured batches / 32 trajectories",
        fontsize=11,
        fontweight="normal",
        pad=18,
    )
    fig.supxlabel(
        "Wave duration includes prefill, decode and backend processing. Active-request count is not GPU utilization.",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"batch_tail.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("data/grpo/grpo-pilot-005-answer-only/observability"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/grpo/cache-pilots/16f6b04668c0")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/latency-analysis"))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    training, batches = load_training(args.training_root)
    recent, waves, recent_batches, lengths = load_cache(args.cache_root)
    assert len({(r["batch_id"], r["episode_id"]) for r in training}) == len(training)
    dump_csv(output / "trajectories.csv", training + recent)
    dump_csv(output / "training_batches.csv", batches)
    dump_csv(output / "generation_waves.csv", waves)
    dump_csv(output / "recent_batches.csv", recent_batches)
    total_correct = sum(r["correct"] for r in training)
    cap_rows = [
        {
            "max_turns": cap,
            "trajectories_finishing_within_cap": sum(
                r["turns"] <= cap for r in training
            ),
            "correct_retained": sum(
                r["correct"] for r in training if r["turns"] <= cap
            ),
            "baseline_correct": total_correct,
            "original_correct_retained_fraction": sum(
                r["correct"] for r in training if r["turns"] <= cap
            )
            / total_correct,
            "accuracy_if_later_answers_are_zero": sum(
                r["correct"] for r in training if r["turns"] <= cap
            )
            / len(training),
        }
        for cap in range(2, 9)
    ]
    dump_csv(output / "turn_cap_tradeoffs.csv", cap_rows)
    plot(training, batches, waves, output)
    generation = sum(w["seconds"] for w in waves)
    by_turn = []
    for turn in sorted({r["turns"] for r in training}):
        peers = [r for r in training if r["turns"] == turn]
        by_turn.append(
            {
                "turns": turn,
                "n": len(peers),
                "median_completion_seconds": statistics.median(
                    r["completion_seconds"] for r in peers
                ),
                "correct": sum(r["correct"] for r in peers),
                "median_output_tokens": statistics.median(
                    r["output_tokens"] for r in peers
                ),
            }
        )
    summary = {
        "sources": {
            "training_root": str(args.training_root),
            "cache_root": str(args.cache_root),
        },
        "training": {
            "trajectories": len(training),
            "fresh_batches": len(batches),
            "correct": total_correct,
            "length_truncated_episodes": sum(
                r["length_truncated_turns"] > 0 for r in training
            ),
            "length_truncated_turns": sum(
                r["length_truncated_turns"] for r in training
            ),
            "generation_turns": sum(r["turns"] for r in training),
            "by_turn": by_turn,
            "turn_time_spearman": spearman(
                [r["turns"] for r in training],
                [r["completion_seconds"] for r in training],
            ),
            "max_turn_batch_time_spearman": spearman(
                [b["max_turns"] for b in batches],
                [b["last_completion_seconds"] for b in batches],
            ),
            "median_tail_after_half_complete_seconds": statistics.median(
                b["tail_after_half_seconds"] for b in batches
            ),
            "median_completion_seconds": statistics.median(
                r["completion_seconds"] for r in training
            ),
        },
        "recent_waves": {
            "trajectories": len(recent),
            "batches": len(recent_batches),
            "generation_wave_seconds": generation,
            "time_with_one_active_fraction": sum(
                w["seconds"] for w in waves if w["active"] == 1
            )
            / generation,
            "time_with_half_or_less_active_fraction": sum(
                w["seconds"] for w in waves if w["active"] <= w["batch_size"] / 2
            )
            / generation,
            "time_after_turn_2_fraction": sum(
                w["seconds"] for w in waves if w["turn"] > 2
            )
            / generation,
            "per_turn_tokens": {
                "median": statistics.median(lengths),
                "max": max(lengths),
                "over_256": sum(n > 256 for n in lengths),
                "over_512": sum(n > 512 for n in lengths),
                "turns": len(lengths),
            },
        },
        "cap_tradeoffs": cap_rows,
        "definitions": {
            "turn": "One sampled generation/action, including search, final answer and format repair.",
            "completion_seconds": "Elapsed collection-attempt time when the trajectory was recorded complete, including shared batch barriers and search waits.",
            "batch_completion": "Last recorded trajectory completion; a proxy for batch end, excluding subsequent training/replay.",
            "generation_wave_seconds": "One shared backend.generate call, including prefill, decode and CPU output processing; counted once.",
            "cap_tradeoffs": "Retrospective terminal hard-stop calculation, with later answers zero; not a rerun with shorter prompts or token caps.",
        },
        "limitations": [
            "Timings correlate within batches; descriptive correlations are not causal or independent-sample significance tests.",
            "Historical BF16 and recent FP8 timings are plotted separately; do not pool throughput across configurations.",
            "No token-level or GPU-kernel timing is available, so decode-only bottleneck is not proven.",
            "No GPU launched and no training limits modified.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
