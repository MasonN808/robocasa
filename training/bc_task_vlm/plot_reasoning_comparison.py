"""Compare v1 SFT against v1.5 per-step reasoning supervision.

Writes ``reasoning_comparison.csv`` plus PNG/PDF figures under ``eval_runs``.
The v1.5 run used a fresh stratified manifest, so the figure deliberately calls
the v1 result an aggregate reference rather than implying a paired ablation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BASE = Path("training/bc_task_vlm/eval_runs")
SPLITS = (
    ("heldout_trajectories", "Held-out trajectories\n(trained tasks)"),
    ("heldout_tasks", "Held-out tasks\n(never trained on)"),
)
METRICS = (
    ("action", "Action\nexact"),
    ("comm", "Communication\njudged"),
    ("judged", "Judged\noverall"),
    ("traj", "Full\ntrajectory"),
)

# The native-format v1 runs were produced on another box and their full result
# directories are not always present locally. These are the published values
# already embedded in artifact_results_explorer.html and documented in
# EXPERIMENT.md; local run files take precedence whenever available.
PUBLISHED_V1_REFERENCE = {
    "heldout_trajectories": {
        "n": 1152,
        "action": 0.984,
        "comm": 0.717,
        "judged": 0.878,
        "traj": 0.173,
    },
    "heldout_tasks": {
        "n": 1097,
        "action": 0.675,
        "comm": 0.674,
        "judged": 0.675,
        "traj": 0.253,
    },
}


def load_run(run_dir: Path) -> dict[str, float]:
    metrics = json.loads(
        (run_dir / "structured_eval_metrics.json").read_text(encoding="utf-8")
    )
    judged = json.loads(
        (run_dir / "comm_judge_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "n": int(metrics["structured_eval_num_samples"]),
        "action": judged["action_exact_call_accuracy"],
        "comm": judged["comm_judged_match_rate"],
        "judged": judged["judged_exact_call_accuracy"],
        "traj": judged["judged_trajectory_all_steps_rate"],
    }


def collect() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    for split, _ in SPLITS:
        baseline_dir = BASE / f"qwen3vl_8b_sft_nativefmt__{split}"
        baseline = (
            load_run(baseline_dir)
            if (baseline_dir / "comm_judge_metrics.json").exists()
            else PUBLISHED_V1_REFERENCE[split]
        )
        reasoning = load_run(BASE / f"qwen3vl_8b_v1_5_reasoning__{split}")
        data[split] = {"v1": baseline, "v1.5": reasoning}
    return data


def write_csv(data: dict[str, dict[str, dict[str, float]]]) -> None:
    path = BASE / "reasoning_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "configuration", "n", "action_exact", "comm_judged", "judged_overall", "full_trajectory"]
        )
        for split, _ in SPLITS:
            for config in ("v1", "v1.5"):
                row = data[split][config]
                writer.writerow(
                    [split, config, row["n"], row["action"], row["comm"], row["judged"], row["traj"]]
                )
    print(f"wrote {path}")


def plot(data: dict[str, dict[str, dict[str, float]]]) -> None:
    ink, muted, grid = "#37352f", "#787066", "#e6e4dd"
    colors = {"v1": "#2a78d6", "v1.5": "#1baf7a"}
    labels = {"v1": "v1 SFT (no rationale)", "v1.5": "v1.5 SFT (+ rationale)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    bar_width = 0.34

    for ax, (split, title) in zip(axes, SPLITS, strict=True):
        xs = list(range(len(METRICS)))
        for index, config in enumerate(("v1", "v1.5")):
            values = [data[split][config][key] for key, _ in METRICS]
            positions = [x + (index - 0.5) * bar_width for x in xs]
            bars = ax.bar(
                positions,
                values,
                width=bar_width - 0.03,
                color=colors[config],
                zorder=3,
            )
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(
                    f"{value:.3f}".lstrip("0"),
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8.5,
                    color=ink,
                )

        for x, (key, _) in zip(xs, METRICS, strict=True):
            delta = data[split]["v1.5"][key] - data[split]["v1"][key]
            top = max(data[split]["v1"][key], data[split]["v1.5"][key])
            ax.annotate(
                f"{delta:+.3f}",
                (x, min(top + 0.105, 1.025)),
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1b7f59" if delta >= 0 else "#c2410c",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in METRICS], fontsize=9.5)
        ax.set_title(title, fontsize=11.5, color=ink)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=muted)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(grid)
        ax.spines["bottom"].set_color(grid)

    axes[0].set_ylabel("Rate (0–1)", fontsize=10.5, color=ink)
    fig.legend(
        handles=[Patch(facecolor=colors[key], label=labels[key]) for key in ("v1", "v1.5")],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle(
        "Reasoning supervision: small step gains, mixed trajectory transfer",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.025,
        "Deltas are v1.5 − v1. Aggregate reference only: v1.5 used a fresh stratified 75-trajectory manifest, not paired trajectory IDs.",
        ha="center",
        fontsize=9,
        color=muted,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.88])
    for suffix in ("png", "pdf"):
        fig.savefig(BASE / f"reasoning_comparison.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote reasoning_comparison.png/pdf")


def main() -> None:
    data = collect()
    write_csv(data)
    plot(data)


if __name__ == "__main__":
    main()
