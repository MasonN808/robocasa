"""Aggregates standalone eval runs into the SFT-necessity comparison figure.

Reads structured_eval_metrics.json (+ predictions jsonl for per-tool rates)
from one eval run dir per (model, split) and emits:

- model_comparison.csv — one row per (model, split) with the headline metrics
- per_tool_accuracy.csv — tool-name accuracy per (model, split, target tool)
- model_comparison.png / .pdf — grouped bars per split, Wilson 95% CIs

Usage:
    python -m training.bc_task_vlm.plot_model_comparison \
        --run "Gemini-3 Flash (no SFT)=heldout_trajectories=eval_runs/gemini__ht" \
        --run "Gemini-3 Flash (no SFT)=heldout_tasks=eval_runs/gemini__hk" \
        --run "Qwen3.6-27B (no SFT)=heldout_trajectories=eval_runs/qwen_base__ht" \
        ... \
        --output-dir training/bc_task_vlm/eval_runs/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from training.bc_task_vlm.evaluation import load_prediction_records

# Pre-validated reference categorical palette (dataviz skill), slots 1-3 in
# fixed order: one hue per metric series across every panel.
METRIC_SERIES = (
    ("structured_eval_tool_name_accuracy", "Tool-name accuracy", "#2a78d6"),
    ("structured_eval_exact_tool_call_accuracy", "Exact tool-call accuracy", "#1baf7a"),
    (
        "structured_eval_trajectory_all_steps_correct_rate",
        "Trajectory all-steps correct",
        "#eda100",
    ),
)
CSV_METRICS = tuple(key for key, _, _ in METRIC_SERIES) + (
    "structured_eval_tool_call_parse_rate",
    "structured_eval_tool_call_valid_rate",
    "structured_eval_exact_args_match_rate",
    "structured_eval_trajectory_mean_correct_prefix_fraction",
    "structured_eval_num_samples",
    "structured_eval_num_trajectories",
)
SPLIT_TITLES = {
    "heldout_trajectories": "Held-out trajectories (train tasks)",
    "heldout_tasks": "Held-out tasks (never seen in SFT)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=SPLIT=DIR",
        help="Eval run as <model label>=<split name>=<run dir>; repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def wilson_interval(successes: float, total: float) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate."""

    if total <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_runs(specs: list[str]) -> list[dict[str, Any]]:
    runs = []
    for spec in specs:
        label, split, run_dir = spec.split("=", 2)
        run_path = Path(run_dir)
        metrics = json.loads(
            (run_path / "structured_eval_metrics.json").read_text(encoding="utf-8")
        )
        runs.append(
            {
                "label": label,
                "split": split,
                "dir": run_path,
                "metrics": metrics,
            }
        )
    return runs


def per_tool_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        records = load_prediction_records(run["dir"] / "structured_eval_predictions.jsonl")
        totals: Counter[str] = Counter()
        correct: Counter[str] = Counter()
        for record in records:
            tool = (record.get("target_tool_call") or {}).get("name", "unknown")
            totals[tool] += 1
            correct[tool] += bool(record["exact_tool_match"])
        for tool in sorted(totals):
            rows.append(
                {
                    "model": run["label"],
                    "split": run["split"],
                    "target_tool": tool,
                    "n": totals[tool],
                    "tool_name_accuracy": correct[tool] / totals[tool],
                }
            )
    return rows


def denominator_for(metric_key: str, metrics: dict[str, float]) -> float:
    if metric_key.startswith("structured_eval_trajectory_"):
        return metrics.get("structured_eval_num_trajectories", 0.0)
    return metrics.get("structured_eval_num_samples", 0.0)


def plot(runs: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted({run["split"] for run in runs})
    labels = list(dict.fromkeys(run["label"] for run in runs))

    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(6.4 * len(splits), 4.6),
        sharey=True,
    )
    if len(splits) == 1:
        axes = [axes]

    bar_width = 0.24
    for ax, split in zip(axes, splits, strict=True):
        split_runs = {run["label"]: run for run in runs if run["split"] == split}
        for series_index, (metric_key, metric_label, color) in enumerate(
            METRIC_SERIES
        ):
            xs, values, err_low, err_high = [], [], [], []
            for label_index, label in enumerate(labels):
                run = split_runs.get(label)
                if run is None:
                    continue
                metrics = run["metrics"]
                value = metrics.get(metric_key, 0.0)
                total = denominator_for(metric_key, metrics)
                low, high = wilson_interval(value * total, total)
                xs.append(label_index + (series_index - 1) * bar_width)
                values.append(value)
                err_low.append(value - low)
                err_high.append(high - value)
            bars = ax.bar(
                xs,
                values,
                width=bar_width - 0.02,
                color=color,
                label=metric_label,
                zorder=3,
            )
            ax.errorbar(
                xs,
                values,
                yerr=[err_low, err_high],
                fmt="none",
                ecolor="#57564d",
                elinewidth=1,
                capsize=2,
                zorder=4,
            )
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    fontsize=8,
                    color="#37352f",
                )

        sample_counts = {
            label: int(
                split_runs[label]["metrics"].get("structured_eval_num_samples", 0)
            )
            for label in labels
            if label in split_runs
        }
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [
                f"{label}\n(n={sample_counts.get(label, 0)})"
                for label in labels
            ],
            fontsize=9,
        )
        ax.set_title(SPLIT_TITLES.get(split, split), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, color="#e6e4dd", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_ylabel("Rate (per-step teacher-forced eval)", fontsize=10)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Tool-calling accuracy: SFT vs zero-shot task VLMs",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"model_comparison.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run)

    csv_path = args.output_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "split"] + list(CSV_METRICS))
        for run in runs:
            writer.writerow(
                [run["label"], run["split"]]
                + [run["metrics"].get(key, "") for key in CSV_METRICS]
            )

    tool_rows = per_tool_rows(runs)
    per_tool_path = args.output_dir / "per_tool_accuracy.csv"
    with per_tool_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "split", "target_tool", "n", "tool_name_accuracy"],
        )
        writer.writeheader()
        writer.writerows(tool_rows)

    plot(runs, args.output_dir)
    print(f"Wrote {csv_path}, {per_tool_path}, model_comparison.png/pdf")


if __name__ == "__main__":
    main()
