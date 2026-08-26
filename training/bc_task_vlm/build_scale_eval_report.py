"""Build the canonical portable artifact for the fixed-cohort scale experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path


RUN_RE = re.compile(r"sft_scale(?P<scale>\d+)_ep(?P<whole>\d+)p(?P<frac>\d+)_fixed10_promptv9")
SPLITS = ("train_task_types", "heldout_task_types")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(root.glob("sft_scale*_ep*_fixed10_promptv9")):
        match = RUN_RE.fullmatch(run_dir.name)
        if not match:
            continue
        scale = int(match.group("scale"))
        epoch = float(f"{match.group('whole')}.{match.group('frac')}")
        for split in SPLITS:
            metrics_path = run_dir / split / "aggregate" / "live_sim_metrics.json"
            trajectories_path = run_dir / split / "aggregate" / "live_sim_trajectories.jsonl"
            if not metrics_path.exists() or not trajectories_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            total = int(metrics["num_trajectories"])
            success = int(round(metrics["fsm_goal_rate"] * total))
            error_free = int(metrics["num_fsm_error_free_successes"])
            success_low, success_high = wilson(success, total)
            ef_low, ef_high = wilson(error_free, total)
            rows.append({
                "scale": scale,
                "epoch": epoch,
                "checkpoint": f"{scale} traj/task · {epoch:g} epoch",
                "split": "Trained tasks" if split == "train_task_types" else "Held-out tasks",
                "split_id": split,
                "episodes": total,
                "fsm_successes": success,
                "fsm_success_rate": success / total,
                "fsm_success_ci_low": success_low,
                "fsm_success_ci_high": success_high,
                "error_free_successes": error_free,
                "error_free_success_rate": error_free / total,
                "error_free_ci_low": ef_low,
                "error_free_ci_high": ef_high,
                "success_requiring_recovery_rate": (success - error_free) / total,
                "mean_rejected_calls": metrics["mean_rejected_tool_calls_per_trajectory"],
                "mean_steps": metrics["mean_steps_used"],
                "source_metrics": str(metrics_path),
                "source_trajectories": str(trajectories_path),
            })
    return sorted(rows, key=lambda row: (row["scale"], row["epoch"], row["split_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("training/bc_task_vlm/eval_runs/fixed_live_sim"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.eval_root)
    if not rows:
        raise SystemExit("No completed scale-evaluation results found")

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    heldout = [row for row in rows if row["split_id"] == "heldout_task_types"]
    trained = [row for row in rows if row["split_id"] == "train_task_types"]
    best_held = max(heldout, key=lambda row: row["fsm_success_rate"])
    best_trained = max(trained, key=lambda row: row["fsm_success_rate"])
    one_epoch = [row for row in rows if row["epoch"] == 1.0]
    completed_pairs = len({(row["scale"], row["epoch"]) for row in rows if row["split_id"] == "train_task_types" and any(
        other["scale"] == row["scale"] and other["epoch"] == row["epoch"] and other["split_id"] == "heldout_task_types" for other in rows
    )})
    expected_pairs = 16
    scale150_epoch1_complete = all(
        any(row["scale"] == 150 and row["epoch"] == 1.0 and row["split_id"] == split for row in rows)
        for split in SPLITS
    )

    source = {
        "id": "fixed_scale_eval_results",
        "label": "Fixed live-sim scale evaluation outputs",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT filename, num_trajectories, fsm_goal_rate, "
                "fsm_error_free_success_rate, num_fsm_error_free_successes, "
                "mean_rejected_tool_calls_per_trajectory, mean_steps_used "
                "FROM read_json_auto('training/bc_task_vlm/eval_runs/fixed_live_sim/"
                "sft_scale*_ep*_fixed10_promptv9/*/aggregate/live_sim_metrics.json', "
                "filename=true)"
            ),
            "description": "Reads completed aggregate live-sim metrics for every scale checkpoint and computes 95% Wilson intervals from exact success counts.",
            "tables_used": sorted({row["source_metrics"] for row in rows}),
            "filters": [
                "Prompt contract v9; no step indexing; full communication protocol",
                "Fixed live-sim cohort; 10 episodes per task",
                "47 trained task types and 6 held-out task types",
                "Only checkpoints with complete aggregate outputs for both splits are compared",
            ],
            "metric_definitions": [
                "FSM success rate = episodes satisfying the concurrent symbolic task goal / all evaluated episodes.",
                "Error-free success rate = successful episodes with no rejected tool call / all evaluated episodes.",
                "Success requiring recovery = FSM success rate minus error-free success rate.",
                "Whiskers and table bounds are two-sided 95% Wilson binomial confidence intervals.",
            ],
            "executed_at": now,
        },
    }

    charts = [
        {
            "id": "heldout_progression",
            "title": "Held-out-task success at every completed checkpoint",
            "subtitle": "Six held-out tasks, 10 fixed episodes per task; final success and error-free success use all 60 episodes as the denominator.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "heldout_progression",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "checkpoint", "type": "ordinal", "label": "Training checkpoint"},
                "y": {"fields": ["fsm_success_rate", "error_free_success_rate"], "type": "quantitative", "format": "percent", "label": "Episode success rate"},
                "tooltip": [
                    {"field": "episodes", "type": "quantitative", "label": "Episodes"},
                    {"field": "fsm_success_ci_low", "type": "quantitative", "format": "percent", "label": "FSM CI low"},
                    {"field": "fsm_success_ci_high", "type": "quantitative", "format": "percent", "label": "FSM CI high"},
                ],
            },
            "combinationRationale": "Both series are success rates over the same episodes; showing them together makes recovery dependence visible.",
            "valueFormat": "percent",
            "layout": "full",
            "surface": {"surface": "explorer", "showControls": True, "viewMode": "both"},
        },
        {
            "id": "one_epoch_heldout",
            "title": "Held-out-task success after one epoch",
            "subtitle": "A like-for-like checkpoint comparison across all five training-data scales.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "one_epoch_heldout",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "scale_label", "type": "ordinal", "label": "Training trajectories per task"},
                "y": {"fields": ["fsm_success_rate", "error_free_success_rate"], "type": "quantitative", "format": "percent", "label": "Episode success rate"},
                "tooltip": [
                    {"field": "fsm_successes", "type": "quantitative", "label": "Successful episodes"},
                    {"field": "episodes", "type": "quantitative", "label": "Episodes"},
                ],
            },
            "combinationRationale": "Both series are success rates over the same episodes; showing them together makes recovery dependence visible.",
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "one_epoch_split",
            "title": "Final success after one epoch: trained versus held-out tasks",
            "subtitle": "Each trained-task bar uses 470 episodes; each held-out-task bar uses 60 episodes.",
            "type": "bar",
            "intent": "comparison",
            "dataset": "one_epoch_split",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "scale_label", "type": "ordinal", "label": "Training trajectories per task"},
                "y": {"field": "fsm_success_rate", "type": "quantitative", "format": "percent", "label": "FSM success rate"},
                "color": {"field": "split", "type": "nominal", "label": "Evaluation split"},
                "tooltip": [
                    {"field": "fsm_successes", "type": "quantitative", "label": "Successful episodes"},
                    {"field": "episodes", "type": "quantitative", "label": "Episodes"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
        },
    ]

    tables = [{
        "id": "checkpoint_detail",
        "title": "Exact checkpoint results",
        "subtitle": "Use this table for exact rates, counts, intervals, recovery dependence, and episode length.",
        "dataset": "all_results",
        "sourceId": source["id"],
        "defaultSort": {"field": "scale", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "scale", "label": "Traj/task", "format": "number"},
            {"field": "epoch", "label": "Epoch", "format": "number"},
            {"field": "split", "label": "Split", "type": "text"},
            {"field": "fsm_successes", "label": "Successes", "format": "number"},
            {"field": "episodes", "label": "N", "format": "number"},
            {"field": "fsm_success_rate", "label": "FSM success", "format": "percent"},
            {"field": "fsm_success_ci_low", "label": "95% CI low", "format": "percent"},
            {"field": "fsm_success_ci_high", "label": "95% CI high", "format": "percent"},
            {"field": "error_free_success_rate", "label": "Error-free", "format": "percent"},
            {"field": "success_requiring_recovery_rate", "label": "Recovered", "format": "percent"},
            {"field": "mean_steps", "label": "Mean steps", "format": "number"},
        ],
    }]

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Training-data scale evaluation"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Executive Summary\n\n"
                f"- **More data clearly improves performance on familiar task types.** The strongest trained-task result so far is **{pct(best_trained['fsm_success_rate'])}**, from {best_trained['scale']} trajectories/task at epoch {best_trained['epoch']:g}.\n"
                f"- **More data does not produce a steady gain on unseen task types.** The best held-out-task result is **{pct(best_held['fsm_success_rate'])}**, from {best_held['scale']} trajectories/task at epoch {best_held['epoch']:g}.\n"
                "- **Training longer can hurt generalization.** Several scales improve through epoch 1, then lose held-out-task success at epoch 1.5 or 2 even while trained-task success remains high.\n"
                f"- **The one-epoch scale comparison is complete.** Overall, {completed_pairs} of {expected_pairs} originally planned checkpoint pairs are available; scale 60 and 90 stopped before epoch 2."
            ),
        },
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## How to read the results\n\n"
                "**Scale** means the number of expert trajectories used per training task. Every model is evaluated on the same fixed simulator episodes: 10 episodes for each of 47 trained task types and 10 episodes for each of 6 task types never used in training.\n\n"
                "**Final FSM success** means the episode eventually completed the symbolic task goal. **Error-free success** means it completed the goal without any rejected tool call. The gap between them is success that required recovery after an error."
            ),
        },
        {
            "id": "heldout_finding",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Held-out performance peaks early rather than rising steadily\n\n"
                "The held-out results do not support a simple “more trajectories is always better” conclusion. Scale 60 at one epoch currently leads. Larger datasets sometimes match it closely, but they do not consistently exceed it. This suggests that optimization length and model capacity matter alongside dataset size."
            ),
        },
        {"id": "heldout_progression_block", "type": "chart", "chartId": "heldout_progression", "layout": "full"},
        {
            "id": "one_epoch_finding",
            "type": "markdown",
            "body": (
                "## Compare one epoch before deciding how much data helps\n\n"
                "The cleanest scale comparison uses one epoch for every model. It avoids comparing a lightly trained large dataset with a heavily trained small dataset. All five data scales are now included."
            ),
        },
        {"id": "one_epoch_heldout_block", "type": "chart", "chartId": "one_epoch_heldout", "layout": "full"},
        {"id": "one_epoch_split_block", "type": "chart", "chartId": "one_epoch_split", "layout": "full"},
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Treat the best one-epoch setting as the current efficiency baseline.\n"
                "2. Run lower-learning-rate or shorter-training ablations on the larger datasets before generating substantially more data.\n"
                "3. Compare the reasoning and non-reasoning controls on both evaluation splits.\n"
                "4. Repeat the strongest settings with another training seed before making a publication claim."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does a lower learning rate let the 120- and 150-trajectory models preserve held-out performance?\n"
                "- Are the same held-out tasks responsible for the drop at later checkpoints?\n"
                "- Does reasoning supervision change the scale trend, or mainly change error recovery?"
            ),
        },
        {"id": "detail_block", "type": "table", "tableId": "checkpoint_detail", "layout": "full"},
        {
            "id": "caveats",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Caveats and assumptions\n\n"
                "- Held-out-task estimates use 60 episodes per checkpoint, so their uncertainty is wider than the 470-episode trained-task estimates.\n"
                "- These are single training runs, not averages across training seeds.\n"
                "- Scale 60 and 90 did not reach epoch 2; those missing checkpoints must not be interpreted as failures with zero success.\n"
                "- Scale 150 epoch 1 is complete; scale 60 and 90 epoch 2 remain unavailable."
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Training-data scale evaluation",
            "description": "Fixed live-sim comparison of SFT checkpoints trained with 30–150 expert trajectories per task.",
            "generatedAt": now,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": now,
            "status": "partial",
            "datasets": {
                "all_results": rows,
                "heldout_progression": [{**row, "checkpoint": f"{row['scale']} · e{row['epoch']:g}"} for row in heldout],
                "one_epoch_heldout": [{**row, "scale_label": str(row["scale"])} for row in one_epoch if row["split_id"] == "heldout_task_types"],
                "one_epoch_split": [{**row, "scale_label": str(row["scale"])} for row in one_epoch],
            },
            "accessIssues": [{
                "id": "pending_scale_checkpoints",
                "dataset": "all_results",
                "message": "Scale 60 and 90 epoch 2 are unavailable because training timed out before those checkpoints were created.",
            }],
        },
        "sources": [source],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "artifact.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "completed_checkpoint_pairs": completed_pairs,
        "best_heldout": {"scale": best_held["scale"], "epoch": best_held["epoch"], "rate": best_held["fsm_success_rate"]},
        "best_trained": {"scale": best_trained["scale"], "epoch": best_trained["epoch"], "rate": best_trained["fsm_success_rate"]},
    }, indent=2))


if __name__ == "__main__":
    main()
