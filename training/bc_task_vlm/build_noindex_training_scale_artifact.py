#!/usr/bin/env python3
"""Build the current 90/10 model comparison and matched-size 27/3 ablation."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/noindex_training_scale_comparison_artifact"
OLD = {
    "previous_30_sft": {"heldout_trajectories": (75, 141), "heldout_tasks": (31, 90)},
    "previous_30_base_qwen": {"heldout_trajectories": (43, 141), "heldout_tasks": (35, 90)},
    "previous_30_gemini_flash": {"heldout_trajectories": (104, 141), "heldout_tasks": (73, 90)},
}
RUNS = {
    "current_30_sft": {
        "label": "Current SFT — 27/3",
        "source": "current_30",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_qwen3vl-8b-v3-tick30-revalidated-v3-partial-idxnone-noreason-schemafix__{split}-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 141, "heldout_tasks": 90},
    },
    "current_100_sft": {
        "label": "Current SFT — 90/10",
        "source": "current_100",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_qwen3vl-8b-v3-tick100-from150-v3-partial-idxnone-noreason-schemafix__{split}-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 470, "heldout_tasks": 60},
    },
    "current_30_sft_ep1": {
        "label": "Current SFT — 27/3 — 1 epoch",
        "source": "current_30",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_qwen3vl-8b-v3-tick30-revalidated-v3-partial-idxnone-noreason-fixedcohort-ablation-ep1-lr2e-4__{split}-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 141, "heldout_tasks": 90},
    },
    "current_100_sft_ep1": {
        "label": "Current SFT — 90/10 — 1 epoch",
        "source": "current_100",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_qwen3vl-8b-v3-tick100-from150-v3-partial-idxnone-noreason-fixedcohort-ablation-ep1-lr2e-4__{split}-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 470, "heldout_tasks": 60},
        "optional_until_complete": True,
    },
    **{
        f"current_30_{run_id}": {
            "label": label,
            "source": "current_30",
            "paths": {
                split: ROOT / "training/bc_task_vlm/eval_runs" /
                f"livesim_ots-noindex-{tag}-tick30_revalidated_v3__{split}-schemafix-contentionfix" /
                "live_sim_trajectories.jsonl"
                for split in ("heldout_trajectories", "heldout_tasks")
            },
            "expected": {"heldout_trajectories": 141, "heldout_tasks": 90},
        }
        for run_id, tag, label in (
            ("base_qwen", "qwen3vl8b", "Base Qwen3-VL-8B"),
            ("gemini_flash", "gemini3flash", "Gemini 3 Flash"),
        )
    },
    "base_qwen": {
        "label": "Base Qwen3-VL-8B",
        "source": "current_100",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_ots-noindex-qwen3vl8b-tick100_from150_v3__{split}-schemafix-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 470, "heldout_tasks": 60},
    },
    "gemini_flash": {
        "label": "Gemini 3 Flash",
        "source": "current_100",
        "paths": {
            split: ROOT / "training/bc_task_vlm/eval_runs" /
            f"livesim_ots-noindex-gemini3flash-tick100_from150_v3__{split}-schemafix-contentionfix" /
            "live_sim_trajectories.jsonl"
            for split in ("heldout_trajectories", "heldout_tasks")
        },
        "expected": {"heldout_trajectories": 470, "heldout_tasks": 60},
        "optional_until_complete": True,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def metric(successes: int, attempts: int) -> dict:
    low, high = wilson(successes, attempts)
    return {
        "successes": successes,
        "attempts": attempts,
        "rate": successes / attempts,
        "ci_low": low,
        "ci_high": high,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metric_from_rows(rows: list[dict], *, seed: int) -> dict:
    result = metric(sum(bool(row.get("fsm_goal_satisfied")) for row in rows), len(rows))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_name"]].append(row)
    task_rates = [
        sum(bool(row.get("fsm_goal_satisfied")) for row in task_rows) / len(task_rows)
        for task_rows in grouped.values()
    ]
    macro = sum(task_rates) / len(task_rates)
    rng = random.Random(seed)
    bootstraps = [
        sum(rng.choice(task_rates) for _ in task_rates) / len(task_rates)
        for _ in range(10_000)
    ]
    result.update({
        "macro_rate": macro,
        "macro_ci_low": _percentile(bootstraps, 0.025),
        "macro_ci_high": _percentile(bootstraps, 0.975),
        "num_tasks": len(task_rates),
    })
    return result


def load_results() -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {
        run_id: {split: metric(*counts) for split, counts in splits.items()}
        for run_id, splits in OLD.items()
    }
    for run_id, spec in RUNS.items():
        split_rows = {}
        for split, path in spec["paths"].items():
            if not path.is_file():
                if not spec.get("optional_until_complete"):
                    raise FileNotFoundError(path)
                split_rows = {}
                break
            rows = read_jsonl(path)
            expected = spec["expected"][split]
            if len(rows) != expected:
                if spec.get("optional_until_complete"):
                    split_rows = {}
                    break
                raise RuntimeError(f"{run_id}/{split}: expected {expected}, found {len(rows)}")
            split_rows[split] = metric_from_rows(
                rows, seed=sum(ord(char) for char in f"{run_id}:{split}")
            )
        if split_rows:
            results[run_id] = split_rows
    return results


def heldout_task_rows(run_id: str) -> dict[str, dict[str, int]]:
    rows = read_jsonl(RUNS[run_id]["paths"]["heldout_tasks"])
    output: dict[str, dict[str, int]] = {}
    for task in sorted({row["task_name"] for row in rows}):
        task_rows = [row for row in rows if row["task_name"] == task]
        output[task] = {
            "successes": sum(bool(row.get("fsm_goal_satisfied")) for row in task_rows),
            "attempts": len(task_rows),
        }
    return output


def current_100_task_groups() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in RUNS["current_100_sft"]["paths"].values():
        for row in read_jsonl(path):
            grouped[row["task_name"]].append(row)
    output: dict[str, list[dict]] = {
        "all_success": [], "mixed": [], "all_failure": []
    }
    for task, rows in sorted(grouped.items()):
        successes = sum(bool(row.get("fsm_goal_satisfied")) for row in rows)
        group = "all_success" if successes == len(rows) else (
            "all_failure" if successes == 0 else "mixed"
        )
        output[group].append({
            "task": task,
            "successes": successes,
            "attempts": len(rows),
            "mean_rejections": sum(row.get("rejected_total", 0) for row in rows) / len(rows),
            "mean_conflicts": sum(row.get("resource_conflicts", 0) for row in rows) / len(rows),
            "mean_partial_goal": sum(row.get("partial_goal_fraction", 0) for row in rows) / len(rows),
            "mean_steps": sum(row.get("steps_used", 0) for row in rows) / len(rows),
        })
    return output


def wide_row(run_id: str, label: str, results: dict, source: str) -> dict:
    task = results[run_id]["heldout_tasks"]
    traj = results[run_id]["heldout_trajectories"]
    return {
        "run_id": run_id,
        "model": label,
        "source": source,
        "heldout_task_rate": task["rate"],
        "heldout_task_ci_low": task["ci_low"],
        "heldout_task_ci_high": task["ci_high"],
        "heldout_task_macro_rate": task.get("macro_rate"),
        "heldout_task_macro_ci_low": task.get("macro_ci_low"),
        "heldout_task_macro_ci_high": task.get("macro_ci_high"),
        "heldout_task_successes": task["successes"],
        "heldout_task_attempts": task["attempts"],
        "heldout_trajectory_rate": traj["rate"],
        "heldout_trajectory_ci_low": traj["ci_low"],
        "heldout_trajectory_ci_high": traj["ci_high"],
        "heldout_trajectory_macro_rate": traj.get("macro_rate"),
        "heldout_trajectory_macro_ci_low": traj.get("macro_ci_low"),
        "heldout_trajectory_macro_ci_high": traj.get("macro_ci_high"),
        "heldout_trajectory_successes": traj["successes"],
        "heldout_trajectory_attempts": traj["attempts"],
    }


def interval(rate: float | None, low: float | None, high: float | None) -> str:
    if rate is None or low is None or high is None:
        return "Not available"
    return f"{rate:.1%} ({low:.1%}–{high:.1%})"


def task_consistency_rows(run_id: str, results: dict, source: str) -> dict:
    output = []
    for split, path in RUNS[run_id]["paths"].items():
        rows = read_jsonl(path)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["task_name"]].append(row)
        counts = {"all_success": 0, "mixed": 0, "no_success": 0}
        for task_rows in grouped.values():
            successes = sum(bool(row.get("fsm_goal_satisfied")) for row in task_rows)
            if successes == len(task_rows):
                counts["all_success"] += 1
            elif successes == 0:
                counts["no_success"] += 1
            else:
                counts["mixed"] += 1
        output.append({
            "run_id": run_id,
            "model": RUNS[run_id]["label"],
            "source": source,
            "split": "Trained task types" if split == "heldout_trajectories" else "Held-out task types",
            "model_and_split": f"{RUNS[run_id]['label']} — " + (
                "trained tasks" if split == "heldout_trajectories" else "held-out tasks"
            ),
            "all_success": counts["all_success"],
            "mixed": counts["mixed"],
            "no_success": counts["no_success"],
            "num_tasks": len(grouped),
        })
    return output


def pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> int:
    results = load_results()
    current_90_rows = [
        wide_row(run_id, RUNS[run_id]["label"], results, "current_100")
        for run_id in ("current_100_sft", "current_100_sft_ep1", "base_qwen", "gemini_flash")
        if run_id in results
    ]
    current_27_rows = [
        wide_row(run_id, RUNS[run_id]["label"], results, "current_30")
        for run_id in (
            "current_30_sft", "current_30_sft_ep1", "current_30_base_qwen",
            "current_30_gemini_flash",
        )
    ]
    current_27_consistency = [
        row
        for run_id in (
            "current_30_sft", "current_30_sft_ep1", "current_30_base_qwen",
            "current_30_gemini_flash",
        )
        for row in task_consistency_rows(run_id, results, "current_30")
    ]
    current_90_consistency = [
        row
        for run_id in ("current_100_sft", "current_100_sft_ep1", "base_qwen", "gemini_flash")
        if run_id in results
        for row in task_consistency_rows(run_id, results, "current_100")
    ]
    fixed_rows = []
    for period, source in (("previous", "historical"), ("current", "current_30")):
        for model_id, label in (
            ("sft", "SFT Qwen3-VL-8B"),
            ("base_qwen", "Base Qwen3-VL-8B"),
            ("gemini_flash", "Gemini 3 Flash"),
        ):
            run_id = f"{period}_30_{model_id}"
            row = wide_row(run_id, f"{period.title()} — {label}", results, source)
            row["pipeline"] = period.title()
            row["model_family"] = label
            fixed_rows.append(row)
    scale_rows = [
        wide_row("current_30_sft", "Current SFT — 27/3", results, "current_30"),
        wide_row("current_100_sft", "Current SFT — 90/10", results, "current_100"),
    ]
    task_27 = heldout_task_rows("current_30_sft")
    task_90 = heldout_task_rows("current_100_sft")
    task_groups = current_100_task_groups()
    task_regression_rows = []
    for task in sorted(task_27):
        rate_27 = task_27[task]["successes"] / task_27[task]["attempts"]
        rate_90 = task_90[task]["successes"] / task_90[task]["attempts"]
        task_regression_rows.append({
            "task": task,
            "sft_27_successes": task_27[task]["successes"],
            "sft_27_attempts": task_27[task]["attempts"],
            "sft_27_rate": rate_27,
            "sft_90_successes": task_90[task]["successes"],
            "sft_90_attempts": task_90[task]["attempts"],
            "sft_90_rate": rate_90,
            "aggregate_gap_contribution_pp": (rate_90 - rate_27) * 100 / len(task_27),
        })
    all_rows = fixed_rows + [
        row for row in current_27_rows if row["run_id"] == "current_30_sft_ep1"
    ] + current_90_rows
    changes = [
        {"area": "Opening plan", "previous": "Simultaneous free-form plans", "current": "Sampled coordinator proposes; partner confirms causally"},
        {"area": "Concurrency", "previous": "Separate scheduling interpretations", "current": "Canonical ticks and one shared scheduler"},
        {"area": "Wait/release", "previous": "Same-tick/order ambiguities possible", "current": "Exact resource match; wake-up begins next tick"},
        {"area": "Completion tails", "previous": "Premature/filler completion messages", "current": "Scoped portion completion followed by waiting"},
        {"area": "Tool validity", "previous": "Looser argument and placement handling", "current": "Strict schemas, grounded targets, single-holder invariant"},
        {"area": "Observations", "previous": "Inserted images could shift validation", "current": "get_image is supervised but scheduler-transparent"},
        {"area": "Partitioning", "previous": "Generated ownership/work partitions", "current": "No forced partition; model chooses an efficient plan"},
        {"area": "Evaluation failures", "previous": "Correction behavior could obscure first errors", "current": "No silent retry; report_failed and first-rejection accounting"},
        {"area": "Context/indexing", "previous": "Corrected no-index baseline", "current": "No indices plus explicit 8,192-token overflow accounting"},
        {"area": "Data gate", "previous": "Earlier validator/render contract", "current": "Current FSM revalidation plus successful simulator render required"},
    ]

    generated = datetime.now(timezone.utc).isoformat()
    comparison_source_path = "training/bc_task_vlm/eval_runs/noindex_training_scale_comparison_artifact/source_queries.sql"
    sources = [
        {"id": "historical", "label": "Previous corrected no-index 27/3 evaluation", "path": "training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/artifact.json"},
        {"id": "current_30", "label": "Current contention-fixed 27/3 SFT and out-of-box live-sim outputs", "path": "training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-revalidated-v3-partial-idxnone-noreason-schemafix__heldout_trajectories-contentionfix/live_sim_trajectories.jsonl"},
        {"id": "current_100", "label": "Current contention-fixed 90/10 SFT and out-of-box live-sim outputs", "path": "training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick100-from150-v3-partial-idxnone-noreason-schemafix__heldout_trajectories-contentionfix/live_sim_trajectories.jsonl"},
        {"id": "design", "label": "No-index training-scale ablation design and change inventory", "path": "training/bc_task_vlm/reports/noindex_training_scale_ablation_design.md"},
        {"id": "comparison_inputs", "label": "All comparison inputs, denominators, and calculation rules", "path": comparison_source_path},
    ]
    charts = [
        {
            "id": "current_27_models", "title": "Success rates on the dataset-derived 27/3 evaluation",
            "subtitle": "FSM success rate on trained and held-out task types, with 141 and 90 evaluated episodes respectively.",
            "showDescription": True, "intent": "comparison",
            "question": "How often does each model complete an evaluated episode successfully?",
            "rationale": "Success rate remains the primary model-performance measure; task consistency is shown separately below.",
            "type": "bar", "dataset": "current_27_models", "sourceId": "comparison_inputs",
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"fields": ["heldout_task_rate", "heldout_trajectory_rate"], "type": "quantitative", "format": "percent", "label": "FSM success rate"}},
            "layout": "full",
        },
        {
            "id": "current_models", "title": "Success rates on the dataset-derived 90/10 evaluation",
            "subtitle": "FSM success rate on trained and held-out task types, with 470 and 60 evaluated episodes respectively.",
            "showDescription": True, "intent": "comparison",
            "question": "How often does each complete 90/10 model complete an evaluated episode successfully?",
            "rationale": "Success rate remains the primary model-performance measure; task consistency is shown separately below.",
            "type": "bar", "dataset": "current_models", "sourceId": "comparison_inputs",
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Model"}, "y": {"fields": ["heldout_task_rate", "heldout_trajectory_rate"], "type": "quantitative", "format": "percent", "label": "FSM success rate"}},
            "layout": "full",
        },
        {
            "id": "scale", "title": "Current SFT performance at 27/3 and 90/10",
            "subtitle": "Both use current pipeline logic, but evaluation splits differ: 27/3 uses N=141/90 and 90/10 uses N=470/60 for held-out trajectories/tasks.",
            "showDescription": True, "intent": "comparison",
            "question": "What is the descriptive effect of increasing current training data?",
            "rationale": "This is the report's only mixed-evaluation-split graph; keeping it separate prevents cohort mixing in the model-comparison views.",
            "type": "bar", "dataset": "scale", "sourceId": "comparison_inputs",
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "Current SFT size"}, "y": {"fields": ["heldout_task_rate", "heldout_trajectory_rate"], "type": "quantitative", "format": "percent", "label": "FSM success rate"}},
            "layout": "full",
        },
        {
            "id": "fixed_size", "title": "Previous versus current results on the 27/3 evaluation design",
            "subtitle": "SFT and all three out-of-box models; 141 held-out trajectories and 90 held-out-task episodes per model and pipeline.",
            "showDescription": True, "intent": "comparison",
            "question": "What changed at the same amount of SFT data?",
            "rationale": "Including unchanged out-of-box models shows how much of the before/after difference comes from the evaluation pipeline versus SFT training.",
            "type": "bar", "dataset": "fixed_size", "sourceId": "comparison_inputs",
            "encodings": {"x": {"field": "model", "type": "nominal", "label": "SFT run"}, "y": {"fields": ["heldout_task_rate", "heldout_trajectory_rate"], "type": "quantitative", "format": "percent", "label": "FSM success rate"}},
            "layout": "full",
        },
    ]
    tables = [
        *[
            {
                "id": f"current_{size}_consistency",
                "title": f"Current {size}/{'3' if size == '27' else '10'} task-consistency counts",
                "subtitle": "These are exact counts over the evaluated task types, so sampling confidence intervals do not apply.",
                "showDescription": True,
                "dataset": f"current_{size}_consistency",
                "sourceId": "comparison_inputs", "density": "spacious", "layout": "full",
                "defaultSort": {"field": "model", "direction": "asc"},
                "columns": [
                    {"field": "model", "label": "Model", "type": "text"},
                    {"field": "split", "label": "Split", "type": "text"},
                    {"field": "all_success", "label": "All episodes succeed", "format": "number"},
                    {"field": "mixed", "label": "Some episodes succeed", "format": "number"},
                    {"field": "no_success", "label": "No episodes succeed", "format": "number"},
                    {"field": "num_tasks", "label": "Task types", "format": "number"},
                ],
            }
            for size in ("27", "90")
        ],
        {
            "id": "exact_results", "title": "Exact success counts and denominators",
            "subtitle": "Do not pool held-out-task and held-out-trajectory cohorts.",
            "showDescription": True, "dataset": "all_results", "sourceId": "comparison_inputs",
            "defaultSort": {"field": "heldout_trajectory_rate", "direction": "desc"}, "density": "spacious", "layout": "full",
            "columns": [
                {"field": "model", "label": "Model / arm", "type": "text"},
                {"field": "heldout_task_successes", "label": "Task successes", "format": "number"},
                {"field": "heldout_task_attempts", "label": "Task N", "format": "number"},
                {"field": "heldout_task_rate", "label": "Task rate", "format": "percent"},
                {"field": "heldout_trajectory_successes", "label": "Trajectory successes", "format": "number"},
                {"field": "heldout_trajectory_attempts", "label": "Trajectory N", "format": "number"},
                {"field": "heldout_trajectory_rate", "label": "Trajectory rate", "format": "percent"},
            ],
        },
        {
            "id": "changes", "title": "Material changes since the previous 27/3 model",
            "subtitle": "The fixed-size comparison estimates their combined effect, not any single row's causal effect.",
            "showDescription": True, "dataset": "changes", "sourceId": "comparison_inputs",
            "defaultSort": {"field": "area", "direction": "asc"}, "density": "spacious", "layout": "full",
            "columns": [
                {"field": "area", "label": "Area", "type": "text"},
                {"field": "previous", "label": "Previous 27/3", "type": "text"},
                {"field": "current", "label": "Current pipeline", "type": "text"},
            ],
        },
        {
            "id": "task_regression", "title": "Held-out-task regression by task",
            "subtitle": "The two runs use the same six task types but different sampled trajectories; contribution is to the unweighted six-task rate gap.",
            "showDescription": True, "dataset": "task_regression", "sourceId": "comparison_inputs",
            "defaultSort": {"field": "aggregate_gap_contribution_pp", "direction": "asc"},
            "density": "spacious", "layout": "full",
            "columns": [
                {"field": "task", "label": "Held-out task", "type": "text"},
                {"field": "sft_27_successes", "label": "27/3 successes", "format": "number"},
                {"field": "sft_27_attempts", "label": "27/3 N", "format": "number"},
                {"field": "sft_27_rate", "label": "27/3 rate", "format": "percent"},
                {"field": "sft_90_successes", "label": "90/10 successes", "format": "number"},
                {"field": "sft_90_attempts", "label": "90/10 N", "format": "number"},
                {"field": "sft_90_rate", "label": "90/10 rate", "format": "percent"},
                {"field": "aggregate_gap_contribution_pp", "label": "Gap contribution (pp)", "format": "number", "movement": True},
            ],
        },
    ]
    s100 = next(row for row in current_90_rows if row["run_id"] == "current_100_sft")
    s30 = next(row for row in fixed_rows if row["run_id"] == "current_30_sft")
    old = next(row for row in fixed_rows if row["run_id"] == "previous_30_sft")
    blocks = [
        {"id": "title", "type": "markdown", "body": "# No-Index SFT: Data Scale and Pipeline Improvements", "layout": "full"},
        {"id": "summary", "type": "markdown", "sourceId": "comparison_inputs", "body": (
            "## The three comparisons answer different questions\n\n"
            f"The current 90/10 SFT model reaches **{pct(s100['heldout_task_rate'])}** on held-out tasks and **{pct(s100['heldout_trajectory_rate'])}** on held-out trajectories. "
            f"At the historical 27/3 training size, the current pipeline reaches **{pct(s30['heldout_task_rate'])} / {pct(s30['heldout_trajectory_rate'])}**, versus **{pct(old['heldout_task_rate'])} / {pct(old['heldout_trajectory_rate'])}** previously. "
            "The fixed-size difference measures the combined pipeline improvement; the current 27/3-to-90/10 difference measures data-scale sensitivity under the new pipeline."
        ), "layout": "full"},
        {"id": "current_27_heading", "type": "markdown", "body": "## Current 27/3 success rates and task consistency\n\nBoth SFT schedules, base Qwen, and Gemini Flash use the same current 27/3 manifests and evaluator contract. The chart reports episode-level FSM success rates. The table immediately below complements those rates by counting task types where all episodes succeed, only some succeed, or none succeed.", "layout": "full"},
        {"id": "current_27_chart", "type": "chart", "chartId": "current_27_models", "layout": "full"},
        {"id": "current_27_intervals", "type": "table", "tableId": "current_27_consistency", "layout": "full"},
        {"id": "current_heading", "type": "markdown", "body": "## Current 90/10 success rates and task consistency\n\nThe completed comparison includes the three-epoch SFT, one-epoch SFT, base Qwen3-VL-8B, and Gemini 3 Flash on both trained and held-out task types. The chart reports episode-level FSM success rates. The table immediately below reports exact all/some/none task-consistency counts.", "layout": "full"},
        {"id": "current_chart", "type": "chart", "chartId": "current_models", "layout": "full"},
        {"id": "current_intervals", "type": "table", "tableId": "current_90_consistency", "layout": "full"},
        {"id": "fixed_heading", "type": "markdown", "body": "## Previous versus current results under the 27/3 design\n\nBoth SFT arms use 27 training and three validation trajectories per trained task. Every previous/current model pair uses 141 held-out trajectories and 90 held-out-task episodes. The out-of-box pairs help reveal whether a shift is caused by the evaluation pipeline itself rather than SFT training.", "layout": "full"},
        {"id": "fixed_chart", "type": "chart", "chartId": "fixed_size", "layout": "full"},
        {"id": "fixed_key_changes", "type": "markdown", "sourceId": "design", "body": (
            "### Most important changes in the current 27/3 pipeline\n\n"
            "- One sampled coordinator proposes the opening plan; the other agent confirms.\n"
            "- Releases take effect on the following tick, eliminating same-tick ordering dependence.\n"
            "- Agents report only their own portion complete and then wait; they cannot announce global completion prematurely.\n\n"
            "The graph measures the combined effect of these and the other documented pipeline changes; it does not isolate any single change."
        ), "layout": "full"},
        {"id": "scale_heading", "type": "markdown", "body": "## Current 27/3 versus 90/10 isolates training scale within the current pipeline\n\nThis is intentionally the only graph that mixes evaluation splits. Both arms use current logic and the 30-trajectory corpus is a seeded subset of the 100-trajectory corpus, but the 27/3 arm is evaluated on 141 held-out trajectories and 90 held-out-task episodes while the 90/10 arm uses 470 and 60, respectively. Treat the comparison as descriptive rather than paired.", "layout": "full"},
        {"id": "scale_chart", "type": "chart", "chartId": "scale", "layout": "full"},
        {"id": "scale_diagnosis", "type": "markdown", "sourceId": "comparison_inputs", "body": (
            "## The unseen-task drop is concentrated, not a uniform collapse\n\n"
            "The current 90/10 model's held-out-task rate is 23.3 percentage points below the current 27/3 model. "
            "About 16.7 points of that gap (71%) come from `meat_skewer_assembly`, which moved from 15/15 to 0/10; another 5.0 points come from `cluster_items_for_clearing`. "
            "On meat skewers, every large-model episode reaches half of the goal and then terminates after repeated resource-conflict rejections. The smaller model also encountered an initial resource conflict on all 15 episodes but recovered, indicating a loss of recovery behavior rather than failure to understand the task from the outset."
        ), "layout": "full"},
        {"id": "task_regression_table", "type": "table", "tableId": "task_regression", "layout": "full"},
        {"id": "scale_validation", "type": "markdown", "sourceId": "comparison_inputs", "body": (
            "## Current validation cannot select for unseen-task generalization\n\n"
            "Both Trainer validation sets contain the same 47 task types as training and split only by trajectory. The 90/10 model's token-level validation loss improves to 0.033, versus 0.049 for 27/3, while unseen-task execution gets worse. "
            "The scale comparison is also unpaired: only 18 small-run validation trajectories overlap the large-run validation set, and only five held-out-task episodes overlap. This prevents the headline rate difference from being treated as a clean causal effect of data volume. "
            "Training examples are sampled per agent turn, so longer tasks receive up to 3.2 times the weight of shorter tasks even though every task has the same number of trajectories."
        ), "layout": "full"},
        {"id": "task_pattern_diagnosis", "type": "markdown", "sourceId": "comparison_inputs", "body": (
            "## Universal failures split into contention and grounding families\n\n"
            f"Across all 53 tasks, **{len(task_groups['all_success'])}** succeed in every rollout, **{len(task_groups['mixed'])}** have mixed results, and **{len(task_groups['all_failure'])}** fail every rollout. "
            "The universally failing tasks average 4.0 rejected calls, 1.0 resource conflicts, 49% partial-goal completion, and 67.6 model steps. Universal successes average 0.8 rejections, 0.15 conflicts, full goal completion, and 37.2 steps. "
            "Four universal failures—`colorful_salsa`, `garnish_cake`, `meat_skewer_assembly`, and `portion_yogurt`—begin every episode with shared-resource contention. The other four—`hot_dog_setup`, `plate_store_dinner`, `prepare_sandwich_station`, and `prepare_soup_serving`—primarily fail on navigation, object-source, or prerequisite grounding. "
            "Before interpreting the contention family as a learning failure, audit the evaluator's fixture locks: it currently treats counters, dining counters, and movable source plates as exclusive resources, although the intended task semantics allow multiple agents at shared counters and may allow simultaneous access to distinct objects on one plate."
        ), "layout": "full"},
        {"id": "scale_recommendation", "type": "markdown", "body": (
            "## Test optimization strength before increasing adapter capacity\n\n"
            "The next controlled run should keep LoRA rank 16 and train the 90-trajectory corpus for 0.5, 1, and 2 epochs at a lower learning rate, retaining an evaluable checkpoint at each point. Select checkpoints on a separate task-level development fold plus a fixed common trajectory cohort. "
            "Higher LoRA rank is not the first remedy because rank 16 already fits same-task validation well and more capacity can strengthen template specialization. A larger base model is promising for compositional generalization, but should first be tested with the shorter, lower-learning-rate schedule so SFT does not overwrite its out-of-box recovery behavior."
        ), "layout": "full"},
        {"id": "definitions", "type": "markdown", "body": "## Scope and metric definitions\n\n**FSM success** means the concurrent task FSM goal was satisfied. Held-out trajectories are unseen initial-state trajectories from the 47 trained task types. Held-out tasks are six task types excluded from SFT. Native RoboCasa success remains a separate physical diagnostic and is not substituted for FSM learning success.", "layout": "full"},
        {"id": "results_table", "type": "table", "tableId": "exact_results", "layout": "full"},
        {"id": "methods", "type": "markdown", "body": "## Experimental design and material treatment changes\n\nThe table below summarizes the main changes. The complete implementation-level inventory is retained in the experiment-design source.", "layout": "full"},
        {"id": "changes_table", "type": "table", "tableId": "changes", "layout": "full"},
        {"id": "limits", "type": "markdown", "body": "## Limitations and robustness\n\nThis is a descriptive two-arm ablation, not a multi-seed causal estimate. The old and current 27/3 arms match sample count and evaluation denominators but not individual expert trajectories. The current 90/10 and 27/3 held-out-task cohorts also have different sizes, so compare their rates with their exact denominators and uncertainty rather than treating small differences as definitive. A second training seed would measure optimization variance.", "layout": "full"},
        {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\nUse the fixed-size result to judge whether the corrected pipeline improved learning independently of data volume. Use the current scale comparison to decide whether generating and training on more trajectories is worthwhile. If either difference is modest relative to sampling uncertainty, repeat both current arms with a second training seed before changing the data-generation protocol again.", "layout": "full"},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nWhich task types account for the fixed-size improvement? Does additional data improve recovery after rejected calls or only clean first-attempt execution? Are gains concentrated in tasks with cross-agent handovers, appliances, or long contexts?", "layout": "full"},
    ]
    artifact = {
        "surface": "report",
        "manifest": {"version": 1, "surface": "report", "title": "No-Index SFT: Data Scale and Pipeline Improvements", "description": "Separates current model capability, fixed-size pipeline improvements, and training-data scale.", "generatedAt": generated, "sources": sources, "cards": [], "charts": charts, "tables": tables, "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": generated, "status": "ready", "datasets": {"current_27_models": current_27_rows, "current_models": current_90_rows, "current_27_consistency": current_27_consistency, "current_90_consistency": current_90_consistency, "fixed_size": fixed_rows, "scale": scale_rows, "task_regression": task_regression_rows, "all_results": all_rows, "changes": changes}},
        "sources": sources,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    source_lines = [
        "# Comparison inputs and calculation rules",
        "",
        "This report reads the historical baseline and the following completed live-simulator outputs:",
        "",
        "- `training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/artifact.json` (historical 27/3 SFT plus Qwen, Gemini 3 Flash, and ER2 out-of-box results; N=141 held-out trajectories and N=90 held-out tasks per model)",
    ]
    for run_id, spec in RUNS.items():
        for split, path in spec["paths"].items():
            source_lines.append(f"- `{path.relative_to(ROOT)}` ({run_id}, {split}, expected N={spec['expected'][split]})")
    source_lines.extend([
        "",
        "For every current output, success is `bool(fsm_goal_satisfied)`. The rate is successes divided by the exact row count. The builder rejects missing files or unexpected row counts. Wilson 95% intervals are retained in the analysis snapshot.",
        "",
        "The historical and current 27/3 arms match training-example counts and evaluation denominators, but they do not contain identical expert trajectories. The current 27/3 corpus is a seeded subset of the current 90/10 corpus.",
    ])
    (OUT / "source_notes.md").write_text("\n".join(source_lines) + "\n")
    source_queries = """-- DuckDB-compatible reproducibility queries for the no-index scale report.
-- Run from the repository root. The report builder applies the same boolean
-- FSM-success aggregation after verifying each file's expected row count.
CREATE OR REPLACE VIEW noindex_scale_eval_rows AS
SELECT 'Current SFT — 90/10' AS model, 'heldout_trajectories' AS split, *
FROM read_json_auto('training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick100-from150-v3-partial-idxnone-noreason-schemafix__heldout_trajectories/live_sim_trajectories.jsonl')
UNION ALL BY NAME
SELECT 'Current SFT — 90/10', 'heldout_tasks', *
FROM read_json_auto('training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick100-from150-v3-partial-idxnone-noreason-schemafix__heldout_tasks/live_sim_trajectories.jsonl')
UNION ALL BY NAME
SELECT 'Current SFT — 27/3', 'heldout_trajectories', *
FROM read_json_auto('training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-revalidated-v3-partial-idxnone-noreason-schemafix__heldout_trajectories/live_sim_trajectories.jsonl')
UNION ALL BY NAME
SELECT 'Current SFT — 27/3', 'heldout_tasks', *
FROM read_json_auto('training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-revalidated-v3-partial-idxnone-noreason-schemafix__heldout_tasks/live_sim_trajectories.jsonl');

SELECT model, split, count(*) AS attempts,
       count(*) FILTER (WHERE fsm_goal_satisfied) AS fsm_successes,
       avg(fsm_goal_satisfied::INTEGER) AS fsm_success_rate
FROM noindex_scale_eval_rows
GROUP BY ALL ORDER BY model, split;

-- The builder repeats the same UNION and aggregation for the Qwen3-VL-8B,
-- Gemini 3 Flash, and Gemini Robotics ER2 JSONL paths enumerated in
-- source_notes.md, and joins the saved historical counts in OLD.

SELECT * FROM (VALUES
  ('Opening plan', 'Simultaneous free-form plans', 'Sampled coordinator proposes; partner confirms causally'),
  ('Concurrency', 'Separate scheduling interpretations', 'Canonical ticks and one shared scheduler'),
  ('Wait/release', 'Same-tick/order ambiguities possible', 'Exact resource match; wake-up begins next tick'),
  ('Completion tails', 'Premature/filler completion messages', 'Scoped portion completion followed by waiting'),
  ('Tool validity', 'Looser argument and placement handling', 'Strict schemas, grounded targets, single-holder invariant')
) AS changes(area, previous, current);
"""
    (OUT / "source_queries.sql").write_text(source_queries)
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUT / "analysis_snapshot.json").write_text(json.dumps(artifact["snapshot"]["datasets"], indent=2) + "\n")
    (OUT / "chart_map.json").write_text(json.dumps({"charts": [{"id": c["id"], "question": c["question"], "type": c["type"], "dataset": c["dataset"]} for c in charts]}, indent=2) + "\n")
    print(OUT / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
