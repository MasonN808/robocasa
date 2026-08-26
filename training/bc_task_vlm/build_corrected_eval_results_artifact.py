#!/usr/bin/env python3
"""Build the comprehensive no-index corrected-evaluation report."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact"
RUNS = {
    "heldout_tasks": {
        "label": "Held-out tasks",
        "job_id": "290438",
        "expected": 90,
        "path": REPO_ROOT / "training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxnone-noreason-indexab-savefix-8h__heldout_tasks-bounded-v3-contextfix/live_sim_trajectories.jsonl",
    },
    "heldout_trajectories": {
        "label": "Held-out trajectories",
        "job_id": "290439",
        "expected": 141,
        "path": REPO_ROOT / "training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxnone-noreason-indexab-savefix-8h__heldout_trajectories-bounded-v3-contextfix/live_sim_trajectories.jsonl",
    },
}

BASELINE_RUNS = {
    "qwen3vl8b": {
        "label": "Qwen3-VL-8B (out of box)",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "gemini3flash": {
        "label": "Gemini 3 Flash (out of box)",
        "model": "gemini-3-flash-preview",
    },
    "er2": {
        "label": "Gemini Robotics ER2 (out of box)",
        "model": "gemini-robotics-er-2-preview",
    },
}
for _model_tag, _model_spec in BASELINE_RUNS.items():
    _model_spec["runs"] = {
        split: {
            "expected": run_spec["expected"],
            "path": REPO_ROOT / (
                "training/bc_task_vlm/eval_runs/"
                f"livesim_ots-noindex-{_model_tag}__{split}/live_sim_trajectories.jsonl"
            ),
        }
        for split, run_spec in RUNS.items()
    }

TRAJECTORY_EXAMPLES = [
    {
        "split": "heldout_trajectories",
        "task": "distribute_chicken",
        "trajectory_id": "traj_000027",
        "title": "One agent waits while the other clears the stove",
        "why": "A short, clean example of a single wait-and-release handoff followed by parallel work.",
    },
    {
        "split": "heldout_trajectories",
        "task": "add_sugar_cubes",
        "trajectory_id": "traj_000027",
        "title": "The agents take turns at the same counter",
        "why": "Both agents wait at different points, so the same shared space is handed over in both directions.",
    },
    {
        "split": "heldout_tasks",
        "task": "meat_skewer_assembly",
        "trajectory_id": "traj_000001",
        "title": "Two separate signals coordinate an unseen task",
        "why": "A held-out task that uses one signal for the counter and another for the oven tray, including a rejected action and retry.",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def task_rows(rows: list[dict], split: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_name"]].append(row)
    result = []
    for task, items in grouped.items():
        attempts = len(items)
        successes = sum(bool(item.get("fsm_goal_satisfied")) for item in items)
        native = sum(bool(item.get("native_success")) for item in items)
        partials = [float(item.get("partial_goal_fraction") or 0) for item in items]
        rejects = [int(item.get("rejected_total") or 0) for item in items]
        result.append({
            "split": split,
            "task": task,
            "attempts": attempts,
            "fsm_successes": successes,
            "fsm_rate": successes / attempts,
            "native_successes": native,
            "native_rate": native / attempts,
            "mean_partial_goal": statistics.mean(partials),
            "mean_rejections": statistics.mean(rejects),
            "budget_exhausted": sum(item.get("termination") == "budget_exhausted" for item in items),
        })
    return sorted(result, key=lambda row: (-row["fsm_rate"], row["task"]))


def format_tool_call(tool: str | None, args: dict) -> str:
    if not tool:
        return "invalid tool call"
    if not args:
        return f"{tool}()"
    rendered = ", ".join(f"{key}={json.dumps(value)}" for key, value in args.items())
    return f"{tool}({rendered})"


def explain_tool_call(tool: str | None, args: dict) -> str:
    if tool == "communicate":
        text = f'Sends: “{args.get("message", "")}”'
        if args.get("releases"):
            text += f'; the release tag `{args["releases"]}` wakes a matching waiter'
        return text
    if tool == "wait_for_signal":
        return f'Pauses until {args.get("from", "the other agent")} sends a release tagged `{args.get("about", "")}`'
    if tool == "give_space":
        return f'Moves clear of `{args.get("fixture_id", "the shared area")}`'
    if tool == "navigate_to_fixture":
        return f'Moves to `{args.get("fixture_id", "the target fixture")}`'
    if tool == "pick_up_object":
        return f'Picks up `{args.get("object_id", "the object")}` from `{args.get("source_id", "its source")}`'
    if tool == "place_on_object":
        return f'Places `{args.get("object_id", "the object")}` on `{args.get("support_object_id", "the support")}`'
    if tool == "place_in_receptacle":
        return f'Places `{args.get("object_id", "the object")}` in `{args.get("receptacle_id", "the receptacle")}`'
    if tool == "open_hinged_part":
        return f'Opens the `{args.get("part_id", "part")}` of `{args.get("target_id", "the fixture")}`'
    return "The proposed call could not be parsed or does not need a special explanation."


def build_trajectory_examples(loaded: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[dict]]]:
    summaries = []
    timelines = {}
    for example_number, spec in enumerate(TRAJECTORY_EXAMPLES, 1):
        episode = next(
            (
                row for row in loaded[spec["split"]]
                if row.get("task_name") == spec["task"] and row.get("trajectory_id") == spec["trajectory_id"]
            ),
            None,
        )
        if episode is None:
            raise RuntimeError(f"Missing trajectory example: {spec}")
        waits = 0
        releases = 0
        calls_by_tick: dict[int, dict[str, list[str]]] = defaultdict(
            lambda: {"agent_0": [], "agent_1": []}
        )
        for step in episode.get("steps", []):
            proposal = step.get("proposal") or {}
            tool = proposal.get("tool")
            args = proposal.get("args") or {}
            waits += tool == "wait_for_signal"
            releases += tool == "communicate" and bool(args.get("releases"))
            if step.get("executed"):
                outcome = "Executed"
                if step.get("fsm_goal") is True:
                    outcome = "Executed — FSM goal satisfied"
            else:
                reason = str(step.get("reason") or "rejected")
                outcome = f"Rejected — {reason.split(':', 1)[-1].strip()}"
            tick = int(float(step.get("sim_time") or 0))
            agent = str(step.get("agent") or "unknown")
            calls_by_tick[tick].setdefault(agent, []).append(
                f"{format_tool_call(tool, args)} — {outcome}"
            )
        timeline = []
        for tick, agent_calls in sorted(calls_by_tick.items()):
            invocation_count = max(
                len(agent_calls["agent_0"]), len(agent_calls["agent_1"]), 1
            )
            for invocation_index in range(invocation_count):
                step_label = (
                    str(tick)
                    if invocation_count == 1
                    else f"{tick}.{invocation_index + 1}"
                )
                timeline.append({
                    "step_number": step_label,
                    "agent_0_tool_call": (
                        agent_calls["agent_0"][invocation_index]
                        if invocation_index < len(agent_calls["agent_0"])
                        else "Not Invoked"
                    ),
                    "agent_1_tool_call": (
                        agent_calls["agent_1"][invocation_index]
                        if invocation_index < len(agent_calls["agent_1"])
                        else "Not Invoked"
                    ),
                })
        timelines[f"trajectory_timeline_{example_number}"] = timeline
        summaries.append({
            "example": f"Example {example_number}",
            "title": spec["title"],
            "split": RUNS[spec["split"]]["label"],
            "task": spec["task"],
            "trajectory_id": spec["trajectory_id"],
            "fsm_success": "Yes" if episode.get("fsm_goal_satisfied") else "No",
            "native_success": "Yes" if episode.get("native_success") else "No",
            "termination": episode.get("termination"),
            "wait_calls": waits,
            "release_messages": releases,
            "total_tool_calls": len(episode.get("steps", [])),
            "why_selected": spec["why"],
        })
    return summaries, timelines


def main() -> None:
    loaded = {name: load_jsonl(spec["path"]) for name, spec in RUNS.items()}
    for name, rows in loaded.items():
        if len(rows) != RUNS[name]["expected"]:
            raise RuntimeError(f"{name}: expected {RUNS[name]['expected']} rows, found {len(rows)}")

    generated_at = datetime.now(timezone.utc).isoformat()
    summary = []
    progress_composition = []
    for name, spec in RUNS.items():
        rows = loaded[name]
        total = len(rows)
        fsm = sum(bool(row.get("fsm_goal_satisfied")) for row in rows)
        native = sum(bool(row.get("native_success")) for row in rows)
        low, high = wilson(fsm, total)
        terms = Counter(row.get("termination") for row in rows)
        partial = sum(
            not row.get("fsm_goal_satisfied") and float(row.get("partial_goal_fraction") or 0) > 0
            for row in rows
        )
        zero = total - fsm - partial
        summary.append({
            "run_id": name,
            "split": spec["label"],
            "job_id": spec["job_id"],
            "attempts": total,
            "fsm_successes": fsm,
            "fsm_rate": fsm / total,
            "fsm_ci_low": low,
            "fsm_ci_high": high,
            "native_successes": native,
            "native_rate": native / total,
            "fsm_native_gap_pp": (fsm - native) / total * 100,
            "goal_satisfied": terms["goal_satisfied"],
            "budget_exhausted": terms["budget_exhausted"],
            "max_rejections": terms["max_consecutive_rejections"],
            "mean_partial_goal": statistics.mean(float(row.get("partial_goal_fraction") or 0) for row in rows),
            "mean_rejections": statistics.mean(int(row.get("rejected_total") or 0) for row in rows),
            "json_errors": 0,
            "release_schema_errors": sum(
                "releases must be a string" in str(step.get("reason"))
                for row in rows for step in row.get("steps", [])
            ),
        })
        progress_composition.append({
            "split": spec["label"],
            "fsm_success": fsm,
            "partial_progress_failure": partial,
            "zero_progress_failure": zero,
            "attempts": total,
        })

    all_tasks = task_rows(loaded["heldout_tasks"], "Held-out tasks") + task_rows(
        loaded["heldout_trajectories"], "Held-out trajectories"
    )
    heldout_tasks = [row for row in all_tasks if row["split"] == "Held-out tasks"]
    trajectory_tasks = [row for row in all_tasks if row["split"] == "Held-out trajectories"]
    rate_distribution = []
    for successes in range(4):
        matching = [row for row in trajectory_tasks if row["fsm_successes"] == successes]
        rate_distribution.append({
            "outcome": f"{successes}/3",
            "successes": successes,
            "task_count": len(matching),
            "share": len(matching) / len(trajectory_tasks),
            "tasks": ", ".join(row["task"] for row in matching),
        })

    perfect = [row["task"] for row in trajectory_tasks if row["fsm_successes"] == 3]
    zero = [row["task"] for row in trajectory_tasks if row["fsm_successes"] == 0]
    overall = {row["run_id"]: row for row in summary}
    task_run = overall["heldout_tasks"]
    trajectory_run = overall["heldout_trajectories"]
    trajectory_examples, trajectory_timelines = build_trajectory_examples(loaded)

    available_baselines: dict[str, dict[str, list[dict]]] = {}
    for model_tag, model_spec in BASELINE_RUNS.items():
        paths = [spec["path"] for spec in model_spec["runs"].values()]
        if not any(path.exists() for path in paths):
            continue
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"{model_tag}: only one comparison split exists")
        model_rows = {
            split: load_jsonl(spec["path"])
            for split, spec in model_spec["runs"].items()
        }
        for split, rows in model_rows.items():
            expected = model_spec["runs"][split]["expected"]
            if len(rows) != expected:
                raise RuntimeError(
                    f"{model_tag}/{split}: expected {expected} rows, found {len(rows)}"
                )
        available_baselines[model_tag] = model_rows

    comparison_rows = [{
        "model": "SFT Qwen3-VL-8B (no index)",
        "heldout_task_rate": task_run["fsm_rate"],
        "heldout_task_successes": task_run["fsm_successes"],
        "heldout_task_attempts": task_run["attempts"],
        "heldout_trajectory_rate": trajectory_run["fsm_rate"],
        "heldout_trajectory_successes": trajectory_run["fsm_successes"],
        "heldout_trajectory_attempts": trajectory_run["attempts"],
    }]
    for model_tag, model_rows in available_baselines.items():
        task_items = model_rows["heldout_tasks"]
        trajectory_items = model_rows["heldout_trajectories"]
        task_successes = sum(bool(row.get("fsm_goal_satisfied")) for row in task_items)
        trajectory_successes = sum(
            bool(row.get("fsm_goal_satisfied")) for row in trajectory_items
        )
        comparison_rows.append({
            "model": BASELINE_RUNS[model_tag]["label"],
            "heldout_task_rate": task_successes / len(task_items),
            "heldout_task_successes": task_successes,
            "heldout_task_attempts": len(task_items),
            "heldout_trajectory_rate": trajectory_successes / len(trajectory_items),
            "heldout_trajectory_successes": trajectory_successes,
            "heldout_trajectory_attempts": len(trajectory_items),
        })

    tool_glossary = [
        {"tool": "get_image", "plain_meaning": "Look again", "world_effect": "No", "coordination_role": "Refreshes only that agent's visual observation."},
        {"tool": "communicate", "plain_meaning": "Send a message", "world_effect": "No", "coordination_role": "Shares text; a `releases` tag can wake a matching waiter."},
        {"tool": "wait_for_signal", "plain_meaning": "Pause until released", "world_effect": "No", "coordination_role": "Blocks this agent until the named partner sends the matching release tag."},
        {"tool": "give_space", "plain_meaning": "Move out of the way", "world_effect": "Yes", "coordination_role": "Physically clears a shared fixture; it does not send the release message itself."},
        {"tool": "navigate_to_fixture", "plain_meaning": "Move to a fixture", "world_effect": "Yes", "coordination_role": "Positions the agent before manipulation."},
        {"tool": "pick_up_object", "plain_meaning": "Pick something up", "world_effect": "Yes", "coordination_role": "Changes the object held by the agent."},
        {"tool": "place_on_object / place_in_receptacle", "plain_meaning": "Put something at its goal", "world_effect": "Yes", "coordination_role": "Often completes an FSM goal predicate."},
    ]

    source = {
        "id": "no_index_corrected_eval",
        "label": "Corrected no-index evaluation outputs (jobs 290438 and 290439)",
        "path": "training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/source_queries.sql",
    }
    comparison_source = {
        "id": "no_index_model_comparison",
        "label": "Corrected tick30 no-index live-sim outputs for SFT Qwen, base Qwen, Gemini 3 Flash, and Gemini Robotics ER2",
        "path": "training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/source_queries.sql",
    }

    cards = [
        {
            "id": "heldout_tasks_fsm",
            "dataset": "summary",
            "filter": {"run_id": "heldout_tasks"},
            "description": "FSM goal satisfaction on six task types excluded from training.",
            "sourceId": source["id"],
            "metrics": [
                {"label": "Held-out task FSM success", "field": "fsm_rate", "format": "percent"},
                {"label": "95% CI low", "field": "fsm_ci_low", "format": "percent"},
                {"label": "95% CI high", "field": "fsm_ci_high", "format": "percent"},
            ],
        },
        {
            "id": "heldout_traj_fsm",
            "dataset": "summary",
            "filter": {"run_id": "heldout_trajectories"},
            "description": "FSM goal satisfaction on unseen trajectories from 47 trained task types.",
            "sourceId": source["id"],
            "metrics": [
                {"label": "Held-out trajectory FSM success", "field": "fsm_rate", "format": "percent"},
                {"label": "95% CI low", "field": "fsm_ci_low", "format": "percent"},
                {"label": "95% CI high", "field": "fsm_ci_high", "format": "percent"},
            ],
        },
        {
            "id": "heldout_task_coverage",
            "dataset": "summary",
            "filter": {"run_id": "heldout_tasks"},
            "description": "Completed evaluation episodes across six unseen task types.",
            "sourceId": source["id"],
            "metrics": [{"label": "Held-out task episodes", "field": "attempts", "format": "number"}],
        },
        {
            "id": "heldout_traj_coverage",
            "dataset": "summary",
            "filter": {"run_id": "heldout_trajectories"},
            "description": "Completed unseen trajectories across 47 trained task types.",
            "sourceId": source["id"],
            "metrics": [{"label": "Held-out trajectory episodes", "field": "attempts", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "success_by_split",
            "title": "FSM success rate by generalization split",
            "subtitle": "95% Wilson intervals are 25.4–44.7% and 45.0–61.3%, respectively.",
            "showDescription": True,
            "intent": "comparison",
            "question": "How well does the no-index model generalize to new task types and new trajectories?",
            "rationale": "Two discrete evaluation cohorts are best compared with a simple bar chart and exact rates.",
            "type": "bar",
            "dataset": "summary",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Evaluation split"},
                "y": {"field": "fsm_rate", "type": "quantitative", "format": "percent", "label": "FSM success rate"},
                "tooltip": [
                    {"field": "fsm_successes", "type": "quantitative", "label": "FSM successes"},
                    {"field": "attempts", "type": "quantitative", "label": "Attempts"},
                    {"field": "fsm_ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"},
                    {"field": "fsm_ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "progress_composition",
            "title": "Goal progress at episode termination",
            "subtitle": "Failures are separated into episodes with some FSM subgoals completed and episodes with no recorded goal progress.",
            "showDescription": True,
            "intent": "composition",
            "question": "When the model fails, does it still make useful task progress?",
            "rationale": "Stacked composition distinguishes full success, partial completion, and zero-progress failure within each fixed cohort.",
            "type": "stackedBar",
            "dataset": "progress_composition",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Evaluation split"},
                "y": {"fields": ["fsm_success", "partial_progress_failure", "zero_progress_failure"], "type": "quantitative", "format": "number", "label": "Episodes"},
            },
            "layout": "full",
        },
        {
            "id": "heldout_task_rates",
            "title": "FSM success by held-out task type",
            "subtitle": "Fifteen episodes per task; these six task types were excluded from training.",
            "showDescription": True,
            "intent": "comparison",
            "question": "Which unseen task types transfer successfully?",
            "rationale": "A horizontal bar chart keeps long task names readable and exposes task-specific transfer.",
            "type": "horizontalBar",
            "dataset": "heldout_tasks",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "task", "type": "nominal", "label": "Task"},
                "y": {"field": "fsm_rate", "type": "quantitative", "format": "percent", "label": "FSM success rate"},
                "tooltip": [
                    {"field": "fsm_successes", "type": "quantitative", "label": "Successes"},
                    {"field": "attempts", "type": "quantitative", "label": "Attempts"},
                    {"field": "mean_partial_goal", "type": "quantitative", "format": "percent", "label": "Mean goal completion"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "trained_task_distribution",
            "title": "Task-level success distribution on held-out trajectories",
            "subtitle": "Forty-seven trained task types, each evaluated on three unseen trajectories.",
            "showDescription": True,
            "intent": "distribution",
            "question": "Is performance broad across trained tasks or concentrated in a subset?",
            "rationale": "Four exact outcome bins show how many tasks succeed consistently, inconsistently, or never.",
            "type": "bar",
            "dataset": "rate_distribution",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "outcome", "type": "ordinal", "label": "Successful trajectories per task"},
                "y": {"field": "task_count", "type": "quantitative", "format": "number", "label": "Task types"},
                "tooltip": [
                    {"field": "share", "type": "quantitative", "format": "percent", "label": "Share of task types"},
                    {"field": "tasks", "type": "text", "label": "Tasks"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "fsm_native",
            "title": "FSM and native success rates",
            "subtitle": "Native success is shown only as a physical-execution diagnostic, not as the model-learning score.",
            "showDescription": True,
            "intent": "comparison",
            "question": "How often does symbolic success also satisfy RoboCasa's physical success check?",
            "rationale": "Grouped bars make the diagnostic gap visible without conflating the two definitions.",
            "type": "bar",
            "dataset": "summary",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "split", "type": "nominal", "label": "Evaluation split"},
                "y": {"fields": ["fsm_rate", "native_rate"], "type": "quantitative", "format": "percent", "label": "Success rate"},
            },
            "layout": "full",
        },
    ]
    comparison_complete = len(comparison_rows) == 1 + len(BASELINE_RUNS)
    if comparison_complete:
        charts.append({
            "id": "model_comparison",
            "title": "SFT versus out-of-the-box models",
            "subtitle": "FSM success on identical corrected tick30 no-index live-sim cohorts: 90 held-out-task and 141 held-out-trajectory episodes per model.",
            "showDescription": True,
            "intent": "comparison",
            "question": "How does the no-index SFT model compare with its base Qwen checkpoint, Gemini 3 Flash, and Gemini Robotics ER2 out of the box?",
            "rationale": "One grouped bar chart compares both generalization splits across all four models without pooling their denominators.",
            "type": "bar",
            "dataset": "model_comparison",
            "sourceId": comparison_source["id"],
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {
                    "fields": ["heldout_task_rate", "heldout_trajectory_rate"],
                    "type": "quantitative",
                    "format": "percent",
                    "label": "FSM success rate",
                },
            },
            "layout": "full",
        })

    tables = [
        {
            "id": "run_results",
            "title": "Run-level results",
            "subtitle": "Exact counts and diagnostics for both completed no-index evaluation cohorts.",
            "showDescription": True,
            "dataset": "summary",
            "defaultSort": {"field": "fsm_rate", "direction": "desc"},
            "density": "spacious",
            "sourceId": source["id"],
            "layout": "full",
            "columns": [
                {"field": "split", "label": "Split", "type": "text"},
                {"field": "attempts", "label": "N", "format": "number"},
                {"field": "fsm_successes", "label": "FSM successes", "format": "number"},
                {"field": "fsm_rate", "label": "FSM rate", "format": "percent"},
                {"field": "fsm_ci_low", "label": "95% CI low", "format": "percent"},
                {"field": "fsm_ci_high", "label": "95% CI high", "format": "percent"},
                {"field": "native_successes", "label": "Native successes", "format": "number"},
                {"field": "mean_partial_goal", "label": "Mean goal completion", "format": "percent"},
                {"field": "mean_rejections", "label": "Mean rejections", "format": "number"},
            ],
        },
        {
            "id": "task_results",
            "title": "Complete task-level results",
            "subtitle": "All six held-out task types and all 47 trained task types represented in the held-out-trajectory evaluation.",
            "showDescription": True,
            "dataset": "all_tasks",
            "defaultSort": {"field": "fsm_rate", "direction": "desc"},
            "density": "dense",
            "sourceId": source["id"],
            "layout": "full",
            "columns": [
                {"field": "split", "label": "Split", "type": "text"},
                {"field": "task", "label": "Task", "type": "text"},
                {"field": "attempts", "label": "N", "format": "number"},
                {"field": "fsm_successes", "label": "FSM successes", "format": "number"},
                {"field": "fsm_rate", "label": "FSM rate", "format": "percent"},
                {"field": "mean_partial_goal", "label": "Mean goal completion", "format": "percent"},
                {"field": "mean_rejections", "label": "Mean rejections", "format": "number"},
                {"field": "native_rate", "label": "Native rate", "format": "percent"},
            ],
        },
        {
            "id": "tool_glossary",
            "title": "Tool calls in plain English",
            "subtitle": "Observation and communication calls do not move objects; manipulation calls do.",
            "showDescription": True,
            "dataset": "tool_glossary",
            "density": "spacious",
            "sourceId": source["id"],
            "layout": "full",
            "columns": [
                {"field": "tool", "label": "Tool", "type": "text"},
                {"field": "plain_meaning", "label": "Plain meaning", "type": "text"},
                {"field": "world_effect", "label": "Changes sim?", "type": "text"},
                {"field": "coordination_role", "label": "What it does in the process", "type": "text"},
            ],
        },
        {
            "id": "trajectory_examples",
            "title": "The three real episodes used below",
            "subtitle": "All are successful corrected no-index live-sim rollouts; native success is shown separately as a physical diagnostic.",
            "showDescription": True,
            "dataset": "trajectory_examples",
            "density": "spacious",
            "sourceId": source["id"],
            "layout": "full",
            "columns": [
                {"field": "example", "label": "Example", "type": "text"},
                {"field": "title", "label": "Coordination pattern", "type": "text"},
                {"field": "split", "label": "Split", "type": "text"},
                {"field": "task", "label": "Task", "type": "text"},
                {"field": "trajectory_id", "label": "Trajectory", "type": "text"},
                {"field": "fsm_success", "label": "FSM success", "type": "text"},
                {"field": "native_success", "label": "Native success", "type": "text"},
                {"field": "wait_calls", "label": "Waits", "format": "number"},
                {"field": "total_tool_calls", "label": "Total calls", "format": "number"},
            ],
        },
    ]
    for example_number, example in enumerate(trajectory_examples, 1):
        tables.append({
            "id": f"trajectory_timeline_{example_number}",
            "title": f"Example {example_number}: {example['task']} / {example['trajectory_id']}",
            "subtitle": "Complete generated trajectory; one invocation per cell so observation, communication, and physical calls remain visible.",
            "showDescription": True,
            "dataset": f"trajectory_timeline_{example_number}",
            "density": "spacious",
            "sourceId": source["id"],
            "layout": "full",
            "columns": [
                {"field": "step_number", "label": "Step # (tick.invocation)", "type": "text"},
                {"field": "agent_0_tool_call", "label": "Agent 0 tool call", "type": "text"},
                {"field": "agent_1_tool_call", "label": "Agent 1 tool call", "type": "text"},
            ],
        })

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Corrected No-Index Evaluation Results", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## The no-index model generalizes better to new trajectories than to entirely new tasks\n\n"
                f"The model satisfied the concurrent FSM in **{trajectory_run['fsm_successes']}/{trajectory_run['attempts']} held-out trajectories ({trajectory_run['fsm_rate']:.1%})** and "
                f"**{task_run['fsm_successes']}/{task_run['attempts']} held-out-task episodes ({task_run['fsm_rate']:.1%})**. "
                "This is a clear descriptive gap between generalizing within trained task types and transferring to unseen task types. Performance is also strongly task-specific: some task types succeed consistently, while others never succeed in the available trials.\n\n"
                "The two jobs completed all expected episodes with valid JSON, no native-evaluation errors, and zero invalid array-valued `communicate.releases` outputs."
            ),
            "layout": "full",
        },
        {"id": "metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"},
        {
            "id": "overall_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Overall generalization performance\n\n"
                f"Held-out trajectory success is {trajectory_run['fsm_rate'] - task_run['fsm_rate']:.1%} higher than held-out-task success. "
                f"The corresponding 95% Wilson intervals are {trajectory_run['fsm_ci_low']:.1%}–{trajectory_run['fsm_ci_high']:.1%} and {task_run['fsm_ci_low']:.1%}–{task_run['fsm_ci_high']:.1%}. "
                "The cohorts measure different forms of generalization and should not be pooled into a single headline rate."
            ),
            "layout": "full",
        },
        {"id": "success_chart", "type": "chart", "chartId": "success_by_split", "layout": "full"},
        {
            "id": "progress_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Failed episodes often still complete part of the goal\n\n"
                "The progress composition separates failures that achieved at least one FSM subgoal from failures with no recorded goal completion. This matters because a budget-exhausted episode at substantial partial completion reflects a different weakness from an episode that never advances the task."
            ),
            "layout": "full",
        },
        {"id": "progress_chart", "type": "chart", "chartId": "progress_composition", "layout": "full"},
        {
            "id": "unseen_tasks_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Transfer to unseen task types is uneven\n\n"
                "Each held-out task has 15 trials, making this the most stable task-level view in the report. The spread shows that the aggregate 34.4% rate is not a uniform level of competence: it combines tasks with substantial transfer and tasks with little or no transfer."
            ),
            "layout": "full",
        },
        {"id": "heldout_task_chart", "type": "chart", "chartId": "heldout_task_rates", "layout": "full"},
        {
            "id": "trained_tasks_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Success on trained task types is concentrated\n\n"
                f"Among 47 trained task types, **{len(perfect)} succeeded on all three unseen trajectories** and **{len(zero)} succeeded on none**. "
                "Because there are only three trials per task, these bins are coarse; they are best used to find consistently easy or difficult task families for qualitative inspection."
            ),
            "layout": "full",
        },
        {"id": "distribution_chart", "type": "chart", "chartId": "trained_task_distribution", "layout": "full"},
        {
            "id": "easy_hard_examples",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Consistently successful and unsuccessful trained tasks\n\n"
                "### 3/3 FSM success\n" + "\n".join(f"- `{task}`" for task in perfect) +
                "\n\n### 0/3 FSM success\n" + "\n".join(f"- `{task}`" for task in zero) +
                "\n\nThese lists describe the three sampled trajectories, not guaranteed task-level probabilities."
            ),
            "layout": "full",
        },
        {
            "id": "fsm_native_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## FSM success remains the model-learning metric\n\n"
                "FSM success asks whether the generated symbolic tool sequence satisfies the concurrent task goal. Native success asks whether RoboCasa's physical `_check_success()` accepts the resulting MuJoCo state. Native failures after FSM success can reflect teleportation precision, contact geometry, physics settling, or native thresholds, so native success is retained as a diagnostic rather than used to judge learned task completion."
            ),
            "layout": "full",
        },
        {"id": "fsm_native_chart", "type": "chart", "chartId": "fsm_native", "layout": "full"},
        {
            "id": "definitions",
            "type": "markdown",
            "body": (
                "## Scope and metric definitions\n\n"
                "- **Held-out tasks:** six task types excluded from training, 15 trajectories per task, 90 episodes total.\n"
                "- **Held-out trajectories:** three unseen trajectories for each of 47 task types represented in training, 141 episodes total.\n"
                "- **FSM success rate:** episodes with `fsm_goal_satisfied=true` divided by all completed episodes in that cohort.\n"
                "- **Partial goal completion:** the fraction of FSM goal predicates satisfied at termination.\n"
                "- **Rejected steps:** generated actions rejected by schema, semantic precondition, or resource-conflict checks."
            ),
            "layout": "full",
        },
        {
            "id": "trajectory_process",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## How a live-sim trajectory works, in plain English\n\n"
                "A trajectory is the saved record of two agents taking turns in one running simulator. Time is divided into **ticks**. At each tick, every agent that is ready sees its own images and history, then proposes one tool call. Calls at the same tick may run concurrently when they do not compete for the same object or fixture.\n\n"
                "The runtime checks each proposal before executing it. A legal physical call changes the simulated kitchen; an illegal or conflicting call is rejected and the agent can try again on a later tick. After every executed physical action, the concurrent FSM checks how much of the task goal is complete. The episode ends successfully when every required FSM condition is satisfied."
            ),
            "layout": "full",
        },
        {"id": "tool_glossary_block", "type": "table", "tableId": "tool_glossary", "layout": "full"},
        {
            "id": "wait_for_signal_explainer",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## `wait_for_signal` is a precise pause-and-release handshake\n\n"
                "Suppose Agent 1 needs the counter but Agent 0 is using it:\n\n"
                "1. Agent 1 calls `wait_for_signal(from=\"agent_0\", about=\"dining_counter\")`. Agent 1 is now paused.\n"
                "2. Agent 0 finishes, calls `give_space(fixture_id=\"dining_counter\")`, and physically moves away.\n"
                "3. Agent 0 then calls `communicate(..., releases=\"dining_counter\")`. The ordinary message explains what happened; the exact `releases` tag is what wakes Agent 1.\n"
                "4. Agent 1 becomes eligible to act on the next tick.\n\n"
                "A normal chat message is not enough, and `give_space` does not wake anyone by itself. The sender, receiver, and resource tag must match the pending wait. While one agent is paused, the other can continue acting; the whole simulator does not stop."
            ),
            "layout": "full",
        },
        {
            "id": "real_trajectory_examples_text",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Three real trajectories from the corrected evaluation\n\n"
                "These are not hand-written demonstrations. They are complete successful tool sequences emitted by the trained no-index model during the corrected live-sim evaluation. The first is a compact one-way handoff, the second hands the same counter back and forth, and the third coordinates two different resources on an unseen task. Every generated call is retained, including `get_image`, communication, waits, physical actions, rejected calls, and retries."
            ),
            "layout": "full",
        },
        {"id": "trajectory_examples_block", "type": "table", "tableId": "trajectory_examples", "layout": "full"},
        {
            "id": "trajectory_timeline_guide",
            "type": "markdown",
            "body": (
                "### How to read the timeline\n\n"
                "Each example now has its own table. Read downward by virtual simulator tick. When an agent was invoked more than once during one tick, suffixes such as `.1` and `.2` put each invocation in its own row so no call is hidden inside a multiline cell. **Not Invoked** means the scheduler did not ask that agent to act for that invocation—for example, because it was waiting for a signal. `Executed` means the runtime accepted the call. `Rejected` means the world did not change and the model had to replan. The call marked **FSM goal satisfied** completed the symbolic task goal."
            ),
            "layout": "full",
        },
        {"id": "trajectory_timeline_1_block", "type": "table", "tableId": "trajectory_timeline_1", "layout": "full"},
        {"id": "trajectory_timeline_2_block", "type": "table", "tableId": "trajectory_timeline_2", "layout": "full"},
        {"id": "trajectory_timeline_3_block", "type": "table", "tableId": "trajectory_timeline_3", "layout": "full"},
        {"id": "run_table", "type": "table", "tableId": "run_results", "layout": "full"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": source["id"],
            "body": (
                "## Data validation and methodology\n\n"
                "Jobs 290438 and 290439 completed with exit code 0 and produced exactly 90 and 141 records. Every non-empty JSONL line parsed successfully. No episodes were excluded. Rates use the episode as the denominator; task-level rates use all episodes for that task. Overall 95% confidence intervals use the Wilson binomial interval. Task-level three-trial outcomes are reported without confidence intervals because their main purpose is descriptive segmentation."
            ),
            "layout": "full",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and uncertainty\n\n"
                "- This is one trained checkpoint, not a multi-seed estimate of training variability.\n"
                "- The two evaluation splits differ in both task novelty and sample construction, so their rate difference is descriptive rather than a controlled causal effect.\n"
                "- Three held-out trajectories per trained task are insufficient for precise task ranking.\n"
                "- Native success is sensitive to physical execution details that are intentionally outside the primary learning metric.\n"
                "- Task difficulty mechanisms require follow-up trajectory inspection; aggregate associations alone cannot distinguish perception, planning, coordination, and tool-format causes."
            ),
            "layout": "full",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Use this checkpoint and configuration as the no-index baseline for subsequent experiments.\n"
                "2. Inspect representative 3/3, mixed, and 0/3 tasks to separate perception, planning, coordination, and semantic-precondition failures.\n"
                "3. Repeat training across multiple seeds before treating the observed rates as stable configuration performance.\n"
                "4. Build a prospective task-difficulty model using task/FSM and expert-trajectory features, then validate it against these corrected task outcomes."
            ),
            "layout": "full",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Which failure categories dominate within the consistently unsuccessful tasks?\n"
                "- Does success correlate more strongly with goal-predicate type, expert-plan length, visual ambiguity, or coordination dependencies?\n"
                "- How stable are the held-out-task results across additional initial states and training seeds?"
            ),
            "layout": "full",
        },
        {"id": "task_table", "type": "table", "tableId": "task_results", "layout": "full"},
    ]
    if comparison_complete:
        comparison_block_index = next(
            index for index, block in enumerate(blocks)
            if block["id"] == "progress_text"
        )
        comparison_by_model = {row["model"]: row for row in comparison_rows}
        sft_row = comparison_by_model["SFT Qwen3-VL-8B (no index)"]
        qwen_row = comparison_by_model["Qwen3-VL-8B (out of box)"]
        flash_row = comparison_by_model["Gemini 3 Flash (out of box)"]
        er2_row = comparison_by_model["Gemini Robotics ER2 (out of box)"]
        blocks[comparison_block_index:comparison_block_index] = [
            {
                "id": "model_comparison_text",
                "type": "markdown",
                "sourceId": comparison_source["id"],
                "body": (
                    "## Fine-tuning versus out-of-the-box capability\n\n"
                    "All four models use the same corrected tick30 manifests, private per-agent context, consume-once images, no step indices, runtime tools, budgets, and FSM success criterion. The hosted Gemini baselines use native function calls; base Qwen and the SFT checkpoint emit the equivalent tool-call structure in text.\n\n"
                    f"The SFT model reaches **{sft_row['heldout_task_rate']:.1%} / {sft_row['heldout_trajectory_rate']:.1%}** on held-out tasks / trajectories, compared with "
                    f"**{qwen_row['heldout_task_rate']:.1%} / {qwen_row['heldout_trajectory_rate']:.1%}** for the untouched Qwen3-VL-8B base checkpoint, "
                    f"**{flash_row['heldout_task_rate']:.1%} / {flash_row['heldout_trajectory_rate']:.1%}** for Gemini 3 Flash and "
                    f"**{er2_row['heldout_task_rate']:.1%} / {er2_row['heldout_trajectory_rate']:.1%}** for Gemini Robotics ER2."
                ),
                "layout": "full",
            },
            {"id": "model_comparison_chart", "type": "chart", "chartId": "model_comparison", "layout": "full"},
        ]

    datasets = {
        "summary": summary,
        "progress_composition": progress_composition,
        "heldout_tasks": heldout_tasks,
        "trajectory_tasks": trajectory_tasks,
        "rate_distribution": rate_distribution,
        "all_tasks": all_tasks,
        "tool_glossary": tool_glossary,
        "trajectory_examples": trajectory_examples,
        **trajectory_timelines,
        "model_comparison": comparison_rows,
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Corrected No-Index Evaluation Results",
            "description": "Comprehensive FSM-first analysis of no-index performance on held-out tasks and held-out trajectories.",
            "generatedAt": generated_at,
            "sources": [source] + ([comparison_source] if comparison_complete else []),
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": datasets},
        "sources": [source] + ([comparison_source] if comparison_complete else []),
    }
    chart_map = {
        "charts": [
            {"section": "Overall", "question": "How does generalization differ by split?", "family": "comparison", "type": "bar", "fields": ["split", "fsm_rate"], "claim": "Within-task trajectory generalization exceeds unseen-task transfer"},
            {"section": "Progress", "question": "Do failed episodes make partial progress?", "family": "composition", "type": "stacked bar", "fields": ["fsm_success", "partial_progress_failure", "zero_progress_failure"], "claim": "Failure severity varies"},
            {"section": "Unseen tasks", "question": "Which unseen tasks transfer?", "family": "ranking", "type": "horizontal bar", "fields": ["task", "fsm_rate"], "claim": "Task transfer is uneven"},
            {"section": "Trained tasks", "question": "How broad is success?", "family": "distribution", "type": "bar", "fields": ["outcome", "task_count"], "claim": "Success is concentrated"},
            {"section": "Physical diagnostic", "question": "How often does native success accompany FSM success?", "family": "comparison", "type": "grouped bar", "fields": ["fsm_rate", "native_rate"], "claim": "Physical checks are stricter than symbolic completion"},
        ]
    }
    if comparison_complete:
        chart_map["charts"].append({
            "section": "Model comparison",
            "question": "How does SFT compare with out-of-the-box Gemini models under the same no-index live-sim protocol?",
            "family": "comparison",
            "type": "grouped bar",
            "fields": ["model", "heldout_task_rate", "heldout_trajectory_rate"],
            "claim": "Directly compares both generalization cohorts across the SFT model, Gemini 3 Flash, and Gemini Robotics ER2",
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "analysis_snapshot.json").write_text(json.dumps(datasets, indent=2) + "\n")
    (OUTPUT_DIR / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (OUTPUT_DIR / "chart_map.json").write_text(json.dumps(chart_map, indent=2) + "\n")
    print(OUTPUT_DIR / "artifact.json")


if __name__ == "__main__":
    main()
