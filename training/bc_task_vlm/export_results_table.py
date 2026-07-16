"""Exports the consolidated out-of-the-box/SFT eval table (markdown + CSV).

Collects structured_eval_metrics.json + comm_judge_metrics.json from every
known run dir under eval_runs/ and writes eval_runs/results_table.{csv,md}.
Runs whose outputs are missing are listed as pending rather than failing.

Usage:
    python -m training.bc_task_vlm.export_results_table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path("training/bc_task_vlm/eval_runs")

# (model, prompt variant, split, run dir)
RUNS = [
    ("Gemini-3 Flash", "baseline", "heldout_trajectories", "gemini3flash__heldout_trajectories"),
    ("Gemini-3 Flash", "baseline", "heldout_tasks", "gemini3flash__heldout_tasks"),
    ("Gemini-3 Flash", "+task spec", "heldout_trajectories", "gemini3flash_specdetail__heldout_trajectories"),
    ("Gemini-3 Flash", "+task spec", "heldout_tasks", "gemini3flash_specdetail__heldout_tasks"),
    ("Qwen3.6-27B base", "baseline", "heldout_trajectories", "qwen36_27b_base__heldout_trajectories"),
    ("Qwen3.6-27B base", "baseline", "heldout_tasks", "qwen36_27b_base__heldout_tasks"),
    ("Qwen3.6-27B base", "+task spec", "heldout_trajectories", "qwen36_27b_base_specdetail__heldout_trajectories"),
    ("Qwen3.6-27B base", "+task spec", "heldout_tasks", "qwen36_27b_base_specdetail__heldout_tasks"),
    ("Qwen3.6-27B base", "thinking default", "heldout_trajectories", "qwen36_27b_base_thinking__heldout_trajectories"),
    ("Qwen3.6-27B base", "thinking default", "heldout_tasks", "qwen36_27b_base_thinking__heldout_tasks"),
    ("Qwen3.6-27B base", "+few-shot", "heldout_trajectories", "qwen36_27b_base_fewshot__heldout_trajectories"),
    ("Qwen3.6-27B base", "+few-shot", "heldout_tasks", "qwen36_27b_base_fewshot__heldout_tasks"),
    ("Qwen3.6-27B base", "+spec+few-shot", "heldout_trajectories", "qwen36_27b_base_spec_fewshot__heldout_trajectories"),
    ("Qwen3.6-27B base", "+spec+few-shot", "heldout_tasks", "qwen36_27b_base_spec_fewshot__heldout_tasks"),
    ("Qwen3.6-27B base", "think+spec+few-shot", "heldout_trajectories", "qwen36_27b_base_think_spec_fewshot__heldout_trajectories"),
    ("Qwen3.6-27B base", "think+spec+few-shot", "heldout_tasks", "qwen36_27b_base_think_spec_fewshot__heldout_tasks"),
    ("Qwen3.6-27B base", "temp 0.7", "heldout_trajectories", "qwen36_27b_base_temp07__heldout_trajectories"),
    ("Qwen3.6-27B base", "temp 0.7", "heldout_tasks", "qwen36_27b_base_temp07__heldout_tasks"),
    ("Gemini-3 Flash", "thinking default", "heldout_trajectories", "gemini3flash_thinkdefault__heldout_trajectories"),
    ("Gemini-3 Flash", "thinking default", "heldout_tasks", "gemini3flash_thinkdefault__heldout_tasks"),
    ("Gemini-3 Flash", "temp 0.7", "heldout_trajectories", "gemini3flash_temp07__heldout_trajectories"),
    ("Gemini-3 Flash", "temp 0.7", "heldout_tasks", "gemini3flash_temp07__heldout_tasks"),
    ("Gemini-3 Flash", "+few-shot", "heldout_trajectories", "gemini3flash_fewshot__heldout_trajectories"),
    ("Gemini-3 Flash", "+few-shot", "heldout_tasks", "gemini3flash_fewshot__heldout_tasks"),
    ("Gemini-3 Flash", "+spec+few-shot", "heldout_trajectories", "gemini3flash_spec_fewshot__heldout_trajectories"),
    ("Gemini-3 Flash", "+spec+few-shot", "heldout_tasks", "gemini3flash_spec_fewshot__heldout_tasks"),
    ("Gemini-3 Flash", "think+spec+few-shot", "heldout_trajectories", "gemini3flash_think_spec_fewshot__heldout_trajectories"),
    ("Gemini-3 Flash", "think+spec+few-shot", "heldout_tasks", "gemini3flash_think_spec_fewshot__heldout_tasks"),
    ("Gemini-3.5 Flash", "baseline", "heldout_trajectories", "gemini35flash_base__heldout_trajectories"),
    ("Gemini-3.5 Flash", "baseline", "heldout_tasks", "gemini35flash_base__heldout_tasks"),
    ("Gemini-3.5 Flash", "+task spec", "heldout_trajectories", "gemini35flash_specdetail__heldout_trajectories"),
    ("Gemini-3.5 Flash", "+task spec", "heldout_tasks", "gemini35flash_specdetail__heldout_tasks"),
    ("Gemini-3.5 Flash", "thinking default", "heldout_trajectories", "gemini35flash_thinkdefault__heldout_trajectories"),
    ("Gemini-3.5 Flash", "thinking default", "heldout_tasks", "gemini35flash_thinkdefault__heldout_tasks"),
    ("Gemini-3.5 Flash", "+few-shot", "heldout_trajectories", "gemini35flash_fewshot__heldout_trajectories"),
    ("Gemini-3.5 Flash", "+few-shot", "heldout_tasks", "gemini35flash_fewshot__heldout_tasks"),
    ("Gemini-3.5 Flash", "+spec+few-shot", "heldout_trajectories", "gemini35flash_spec_fewshot__heldout_trajectories"),
    ("Gemini-3.5 Flash", "+spec+few-shot", "heldout_tasks", "gemini35flash_spec_fewshot__heldout_tasks"),
    ("Gemini-3.5 Flash", "+spec+few-shot (native tools control)", "heldout_trajectories", "gemini35flash_nativetools_spec_fewshot__heldout_trajectories"),
    ("Gemini-3.5 Flash", "+spec+few-shot (native tools control)", "heldout_tasks", "gemini35flash_nativetools_spec_fewshot__heldout_tasks"),
    ("Gemini-3.5 Flash", "think+spec+few-shot", "heldout_trajectories", "gemini35flash_think_spec_fewshot__heldout_trajectories"),
    ("Gemini-3.5 Flash", "think+spec+few-shot", "heldout_tasks", "gemini35flash_think_spec_fewshot__heldout_tasks"),
    ("Gemini-3.1 Pro", "baseline", "heldout_trajectories", "gemini31pro_base__heldout_trajectories"),
    ("Gemini-3.1 Pro", "baseline", "heldout_tasks", "gemini31pro_base__heldout_tasks"),
    ("Gemini-3.1 Pro", "+task spec", "heldout_trajectories", "gemini31pro_specdetail__heldout_trajectories"),
    ("Gemini-3.1 Pro", "+task spec", "heldout_tasks", "gemini31pro_specdetail__heldout_tasks"),
    ("Gemini-3.1 Pro", "think+spec+few-shot", "heldout_trajectories", "gemini31pro_think_spec_fewshot__heldout_trajectories"),
    ("Gemini-3.1 Pro", "think+spec+few-shot", "heldout_tasks", "gemini31pro_think_spec_fewshot__heldout_tasks"),
    # The 8B runs are plain "baseline" prompts (no spec/few-shot/thinking); they
    # differ only in output interface, marked with * = native tool-calling.
    ("Qwen3-VL-8B base", "baseline", "heldout_trajectories", "qwen3vl_8b_base_tc__heldout_trajectories"),
    ("Qwen3-VL-8B base", "baseline", "heldout_tasks", "qwen3vl_8b_base_tc__heldout_tasks"),
    ("Qwen3-VL-8B base", "baseline (plain fmt control)", "heldout_trajectories", "qwen3vl_8b_base_plain__heldout_trajectories"),
    ("Qwen3-VL-8B base", "baseline (plain fmt control)", "heldout_tasks", "qwen3vl_8b_base_plain__heldout_tasks"),
    ("Qwen3-VL-8B SFT", "baseline", "heldout_trajectories", "qwen3vl_8b_sft__heldout_trajectories"),
    ("Qwen3-VL-8B SFT", "baseline", "heldout_tasks", "qwen3vl_8b_sft__heldout_tasks"),
    ("Qwen3.6-27B SFT", "baseline", "heldout_trajectories", "qwen36_27b_sft__heldout_trajectories"),
    ("Qwen3.6-27B SFT", "baseline", "heldout_tasks", "qwen36_27b_sft__heldout_tasks"),
]

COLUMNS = [
    ("n", "num_samples", "{:.0f}"),
    ("parse_rate", "tool_call_parse_rate", "{:.3f}"),
    ("valid_rate", "tool_call_valid_rate", "{:.3f}"),
    ("tool_name_acc", "tool_name_accuracy", "{:.3f}"),
    ("exact_call_acc", "exact_tool_call_accuracy", "{:.3f}"),
    ("action_tool_acc", "action_tool_name_accuracy", "{:.3f}"),
    ("action_exact_acc", "action_exact_call_accuracy", "{:.3f}"),
    ("comm_fraction", "comm_step_fraction", "{:.2f}"),
    ("comm_tool_rate", "comm_tool_selected_rate", "{:.3f}"),
    ("traj_all_steps", "trajectory_all_steps_correct_rate", "{:.3f}"),
    ("traj_prefix", "trajectory_mean_correct_prefix_fraction", "{:.3f}"),
]
JUDGE_COLUMNS = [
    ("comm_judged_acc", "comm_judged_match_rate", "{:.3f}"),
    ("judged_overall_acc", "judged_exact_call_accuracy", "{:.3f}"),
    ("judged_traj_all", "judged_trajectory_all_steps_rate", "{:.3f}"),
    ("judged_traj_prefix", "judged_trajectory_mean_prefix_fraction", "{:.3f}"),
]


def collect_row(run_dir: Path) -> dict[str, str] | None:
    metrics_path = run_dir / "structured_eval_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    if "structured_eval_action_exact_call_accuracy" not in metrics:
        # Older run predating the comm/action split: recompute from records.
        from training.bc_task_vlm.evaluation import (
            load_prediction_records,
            metrics_from_prediction_records,
        )

        metrics.update(
            metrics_from_prediction_records(
                load_prediction_records(run_dir / "structured_eval_predictions.jsonl")
            )
        )
    judge_path = run_dir / "comm_judge_metrics.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else {}
    row: dict[str, str] = {}
    for column, key, fmt in COLUMNS:
        value = metrics.get("structured_eval_" + key)
        row[column] = fmt.format(value) if value is not None else ""
    for column, key, fmt in JUDGE_COLUMNS:
        value = judge.get(key)
        row[column] = fmt.format(value) if value is not None else ""
    return row


def main() -> None:
    header = ["model", "prompt", "split"] + [c for c, _, _ in COLUMNS] + [
        c for c, _, _ in JUDGE_COLUMNS
    ]
    rows: list[list[str]] = []
    pending: list[str] = []
    for model, variant, split, dirname in RUNS:
        row = collect_row(BASE / dirname)
        if row is None:
            pending.append(f"{model} / {variant} / {split} ({dirname})")
            continue
        rows.append([model, variant, split] + [row[c] for c in header[3:]])

    csv_path = BASE / "results_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    md_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    md_lines += ["| " + " | ".join(row) + " |" for row in rows]
    if pending:
        md_lines += ["", "Pending runs:"] + [f"- {p}" for p in pending]
    md_path = BASE / "results_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {md_path} ({len(rows)} rows, {len(pending)} pending)")


if __name__ == "__main__":
    main()
