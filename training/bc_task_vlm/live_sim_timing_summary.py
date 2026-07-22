"""Summarize component timings emitted by live_sim_timing_benchmark."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
ROOT = args.output_dir


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def stats(values):
    values = list(values)
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p95": pct(values, 0.95),
        "max": max(values) if values else None,
    }


payload = json.loads((ROOT / "timing_events.json").read_text())
events = payload["events"]
rows = [
    json.loads(line)
    for line in (ROOT / "live_sim_trajectories.jsonl").read_text().splitlines()
]
metrics = json.loads((ROOT / "live_sim_metrics.json").read_text())
keys = ("trajectory_id", "step_index", "pass_ordinal")
pass_by_key = {tuple(row.get(k) for k in keys): row for row in events["inference_pass"]}
generation_rows = []
for row in events["generation"]:
    enriched = dict(row)
    matched = pass_by_key[tuple(row.get(k) for k in keys)]
    enriched["result_tool"] = matched.get("result_tool")
    enriched["result_error"] = matched.get("result_error")
    generation_rows.append(enriched)


def group_stats(items, key_fn, value_key="elapsed_s"):
    grouped = defaultdict(list)
    for item in items:
        grouped[str(key_fn(item))].append(item[value_key])
    return {key: stats(values) for key, values in sorted(grouped.items())}


summary = {
    "metrics": metrics,
    "event_counts": {key: len(value) for key, value in events.items()},
    "proposal": stats(row["elapsed_s"] for row in events["proposal_turn"]),
    "proposal_by_final_tool": group_stats(
        events["proposal_turn"], lambda row: row.get("final_tool")
    ),
    "inference_pass": stats(row["elapsed_s"] for row in events["inference_pass"]),
    "inference_pass_by_ordinal": group_stats(
        events["inference_pass"], lambda row: row["pass_ordinal"]
    ),
    "inference_pass_by_result_tool": group_stats(
        events["inference_pass"], lambda row: row.get("result_tool")
    ),
    "generation_pipeline": stats(row["total_s"] for row in generation_rows),
    "generation_model": stats(row["model_generate_s"] for row in generation_rows),
    "generation_collate": stats(row["collate_s"] for row in generation_rows),
    "generation_h2d": stats(row["host_to_device_s"] for row in generation_rows),
    "generation_decode": stats(row["decode_s"] for row in generation_rows),
    "generation_model_by_ordinal": group_stats(
        generation_rows, lambda row: row["pass_ordinal"], "model_generate_s"
    ),
    "generation_pipeline_by_ordinal": group_stats(
        generation_rows, lambda row: row["pass_ordinal"], "total_s"
    ),
    "generation_model_by_result_tool": group_stats(
        generation_rows, lambda row: row.get("result_tool"), "model_generate_s"
    ),
    "tokens": {
        "prompt": stats(row["prompt_tokens"] for row in generation_rows),
        "output": stats(row["output_tokens"] for row in generation_rows),
    },
    "render": stats(row["elapsed_s"] for row in events["render"]),
    "render_by_views": group_stats(events["render"], lambda row: "+".join(row["views"])),
    "render_by_contains_map": group_stats(events["render"], lambda row: "map" in row["views"]),
    "render_by_ordinal": group_stats(events["render"], lambda row: row["pass_ordinal"]),
    "sim_execute": stats(row["elapsed_s"] for row in events["sim_execute"]),
    "sim_execute_by_tool": group_stats(events["sim_execute"], lambda row: row["tool"]),
    "fsm_step": stats(row["elapsed_s"] for row in events["fsm_step"]),
    "adapt_step": stats(row["elapsed_s"] for row in events["adapt_step"]),
    "trajectory_load": stats(row["elapsed_s"] for row in events["trajectory_load"]),
    "native_check": stats(row["elapsed_s"] for row in events["native_check"]),
    "peak_cuda": {
        "allocated_bytes": max(row["peak_allocated_bytes"] for row in generation_rows),
        "reserved_bytes": max(row["peak_reserved_bytes"] for row in generation_rows),
    },
}

component_totals = {
    "proposal_outer": sum(row["elapsed_s"] for row in events["proposal_turn"]),
    "inference_pass_outer": sum(row["elapsed_s"] for row in events["inference_pass"]),
    "render": sum(row["elapsed_s"] for row in events["render"]),
    "generation_pipeline": sum(row["total_s"] for row in generation_rows),
    "model_generate": sum(row["model_generate_s"] for row in generation_rows),
    "sim_execute": sum(row["elapsed_s"] for row in events["sim_execute"]),
    "fsm_step": sum(row["elapsed_s"] for row in events["fsm_step"]),
    "adapt_step": sum(row["elapsed_s"] for row in events["adapt_step"]),
    "trajectory_load": sum(row["elapsed_s"] for row in events["trajectory_load"]),
    "native_check": sum(row["elapsed_s"] for row in events["native_check"]),
    "episode_elapsed": sum(row["elapsed_s"] for row in rows),
}
component_totals["proposal_unattributed"] = (
    component_totals["proposal_outer"]
    - component_totals["render"]
    - component_totals["generation_pipeline"]
)
component_totals["nonproposal_episode"] = (
    component_totals["episode_elapsed"] - component_totals["proposal_outer"]
)
summary["component_totals"] = component_totals

per_trajectory = []
for result in rows:
    tid = result["trajectory_id"]
    subset = lambda name: [row for row in events[name] if row.get("trajectory_id") == tid]
    proposals = subset("proposal_turn")
    passes = subset("inference_pass")
    renders = subset("render")
    gens = [row for row in generation_rows if row.get("trajectory_id") == tid]
    sims = subset("sim_execute")
    per_trajectory.append(
        {
            "task": result["task_name"],
            "trajectory_id": tid,
            "termination": result["termination"],
            "native": result["native_success"],
            "fsm": result["fsm_goal_satisfied"],
            "episode_elapsed_s": result["elapsed_s"],
            "steps_used": result["steps_used"],
            "executed_steps": result["executed_steps"],
            "rejected_steps": result["rejected_steps"],
            "proposals": len(proposals),
            "proposal_sum_s": sum(row["elapsed_s"] for row in proposals),
            "proposal_mean_s": statistics.fmean(row["elapsed_s"] for row in proposals),
            "inference_passes": len(passes),
            "render_sum_s": sum(row["elapsed_s"] for row in renders),
            "map_renders": sum("map" in row["views"] for row in renders),
            "generation_pipeline_sum_s": sum(row["total_s"] for row in gens),
            "model_generate_sum_s": sum(row["model_generate_s"] for row in gens),
            "sim_actions": len(sims),
            "sim_execute_sum_s": sum(row["elapsed_s"] for row in sims),
        }
    )
summary["per_trajectory"] = per_trajectory

summary["view_counts"] = Counter("+".join(row["views"]) for row in events["render"])
summary["result_tool_counts"] = Counter(
    str(row.get("result_tool")) for row in events["inference_pass"]
)
summary["proposal_tool_counts"] = Counter(
    str(row.get("final_tool")) for row in events["proposal_turn"]
)
summary["errors"] = {
    "inference_pass_errors": Counter(
        str(row.get("result_error"))
        for row in events["inference_pass"]
        if row.get("result_error")
    ),
    "proposal_errors": Counter(
        str(row.get("final_error"))
        for row in events["proposal_turn"]
        if row.get("final_error")
    ),
    "sim_errors": Counter(
        str(row.get("error"))
        for row in events["sim_execute"]
        if row.get("error")
    ),
    "native_errors": Counter(
        str(row.get("error"))
        for row in events["native_check"]
        if row.get("error")
    ),
}

rendered = json.dumps(summary, indent=2, sort_keys=True)
if args.output is None:
    print(rendered)
else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
