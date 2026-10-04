#!/usr/bin/env python3
"""Build a refreshable 43/10 live-sim evaluation dashboard."""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "outputs/eval/fixed_live_sim"
COMM_ROOT = ROOT / "outputs/eval/communication_ablation/summaries"
SPLIT_PATH = ROOT / "configs/splits/43_train_10_heldout.json"
OUT = ROOT / "outputs/figures/43_10"
RUN_RE = re.compile(
    r"sft_s(?P<scale>30|60|90|120|150)_noreason_train43_held10_lr1e4_wd001_"
    r"ep(?P<epoch>0p5|1p0)_fixed10_promptv9"
)
NON_TASK_ACTION_TOOLS = {
    "communicate", "get_image", "give_space", "navigate_to_fixture",
    "report_failed", "task_complete", "wait_for_signal",
}


def load_trajectory_rows(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("trajectory_id") or row.get("episode_id"))
        latest[key] = row
    return list(latest.values())


def single_physical_agent_fsm_success(row: dict, *, require_error_free: bool) -> bool:
    counts = {"agent_0": 0, "agent_1": 0}
    for step in row.get("steps", []):
        proposal = step.get("proposal") or {}
        agent = step.get("agent") or proposal.get("agent")
        if (
            agent in counts and step.get("executed")
            and step.get("sim_success", True) is not False
            and proposal.get("tool") not in NON_TASK_ACTION_TOOLS
        ):
            counts[agent] += 1
    rejected = int(row.get("rejected_total", row.get("rejected_steps", 0))) > 0
    return (
        bool(row.get("fsm_goal_satisfied"))
        and (not require_error_free or not rejected)
        and sum(value > 0 for value in counts.values()) == 1
    )


def error_free_single_agent(row: dict) -> bool:
    return single_physical_agent_fsm_success(row, require_error_free=True)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def metric_row(*, model: str, checkpoint: str, split: str, successes: int, episodes: int,
               source_kind: str, source_path: str, communication_mode: str = "Full",
               single_agent_successes: int = 0, final_successes: int | None = None,
               final_single_agent_successes: int = 0) -> dict:
    final_successes = successes if final_successes is None else final_successes
    if not 0 <= single_agent_successes <= successes <= episodes:
        raise ValueError(
            f"invalid nested counts: one-agent error-free={single_agent_successes}, "
            f"error-free={successes}, episodes={episodes}"
        )
    if not 0 <= final_single_agent_successes <= final_successes <= episodes:
        raise ValueError(
            f"invalid final nested counts: one-agent final={final_single_agent_successes}, "
            f"final={final_successes}, episodes={episodes}"
        )
    low, high = wilson(successes, episodes)
    single_low, single_high = wilson(single_agent_successes, episodes)
    final_low, final_high = wilson(final_successes, episodes)
    return {
        "model": model,
        "checkpoint": checkpoint,
        "model_checkpoint": f"{model} · {checkpoint}",
        "split": split,
        "communication_mode": communication_mode,
        "model_and_mode": f"{communication_mode} · {model}",
        "error_free_successes": successes,
        "episodes": episodes,
        "error_free_success_rate": successes / episodes if episodes else 0.0,
        "ci_low": low,
        "ci_high": high,
        "error_free_single_agent_successes": single_agent_successes,
        "error_free_single_agent_success_rate": single_agent_successes / episodes if episodes else 0.0,
        "single_agent_ci_low": single_low,
        "single_agent_ci_high": single_high,
        "fsm_successes": final_successes,
        "fsm_success_rate": final_successes / episodes if episodes else 0.0,
        "fsm_ci_low": final_low,
        "fsm_ci_high": final_high,
        "fsm_single_physical_agent_successes": final_single_agent_successes,
        "fsm_single_physical_agent_success_rate": final_single_agent_successes / episodes if episodes else 0.0,
        "source_kind": source_kind,
        "source_path": source_path,
    }


def recorded_checkpoint_epoch(split_dir: Path) -> str:
    """Use evaluated checkpoint metadata, not the historical output-folder label."""
    metadata = json.loads((split_dir / "parallel_run.json").read_text())
    args = metadata["identity"]["live_args_template"]
    adapter = Path(args[args.index("--adapter-path") + 1])
    state = json.loads((adapter / "trainer_state.json").read_text())
    epoch = float(state["epoch"])
    nearest = min((0.5, 1.0), key=lambda value: abs(epoch - value))
    if abs(epoch - nearest) > 0.01:
        raise ValueError(f"Unexpected evaluated epoch {epoch}: {adapter}")
    return f"epoch {nearest:.1f}"


def native_rows() -> list[dict]:
    rows: list[dict] = []
    candidates: list[tuple[Path, str, str]] = []
    for run_dir in EVAL_ROOT.iterdir():
        match = RUN_RE.fullmatch(run_dir.name)
        if match:
            scale = match.group("scale")
            epoch = match.group("epoch").replace("p", ".")
            candidates.append((run_dir, f"SFT {scale}/task", f"epoch {epoch}"))
        elif run_dir.name.startswith("thinking_rationale_s30_train43_held10_lr1e4_wd001_ep"):
            epoch = "0.5" if "ep0p5" in run_dir.name else "1.0"
            candidates.append((run_dir, "Thinking + rationale SFT 30/task", f"epoch {epoch}"))
        elif re.fullmatch(r"thinking_rationale_s(60|90|120)_train43_held10_lr1e4_wd001_ep(0p5|1p0)_fixed10_promptv9", run_dir.name):
            match = re.fullmatch(r"thinking_rationale_s(60|90|120)_train43_held10_lr1e4_wd001_ep(0p5|1p0)_fixed10_promptv9", run_dir.name)
            assert match is not None
            scale, epoch = match.groups()
            candidates.append((run_dir, f"Thinking + rationale SFT {scale}/task", f"epoch {epoch.replace('p', '.')}"))
        elif run_dir.name == "thinking_rationale_s150_train43_held10_lr1e4_wd001_ep1p0_fixed10_promptv9":
            candidates.append((run_dir, "Thinking + rationale SFT 150/task", "epoch 1.0"))
        elif run_dir.name == "thinking_noreason_s30_train43_held10_lr1e4_wd001_ep1p0_fixed10_recovery_v1":
            candidates.append((run_dir, "Thinking + no rationale SFT 30/task", "epoch 1.0"))
        elif run_dir.name == "instruct_rationale_s30_train43_held10_lr1e4_wd001_ep1p0_fixed10_promptv9":
            candidates.append((run_dir, "Instruct + rationale SFT 30/task", "epoch 1.0"))
        elif run_dir.name.startswith("sft_dagger89mix_s30_train43_held10_ep"):
            epoch = "0.5" if "ep0p5" in run_dir.name else "1.0"
            candidates.append((run_dir, "DAgger-89 mixture", f"epoch {epoch}"))
        elif re.fullmatch(r"dagger_strict100_regsft_(flash|pro31)_v2_ep(0p5|1p0)_fixed10_promptv9", run_dir.name):
            match = re.fullmatch(r"dagger_strict100_regsft_(flash|pro31)_v2_ep(0p5|1p0)_fixed10_promptv9", run_dir.name)
            assert match is not None
            expert, epoch = match.groups()
            label = "DAgger-100 · Gemini Flash" if expert == "flash" else "DAgger-100 · Gemini 3.1 Pro"
            candidates.append((run_dir, label, f"epoch {epoch.replace('p', '.')}"))
    for run_dir, model, checkpoint in sorted(candidates):
        for split_id, split_label in (
            ("train_task_types", "Trained tasks (43)"),
            ("heldout_task_types", "Held-out tasks (10)"),
        ):
            path = run_dir / split_id / "aggregate" / "live_sim_metrics.json"
            if (split_id == "heldout_task_types" and checkpoint == "epoch 1.0"
                    and model in {f"Thinking + rationale SFT {n}/task" for n in (60, 90, 120)}):
                path = run_dir.with_name(run_dir.name + "_native43matched") / split_id / "aggregate" / "live_sim_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Build the native43 replacement results first: {path}")
            if not path.is_file():
                continue
            actual_checkpoint = (
                recorded_checkpoint_epoch(run_dir / split_id)
                if RUN_RE.fullmatch(run_dir.name) else checkpoint
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            episodes = int(data["num_trajectories"])
            successes = int(data["num_fsm_error_free_successes"])
            trajectories_path = path.with_name("live_sim_trajectories.jsonl")
            trajectory_rows = load_trajectory_rows(trajectories_path)
            if len(trajectory_rows) != episodes:
                raise RuntimeError(f"{trajectories_path}: expected {episodes} latest trajectories, found {len(trajectory_rows)}")
            single_agent_successes = sum(error_free_single_agent(row) for row in trajectory_rows)
            final_successes = sum(bool(row.get("fsm_goal_satisfied")) for row in trajectory_rows)
            final_single_agent_successes = sum(
                single_physical_agent_fsm_success(row, require_error_free=False)
                for row in trajectory_rows
            )
            rows.append(metric_row(
                model=model,
                checkpoint=actual_checkpoint,
                split=split_label,
                successes=successes,
                episodes=episodes,
                source_kind="native_43_10",
                source_path=str(path.relative_to(ROOT)),
                single_agent_successes=single_agent_successes,
                final_successes=final_successes,
                final_single_agent_successes=final_single_agent_successes,
            ))
    def order(row: dict) -> tuple:
        scale_match = re.fullmatch(r"SFT (\d+)/task", row["model"])
        if scale_match:
            model_rank = (0, int(scale_match.group(1)))
        elif row["model"] == "Instruct + rationale SFT 30/task":
            model_rank = (1, 30)
        elif row["model"] == "Thinking + no rationale SFT 30/task":
            model_rank = (2, 30)
        elif row["model"] == "Thinking + rationale SFT 30/task":
            model_rank = (3, 30)
        elif row["model"] == "Thinking + rationale SFT 150/task":
            model_rank = (3, 150)
        elif row["model"] == "Thinking + rationale SFT 60/task":
            model_rank = (3, 60)
        elif row["model"] == "Thinking + rationale SFT 90/task":
            model_rank = (3, 90)
        elif row["model"] == "Thinking + rationale SFT 120/task":
            model_rank = (3, 120)
        elif row["model"] == "DAgger-100 · Gemini Flash":
            model_rank = (4, 1)
        elif row["model"] == "DAgger-100 · Gemini 3.1 Pro":
            model_rank = (4, 2)
        else:
            model_rank = (4, 0)
        epoch = float(row["checkpoint"].removeprefix("epoch "))
        split_rank = 0 if row["split"].startswith("Trained") else 1
        return (*model_rank, epoch, split_rank)
    return sorted(rows, key=order)


def communication_rows() -> list[dict]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    memberships = {
        "Trained tasks (43)": set(split["train_tasks"]),
        "Held-out tasks (10)": set(split["held_out_tasks"]),
    }
    rows: list[dict] = []
    raw_dirs = {
        ("gemini3flash", "full"): "ootb_gemini3flash_fixed10_promptv9",
        ("gemini3flash", "intermediate"): "ootb_gemini3flash_fixed10_promptv9_comm_intermediate_waitfix_v3_job853823",
        ("gemini3flash", "minimal"): "ootb_gemini3flash_fixed10_promptv9_comm_minimal_waitfix_v3_job853823",
        ("gemini3flash", "none"): "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        ("gemini3flash", "unguided"): "ootb_gemini3flash_fixed10_promptv9_comm_unguided_waitfix_v3_job853823",
        ("qwen3vl8b", "full"): "ootb_qwen3vl8b_fixed10_promptv9_vllm",
        ("qwen3vl8b", "intermediate"): "ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_intermediate_waitfix_v3_job853823",
        ("qwen3vl8b", "minimal"): "ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_minimal_waitfix_v3_job853823",
        ("qwen3vl8b", "none"): "ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_none_v2",
        ("qwen3vl8b", "unguided"): "ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_unguided_waitfix_v3_job853823",
    }
    for path in sorted(COMM_ROOT.glob("*.json")):
        match = re.fullmatch(r"(gemini3flash|qwen3vl8b)_(full|intermediate|minimal|none|unguided)\.json", path.name)
        if not match:
            continue
        model_id, mode = match.groups()
        model = "Gemini 3 Flash" if model_id == "gemini3flash" else "Base Qwen3-VL-8B"
        mode_label = {"full": "Full", "intermediate": "Intermediate", "minimal": "Minimal", "none": "None", "unguided": "Unguided"}[mode]
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = []
        raw_dir = EVAL_ROOT / raw_dirs[(model_id, mode)]
        for old_split_id in ("train_task_types", "heldout_task_types"):
            raw_rows.extend(load_trajectory_rows(raw_dir / old_split_id / "aggregate" / "live_sim_trajectories.jsonl"))
        raw_by_task: dict[str, list[dict]] = {}
        for raw_row in raw_rows:
            raw_by_task.setdefault(raw_row["task_name"], []).append(raw_row)
        per_task: dict[str, dict] = {}
        for old_split in data["splits"].values():
            per_task.update(old_split["per_task"])
        for split_label, tasks in memberships.items():
            selected = [per_task[task] for task in sorted(tasks) if task in per_task]
            episodes = sum(int(item["num_episodes"]) for item in selected)
            successes = sum(int(item["metrics"]["fsm_error_free_success"]["count"]) for item in selected)
            selected_raw = [row for task in tasks for row in raw_by_task.get(task, [])]
            if len(selected_raw) != episodes:
                raise RuntimeError(f"{model_id}/{mode}/{split_label}: summary has {episodes} episodes, raw outputs have {len(selected_raw)}")
            raw_error_free = sum(bool(row.get("fsm_goal_satisfied")) and not int(row.get("rejected_total", row.get("rejected_steps", 0))) for row in selected_raw)
            if raw_error_free != successes:
                raise RuntimeError(f"{model_id}/{mode}/{split_label}: summary error-free={successes}, raw={raw_error_free}")
            single_agent_successes = sum(error_free_single_agent(row) for row in selected_raw)
            final_successes = sum(bool(row.get("fsm_goal_satisfied")) for row in selected_raw)
            summary_final_successes = sum(int(item["metrics"]["fsm_success"]["count"]) for item in selected)
            if final_successes != summary_final_successes:
                raise RuntimeError(
                    f"{model_id}/{mode}/{split_label}: summary final={summary_final_successes}, "
                    f"raw={final_successes}"
                )
            final_single_agent_successes = sum(
                single_physical_agent_fsm_success(row, require_error_free=False)
                for row in selected_raw
            )
            rows.append(metric_row(
                model=model,
                checkpoint="out of box",
                split=split_label,
                successes=successes,
                episodes=episodes,
                source_kind="migrated_47_6_ablation",
                source_path=str(path.relative_to(ROOT)),
                communication_mode=mode_label,
                single_agent_successes=single_agent_successes,
                final_successes=final_successes,
                final_single_agent_successes=final_single_agent_successes,
            ))
    model_rank = {"Base Qwen3-VL-8B": 0, "Gemini 3 Flash": 1}
    mode_rank = {"None": 0, "Unguided": 1, "Minimal": 2, "Intermediate": 3, "Full": 4}
    return sorted(rows, key=lambda row: (
        mode_rank[row["communication_mode"]], model_rank[row["model"]],
        0 if row["split"].startswith("Trained") else 1,
    ))


def main() -> None:
    native = native_rows()
    communication = communication_rows()
    scale_results: list[dict] = []
    for scale in (30, 60, 90, 120, 150):
        for model in (f"SFT {scale}/task", f"Thinking + rationale SFT {scale}/task"):
            scale_results.extend(
                row for row in native
                if row["model"] == model and row["checkpoint"] == "epoch 1.0"
            )
    thirty_results = [
        row for row in native
        if row["checkpoint"] == "epoch 1.0"
        and row["model"] in {
            "SFT 30/task",
            "Instruct + rationale SFT 30/task",
            "Thinking + no rationale SFT 30/task",
            "Thinking + rationale SFT 30/task",
            "DAgger-100 · Gemini Flash",
            "DAgger-100 · Gemini 3.1 Pro",
        }
    ]
    sixty_results = [
        row for row in native
        if row["checkpoint"] == "epoch 1.0"
        and row["model"] in {"SFT 60/task", "Thinking + rationale SFT 60/task"}
    ]
    ninety_results = [
        row for row in native
        if row["checkpoint"] == "epoch 1.0"
        and row["model"] in {"SFT 90/task", "Thinking + rationale SFT 90/task"}
    ]
    one_twenty_results = [
        row for row in native
        if row["checkpoint"] == "epoch 1.0"
        and row["model"] in {"SFT 120/task", "Thinking + rationale SFT 120/task"}
    ]
    one_fifty_results = [
        row for row in native
        if row["checkpoint"] == "epoch 1.0"
        and row["model"] in {
            "SFT 150/task",
            "Thinking + rationale SFT 150/task",
        }
    ]
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    expected = [
        {"model": f"SFT {scale}/task", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == f"SFT {scale}/task" for r in native) == 4 else "Pending"}
        for scale in (30, 60, 90, 120, 150)
    ] + [
        {"model": "Thinking + rationale SFT 30/task", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "Thinking + rationale SFT 30/task" for r in native) == 4 else "Pending"},
        {"model": "Thinking + rationale SFT 150/task", "checkpoints": "1.0", "status": "Complete" if sum(r["model"] == "Thinking + rationale SFT 150/task" for r in native) == 2 else "Pending"},
        {"model": "Thinking + rationale SFT 60/task", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "Thinking + rationale SFT 60/task" for r in native) == 4 else "Pending"},
        {"model": "Thinking + rationale SFT 90/task", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "Thinking + rationale SFT 90/task" for r in native) == 4 else "Pending"},
        {"model": "Thinking + rationale SFT 120/task", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "Thinking + rationale SFT 120/task" for r in native) == 4 else "Pending"},
        {"model": "Thinking + no rationale SFT 30/task", "checkpoints": "1.0", "status": "Complete" if sum(r["model"] == "Thinking + no rationale SFT 30/task" for r in native) == 2 else "Pending"},
        {"model": "Instruct + rationale SFT 30/task", "checkpoints": "1.0", "status": "Complete" if sum(r["model"] == "Instruct + rationale SFT 30/task" for r in native) == 2 else "Pending"},
        {"model": "DAgger-89 mixture", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "DAgger-89 mixture" for r in native) == 4 else "Pending"},
        {"model": "DAgger-100 · Gemini Flash", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "DAgger-100 · Gemini Flash" for r in native) == 4 else "Pending"},
        {"model": "DAgger-100 · Gemini 3.1 Pro", "checkpoints": "0.5, 1.0", "status": "Complete" if sum(r["model"] == "DAgger-100 · Gemini 3.1 Pro" for r in native) == 4 else "Pending"},
    ]
    native_source = {
        "id": "native_43_10",
        "label": "Native fixed-cohort 43/10 live-sim evaluations",
        "query": {
            "engine": "filesystem-json",
            "language": "python",
            "sql": "SELECT * FROM completed 43/10 live_sim_metrics.json files; compute error_free_successes / num_trajectories and Wilson intervals",
            "description": "Completed 43/10 aggregate evaluation metrics discovered by run naming contract.",
            "tables_used": sorted({row["source_path"] for row in native}),
            "filters": ["43 trained and 10 held-out task types", "10 fixed live-sim episodes per task", "full communication protocol", "no step indexing"],
            "metric_definitions": ["Error-free FSM success = FSM goal satisfied with zero rejected tool calls / all evaluated episodes.", "Inset subset = error-free FSM success where exactly one agent performed every task-changing physical action / all evaluated episodes.", "Intervals are two-sided 95% Wilson binomial intervals."],
            "executed_at": now,
        },
    }
    communication_source = {
        "id": "migrated_communication",
        "label": "Communication ablations regrouped from 47/6 to 43/10 task membership",
        "query": {
            "engine": "filesystem-json",
            "language": "python",
            "sql": "SELECT task, mode, model, SUM(error_free_successes), SUM(episodes) FROM communication_ablation_per_task GROUP BY task_membership_43_10, mode, model",
            "description": "Reuses the original per-task episodes and outcomes, changing only whether four moved tasks are counted as trained or held-out under the 43/10 split.",
            "tables_used": sorted({row["source_path"] for row in communication}) + [str(SPLIT_PATH.relative_to(ROOT))],
            "filters": ["No new episodes generated", "Original 10 episodes per task retained", "Only Base Qwen3-VL-8B and Gemini 3 Flash communication arms"],
            "metric_definitions": ["Error-free FSM success = FSM goal satisfied with zero rejected tool calls / all evaluated episodes.", "Final FSM success with error feedback permitted = FSM goal satisfied at episode end, whether or not rejected calls occurred / all evaluated episodes.", "Each inset is the corresponding success numerator restricted to episodes where exactly one agent performed every task-changing physical action, still divided by all evaluated episodes.", "Migrated means per-task outcomes were reassigned to the current 43/10 task grouping; model outputs were not changed."],
            "executed_at": now,
        },
    }
    charts = [
        {
            "id": "native_error_free", "title": "SFT scale after one epoch on the native 43/10 cohort",
            "subtitle": "At each data scale, standard SFT is followed by Thinking-base SFT with supervised rationales; bars separate trained and held-out tasks.", "type": "bar", "intent": "comparison",
            "dataset": "scale_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "model_checkpoint", "type": "ordinal", "label": "Model checkpoint"},
                "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"},
                "color": {"field": "split", "type": "nominal", "label": "43/10 split"},
                "tooltip": [{"field": "error_free_successes", "type": "quantitative", "label": "Error-free successes"}, {"field": "error_free_single_agent_successes", "type": "quantitative", "label": "Error-free successes with one physical agent"}, {"field": "error_free_single_agent_success_rate", "type": "quantitative", "format": "percent", "label": "Inset subset rate"}, {"field": "episodes", "type": "quantitative", "label": "Episodes"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"}],
            },
        },
        {
            "id": "thirty_epoch_one", "title": "Thirty trajectories per task after one epoch",
            "subtitle": "One-epoch 30/task variants on the same fixed 43/10 cohort. Thinking + no rationale uses the documented missing-opening-tag compatibility recovery.", "type": "bar", "intent": "comparison",
            "dataset": "thirty_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "model", "type": "ordinal", "label": "Training approach"},
                "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"},
                "color": {"field": "split", "type": "nominal", "label": "43/10 split"},
                "tooltip": [{"field": "error_free_successes", "type": "quantitative", "label": "Error-free successes"}, {"field": "error_free_single_agent_successes", "type": "quantitative", "label": "Error-free successes with one physical agent"}, {"field": "error_free_single_agent_success_rate", "type": "quantitative", "format": "percent", "label": "Inset subset rate"}, {"field": "episodes", "type": "quantitative", "label": "Episodes"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"}],
            },
        },
        {
            "id": "one_fifty_epoch_one", "title": "One hundred fifty trajectories per task after one epoch",
            "subtitle": "Standard Instruct SFT versus Thinking-base SFT with supervised rationales on the same fixed 43/10 cohort.", "type": "bar", "intent": "comparison",
            "dataset": "one_fifty_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "model", "type": "ordinal", "label": "Training approach"},
                "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"},
                "color": {"field": "split", "type": "nominal", "label": "43/10 split"},
                "tooltip": [{"field": "error_free_successes", "type": "quantitative", "label": "Error-free successes"}, {"field": "episodes", "type": "quantitative", "label": "Episodes"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"}],
            },
        },
        {
            "id": "sixty_epoch_one", "title": "Sixty trajectories per task after one epoch", "subtitle": "Standard SFT versus Thinking-base SFT with supervised rationales on the same fixed 43/10 cohort.", "type": "bar", "intent": "comparison", "dataset": "sixty_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {"x": {"field": "model", "type": "ordinal", "label": "Training approach"}, "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"}, "color": {"field": "split", "type": "nominal", "label": "43/10 split"}},
        },
        {
            "id": "ninety_epoch_one", "title": "Ninety trajectories per task after one epoch", "subtitle": "Standard SFT versus Thinking-base SFT with supervised rationales on the same fixed 43/10 cohort.", "type": "bar", "intent": "comparison", "dataset": "ninety_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {"x": {"field": "model", "type": "ordinal", "label": "Training approach"}, "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"}, "color": {"field": "split", "type": "nominal", "label": "43/10 split"}},
        },
        {
            "id": "one_twenty_epoch_one", "title": "One hundred twenty trajectories per task after one epoch", "subtitle": "Standard SFT versus Thinking-base SFT with supervised rationales on the same fixed 43/10 cohort.", "type": "bar", "intent": "comparison", "dataset": "one_twenty_results", "sourceId": "native_43_10", "layout": "full", "valueFormat": "percent",
            "encodings": {"x": {"field": "model", "type": "ordinal", "label": "Training approach"}, "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"}, "color": {"field": "split", "type": "nominal", "label": "43/10 split"}},
        },
        {
            "id": "communication_error_free", "title": "Error-free success by communication mode, regrouped to 43/10",
            "subtitle": "Same original episodes and model outputs; only task split membership is corrected.", "type": "bar", "intent": "comparison",
            "dataset": "communication_results", "sourceId": "migrated_communication", "layout": "full", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "model_and_mode", "type": "ordinal", "label": "Model and communication mode"},
                "y": {"field": "error_free_success_rate", "type": "quantitative", "format": "percent", "label": "Error-free success rate"},
                "color": {"field": "split", "type": "nominal", "label": "43/10 split"},
                "tooltip": [{"field": "error_free_successes", "type": "quantitative", "label": "Error-free successes"}, {"field": "error_free_single_agent_successes", "type": "quantitative", "label": "Error-free successes with one physical agent"}, {"field": "error_free_single_agent_success_rate", "type": "quantitative", "format": "percent", "label": "Inset subset rate"}, {"field": "episodes", "type": "quantitative", "label": "Episodes"}, {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"}, {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"}],
            },
        },
        {
            "id": "communication_final_success", "title": "Final FSM success with error feedback permitted",
            "subtitle": "Episodes continue after rejected calls; bars include both error-free successes and successes that required recovery.", "type": "bar", "intent": "comparison",
            "dataset": "communication_results", "sourceId": "migrated_communication", "layout": "full", "valueFormat": "percent",
            "encodings": {
                "x": {"field": "model_and_mode", "type": "ordinal", "label": "Model and communication mode"},
                "y": {"field": "fsm_success_rate", "type": "quantitative", "format": "percent", "label": "Final FSM success rate"},
                "color": {"field": "split", "type": "nominal", "label": "43/10 split"},
                "tooltip": [{"field": "fsm_successes", "type": "quantitative", "label": "Final FSM successes"}, {"field": "fsm_single_physical_agent_successes", "type": "quantitative", "label": "Final successes with exactly one physical-work agent"}, {"field": "fsm_single_physical_agent_success_rate", "type": "quantitative", "format": "percent", "label": "Inset subset rate"}, {"field": "episodes", "type": "quantitative", "label": "Episodes"}, {"field": "fsm_ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"}, {"field": "fsm_ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"}],
            },
        },
    ]
    detail_columns = [
        {"field": "model", "label": "Model", "type": "text"}, {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
        {"field": "communication_mode", "label": "Communication", "type": "text"}, {"field": "split", "label": "Split", "type": "text"},
        {"field": "error_free_successes", "label": "Error-free successes", "format": "number"}, {"field": "episodes", "label": "N", "format": "number"},
        {"field": "error_free_success_rate", "label": "Rate", "format": "percent"}, {"field": "ci_low", "label": "95% CI low", "format": "percent"},
        {"field": "ci_high", "label": "95% CI high", "format": "percent"},
        {"field": "error_free_single_agent_successes", "label": "Error-free + exactly one physical agent", "format": "number"},
        {"field": "error_free_single_agent_success_rate", "label": "Subset rate", "format": "percent"},
        {"field": "fsm_successes", "label": "Final FSM successes", "format": "number"},
        {"field": "fsm_success_rate", "label": "Final FSM rate", "format": "percent"},
        {"field": "fsm_single_physical_agent_successes", "label": "Final + exactly one physical-work agent", "format": "number"},
        {"field": "fsm_single_physical_agent_success_rate", "label": "Final subset rate", "format": "percent"},
    ]
    tables = [
        {"id": "native_detail", "title": "Native 43/10 exact results", "dataset": "native_results", "sourceId": "native_43_10", "defaultSort": {"field": "model", "direction": "asc"}, "density": "dense", "layout": "full", "columns": detail_columns},
        {"id": "communication_detail", "title": "Migrated communication-ablation exact results", "dataset": "communication_results", "sourceId": "migrated_communication", "defaultSort": {"field": "model", "direction": "asc"}, "density": "dense", "layout": "full", "columns": detail_columns},
        {"id": "coverage", "title": "Evaluation population status", "dataset": "coverage", "sourceId": "native_43_10", "defaultSort": {"field": "model", "direction": "asc"}, "density": "spacious", "layout": "full", "columns": [{"field": "model", "label": "Model", "type": "text"}, {"field": "checkpoints", "label": "Expected checkpoints", "type": "text"}, {"field": "status", "label": "Status", "type": "text"}]},
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Fixed live-sim evaluation: 43 trained / 10 held-out tasks"},
        {"id": "scope", "type": "markdown", "body": "## Primary metric\n\nThis dashboard prioritizes **error-free FSM success**: the task goal is reached without any rejected model tool call. Native 43/10 evaluations and migrated communication ablations are kept separate because only the native runs share the current fixed cohort directly."},
        {"id": "native_chart", "type": "chart", "chartId": "native_error_free", "layout": "full"},
        {"id": "thirty_chart", "type": "chart", "chartId": "thirty_epoch_one", "layout": "full"},
        {"id": "sixty_chart", "type": "chart", "chartId": "sixty_epoch_one", "layout": "full"},
        {"id": "ninety_chart", "type": "chart", "chartId": "ninety_epoch_one", "layout": "full"},
        {"id": "one_twenty_chart", "type": "chart", "chartId": "one_twenty_epoch_one", "layout": "full"},
        {"id": "one_fifty_chart", "type": "chart", "chartId": "one_fifty_epoch_one", "layout": "full"},
        {"id": "native_table", "type": "table", "tableId": "native_detail", "layout": "full"},
        {"id": "comm_note", "type": "markdown", "body": "## Communication ablations\n\nThese results were originally summarized as a 47/6 split. The artifact recomputes them from the retained per-task counts using the current 43/10 membership. This is a bookkeeping correction, not a new evaluation: scenes, prompts, outputs, and episode outcomes are unchanged."},
        {"id": "comm_chart", "type": "chart", "chartId": "communication_error_free", "layout": "full"},
        {"id": "comm_final_chart", "type": "chart", "chartId": "communication_final_success", "layout": "full"},
        {"id": "comm_table", "type": "table", "tableId": "communication_detail", "layout": "full"},
        {"id": "coverage_block", "type": "table", "tableId": "coverage", "layout": "full"},
        {"id": "definitions", "type": "markdown", "body": "## Definitions and caveats\n\n- **Trained tasks** are the 43 task types used for the current training split; **held-out tasks** are the other 10.\n- In both communication-ablation plots, the dark inset restricts the main success definition to episodes in which exactly one agent performed every task-changing physical action. Navigation, images, communication, waiting, and giving space are not counted as task-changing physical work.\n- Every main bar and inset uses all evaluated episodes as the denominator.\n- Final FSM success with error feedback permitted includes both error-free successes and successes achieved after rejected calls and error messages.\n- All confidence intervals shown are two-sided 95% Wilson binomial intervals for the main bars.\n- Communication-ablation results are directly comparable across modes within each model because they reuse the same task population. They should not be treated as a rerun on the newer native 43/10 cohort.\n- Rerun this builder after pending evaluations finish to populate additional checkpoints."},
    ]
    complete = all(item["status"] == "Complete" for item in expected)
    artifact = {
        "surface": "dashboard",
        "manifest": {"version": 1, "surface": "dashboard", "title": "Fixed live-sim evaluation: 43/10", "description": "Error-free success for current 43/10 evaluations and corrected communication-ablation groupings.", "generatedAt": now, "charts": charts, "tables": tables, "sources": [native_source, communication_source], "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": now, "status": "ready" if complete else "partial", "datasets": {"native_results": native, "scale_results": scale_results, "thirty_results": thirty_results, "sixty_results": sixty_results, "ninety_results": ninety_results, "one_twenty_results": one_twenty_results, "one_fifty_results": one_fifty_results, "communication_results": communication, "coverage": expected}, "accessIssues": [] if complete else [{"id": "pending_evaluations", "dataset": "native_results", "message": "Some expected evaluation checkpoints are still running or queued; completed outputs are shown and the artifact is refreshable."}]},
        "sources": [native_source, communication_source],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (OUT / "DESIGN.md").write_text("""# Dashboard design\n\n## Intent\nMonitor error-free live-sim success across current 43/10 checkpoints and communication modes.\n\n## Layout\nSummary-first: metric definition, native comparison, migrated communication comparison, then exact tables and coverage.\n\n## Visual system\nNeutral comparison bars, consistent split colors, percent axes, and compact exact-result tables.\n\n## Data states\nPartial while scheduled evaluations are incomplete; missing checkpoints are labeled pending rather than shown as zero.\n""", encoding="utf-8")
    def bar_chart(
        rows: list[dict], key: str, *, paired_groups: bool = False,
        within_pair_spacing: int = 104,
        show_single_agent_subset: bool = False,
        value_key: str = "error_free_success_rate",
        successes_key: str = "error_free_successes",
        ci_low_key: str = "ci_low",
        ci_high_key: str = "ci_high",
        subset_rate_key: str = "error_free_single_agent_success_rate",
        subset_count_key: str = "error_free_single_agent_successes",
        y_label: str = "Error-free FSM success",
        metric_label: str = "error-free FSM success",
    ) -> str:
        groups: dict[str, dict[str, dict]] = {}
        for row in rows:
            groups.setdefault(row[key], {})["held" if row["split"].startswith("Held") else "train"] = row
        labels = list(groups)
        plot_h, top, bottom, left, group_w = 260, 30, 115, 64, 142
        if paired_groups:
            within_pair, between_pairs = within_pair_spacing, 48
            centers = [left + 58 + (i // 2) * (2 * within_pair + between_pairs) + (i % 2) * within_pair for i in range(len(labels))]
            width = int(centers[-1] + 88)
        else:
            centers = [left + i * group_w + group_w / 2 for i in range(len(labels))]
            width = left + len(labels) * group_w + 28
        height = top + plot_h + bottom
        svg = [f'<div class="chart-scroll"><svg class="proper-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Grouped error-free success bar chart">']
        for tick in range(0, 101, 20):
            y = top + plot_h * (1 - tick / 100)
            svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-18}" y2="{y:.1f}"/><text class="tick" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick}%</text>')
        svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/><line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-18}" y2="{top+plot_h}"/>')
        for i, label in enumerate(labels):
            center = centers[i]
            for split_id, offset, css in (("train", -24, "train"), ("held", 8, "held")):
                row = groups[label].get(split_id)
                if not row:
                    continue
                x = center + offset
                value = row[value_key]
                bar_h = plot_h * value
                y = top + plot_h - bar_h
                lo_y = top + plot_h * (1 - row[ci_low_key])
                hi_y = top + plot_h * (1 - row[ci_high_key])
                title = f'{label} · {row["split"]}: {metric_label} {100*value:.1f}% ({row[successes_key]}/{row["episodes"]}); 95% CI {100*row[ci_low_key]:.1f}–{100*row[ci_high_key]:.1f}%'
                subset_rect = ""
                if show_single_agent_subset:
                    subset = row[subset_rate_key]
                    subset_h = plot_h * subset
                    subset_y = top + plot_h - subset_h
                    title += f'; exactly-one-physical-work-agent subset {100*subset:.1f}% ({row[subset_count_key]}/{row["episodes"]})'
                    subset_rect = f'<rect class="subset {css}" x="{x+5:.1f}" y="{subset_y:.1f}" width="14" height="{subset_h:.1f}" rx="2"/>'
                svg.append(f'<g><title>{html.escape(title)}</title><rect class="bar {css}" x="{x:.1f}" y="{y:.1f}" width="24" height="{bar_h:.1f}" rx="3"/>{subset_rect}<line class="whisker" x1="{x+12:.1f}" y1="{hi_y:.1f}" x2="{x+12:.1f}" y2="{lo_y:.1f}"/><line class="whisker" x1="{x+7:.1f}" y1="{hi_y:.1f}" x2="{x+17:.1f}" y2="{hi_y:.1f}"/><line class="whisker" x1="{x+7:.1f}" y1="{lo_y:.1f}" x2="{x+17:.1f}" y2="{lo_y:.1f}"/><text class="value" x="{x+12:.1f}" y="{max(14,y-6):.1f}" text-anchor="middle">{100*value:.0f}%</text></g>')
            svg.append(f'<text class="xlabel" transform="translate({center+6:.1f},{top+plot_h+18}) rotate(38)" text-anchor="start">{html.escape(label)}</text>')
        svg.append(f'<text class="ylabel" transform="translate(17,{top+plot_h/2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text></svg></div>')
        return "".join(svg)

    def result_table(rows: list[dict]) -> str:
        body = "".join(
            "<tr>" +
            f"<td>{html.escape(row['model'])}</td><td>{html.escape(row['checkpoint'])}</td>" +
            f"<td>{html.escape(row['communication_mode'])}</td><td>{html.escape(row['split'])}</td>" +
            f"<td>{row['error_free_successes']}/{row['episodes']}</td><td>{100*row['error_free_success_rate']:.1f}%</td>" +
            f"<td>{row['error_free_single_agent_successes']}/{row['episodes']} ({100*row['error_free_single_agent_success_rate']:.1f}%)</td>" +
            f"<td>{100*row['ci_low']:.1f}–{100*row['ci_high']:.1f}%</td>" +
            f"<td>{row['fsm_successes']}/{row['episodes']} ({100*row['fsm_success_rate']:.1f}%)</td>" +
            f"<td>{row['fsm_single_physical_agent_successes']}/{row['episodes']} ({100*row['fsm_single_physical_agent_success_rate']:.1f}%)</td></tr>"
            for row in rows
        )
        return "<table><thead><tr><th>Model</th><th>Checkpoint</th><th>Communication</th><th>Split</th><th>Error-free successes / N</th><th>Error-free rate</th><th>Error-free + exactly one physical-work agent</th><th>Error-free 95% Wilson CI</th><th>Final FSM success / N</th><th>Final FSM + exactly one physical-work agent</th></tr></thead><tbody>" + body + "</tbody></table>"

    coverage_rows = "".join(
        f'<tr><td>{html.escape(row["model"])}</td><td>{html.escape(row["checkpoints"])}</td><td><span class="status {"done" if row["status"] == "Complete" else "pending"}">{html.escape(row["status"])}</span></td></tr>'
        for row in expected
    )
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed live-sim evaluation · 43/10</title><style>
:root{{--ink:#172033;--muted:#647087;--paper:#f5f7fb;--card:#fff;--line:#dce2ec;--blue:#3478f6;--amber:#e49a31}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1220px;margin:auto;padding:38px 24px 72px}}h1{{font-size:34px;line-height:1.12;margin:0 0 8px}}h2{{margin:0 0 7px;font-size:22px}}p{{margin:6px 0;color:var(--muted)}}.hero{{margin-bottom:25px}}.pill{{display:inline-block;background:#e7eefc;color:#234d9c;border-radius:999px;padding:5px 10px;font-size:12px;font-weight:700;margin-bottom:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 4px 18px #2030500b}}.legend{{display:flex;flex-wrap:wrap;gap:18px;color:var(--muted);font-size:13px;margin:12px 0 20px}}.dot{{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:6px}}.dot.train,.fill.train{{background:var(--blue)}}.dot.held,.fill.held{{background:var(--amber)}}.dot.subset-key{{background:#263750;border:2px solid #fff;outline:1px solid #263750}}
.chart-scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:#fff}}.proper-chart{{display:block;height:430px;min-width:100%;width:auto}}.grid{{stroke:#e4e8ef;stroke-width:1}}.axis{{stroke:#8d98aa;stroke-width:1.2}}.tick,.xlabel,.ylabel{{fill:#58657a;font-size:11px}}.value{{fill:#2f3b50;font-size:10px;font-weight:700}}.bar.train{{fill:var(--blue)}}.bar.held{{fill:var(--amber)}}.subset.train{{fill:#123d91}}.subset.held{{fill:#864f07}}.whisker{{stroke:#27344a;stroke-width:1.25}}small{{display:block;color:var(--muted);font-weight:400}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:#526077;background:#f7f9fc;position:sticky;top:0}}.scroll{{overflow:auto;max-height:570px;border:1px solid var(--line);border-radius:10px}}.status{{padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700}}.status.done{{background:#dcf4e9;color:#126442}}.status.pending{{background:#fff0d8;color:#875414}}.note{{border-left:4px solid var(--amber);padding-left:14px}}footer{{font-size:12px;color:var(--muted);margin-top:26px}}@media(max-width:780px){{h1{{font-size:28px}}main{{padding:24px 14px 50px}}}}
</style></head><body><main>
<section class="hero"><span class="pill">PARTIAL · REFRESHABLE</span><h1>Fixed live-sim evaluation: 43 trained / 10 held-out tasks</h1><p>Primary metric: <strong>error-free FSM success</strong> — the task reaches its symbolic goal with zero rejected model tool calls.</p><p>Snapshot generated {html.escape(now)}.</p></section>
<section class="card"><h2>SFT scale after one epoch</h2><p>Models are ordered by trajectories per task: baseline SFT first, followed by the matching Thinking+rationale model at each scale. Whiskers show 95% Wilson intervals.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(scale_results, "model", paired_groups=True, within_pair_spacing=64)}</section>
<section class="card"><h2>Thirty trajectories per task after one epoch</h2><p>Compares the original SFT variants with two strict-100 DAgger mixtures. Each DAgger arm starts from the regularized 30/task SFT checkpoint and mixes 100 corrected trajectories with 400 original trajectories.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(thirty_results, "model")}</section>
<section class="card"><h2>Sixty trajectories per task after one epoch</h2><p>Compares standard Instruct SFT with Thinking-base SFT trained on demonstration rationales using the same 60 trajectories per task and fixed 43/10 cohort.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(sixty_results, "model")}</section>
<section class="card"><h2>Ninety trajectories per task after one epoch</h2><p>Compares standard Instruct SFT with Thinking-base SFT trained on demonstration rationales using the same 90 trajectories per task and fixed 43/10 cohort.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(ninety_results, "model")}</section>
<section class="card"><h2>One hundred twenty trajectories per task after one epoch</h2><p>Compares standard Instruct SFT with Thinking-base SFT trained on demonstration rationales using the same 120 trajectories per task and fixed 43/10 cohort.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(one_twenty_results, "model")}</section>
<section class="card"><h2>One hundred fifty trajectories per task after one epoch</h2><p>Compares standard Instruct SFT with Thinking-base SFT trained on the demonstration rationales. Both use 150 trajectories per trained task and the same fixed 43/10 evaluation cohort. Whiskers show 95% Wilson intervals.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span></div>{bar_chart(one_fifty_results, "model")}</section>
<section class="card"><h2>Exact native results</h2><div class="scroll">{result_table(native)}</div></section>
<section class="card"><h2>Communication ablations: error-free FSM success</h2><p class="note">No new evaluation was run. The original per-task episode outcomes are unchanged; four tasks were reassigned from the old trained group to the current held-out group.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span><span><i class="dot subset-key"></i>Inset: error-free FSM success where exactly one agent performed every task-changing physical action, divided by all episodes</span></div>{bar_chart(communication, "model_and_mode", paired_groups=True, show_single_agent_subset=True)}</section>
<section class="card"><h2>Communication ablations: final FSM success with error feedback permitted</h2><p>Episodes continue after rejected calls, so the main bars include both error-free successes and successes achieved after the model received error messages.</p><div class="legend"><span><i class="dot train"></i>Trained tasks</span><span><i class="dot held"></i>Held-out tasks</span><span><i class="dot subset-key"></i>Inset: final FSM success where exactly one agent performed every task-changing physical action, divided by all episodes</span></div>{bar_chart(communication, "model_and_mode", paired_groups=True, show_single_agent_subset=True, value_key="fsm_success_rate", successes_key="fsm_successes", ci_low_key="fsm_ci_low", ci_high_key="fsm_ci_high", subset_rate_key="fsm_single_physical_agent_success_rate", subset_count_key="fsm_single_physical_agent_successes", y_label="Final FSM success", metric_label="final FSM success")}</section>
<section class="card"><h2>Exact communication-ablation results</h2><div class="scroll">{result_table(communication)}</div></section>
<section class="card"><h2>Population status</h2><table><thead><tr><th>Model</th><th>Expected checkpoints</th><th>Status</th></tr></thead><tbody>{coverage_rows}</tbody></table></section>
<section class="card"><h2>How to interpret this page</h2><p>Every native split bar uses 10 episodes per task: N=430 for trained tasks and N=100 for held-out tasks. In each communication-ablation plot, the dark inset applies that plot's success definition and additionally requires exactly one agent to have performed every task-changing physical action. All main bars and insets divide by all episodes. Intervals are two-sided 95% Wilson intervals for the main bars. Missing evaluations are labeled pending and never treated as zero.</p><p>The communication section is comparable across modes within each model. Because it reuses an older evaluation execution, compare it with the native section cautiously.</p></section>
<footer>Exact provenance is embedded in <code>artifact.json</code>. Re-run <code>training/bc_task_vlm/build_43_10_eval_artifact.py</code> as evaluations finish.</footer>
</main></body></html>'''
    (OUT / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT / "artifact.json"), "html": str(OUT / "report.html"), "native_rows": len(native), "communication_rows": len(communication), "status": artifact["snapshot"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
