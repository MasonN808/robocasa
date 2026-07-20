from __future__ import annotations

import json

import pytest

from training.bc_task_vlm.divergence_analysis import analyze
from training.bc_task_vlm.live_sim_eval import (
    OVERHEAD_VIEWS,
    SCOUT_VIEWS,
    WRIST_VIEWS,
    _run_lengths,
    _write_metrics,
    canonical_views_for_tool,
)


def test_canonical_views_follow_tool_kind():
    assert canonical_views_for_tool("navigate_to_fixture") == SCOUT_VIEWS
    assert canonical_views_for_tool("communicate") == OVERHEAD_VIEWS
    assert canonical_views_for_tool("pick_up_object") == WRIST_VIEWS


def test_run_lengths():
    assert _run_lengths([]) == []
    assert _run_lengths(["agent_0", "agent_0", "agent_1", "agent_0"]) == [2, 1, 1]


def test_metrics_include_rejections_partial_goal_and_turns(tmp_path):
    result = {
        "native_success": True,
        "fsm_goal_satisfied": True,
        "declared_complete": True,
        "termination": "task_complete_declared",
        "steps_used": 4,
        "expert_steps": 2,
        "partial_goal_fraction": 1.0,
        "step_efficiency_ratio": 2.0,
        "rejected_steps": 1,
        "agent_turns": ["agent_0", "agent_0", "agent_1"],
    }
    results = tmp_path / "live_sim_trajectories.jsonl"
    results.write_text(json.dumps(result) + "\n", encoding="utf-8")
    _write_metrics(results, tmp_path)
    metrics = json.loads((tmp_path / "live_sim_metrics.json").read_text())
    assert metrics["rejection_rate"] == pytest.approx(0.25)
    assert metrics["mean_partial_goal_fraction"] == 1.0
    assert metrics["same_agent_transition_rate"] == pytest.approx(0.5)
    assert metrics["same_agent_run_lengths"] == {"1": 1, "2": 1}


def test_divergence_analysis_ranks_noncontiguous_step_indices(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "sample_id": "task/traj/step_2", "task_name": "task",
            "trajectory_id": "traj", "step_index": 2,
            "target_tool_call": {"name": "communicate"},
            "exact_action_step_match": False,
        },
        {
            "sample_id": "task/traj/step_7", "task_name": "task",
            "trajectory_id": "traj", "step_index": 7,
            "target_tool_call": {"name": "pick_up_object"},
            "exact_action_step_match": True,
        },
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    judge = tmp_path / "judge.jsonl"
    judge.write_text(json.dumps({"sample_id": rows[0]["sample_id"], "equivalent": False}) + "\n")
    output = analyze(predictions, judge)
    first = next(row for row in output if row["row_type"] == "first_divergence_position")
    assert first["key"] == 0
    positions = [row["key"] for row in output if row["row_type"] == "position_accuracy"]
    assert positions == [0, 1]
