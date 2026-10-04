from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bc_task_vlm.dagger.data_generation.build_mixture import _balanced_original_selection
from training.bc_task_vlm.dagger.data_generation.materialize import build_full_replay_prefix, prepare_prefixes
from training.bc_task_vlm.dagger.loop import _failed, _simulator_only_rejection, _training_ready


def test_full_replay_prefix_keeps_only_executed_legal_calls() -> None:
    row = {
        "trajectory_id": "episode_1",
        "task_name": "prepare_coffee",
        "steps": [
            {"agent": "agent_0", "legal": True, "executed": True, "proposal": {"tool": "get_image", "args": {"views": ["map"]}}},
            {"agent": "agent_1", "legal": False, "executed": False, "proposal": {"tool": "navigate_to_fixture", "args": {"fixture_id": "x"}}},
        ],
    }
    prefix = build_full_replay_prefix(row)
    assert prefix["expert_review"]["correction"]["branch_before_event_position"] == 1
    assert prefix["diagnostic"]["full_trajectory"][0]["proposal"]["tool"] == "get_image"


def test_prepare_prefixes_requires_success(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"trajectory_id": "x", "task_name": "x", "fsm_goal_satisfied": False}) + "\n")
    try:
        prepare_prefixes(source, tmp_path / "out.jsonl")
    except ValueError as exc:
        assert "No successful" in str(exc)
    else:
        raise AssertionError("expected fail-closed behavior")


def test_balanced_original_selection_is_reproducible_and_balanced() -> None:
    candidates = {"a": [f"a{i}" for i in range(5)], "b": [f"b{i}" for i in range(5)], "c": [f"c{i}" for i in range(5)]}
    first = _balanced_original_selection(candidates, 8, seed=7)
    second = _balanced_original_selection(candidates, 8, seed=7)
    assert first == second
    counts = {task: sum(selected_task == task for selected_task, _ in first) for task in candidates}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_dagger_requires_error_free_fsm_success() -> None:
    clean = {"fsm_goal_satisfied": True, "had_rejection": False}
    recovered = {"fsm_goal_satisfied": True, "had_rejection": True}
    assert _training_ready(clean)
    assert not _training_ready(recovered)
    assert _failed([clean, recovered]) == [recovered]


def test_simulator_only_rejection_is_not_a_symbolic_error() -> None:
    simulator = {"steps": [{"legal": True, "executed": True, "sim_success": False}]}
    symbolic = {"steps": [{"legal": False, "executed": False}]}
    mixed = {"steps": [*simulator["steps"], *symbolic["steps"]]}
    assert _simulator_only_rejection(simulator)
    assert not _simulator_only_rejection(symbolic)
    assert not _simulator_only_rejection(mixed)
