"""Compare RoboCasa's original bowl pose with reconstructed placement checks."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_parentfix_ab_v1"
OUT = ROOT / "training/bc_task_vlm/eval_runs/arrange_bread_bowl_original_pose_diagnostic_v1"
SCENE = "reserved_layout_15_style_14_seed_99"
CASE_ID = "4bb75651f97cefb4"


def main() -> None:
    payload = json.loads((AUDIT / SCENE / "audit.json").read_text())
    record = next(row for row in payload["records"] if row["case_id"] == CASE_ID)
    trajectory = {
        "trajectory_id": record["carrier_trajectory_id"],
        "task": "arrange_bread_bowl",
        "initial_state": record["initial_state"],
        "steps": [],
    }
    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    session = SimSession(
        composite_task="ArrangeBreadBowl",
        sample_trajectory=trajectory,
        layout=15,
        style=14,
        seed=99,
        gl_backend="egl",
        render_size=512,
        map_dpi=100,
        map_renderer="raster",
    )
    executor = session.executor
    runner = executor.runner
    original_pos, _ = executor._get_object_pose("bowl")
    adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=True)
    adapted = adapter.adapt(trajectory)
    counter_id = adapted["initial_state"]["objects"]["bowl"]["location"]
    toaster_id = adapted["initial_state"]["objects"]["toaster_oven_bread"]["location"]
    ignored = executor._ignored_fixture_ids_for_placement(counter_id, None)
    context = runner._build_object_collision_context(
        "bowl", counter_id, ignored_fixture_ids=ignored
    )
    original_labels = runner._candidate_scene_overlap_labels(original_pos, context)
    preferred = executor._default_fixture_surface_preference(counter_id, "bowl")
    candidates = runner._iter_object_target_candidates(
        runner._fixtures[counter_id], "bowl", preferred_xy=preferred
    )
    candidate_rows = []
    label_counts: Counter[str] = Counter()
    for candidate in candidates:
        labels = runner._candidate_scene_overlap_labels(candidate, context)
        label_counts.update(labels)
        candidate_rows.append({
            "position": [round(float(value), 6) for value in candidate],
            "inside_counter": bool(runner._validate_object_on_fixture(candidate, counter_id)),
            "overlap_labels": labels,
        })
    refs = {}
    for role, value in getattr(executor.env, "fixture_refs", {}).items():
        fixture = value[0] if isinstance(value, tuple) else value
        refs[str(role)] = next(
            (fixture_id for fixture_id, candidate in runner._fixtures.items() if candidate is fixture),
            None,
        )
    result = {
        "scene": {"layout": 15, "style": 14, "seed": 99},
        "case_id": CASE_ID,
        "resolved_counter_id": counter_id,
        "resolved_toaster_id": toaster_id,
        "fixture_refs": refs,
        "original_bowl_position": [round(float(value), 6) for value in original_pos],
        "original_inside_counter": bool(runner._validate_object_on_fixture(original_pos, counter_id)),
        "original_overlap_labels": original_labels,
        "preferred_position": None if preferred is None else [round(float(value), 6) for value in preferred],
        "candidate_count": len(candidate_rows),
        "candidate_overlap_counts": dict(label_counts),
        "candidates": candidate_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
