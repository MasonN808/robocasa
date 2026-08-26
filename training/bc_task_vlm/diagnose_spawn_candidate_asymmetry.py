"""Inspect candidate geometry and readiness for disputed spawn failures."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from robocasa.utils.placement import get_fixture_aabb
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/spawn_candidate_asymmetry_diagnostic.json"
CASES = (
    ("SetupBowls", 0, 15, 14, 42, ("stool1", "stool2")),
    ("AlcoholServingPrep", 12, 15, 14, 42, ("dining_table", "sink")),
    ("PrepareCoffee", 8, 50, 14, 42, ("coffee_machine", "cabinet_parent_counter")),
)


def _candidate_summary(session: SimSession, fixture_id: str, other_positions=()) -> dict:
    runner = session.executor.runner
    grid = runner._occupancy_grid
    fixture = runner._fixtures[fixture_id]
    fmin, fmax = get_fixture_aabb(fixture)
    require_front = session.executor._surface_fixture_prefers_front_approach(fixture_id)
    faces = grid._get_face_order(fixture)[:1] if require_front else ["neg_y", "pos_y", "neg_x", "pos_x"]
    rows = []
    for standoff in (grid._standoff, grid._standoff + grid.cell_size, grid._standoff + 2 * grid.cell_size):
        for face in faces:
            for pos, _yaw in grid._sample_face(face, fmin, fmax, standoff=standoff):
                rows.append({
                    "face": face,
                    "standoff": float(standoff),
                    "position": pos.tolist(),
                    "standable": bool(grid.is_standable(pos)),
                    "far_from_others": all(float(np.linalg.norm(pos - np.asarray(other)[:2])) >= grid._MIN_ROBOT_SEPARATION for other in other_positions),
                })
    return {
        "fixture_id": fixture_id,
        "fixture_type": session.executor._get_fixture_type_name(fixture_id),
        "aabb_xy": [np.asarray(fmin)[:2].tolist(), np.asarray(fmax)[:2].tolist()],
        "surface_prefers_front": require_front,
        "fixture_requires_front": session.executor._fixture_requires_front_approach(fixture_id),
        "preferred_face": grid.preferred_reachable_approach_face(fixture),
        "candidate_count": len(rows),
        "standable_count": sum(row["standable"] for row in rows),
        "standable_and_separated_count": sum(row["standable"] and row["far_from_others"] for row in rows),
        "candidates": rows,
    }


def main() -> None:
    results = []
    for task, run_index, layout, style, seed, symbols in CASES:
        trajectory = _trajectory(task, run_index)
        session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=layout, style=style, seed=seed, gl_backend="egl", render_size=128, map_dpi=40, map_renderer="raster")
        try:
            adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
            adapted = adapter.adapt(trajectory)
            resolved = {symbol: adapter._fixture_aliases[symbol] for symbol in symbols}
            fixture_rows = {symbol: _candidate_summary(session, fixture_id) for symbol, fixture_id in resolved.items()}
            runner = session.executor.runner
            for index, position in enumerate((np.asarray([100.0, 100.0]), np.asarray([110.0, 110.0]))):
                runner._set_robot_pose(index, position, 0.0)
            runner.env.sim.forward()
            placement_rows = []
            target_pairs = []
            if task == "SetupBowls":
                for symbol, fixture_id in resolved.items():
                    target_pairs.append((symbol, (("agent_0", fixture_id), ("agent_1", fixture_id))))
            else:
                target_pairs.append(("initial_configuration", tuple(
                    (agent_id, state["location"])
                    for agent_id, state in adapted["initial_state"]["agents"].items()
                )))
            for label, targets in target_pairs:
                for index, position in enumerate((np.asarray([100.0, 100.0]), np.asarray([110.0, 110.0]))):
                    runner._set_robot_pose(index, position, 0.0)
                runner.env.sim.forward()
                for agent_id, fixture_id in targets:
                    index = int(agent_id.rsplit("_", 1)[1])
                    before = runner._get_robot_position(index)[:2].copy()
                    placed = bool(runner._move_robot_near_fixture(index, fixture_id, require_front=session.executor._surface_fixture_prefers_front_approach(fixture_id)))
                    after = runner._get_robot_position(index)[:2].copy()
                    placement_rows.append({"probe":label,"agent":agent_id,"fixture":fixture_id,"placed_return":placed,"before":before.tolist(),"after":after.tolist(),"ready_after":bool(session.executor._robot_near_fixture(index,fixture_id)),"valid_after":bool(runner._is_valid_robot_position(after))})
            results.append({"task":task,"scene":{"layout":layout,"style":style,"seed":seed},"resolved":resolved,"fixtures":fixture_rows,"placements":placement_rows})
        finally:
            session.close()
    OUT.write_text(json.dumps(results, indent=2)+"\n")
    print(OUT)


if __name__ == "__main__": main()
