"""Probe every task-visible shared workspace for true two-robot occupancy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import robocasa  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS

from data_generation.task_level.scene_sampling import SCENE_POLICY_VERSION, scene_signature
from data_generation.task_level.tasks import get_task_definition
from data_generation.task_level.tasks.shared.concurrent_fsm import is_exclusive_fixture
from data_generation.task_level.tasks.shared.workspace_semantics import (
    canonical_agent_workspace,
)
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import (
    MANIFEST,
    _trajectory,
)
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata


FAR_POSITIONS = (np.asarray([100.0, 100.0]), np.asarray([110.0, 110.0]))


def _probe_order(session: SimSession, fixture_id: str, order: tuple[int, int]) -> dict[str, Any]:
    executor = session.executor
    runner = executor.runner
    session.executor.restore_baseline_state()
    for robot_idx, position in enumerate(FAR_POSITIONS):
        runner._set_robot_pose(robot_idx, position, 0.0)
    runner.env.sim.forward()
    fixture = runner._fixtures.get(fixture_id)
    if fixture is None:
        return {"order": list(order), "success": False, "error": "fixture_not_grounded"}
    require_front = executor._fixture_requires_front_approach(
        fixture_id
    ) or executor._surface_fixture_prefers_front_approach(fixture_id)
    placements = []
    for robot_idx in order:
        placed = runner._move_robot_near_fixture(
            robot_idx, fixture_id, require_front=require_front
        )
        placements.append(bool(placed))
        if not placed:
            break
    positions = {
        f"agent_{idx}": runner._get_robot_position(idx)[:2].tolist()
        for idx in (0, 1)
    }
    separation = float(
        np.linalg.norm(
            runner._get_robot_position(0)[:2]
            - runner._get_robot_position(1)[:2]
        )
    )
    ready = {
        f"agent_{idx}": bool(executor._robot_near_fixture(idx, fixture_id))
        for idx in (0, 1)
    }
    valid = {
        f"agent_{idx}": bool(
            runner._is_valid_robot_position(runner._get_robot_position(idx)[:2])
        )
        for idx in (0, 1)
    }
    success = (
        len(placements) == 2
        and all(placements)
        and all(ready.values())
        and all(valid.values())
        and separation >= 0.40 - 1e-6
    )
    return {
        "order": list(order),
        "success": success,
        "placements": placements,
        "positions": positions,
        "interaction_ready": ready,
        "valid_positions": valid,
        "separation_m": separation,
        "require_front": bool(require_front),
    }


def _shared_workspaces(
    initial_state: dict[str, Any],
    *,
    actionable_fixture_ids: set[str],
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for fixture_id in sorted(initial_state.get("fixtures") or {}):
        if fixture_id not in actionable_fixture_ids:
            continue
        if is_exclusive_fixture(fixture_id, initial_state):
            continue
        canonical = canonical_agent_workspace(initial_state, fixture_id)
        if isinstance(canonical, str):
            aliases.setdefault(canonical, []).append(fixture_id)
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--style", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", dest="tasks",
        help="Audit only this task; repeat for multiple tasks",
    )
    args = parser.parse_args()
    source = json.loads(MANIFEST.read_text())
    scene = {"layout": args.layout, "style": args.style, "seed": args.seed}
    records: list[dict[str, Any]] = []
    task_errors: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for task_row in source["tasks"]:
        task = task_row["task"]
        if args.tasks and task not in args.tasks:
            continue
        task_class = REGISTERED_ENVS[get_task_metadata(task).composite_task]
        if args.layout in task_class.EXCLUDE_LAYOUTS or args.style in task_class.EXCLUDE_STYLES:
            unsupported.append(task)
            continue
        session = None
        try:
            run_index = task_row["runs"][0]["run_index"]
            trajectory = _trajectory(task, run_index)
            session = SimSession(
                composite_task=task,
                sample_trajectory=trajectory,
                **scene,
                gl_backend="egl",
                render_size=128,
                map_dpi=40,
                map_renderer="raster",
            )
            adapter = TrajectoryAdapter(
                executor=session.executor, allow_approximate_ids=True
            )
            adapted = adapter.adapt(trajectory)
            navigate_spec = (
                get_task_definition(task).validator_factory(None).allowed_tool_specs.get(
                    "navigate_to_fixture", {}
                )
            )
            actionable_fixture_ids = {
                adapter._fixture_aliases.get(symbol, symbol)
                for symbol in navigate_spec.get("allowed_fixture_ids") or ()
            }
            for fixture_id, aliases in _shared_workspaces(
                adapted["initial_state"],
                actionable_fixture_ids=actionable_fixture_ids,
            ).items():
                orders = [
                    _probe_order(session, fixture_id, (0, 1)),
                    _probe_order(session, fixture_id, (1, 0)),
                ]
                records.append(
                    {
                        "task": task,
                        "workspace_fixture_id": fixture_id,
                        "symbolic_aliases": aliases,
                        "compatible": all(row["success"] for row in orders),
                        "orders": orders,
                    }
                )
        except Exception as exc:
            task_errors.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if session is not None:
                session.close()

    payload = {
        "schema_version": 1,
        "scene_policy_version": SCENE_POLICY_VERSION,
        "scene": {**scene, "scene_signature": scene_signature(scene)},
        "records": records,
        "unsupported_tasks": unsupported,
        "task_errors": task_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scene": scene,
                "workspace_count": len(records),
                "compatible": sum(row["compatible"] for row in records),
                "incompatible": sum(not row["compatible"] for row in records),
                "task_errors": len(task_errors),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
