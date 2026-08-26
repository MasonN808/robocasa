"""Audit the repaired stool, sink-handover, and coffee-action paths in one scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata


def _clear(runner) -> None:
    for index in range(runner._num_robots):
        offset = float(index * 10)
        runner._set_robot_pose(index, np.asarray([100.0 + offset, 100.0 + offset]), 0.0)
    runner.env.sim.forward()


def _setup_bowls(session: SimSession, adapter: TrajectoryAdapter) -> dict:
    runner = session.executor.runner
    probes = []
    for symbol in ("stool1", "stool2"):
        fixture_id = adapter._fixture_aliases[symbol]
        for order in ((0, 1), (1, 0)):
            _clear(runner)
            placements = []
            for robot_idx in order:
                result = session.executor.navigate_to_fixture(fixture_id, robot_idx=robot_idx)
                placements.append({
                    "robot_idx": robot_idx,
                    "success": bool(result.success),
                    "ready": bool(session.executor._robot_near_fixture(robot_idx, fixture_id)),
                    "position": runner._get_robot_position(robot_idx)[:2].tolist(),
                })
            probes.append({"fixture": symbol, "resolved_fixture": fixture_id, "order": list(order), "placements": placements, "passed": all(row["success"] and row["ready"] for row in placements)})
    return {"passed": all(row["passed"] for row in probes), "probes": probes}


def _alcohol(session: SimSession, adapter: TrajectoryAdapter, adapted: dict) -> dict:
    executor = session.executor
    executor.restore_baseline_state()
    executor.load_initial_state(adapted["initial_state"])
    agents = adapted["initial_state"]["agents"]
    sink_id = adapter._fixture_aliases["sink"]
    holders = [agent for agent, state in agents.items() if state.get("location") == sink_id]
    if len(holders) == 1:
        holder = holders[0]
        entrant = "agent_1" if holder == "agent_0" else "agent_0"
    else:
        holder, entrant = "agent_1", "agent_0"
        initial_enter = executor.navigate_to_fixture(sink_id, robot_idx=1)
        if not initial_enter.success:
            return {"passed": False, "error": "could not establish initial sink holder", "holders": holders}
    holder_idx, entrant_idx = int(holder[-1]), int(entrant[-1])
    yielded = executor.give_space(sink_id, robot_idx=holder_idx)
    entered = executor.navigate_to_fixture(sink_id, robot_idx=entrant_idx)
    ready = bool(executor._robot_near_fixture(entrant_idx, sink_id))
    return {"passed": bool(yielded.success and entered.success and ready), "holder": holder, "entrant": entrant, "yield_success": bool(yielded.success), "entry_success": bool(entered.success), "entrant_ready": ready, "positions": {f"agent_{idx}": executor.runner._get_robot_position(idx)[:2].tolist() for idx in range(2)}}


def _coffee(session: SimSession, adapter: TrajectoryAdapter, adapted: dict) -> dict:
    executor = session.executor
    executor.restore_baseline_state()
    executor.load_initial_state(adapted["initial_state"])
    agents = adapted["initial_state"]["agents"]
    coffee_id = adapter._fixture_aliases["coffee_machine"]
    actor = next((agent for agent, state in agents.items() if state.get("location") == coffee_id), "agent_0")
    actor_idx = int(actor[-1])
    if agents.get(actor, {}).get("location") != coffee_id:
        entered = executor.navigate_to_fixture(coffee_id, robot_idx=actor_idx)
        if not entered.success:
            return {"passed": False, "error": "coffee navigation failed", "actor": actor}
    mug_id = next(object_id for object_id, state in adapted["initial_state"]["objects"].items() if str(state.get("object_type", "")).lower() == "mug")
    executor._held_objects[actor_idx] = mug_id
    executor._held_object_offsets[actor_idx] = executor._held_pose_offset(mug_id)
    executor._set_support_parent(mug_id, f"held_by_robot_{actor_idx}")
    executor._sync_held_object(actor_idx)
    before = executor.runner._get_robot_position(actor_idx)[:2].copy()
    placed = executor.place_under(mug_id, reference_fixture_id=coffee_id, robot_idx=actor_idx)
    after_place = executor.runner._get_robot_position(actor_idx)[:2].copy()
    pressed = executor.press_button(coffee_id, "start_button", robot_idx=actor_idx)
    after_press = executor.runner._get_robot_position(actor_idx)[:2].copy()
    ready = bool(executor._robot_near_fixture(actor_idx, coffee_id))
    return {"passed": bool(placed.success and pressed.success and ready), "actor": actor, "coffee_fixture": coffee_id, "place_under_success": bool(placed.success), "press_button_success": bool(pressed.success), "ready": ready, "position_before": before.tolist(), "position_after_place": after_place.tolist(), "position_after_press": after_press.tolist(), "movement_during_actions_m": float(np.linalg.norm(after_press - before))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--style", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (("SetupBowls", 0, _setup_bowls), ("AlcoholServingPrep", 12, _alcohol), ("PrepareCoffee", 8, _coffee))
    records = []
    for task, run_index, probe in cases:
        task_class = REGISTERED_ENVS[get_task_metadata(task).composite_task]
        if args.layout in task_class.EXCLUDE_LAYOUTS or args.style in task_class.EXCLUDE_STYLES:
            records.append({"task": task, "passed": True, "supported": False})
            continue
        trajectory = _trajectory(task, run_index)
        session = None
        try:
            session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=args.layout, style=args.style, seed=args.seed, gl_backend="egl", render_size=128, map_dpi=40, map_renderer="raster")
            adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
            adapted = adapter.adapt(trajectory)
            if probe is _setup_bowls:
                result = probe(session, adapter)
            else:
                result = probe(session, adapter, adapted)
            records.append({"task": task, "supported": True, **result})
        except Exception as exc:
            records.append({"task": task, "passed": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if session is not None:
                session.close()
    payload = {"scene": {"layout": args.layout, "style": args.style, "seed": args.seed}, "passed": all(row["passed"] for row in records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"scene": payload["scene"], "passed": payload["passed"], "tasks": {row["task"]: row["passed"] for row in records}}, indent=2))


if __name__ == "__main__":
    main()
