"""Visual probe for cabinet sharing and child-to-parent access semantics.

This is deliberately an experiment, not a production policy change.  It tests
cabinet access without front-pose requirements and parent-surface access while
retaining an exclusive appliance pose.
"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.probe_parent_access_occupied import (
    ROOT,
    SCENE,
    _annotate,
    _positions,
)


OUT = ROOT / "training/bc_task_vlm/eval_runs/cabinet_parent_semantics_probe"
ASSETS = OUT / "assets"


def _trajectory(task: str, run_index: int, num_runs: int, locations: dict[str, str]) -> dict[str, Any]:
    config = _config(
        Namespace(task=task, num_runs=num_runs, run_index=run_index, location="global", temperature=0.6),
        model=FLASH_MODEL,
        thinking="low",
    )
    instance = get_task_definition(task).build_task_instance(run_index, config)
    state = instance.initial_state
    for agent, location in locations.items():
        state["agents"][agent]["location"] = location
    return {
        "trajectory_id": f"{task}_{run_index}",
        "task": task,
        "composite_task": task,
        "initial_state": state,
        "grounding_map": build_grounding_map_for_task(task, state),
        "steps": [],
    }


def _render(session: SimSession, stem: str, title: str, note: str) -> dict[str, Any]:
    session.executor._invalidate_visual_cache()
    top_path = ASSETS / f"{stem}_top.jpg"
    _annotate(session.executor.runner._render_top_view(), title, note).save(top_path, quality=94)
    views: dict[str, dict[str, str]] = {}
    for agent in ("agent_0", "agent_1"):
        paths, names = session.render_views(
            ("agentview_center", "wrist"), agent_id=agent, out_dir=ASSETS, tag=f"{stem}_{agent}"
        )
        views[agent] = dict(zip(names, paths))
    return {"top": str(top_path), "agent_views": views, "positions": _positions(session)}


def _adapt_execute(adapter: Any, adapted: dict[str, Any], session: SimSession, agent: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    call = adapter._adapt_step(
        {"agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )
    result = session.executor.execute(call["tool"], robot_idx=call["robot_idx"], **call["args"])
    return {"call": call, "success": bool(result.success), "details": result.details}


def _is_cabinet(session: SimSession, fixture_id: str) -> bool:
    info = (session.executor.get_scene_description().get("fixtures") or {}).get(fixture_id, {})
    return "cabinet" in str(info.get("fixture_type") or "").lower()


def _install_cabinet_no_front_policy(session: SimSession, *, silent_clear: bool) -> None:
    """Treat a robot at the cabinet's parent as ready, without moving it."""
    executor = session.executor
    original_near = executor._robot_near_fixture
    original_move = executor._move_robot_near_fixture_with_retries

    def near(robot_idx: int, fixture_id: str, threshold: float = 1.5) -> bool:
        if _is_cabinet(session, fixture_id):
            info = (executor.get_scene_description().get("fixtures") or {}).get(fixture_id, {})
            parent = info.get("parent_fixture")
            if isinstance(parent, str):
                return original_near(robot_idx, parent, threshold=threshold)
        return original_near(robot_idx, fixture_id, threshold=threshold)

    def move(robot_idx: int, fixture_id: str, *args: Any, **kwargs: Any) -> bool:
        if _is_cabinet(session, fixture_id):
            # Navigation remains physically testable, but it uses ordinary
            # near-fixture placement rather than a reserved front pose.
            kwargs["require_front"] = False
            return executor.runner._move_robot_near_fixture(
                robot_idx,
                fixture_id,
                ref_object_id=kwargs.get("ref_object_id"),
                ref_pos_override=kwargs.get("ref_pos_override"),
                require_front=False,
            )
        return original_move(robot_idx, fixture_id, *args, **kwargs)

    executor._robot_near_fixture = near
    executor._move_robot_near_fixture_with_retries = move
    if not silent_clear:
        executor._clear_fixture_blockers = lambda *_args, **_kwargs: False


def _cabinet_case(start_location: str, *, silent_clear: bool) -> dict[str, Any]:
    mode = "silent_clear_on" if silent_clear else "silent_clear_off"
    name = f"gather_both_at_{start_location}_{mode}"
    trajectory = _trajectory(
        "GatherMarinadeIngredients", 0, 32,
        {"agent_0": start_location, "agent_1": start_location},
    )
    session = SimSession(
        composite_task="GatherMarinadeIngredients", sample_trajectory=trajectory,
        **SCENE, gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster",
    )
    record: dict[str, Any] = {"name": name, "policy": "cabinet_no_front", "start_location": start_location, "silent_clear": silent_clear}
    try:
        _install_cabinet_no_front_policy(session, silent_clear=silent_clear)
        adapter, adapted = session.start_trajectory(trajectory)
        record["initial"] = _render(session, f"{name}_initial", name, "Both robots start at the same symbolic location.")
        calls = [
            _adapt_execute(adapter, adapted, session, "agent_0", "open_hinged_part", {"target_id": "cabinet", "part_id": "hinged"}),
            _adapt_execute(adapter, adapted, session, "agent_0", "pick_up_object", {"object_id": "oil_or_vinegar_bottle", "source_id": "cabinet"}),
            _adapt_execute(adapter, adapted, session, "agent_1", "pick_up_object", {"object_id": "shaker", "source_id": "cabinet"}),
        ]
        record["calls"] = calls
        record["final"] = _render(session, f"{name}_final", name, "After opening and two distinct pickups from the cabinet.")
        record["success"] = all(call["success"] for call in calls)
        record["partner_displacements"] = {
            agent: round(float(np.linalg.norm(np.asarray(record["final"]["positions"][agent])[:2] - np.asarray(record["initial"]["positions"][agent])[:2])), 4)
            for agent in ("agent_0", "agent_1")
        }
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def _toaster_parent_case(*, retain_pose: bool) -> dict[str, Any]:
    mode = "retain_toaster_pose" if retain_pose else "normal_parent_reposition"
    trajectory = _trajectory(
        "ArrangeBreadBowl", 6, 8,
        {"agent_0": "toaster_oven", "agent_1": "counter"},
    )
    session = SimSession(
        composite_task="ArrangeBreadBowl", sample_trajectory=trajectory,
        **SCENE, gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster",
    )
    record: dict[str, Any] = {"name": f"arrange_bread_{mode}", "retain_pose": retain_pose}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        session.executor._clear_fixture_blockers = lambda *_args, **_kwargs: False
        actor_idx = 0
        original_move = session.executor._move_robot_near_fixture_with_retries
        original_runner_move = session.executor.runner._move_robot_near_fixture

        def keep_toaster(robot_idx: int, fixture_id: str, *args: Any, **kwargs: Any) -> bool:
            if robot_idx == actor_idx and fixture_id == "toaster_oven":
                return True
            return original_move(robot_idx, fixture_id, *args, **kwargs)

        session.executor._move_robot_near_fixture_with_retries = keep_toaster
        calls = [
            _adapt_execute(adapter, adapted, session, "agent_0", "open_hinged_part", {"target_id": "toaster_oven", "part_id": "door"}),
            _adapt_execute(adapter, adapted, session, "agent_0", "pick_up_object", {"object_id": "toaster_oven_bread", "source_id": "toaster_oven"}),
        ]
        record["before_parent_access"] = _render(session, f"arrange_bread_{mode}_before", mode, "Actor holds bread at toaster; partner remains at counter.")

        if retain_pose:
            parent_id = adapter._adapt_step(
                {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": "counter"}},
                resolved_initial_state=adapted["initial_state"], output_dir=None,
            )["args"]["fixture_id"]

            def keep_actor_for_parent(robot_idx: int, fixture_id: str, *args: Any, **kwargs: Any) -> bool:
                if robot_idx == actor_idx and fixture_id == parent_id:
                    return True
                return original_runner_move(robot_idx, fixture_id, *args, **kwargs)

            session.executor.runner._move_robot_near_fixture = keep_actor_for_parent

        calls.append(_adapt_execute(adapter, adapted, session, "agent_0", "place_in_receptacle", {"object_id": "toaster_oven_bread", "receptacle_id": "bowl"}))
        record["calls"] = calls
        record["after_parent_access"] = _render(session, f"arrange_bread_{mode}_after", mode, "Bread placed in bowl on parent counter.")
        record["success"] = all(call["success"] for call in calls)
        record["actor_displacement"] = round(float(np.linalg.norm(
            np.asarray(record["after_parent_access"]["positions"]["agent_0"])[:2]
            - np.asarray(record["before_parent_access"]["positions"]["agent_0"])[:2]
        )), 4)
        record["partner_displacement"] = round(float(np.linalg.norm(
            np.asarray(record["after_parent_access"]["positions"]["agent_1"])[:2]
            - np.asarray(record["before_parent_access"]["positions"]["agent_1"])[:2]
        )), 4)
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [
        _cabinet_case(location, silent_clear=silent)
        for location in ("cabinet", "counter")
        for silent in (True, False)
    ]
    records.extend(_toaster_parent_case(retain_pose=retain) for retain in (False, True))
    payload = {"scene": SCENE, "records": records}
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if all(record.get("success") for record in records) else 1)


if __name__ == "__main__":
    main()
