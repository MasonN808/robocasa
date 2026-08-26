"""Regression probe production cabinet aliases and child-parent access."""

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
from training.bc_task_vlm.probe_parent_access_occupied import ROOT, SCENE, _annotate, _positions


OUT = ROOT / "training/bc_task_vlm/eval_runs/candidate_global_access_semantics_probe"
ASSETS = OUT / "assets"

CABINET_CASES = [
    {
        "name": "gather_counter_access_partner_at_cabinet",
        "task": "GatherMarinadeIngredients", "run_index": 0, "num_runs": 32,
        "cabinet": "cabinet", "parent": "counter", "parent_declared": True,
        "locations": {"agent_0": "counter", "agent_1": "cabinet"},
        "calls": [
            ("agent_0", "open_hinged_part", {"target_id": "cabinet", "part_id": "hinged"}),
            ("agent_0", "pick_up_object", {"object_id": "oil_or_vinegar_bottle", "source_id": "cabinet"}),
            ("agent_1", "pick_up_object", {"object_id": "shaker", "source_id": "cabinet"}),
        ],
    },
    {
        "name": "arrange_tea_parent_access_and_close",
        "task": "ArrangeTea", "run_index": 0, "num_runs": 32,
        "cabinet": "cab", "parent": "counter", "parent_declared": True,
        "locations": {"agent_0": "cab", "agent_1": "counter"},
        "calls": [
            ("agent_1", "open_hinged_part", {"target_id": "cab", "part_id": "hinged"}),
            ("agent_1", "pick_up_object", {"object_id": "mug", "source_id": "cab"}),
            ("agent_1", "place_in_receptacle", {"object_id": "mug", "receptacle_id": "tray"}),
            ("agent_1", "close_hinged_part", {"target_id": "cab", "part_id": "hinged"}),
        ],
    },
    {
        "name": "garnish_cupcake_two_parent_alias_navigations",
        "task": "GarnishCupcake", "run_index": 0, "num_runs": 32,
        "cabinet": "cabinet", "parent": "dining_counter", "parent_declared": False,
        "locations": {"agent_0": "dining_counter", "agent_1": "dining_counter"},
        "calls": [
            ("agent_0", "navigate_to_fixture", {"fixture_id": "cabinet"}),
            ("agent_1", "navigate_to_fixture", {"fixture_id": "cabinet"}),
            ("agent_0", "pick_up_object", {"object_id": "cinnamon", "source_id": "cabinet"}),
        ],
    },
    {
        "name": "set_bowls_two_shared_cabinet_pickups",
        "task": "SetBowlsForSoup", "run_index": 0, "num_runs": 32,
        "cabinet": "cab", "parent": "dining_table", "parent_declared": False,
        "locations": {"agent_0": "dining_table", "agent_1": "dining_table"},
        "calls": [
            ("agent_0", "navigate_to_fixture", {"fixture_id": "cab"}),
            ("agent_1", "navigate_to_fixture", {"fixture_id": "cab"}),
            ("agent_0", "pick_up_object", {"object_id": "bowl1", "source_id": "cab"}),
            ("agent_1", "pick_up_object", {"object_id": "bowl2", "source_id": "cab"}),
        ],
    },
    {
        "name": "spice_station_two_parent_alias_pickups",
        "task": "SetUpSpiceStation", "run_index": 0, "num_runs": 32,
        "cabinet": "cabinet", "parent": "counter_stove", "parent_declared": False,
        "locations": {"agent_0": "counter_stove", "agent_1": "counter_stove"},
        "calls": [
            ("agent_0", "navigate_to_fixture", {"fixture_id": "cabinet"}),
            ("agent_1", "navigate_to_fixture", {"fixture_id": "cabinet"}),
            ("agent_0", "pick_up_object", {"object_id": "spice", "source_id": "cabinet"}),
            ("agent_1", "pick_up_object", {"object_id": "shaker", "source_id": "cabinet"}),
        ],
    },
    {
        "name": "soup_retrieve_return_and_close",
        "task": "PrepareSoupServing", "run_index": 0, "num_runs": 32,
        "cabinet": "cab", "parent": "counter", "parent_declared": False,
        "locations": {"agent_0": "counter", "agent_1": "cab"},
        "calls": [
            ("agent_0", "navigate_to_fixture", {"fixture_id": "cab"}),
            ("agent_0", "pick_up_object", {"object_id": "ladle", "source_id": "cab"}),
            ("agent_0", "navigate_to_fixture", {"fixture_id": "stove"}),
            ("agent_0", "place_in_receptacle", {"object_id": "ladle", "receptacle_id": "pot"}),
            ("agent_0", "navigate_to_fixture", {"fixture_id": "cab"}),
            ("agent_0", "close_hinged_part", {"target_id": "cab", "part_id": "hinged"}),
        ],
    },
]


def _trajectory(task: str, run_index: int, num_runs: int, locations: dict[str, str]) -> dict[str, Any]:
    config = _config(
        Namespace(task=task, num_runs=num_runs, run_index=run_index, location="global", temperature=0.6),
        model=FLASH_MODEL, thinking="low",
    )
    state = get_task_definition(task).build_task_instance(run_index, config).initial_state
    for agent, location in locations.items():
        state["agents"][agent]["location"] = location
    return {
        "trajectory_id": f"probe_{task}_{run_index}", "task": task, "composite_task": task,
        "initial_state": state,
        "grounding_map": build_grounding_map_for_task(task, state), "steps": [],
    }


def _render(session: SimSession, stem: str, note: str) -> dict[str, Any]:
    session.executor._invalidate_visual_cache()
    top = ASSETS / f"{stem}_top.jpg"
    _annotate(session.executor.runner._render_top_view(), stem, note).save(top, quality=94)
    views = {}
    for agent in ("agent_0", "agent_1"):
        paths, names = session.render_views(
            ("agentview_center", "wrist"), agent_id=agent, out_dir=ASSETS, tag=f"{stem}_{agent}"
        )
        views[agent] = dict(zip(names, paths))
    return {"top": str(top), "agent_views": views, "positions": _positions(session)}


def _execute(adapter: Any, adapted: dict[str, Any], session: SimSession, agent: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    call = adapter._adapt_step(
        {"agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"], output_dir=None,
    )
    result = session.executor.execute(call["tool"], robot_idx=call["robot_idx"], **call["args"])
    return {"call": call, "success": bool(result.success), "details": result.details}


def _install_shared_cabinet_policy(session: SimSession, cabinet: str, parent: str) -> None:
    """Map cabinet navigation physically to its parent and allow both aliases."""
    executor = session.executor
    original_near = executor._robot_near_fixture
    original_move = executor._move_robot_near_fixture_with_retries

    def near(robot_idx: int, fixture_id: str, threshold: float = 1.5) -> bool:
        if fixture_id == cabinet:
            return original_near(robot_idx, parent, threshold=threshold)
        return original_near(robot_idx, fixture_id, threshold=threshold)

    def move(robot_idx: int, fixture_id: str, *args: Any, **kwargs: Any) -> bool:
        if fixture_id == cabinet:
            return executor.runner._move_robot_near_fixture(
                robot_idx, parent,
                ref_object_id=kwargs.get("ref_object_id"),
                ref_pos_override=kwargs.get("ref_pos_override"),
                require_front=False,
            )
        return original_move(robot_idx, fixture_id, *args, **kwargs)

    executor._robot_near_fixture = near
    executor._move_robot_near_fixture_with_retries = move
    executor._clear_fixture_blockers = lambda *_args, **_kwargs: False


def _run_cabinet(case: dict[str, Any]) -> dict[str, Any]:
    trajectory = _trajectory(case["task"], case["run_index"], case["num_runs"], case["locations"])
    session = SimSession(
        composite_task=case["task"], sample_trajectory=trajectory, **SCENE,
        gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster",
    )
    record = {k: case[k] for k in ("name", "task", "cabinet", "parent", "parent_declared")}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        # Runtime fixtures have scene-specific IDs (for example,
        # ``cab_1_main_group``). Resolve both aliases before installing the
        # experimental policy; comparing against symbolic task IDs would leave
        # the production front-facing behavior active.
        actual_cabinet = adapter._adapt_step(
            {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": case["cabinet"]}},
            resolved_initial_state=adapted["initial_state"], output_dir=None,
        )["args"]["fixture_id"]
        requested_parent = adapter._adapt_step(
            {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": case["parent"]}},
            resolved_initial_state=adapted["initial_state"], output_dir=None,
        )["args"]["fixture_id"]
        scene_parent = (session.executor.get_scene_description().get("fixtures") or {}).get(
            actual_cabinet, {}
        ).get("parent_fixture")
        actual_parent = scene_parent if isinstance(scene_parent, str) else requested_parent
        record["resolved_cabinet"] = actual_cabinet
        record["resolved_parent"] = actual_parent
        record["requested_parent_symbol"] = case["parent"]
        record["requested_parent_resolution"] = requested_parent
        record["scene_inferred_parent"] = scene_parent
        record["requested_parent_matches_scene_parent"] = requested_parent == scene_parent
        record["initial_state"] = adapted["initial_state"]
        record["initial"] = _render(session, f"{case['name']}_initial", "Before the proposed shared cabinet policy is exercised.")
        calls = []
        for index, (agent, tool, args) in enumerate(case["calls"]):
            calls.append(_execute(adapter, adapted, session, agent, tool, args))
            if tool in {"navigate_to_fixture", "open_hinged_part", "close_hinged_part"}:
                record.setdefault("milestones", []).append({
                    "label": f"{index + 1}: {agent} {tool}",
                    "frame": _render(session, f"{case['name']}_step_{index + 1}", f"After {agent} {tool}.")
                })
        record["calls"] = calls
        record["final"] = _render(session, f"{case['name']}_final", "After all tested cabinet calls.")
        record["success"] = all(call["success"] for call in calls)
        record["displacements"] = {
            agent: round(float(np.linalg.norm(
                np.asarray(record["final"]["positions"][agent])[:2]
                - np.asarray(record["initial"]["positions"][agent])[:2]
            )), 4) for agent in ("agent_0", "agent_1")
        }
        final_xy = [np.asarray(record["final"]["positions"][agent])[:2] for agent in ("agent_0", "agent_1")]
        record["final_inter_robot_distance"] = round(float(np.linalg.norm(final_xy[0] - final_xy[1])), 4)
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def _run_retained_parent(task: str) -> dict[str, Any]:
    if task == "ArrangeBreadBowl":
        run_index, num_runs, child, parent = 6, 8, "toaster_oven", "counter"
        locations = {"agent_0": child, "agent_1": parent}
        calls = [
            ("open_hinged_part", {"target_id": child, "part_id": "door"}),
            ("pick_up_object", {"object_id": "toaster_oven_bread", "source_id": child}),
            ("place_in_receptacle", {"object_id": "toaster_oven_bread", "receptacle_id": "bowl"}),
        ]
    else:
        run_index, num_runs, child, parent = 0, 32, "coffee_machine", "counter"
        locations = {"agent_0": child, "agent_1": parent}
        calls = [
            ("pick_up_object", {"object_id": "sugar_cube", "source_id": "saucer_plate"}),
            ("place_in_receptacle", {"object_id": "sugar_cube", "receptacle_id": "coffee"}),
        ]
    trajectory = _trajectory(task, run_index, num_runs, locations)
    session = SimSession(
        composite_task=task, sample_trajectory=trajectory, **SCENE,
        gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster",
    )
    name = f"{task.lower()}_retain_{child}_pose"
    record: dict[str, Any] = {"name": name, "task": task, "child": child, "parent": parent}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        actual_child = adapter._adapt_step(
            {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": child}},
            resolved_initial_state=adapted["initial_state"], output_dir=None,
        )["args"]["fixture_id"]
        actual_parent = adapter._adapt_step(
            {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": parent}},
            resolved_initial_state=adapted["initial_state"], output_dir=None,
        )["args"]["fixture_id"]
        record["resolved_child"] = actual_child
        record["resolved_parent"] = actual_parent
        record["initial_state"] = adapted["initial_state"]
        record["initial"] = _render(session, f"{name}_initial", "Actor starts at exclusive child; partner occupies parent.")
        executed = [_execute(adapter, adapted, session, "agent_0", tool, args) for tool, args in calls]
        record["calls"] = executed
        record["final"] = _render(session, f"{name}_final", "Parent-side manipulation completed without moving either robot.")
        record["success"] = all(call["success"] for call in executed)
        record["displacements"] = {
            agent: round(float(np.linalg.norm(
                np.asarray(record["final"]["positions"][agent])[:2]
                - np.asarray(record["initial"]["positions"][agent])[:2]
            )), 4) for agent in ("agent_0", "agent_1")
        }
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [_run_cabinet(case) for case in CABINET_CASES]
    records.extend(_run_retained_parent(task) for task in ("ArrangeBreadBowl", "SweetenCoffee"))
    payload = {"scene": SCENE, "production_changes": True, "records": records}
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if all(record.get("success") for record in records) else 1)


if __name__ == "__main__":
    main()
