"""Probe child-to-parent manipulation while the partner occupies the parent."""

from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
DISABLE_SILENT_BLOCKER_CLEARING = os.environ.get(
    "DISABLE_SILENT_BLOCKER_CLEARING", "0"
) == "1"
RETAIN_EXISTING_EXCLUSIVE_POSE = os.environ.get(
    "RETAIN_EXISTING_EXCLUSIVE_POSE", "0"
) == "1"
OUT = ROOT / "training/bc_task_vlm/eval_runs" / (
    "parent_access_retain_exclusive_pose_probe"
    if RETAIN_EXISTING_EXCLUSIVE_POSE
    else (
        "parent_access_no_silent_clear_probe"
        if DISABLE_SILENT_BLOCKER_CLEARING
        else "parent_access_occupied_probe"
    )
)
ASSETS = OUT / "assets"
SCENE = {"layout": 11, "style": 14, "seed": 42}
SCENARIOS = [
    {
        "name": "arrange_bread_place_in_receptacle",
        "task": "ArrangeBreadBowl", "run_index": 6, "num_runs": 8,
        "actor": "agent_0", "child": "toaster_oven", "parent": "counter",
        "setup": [
            ("open_hinged_part", {"target_id": "toaster_oven", "part_id": "door"}),
            ("pick_up_object", {"object_id": "toaster_oven_bread", "source_id": "toaster_oven"}),
        ],
        "action": ("place_in_receptacle", {"object_id": "toaster_oven_bread", "receptacle_id": "bowl"}),
    },
    {
        "name": "arrange_tea_place_in_receptacle",
        "task": "ArrangeTea", "run_index": 0, "num_runs": 32,
        "actor": "agent_0", "child": "cab", "parent": "counter",
        "setup": [
            ("open_hinged_part", {"target_id": "cab", "part_id": "hinged"}),
            ("pick_up_object", {"object_id": "mug", "source_id": "cab"}),
        ],
        "action": ("place_in_receptacle", {"object_id": "mug", "receptacle_id": "tray"}),
    },
    {
        "name": "gather_marinade_place_next_to",
        "task": "GatherMarinadeIngredients", "run_index": 0, "num_runs": 32,
        "actor": "agent_1", "child": "cabinet", "parent": "counter",
        "setup": [
            ("open_hinged_part", {"target_id": "cabinet", "part_id": "hinged"}),
            ("pick_up_object", {"object_id": "oil_or_vinegar_bottle", "source_id": "cabinet"}),
        ],
        "action": ("place_next_to", {"object_id": "oil_or_vinegar_bottle", "reference_object_id": "mixing_bowl"}),
    },
    {
        "name": "spicy_marinade_place_on_surface",
        "task": "SpicyMarinade", "run_index": 0, "num_runs": 32,
        "actor": "agent_1", "child": "cabinet", "parent": "counter",
        "setup": [
            ("open_hinged_part", {"target_id": "cabinet", "part_id": "hinged"}),
            ("pick_up_object", {"object_id": "bowl", "source_id": "cabinet"}),
        ],
        "action": ("place_on_surface", {"object_id": "bowl", "support_id": "counter"}),
    },
]


def _annotate(image: Any, heading: str, note: str) -> Image.Image:
    result = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.rounded_rectangle((10, 10, result.width - 10, 70), 7, fill=(8, 15, 30))
    draw.text((22, 21), heading, fill="white", font=font)
    draw.text((22, 45), note, fill=(226, 232, 240), font=font)
    return result


def _render(session: SimSession, path: Path, heading: str, note: str) -> None:
    session.executor._invalidate_visual_cache()
    _annotate(session.executor.runner._render_top_view(), heading, note).save(path, quality=94)


def _render_agent_views(
    session: SimSession,
    *,
    agent: str,
    tag: str,
) -> dict[str, str]:
    paths, views = session.render_views(
        ("agentview_center", "wrist"),
        agent_id=agent,
        out_dir=ASSETS,
        tag=tag,
    )
    return dict(zip(views, paths))


def _positions(session: SimSession) -> dict[str, list[float]]:
    return {
        f"agent_{idx}": [round(float(v), 6) for v in session.executor.runner._get_robot_position(idx)]
        for idx in (0, 1)
    }


def _execute(adapter: Any, adapted: dict[str, Any], session: SimSession, agent: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    call = adapter._adapt_step(
        {"agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"], output_dir=None,
    )
    result = session.executor.execute(call["tool"], robot_idx=call["robot_idx"], **call["args"])
    if not result.success:
        raise RuntimeError(f"{agent} {tool} failed: {result.details}")
    return {"call": call, "details": result.details}


def _trajectory(scenario: dict[str, Any]) -> dict[str, Any]:
    config = _config(
        Namespace(task=scenario["task"], num_runs=scenario["num_runs"], run_index=scenario["run_index"], location="global", temperature=0.6),
        model=FLASH_MODEL, thinking="low",
    )
    definition = get_task_definition(scenario["task"])
    instance = definition.build_task_instance(scenario["run_index"], config)
    state = instance.initial_state
    partner = next(agent for agent in state["agents"] if agent != scenario["actor"])
    # This is the condition under test, not a post-hoc robot move: simulator
    # initialization receives the symbolic actor-at-child/partner-at-parent state.
    state["agents"][scenario["actor"]]["location"] = scenario["child"]
    state["agents"][partner]["location"] = scenario["parent"]
    return {
        "trajectory_id": scenario["name"], "task": scenario["task"],
        "composite_task": scenario["task"], "initial_state": state,
        "grounding_map": build_grounding_map_for_task(scenario["task"], state), "steps": [],
    }


def _run(scenario: dict[str, Any], navigate_parent: bool) -> dict[str, Any]:
    branch = "explicit_parent_navigation" if navigate_parent else "implicit_parent_access"
    trajectory = _trajectory(scenario)
    session = SimSession(composite_task=scenario["task"], sample_trajectory=trajectory, **SCENE, gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster")
    record: dict[str, Any] = {"scenario": scenario["name"], "task": scenario["task"], "branch": branch}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        if DISABLE_SILENT_BLOCKER_CLEARING:
            # Preserve the partner's pose. A failed front-facing placement must
            # surface as a failed tool call rather than secretly executing a
            # give_space action on a different robot.
            session.executor._clear_fixture_blockers = lambda *_args, **_kwargs: False
        if RETAIN_EXISTING_EXCLUSIVE_POSE:
            original_move = session.executor._move_robot_near_fixture_with_retries
            actor_idx = int(scenario["actor"].split("_")[-1])
            child_fixture = adapter._adapt_step(
                {
                    "agent": scenario["actor"],
                    "tool": "navigate_to_fixture",
                    "args": {"fixture_id": scenario["child"]},
                },
                resolved_initial_state=adapted["initial_state"],
                output_dir=None,
            )["args"]["fixture_id"]

            def retain_actor_child_pose(robot_idx, fixture_id, *args, **kwargs):
                if robot_idx == actor_idx and fixture_id == child_fixture:
                    return True
                return original_move(robot_idx, fixture_id, *args, **kwargs)

            session.executor._move_robot_near_fixture_with_retries = retain_actor_child_pose
        record["initial_positions"] = _positions(session)
        _render(session, ASSETS / f'{scenario["name"]}_{branch}_initial.jpg', f'{scenario["task"]}: {branch}', "Actor starts at child fixture; partner occupies parent counter.")
        record["active_agent_views_initial"] = _render_agent_views(
            session,
            agent=scenario["actor"],
            tag=f'{scenario["name"]}_{branch}_initial',
        )
        calls = []
        for setup_index, (tool, args) in enumerate(scenario["setup"]):
            calls.append(_execute(adapter, adapted, session, scenario["actor"], tool, args))
            if tool == "open_hinged_part":
                record.setdefault("active_agent_views_after_open", []).append(
                    _render_agent_views(
                        session,
                        agent=scenario["actor"],
                        tag=f'{scenario["name"]}_{branch}_after_open_{setup_index}',
                    )
                )
        record["positions_after_setup"] = _positions(session)
        if navigate_parent:
            calls.append(_execute(adapter, adapted, session, scenario["actor"], "navigate_to_fixture", {"fixture_id": scenario["parent"]}))
            record["positions_after_parent_navigation"] = _positions(session)
            _render(
                session,
                ASSETS / f'{scenario["name"]}_{branch}_after_navigation.jpg',
                f'{scenario["task"]}: after navigate({scenario["parent"]})',
                "Active agent explicitly navigated to the parent; partner began there.",
            )
            record["active_agent_views_after_parent_navigation"] = _render_agent_views(
                session,
                agent=scenario["actor"],
                tag=f'{scenario["name"]}_{branch}_after_parent_navigation',
            )
        tool, args = scenario["action"]
        calls.append(_execute(adapter, adapted, session, scenario["actor"], tool, args))
        record["final_positions"] = _positions(session)
        record["calls"] = calls
        record["success"] = True
        _render(session, ASSETS / f'{scenario["name"]}_{branch}_final.jpg', f'{scenario["task"]}: {branch} final', "Inspect whether both robots remain physically valid after the parent-side action.")
        record["active_agent_views_final"] = _render_agent_views(
            session,
            agent=scenario["actor"],
            tag=f'{scenario["name"]}_{branch}_final',
        )
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [_run(scenario, nav) for scenario in SCENARIOS for nav in (False, True)]
    payload = {
        "scene": SCENE,
        "silent_blocker_clearing": not DISABLE_SILENT_BLOCKER_CLEARING,
        "retain_existing_exclusive_pose": RETAIN_EXISTING_EXCLUSIVE_POSE,
        "records": records,
    }
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if all(record["success"] for record in records) else 1)


if __name__ == "__main__":
    main()
