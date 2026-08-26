"""Live-simulator probe for place_next_to on nested movable supports."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import numpy as np
import robocasa.utils.object_utils as OU

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/nested_place_next_to_probe"
ASSETS = OUT / "assets"
SCENE = {"layout": 11, "style": 14, "seed": 42}
CASES = (
    {
        "task": "GarnishCake",
        "run_index": 0,
        "actor": "agent_1",
        "object": "strawberry1",
        "source": "fruit_plate",
        "reference": "cake",
        "expected_support": "cake_plate",
    },
    {
        "task": "SeasoningSteak",
        "run_index": 0,
        "actor": "agent_0",
        "object": "shaker",
        "source": "cabinet",
        "reference": "steak",
        "expected_support": "steak_plate",
    },
    {
        "task": "TongBuffetSetup",
        "run_index": 0,
        "actor": "agent_0",
        "object": "tongs",
        "source": "drawer",
        "reference": "food",
        "expected_support": "food_tray",
    },
    {
        "task": "PrepareCheeseStation",
        "run_index": 0,
        "actor": "agent_0",
        "object": "cheese",
        "source": "fridge",
        "reference": "lettuce",
        "expected_support": "salad_bowl",
    },
)


def _trajectory(case: dict[str, Any]) -> dict[str, Any]:
    config = _config(
        Namespace(
            task=case["task"], num_runs=32, run_index=case["run_index"],
            location="global", temperature=0.6,
        ),
        model=FLASH_MODEL,
        thinking="low",
    )
    instance = get_task_definition(case["task"]).build_task_instance(
        case["run_index"], config
    )
    state = instance.initial_state
    return {
        "trajectory_id": f'nested-place-next-to:{case["task"]}',
        "task": instance.task_goal,
        "composite_task": case["task"],
        "initial_state": state,
        "grounding_map": build_grounding_map_for_task(case["task"], state),
        "steps": [],
    }


def _adapt_execute(
    session: SimSession,
    adapter: Any,
    adapted: dict[str, Any],
    *,
    agent: str,
    tool: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    call = adapter._adapt_step(
        {"agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )
    result = session.executor.execute(
        call["tool"], robot_idx=call["robot_idx"], **call["args"]
    )
    if not result.success:
        raise RuntimeError(f"{tool} failed: {result.details}")
    return {"symbolic": {"tool": tool, "args": args}, "adapted": call, "details": result.details}


def _render_views(session: SimSession, case_slug: str, phase: str) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for agent in ("agent_0", "agent_1"):
        paths, names = session.render_views(
            ("agentview_center",),
            agent_id=agent,
            out_dir=ASSETS,
            tag=f"{case_slug}_{phase}_{agent}",
        )
        rendered[agent] = str(paths[names.index("agentview_center")])
    session.executor._invalidate_visual_cache()
    top_path = ASSETS / f"{case_slug}_{phase}_top.jpg"
    from PIL import Image
    Image.fromarray(session.executor.runner._render_top_view()).convert("RGB").save(
        top_path, quality=94
    )
    rendered["top"] = str(top_path)
    return rendered


def _open_source_if_needed(
    case: dict[str, Any], session: SimSession, adapter: Any, adapted: dict[str, Any]
) -> list[dict[str, Any]]:
    source = case["source"]
    fixture = adapted["initial_state"].get("fixtures", {}).get(source, {})
    calls: list[dict[str, Any]] = []
    for part_id, part in (fixture.get("parts") or {}).items():
        if part.get("state") == "open":
            continue
        part_type = part.get("part_type")
        tool = "open_sliding_part" if part_type == "sliding_part" else "open_hinged_part"
        calls.append(
            _adapt_execute(
                session, adapter, adapted, agent=case["actor"], tool=tool,
                args={"target_id": source, "part_id": part_id},
            )
        )
    return calls


def _run(case: dict[str, Any]) -> dict[str, Any]:
    trajectory = _trajectory(case)
    slug = case["task"].lower()
    session = SimSession(
        composite_task=case["task"], sample_trajectory=trajectory,
        **SCENE, gl_backend="egl", render_size=640,
        map_dpi=120, map_renderer="raster",
    )
    record: dict[str, Any] = {**case, "scene": SCENE}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        record["initial_views"] = _render_views(session, slug, "initial")
        calls = _open_source_if_needed(case, session, adapter, adapted)
        calls.append(
            _adapt_execute(
                session, adapter, adapted, agent=case["actor"],
                tool="pick_up_object",
                args={"object_id": case["object"], "source_id": case["source"]},
            )
        )
        calls.append(
            _adapt_execute(
                session, adapter, adapted, agent=case["actor"],
                tool="place_next_to",
                args={
                    "object_id": case["object"],
                    "reference_object_id": case["reference"],
                },
            )
        )
        record["calls"] = calls
        resolved_support_object_id = calls[-1]["details"].get(
            "resolved_support_object_id"
        )
        record["resolved_support_object_id"] = resolved_support_object_id
        record["final_views"] = _render_views(session, slug, "final")
        object_pos, _ = session.executor._get_object_pose(case["object"])
        reference_pos, _ = session.executor._get_object_pose(case["reference"])
        record["xy_distance_to_reference_m"] = round(
            float(np.linalg.norm(object_pos[:2] - reference_pos[:2])), 4
        )
        record["executor_location"] = session.executor.runner._object_locations.get(
            case["object"]
        )
        record["support_parent"] = session.executor._support_parents.get(case["object"])
        record["physical_on_expected_support"] = bool(
            OU.check_obj_in_receptacle(
                session.executor.env, case["object"], resolved_support_object_id
            )
        )
        record["success"] = (
            isinstance(resolved_support_object_id, str)
            and record["executor_location"] == resolved_support_object_id
            and record["support_parent"] == resolved_support_object_id
            and record["physical_on_expected_support"]
        )
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [_run(case) for case in CASES]
    payload = {"semantics": "adjacent_on_immediate_movable_support", "records": records}
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if all(record["success"] for record in records) else 1)


if __name__ == "__main__":
    main()
