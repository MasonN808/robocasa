"""Visual probe for place_next_to from anchor and support navigation poses."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/reference_fixture_navigation_probe"
ASSETS = OUT / "assets"
SPECS = ROOT / "data_generation/task_level/tasks/specs/verified"
SCENE = {"layout": 11, "style": 34, "seed": 42}

CASES = {
    "SetUpSpiceStation": {
        "spec": "setupspicestation.json",
        "object": "spice",
        "source": "cabinet",
        "source_part": ("cabinet", "hinged"),
        "reference": "stove",
        "support": "counter_stove",
    },
    "PrepareSandwichStation": {
        "spec": "preparesandwichstation.json",
        "object": "baguette",
        "source": "ingredient_source_fixture",
        "source_part": ("ingredient_source_fixture", "door"),
        "reference": "toaster_oven",
        "support": "staging_surface",
    },
    "SetupBowls": {
        "spec": "setupbowls.json",
        "object": "bowl1",
        "source": "cabinet",
        "source_part": ("cabinet", "hinged"),
        "reference": "stool1",
        "support": "dining_counter",
    },
}


def _annotate(image: Any, heading: str, note: str) -> Image.Image:
    result = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.rounded_rectangle((10, 10, result.width - 10, 68), 7, fill=(8, 15, 30))
    draw.text((22, 21), heading, fill="white", font=font)
    draw.text((22, 44), note, fill=(226, 232, 240), font=font)
    return result


def _render(session: SimSession, path: Path, heading: str, note: str) -> None:
    session.executor._invalidate_visual_cache()
    image = session.executor.runner._render_top_view()
    _annotate(image, heading, note).save(path, quality=94)


def _execute(adapter: Any, adapted: dict[str, Any], session: SimSession, step: dict[str, Any]):
    call = adapter._adapt_step(
        step, resolved_initial_state=adapted["initial_state"], output_dir=None
    )
    result = session.executor.execute(
        call["tool"], robot_idx=call.get("robot_idx", 0), **call.get("args", {})
    )
    return call, result


def _geometry(session: SimSession, concrete_id: str) -> dict[str, Any]:
    """Collect enough geometry to distinguish bad grounding from bad posing."""
    fixture = session.executor.runner._fixtures.get(concrete_id)
    if fixture is None:
        return {"concrete_id": concrete_id, "missing": True}
    from robocasa.utils.placement import get_fixture_aabb

    result: dict[str, Any] = {"concrete_id": concrete_id}
    aabb = get_fixture_aabb(fixture)
    if aabb is not None:
        result["aabb"] = [[float(value) for value in corner] for corner in aabb]
    pos = getattr(fixture, "pos", None)
    if pos is not None:
        result["position"] = [float(value) for value in pos]
    return result


def _run_case(task: str, case: dict[str, Any], start: str) -> dict[str, Any]:
    spec = json.loads((SPECS / case["spec"]).read_text())
    trajectory = {
        "trajectory_id": f"reference_fixture_{task}_{start}",
        "task": task,
        "composite_task": task,
        "initial_state": deepcopy(spec["initial_state"]),
        "steps": [],
    }
    # Match generated trajectory records exactly. Anchor-dependent fixture
    # resolution (for example staging_surface relative to toaster_oven) lives
    # in the persisted grounding map rather than the initial-state dictionary.
    trajectory["grounding_map"] = build_grounding_map_for_task(
        task,
        trajectory["initial_state"],
    )
    session = SimSession(
        composite_task=task,
        sample_trajectory=trajectory,
        **SCENE,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )
    slug = task.lower()
    record: dict[str, Any] = {"task": task, "start": start, **case, **SCENE}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        source_fixture, part = case["source_part"]
        # The adapter replaces symbolic fixture keys with simulator ids. Read
        # the symbolic access state from the trajectory, then let _adapt_step
        # resolve the actual fixture for execution.
        fixture_state = trajectory["initial_state"]["fixtures"][source_fixture]
        part_state = (fixture_state.get("parts") or {}).get(part, {}).get("state")
        steps = [
            {"step": 0, "agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": case["source"]}},
        ]
        if part_state != "open":
            steps.append({"step": 1, "agent": "agent_0", "tool": "open_hinged_part", "args": {"target_id": source_fixture, "part_id": part}})
        steps.append({"step": 2, "agent": "agent_0", "tool": "pick_up_object", "args": {"object_id": case["object"], "source_id": case["source"]}})
        for step in steps:
            _, result = _execute(adapter, adapted, session, step)
            if not result.success:
                raise RuntimeError(f"setup call failed: {step}: {result.details}")

        nav_target = case["reference"] if start == "reference" else case["support"]
        nav_call, nav_result = _execute(
            adapter,
            adapted,
            session,
            {"step": 3, "agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": nav_target}},
        )
        record["navigation"] = {"call": nav_call, "success": bool(nav_result.success), "details": nav_result.details}
        if not nav_result.success:
            raise RuntimeError(f"navigation failed: {nav_result.details}")
        record["robot_position_after_navigation"] = [
            float(value) for value in session.executor.runner._get_robot_position(0)
        ]
        record["resolved_geometry"] = {
            "navigation_target": _geometry(session, nav_call["args"]["fixture_id"]),
            "reference": _geometry(
                session,
                adapter._fixture_aliases[case["reference"]],
            ),
            "support": _geometry(
                session,
                adapter._fixture_aliases[case["support"]],
            ),
        }
        _render(
            session,
            ASSETS / f"{slug}_{start}_before.jpg",
            f"{task}: at {nav_target}, before placement",
            f"Holding {case['object']}; reference={case['reference']}, support={case['support']}",
        )

        place_call, place_result = _execute(
            adapter,
            adapted,
            session,
            {"step": 4, "agent": "agent_0", "tool": "place_next_to", "args": {"object_id": case["object"], "reference_fixture_id": case["reference"]}},
        )
        record["placement"] = {"call": place_call, "success": bool(place_result.success), "details": place_result.details}
        record["robot_position_after_placement"] = [
            float(value) for value in session.executor.runner._get_robot_position(0)
        ]
        _render(
            session,
            ASSETS / f"{slug}_{start}_after.jpg",
            f"{task}: after place_next_to",
            f"Started at {nav_target}; placement success={bool(place_result.success)}",
        )
        record["passed"] = bool(nav_result.success and place_result.success)
    except Exception as exc:
        record["passed"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [
        _run_case(task, case, start)
        for task, case in CASES.items()
        for start in ("reference", "support")
    ]
    payload = {
        "scene": SCENE,
        "records": records,
        "summary": {
            "runs": len(records),
            "passed": sum(bool(record.get("passed")) for record in records),
        },
    }
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
