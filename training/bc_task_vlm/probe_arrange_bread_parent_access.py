"""Compare ArrangeBreadBowl placement with and without parent navigation."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data_generation.task_level.generation.raw.cascade_canary import _config, FLASH_MODEL
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/arrange_bread_parent_access_probe"
ASSETS = OUT / "assets"
SCENE = {"layout": 11, "style": 14, "seed": 42}
RUN_INDEX = 6
NUM_RUNS = 8


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
    _annotate(
        session.executor.runner._render_top_view(), heading, note
    ).save(path, quality=94)


def _position(session: SimSession) -> list[float]:
    return [
        round(float(value), 6)
        for value in session.executor.runner._get_robot_position(0)
    ]


def _execute(adapter: Any, adapted: dict[str, Any], session: SimSession, tool: str, args: dict[str, Any]):
    call = adapter._adapt_step(
        {"agent": "agent_0", "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )
    result = session.executor.execute(
        call["tool"], robot_idx=call.get("robot_idx", 0), **call.get("args", {})
    )
    if not result.success:
        raise RuntimeError(f"{tool} failed: {result.details}")
    return {"call": call, "details": result.details}


def _trajectory() -> dict[str, Any]:
    config = _config(
        Namespace(
            task="ArrangeBreadBowl",
            num_runs=NUM_RUNS,
            run_index=RUN_INDEX,
            location="global",
            temperature=0.6,
        ),
        model=FLASH_MODEL,
        thinking="low",
    )
    definition = get_task_definition("ArrangeBreadBowl")
    instance = definition.build_task_instance(RUN_INDEX, config)
    state = deepcopy(instance.initial_state)
    return {
        "trajectory_id": "arrange_bread_parent_access_probe",
        "task": "ArrangeBreadBowl",
        "composite_task": "ArrangeBreadBowl",
        "initial_state": state,
        "grounding_map": build_grounding_map_for_task("ArrangeBreadBowl", state),
        "steps": [],
    }


def _run_branch(*, navigate_parent: bool) -> dict[str, Any]:
    trajectory = _trajectory()
    branch = "explicit_navigation" if navigate_parent else "implicit_parent_access"
    session = SimSession(
        composite_task="ArrangeBreadBowl",
        sample_trajectory=trajectory,
        **SCENE,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )
    record: dict[str, Any] = {"branch": branch, "scene": SCENE}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        calls = []
        calls.append(_execute(adapter, adapted, session, "open_hinged_part", {"target_id": "toaster_oven", "part_id": "door"}))
        calls.append(_execute(adapter, adapted, session, "pick_up_object", {"object_id": "toaster_oven_bread", "source_id": "toaster_oven"}))
        record["position_after_pickup"] = _position(session)
        _render(
            session,
            ASSETS / f"{branch}_after_pickup.jpg",
            f"{branch}: after toaster pickup",
            "Agent_0 holds toaster_oven_bread at the toaster working pose.",
        )
        if navigate_parent:
            calls.append(_execute(adapter, adapted, session, "navigate_to_fixture", {"fixture_id": "counter"}))
            record["position_after_navigation"] = _position(session)
            _render(
                session,
                ASSETS / f"{branch}_after_navigation.jpg",
                f"{branch}: after navigate(counter)",
                "Explicit navigation moves agent_0 to the parent-counter pose.",
            )
        calls.append(_execute(adapter, adapted, session, "place_in_receptacle", {"object_id": "toaster_oven_bread", "receptacle_id": "bowl"}))
        record["position_after_placement"] = _position(session)
        record["calls"] = calls
        record["success"] = True
        _render(
            session,
            ASSETS / f"{branch}_after_placement.jpg",
            f"{branch}: after placement",
            "Bread is placed in the bowl; compare the final robot pose across branches.",
        )
    except Exception as exc:
        record["success"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "ArrangeBreadBowl",
        "run_index": RUN_INDEX,
        "scene": SCENE,
        "branches": [
            _run_branch(navigate_parent=False),
            _run_branch(navigate_parent=True),
        ],
    }
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
