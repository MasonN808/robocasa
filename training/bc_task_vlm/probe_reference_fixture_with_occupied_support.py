"""Probe place_next_to when another robot occupies the adjacent support."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.probe_reference_fixture_navigation import CASES, SCENE, SPECS


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/reference_fixture_occupied_support_probe"
ASSETS = OUT / "assets"


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


def _adapt(adapter, adapted, *, step: int, agent: str, tool: str, args: dict[str, Any]):
    return adapter._adapt_step(
        {"step": step, "agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )


def _execute(session: SimSession, call: dict[str, Any]):
    return session.executor.execute(
        call["tool"], robot_idx=call.get("robot_idx", 0), **call.get("args", {})
    )


def _result(call, result) -> dict[str, Any]:
    return {
        "call": call,
        "success": bool(result.success),
        "details": result.details,
    }


def _positions(session: SimSession) -> list[list[float]]:
    return [
        [round(float(value), 6) for value in session.executor.runner._get_robot_position(i)]
        for i in (0, 1)
    ]


def _run_case(task: str, case: dict[str, Any]) -> dict[str, Any]:
    spec = json.loads((SPECS / case["spec"]).read_text())
    trajectory = {
        "trajectory_id": f"occupied_support_{task}",
        "task": task,
        "composite_task": task,
        "initial_state": deepcopy(spec["initial_state"]),
        "steps": [],
    }
    trajectory["grounding_map"] = build_grounding_map_for_task(
        task, trajectory["initial_state"]
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
    record: dict[str, Any] = {"task": task, **case, **SCENE}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        source_fixture, part = case["source_part"]
        fixture_state = trajectory["initial_state"]["fixtures"][source_fixture]
        part_state = (fixture_state.get("parts") or {}).get(part, {}).get("state")

        setup = [
            _adapt(adapter, adapted, step=0, agent="agent_0", tool="navigate_to_fixture", args={"fixture_id": case["source"]}),
        ]
        if part_state != "open":
            setup.append(_adapt(adapter, adapted, step=1, agent="agent_0", tool="open_hinged_part", args={"target_id": source_fixture, "part_id": part}))
        setup.append(_adapt(adapter, adapted, step=2, agent="agent_0", tool="pick_up_object", args={"object_id": case["object"], "source_id": case["source"]}))
        for call in setup:
            outcome = _execute(session, call)
            if not outcome.success:
                raise RuntimeError(f"setup failed: {call}: {outcome.details}")

        support_call = _adapt(
            adapter, adapted, step=3, agent="agent_1", tool="navigate_to_fixture",
            args={"fixture_id": case["support"]},
        )
        support_result = _execute(session, support_call)
        record["support_navigation"] = _result(support_call, support_result)

        reference_call = _adapt(
            adapter, adapted, step=4, agent="agent_0", tool="navigate_to_fixture",
            args={"fixture_id": case["reference"]},
        )
        reference_result = _execute(session, reference_call)
        record["reference_navigation"] = _result(reference_call, reference_result)
        record["positions_before_placement"] = _positions(session)
        _render(
            session,
            ASSETS / f"{slug}_before.jpg",
            f"{task}: support occupied, before placement",
            f"agent_0 -> {case['reference']}; agent_1 -> {case['support']}",
        )

        if support_result.success and reference_result.success:
            place_call = _adapt(
                adapter, adapted, step=5, agent="agent_0", tool="place_next_to",
                args={"object_id": case["object"], "reference_fixture_id": case["reference"]},
            )
            place_result = _execute(session, place_call)
            record["placement"] = _result(place_call, place_result)
        else:
            record["placement"] = {
                "attempted": False,
                "success": False,
                "details": "Not attempted because one navigation failed.",
            }
        record["positions_after_placement"] = _positions(session)
        _render(
            session,
            ASSETS / f"{slug}_after.jpg",
            f"{task}: after attempted place_next_to",
            f"placement success={record['placement']['success']}",
        )
        record["passed"] = bool(
            support_result.success
            and reference_result.success
            and record["placement"]["success"]
        )
    except Exception as exc:
        record["passed"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [_run_case(task, case) for task, case in CASES.items()]
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
