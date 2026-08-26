"""Append shared reservation/navigation-corridor views to existing artifacts."""

from __future__ import annotations

import json
import os

from training.bc_task_vlm.render_exclusive_workspace_top_views import (
    OUT as TOASTER_OUT,
    _annotate as annotate_toaster,
    _record as toaster_record,
    _session as toaster_session,
)
from training.bc_task_vlm.render_sweeten_coffee_workspace_example import (
    OUT as COFFEE_OUT,
    _annotate as annotate_coffee,
    _positions,
    _record as coffee_record,
    _session as coffee_session,
)


def _navigate(session, adapter, adapted, agent: str, fixture_id: str):
    call = adapter._adapt_step(
        {"agent": agent, "tool": "navigate_to_fixture", "args": {"fixture_id": fixture_id}},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )
    return session.executor.execute(
        call["tool"], robot_idx=call.get("robot_idx", 0), **call.get("args", {})
    )


def render_toaster() -> dict:
    record = toaster_record("baseline")
    trajectory = {
        "trajectory_id": record["carrier_trajectory_id"],
        "task": "arrange_bread_bowl",
        "initial_state": record["initial_state"],
        "steps": [],
    }
    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    session = toaster_session(trajectory)
    adapter, adapted = session.start_trajectory(trajectory)
    result = _navigate(session, adapter, adapted, "agent_1", "toaster_oven")
    session.executor._invalidate_visual_cache()
    annotate_toaster(
        session,
        session.executor.runner._render_top_view(),
        label="NEW — shared 46.5 cm navigation span",
        note="agent_1 samples the same lateral corridor that placement keeps clear.",
    ).save(TOASTER_OUT / "shared_corridor_agent_1_navigates_top_view.jpg", quality=94)
    return {"navigation_success": bool(result.success), "positions": _positions(session)}


def render_coffee() -> dict:
    record = coffee_record("reserved")
    trajectory = {
        "trajectory_id": record["carrier_trajectory_id"],
        "task": "sweeten_coffee",
        "initial_state": record["initial_state"],
        "steps": [],
    }
    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    session = coffee_session(trajectory)
    adapter, adapted = session.start_trajectory(trajectory)
    navigator = record["evidence"]["navigator_agent"]
    result = _navigate(session, adapter, adapted, navigator, "coffee_machine")
    session.executor._invalidate_visual_cache()
    annotate_coffee(
        session,
        session.executor.runner._render_top_view(),
        heading="NEW — shared 46.5 cm navigation span",
        note=f"{navigator} reaches a safe offset within the full counter corridor.",
    ).save(COFFEE_OUT / "shared_corridor_navigator_navigates_top_view.jpg", quality=94)
    return {
        "navigator": navigator,
        "navigation_success": bool(result.success),
        "positions": _positions(session),
    }


def main() -> None:
    results = {"arrange_bread_bowl": render_toaster(), "sweeten_coffee": render_coffee()}
    for out in (TOASTER_OUT, COFFEE_OUT):
        (out / "shared_corridor_metadata.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
