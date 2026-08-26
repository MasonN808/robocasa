"""Render the layout-50 SweetenCoffee residual corridor conflict."""

from __future__ import annotations

import json
import os
from pathlib import Path

from training.bc_task_vlm.audit_access_blocking import _adapt, _execute, _positions
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.render_sweeten_coffee_workspace_example import _annotate
from robocasa.utils.placement import is_in_front_workspace_corridor


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_shared_corridor_ab_v1"
OUT = ROOT / "training/bc_task_vlm/eval_runs/sweeten_coffee_workspace_artifact/assets"
SCENE = "reserved_layout_50_style_34_seed_7"
CASE_ID = "80aae4ea7d4246bf"


def _record() -> dict:
    payload = json.loads((AUDIT / SCENE / "audit.json").read_text())
    return next(row for row in payload["records"] if row["case_id"] == CASE_ID)


def _session(trajectory: dict) -> SimSession:
    return SimSession(
        composite_task="SweetenCoffee",
        sample_trajectory=trajectory,
        layout=50,
        style=34,
        seed=7,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = _record()
    evidence = record["evidence"]
    trajectory = {
        "trajectory_id": record["carrier_trajectory_id"],
        "task": "sweeten_coffee",
        "initial_state": record["initial_state"],
        "steps": [],
    }
    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    session = _session(trajectory)

    adapter, adapted = session.start_trajectory(trajectory)
    initial_positions = _positions(session)
    runner = session.executor.runner
    counter = runner._fixtures.get("counter")
    coffee = runner._fixtures.get(evidence["resolved_fixture_id"])
    attached_children = runner._exclusive_children_for_parent(counter) if counter is not None else []
    blocker_idx = int(evidence["blocker_agent"][-1])
    blocker_pos = runner._get_robot_position(blocker_idx)[:2]
    corridor_face = runner._occupancy_grid.preferred_reachable_approach_face(coffee)
    blocker_inside_corridor = is_in_front_workspace_corridor(
        coffee,
        blocker_pos,
        max_depth=0.75,
        margin=1e-6,
        front_face=corridor_face,
        corridor_width=0.465,
    )
    nav = _adapt(
        adapter, adapted, agent=evidence["navigator_agent"],
        tool="navigate_to_fixture", args={"fixture_id": "coffee_machine"},
    )
    direct = _execute(session, nav)
    session.executor._invalidate_visual_cache()
    _annotate(
        session,
        session.executor.runner._render_top_view(),
        heading="LAYOUT 50 — accepted blocker inside corridor",
        note=f"{evidence['blocker_agent']} is accepted at counter; {evidence['navigator_agent']} navigation fails.",
    ).save(OUT / "layout50_accepted_blocker_top_view.jpg", quality=94)

    adapter, adapted = session.start_trajectory(trajectory)
    blocker_yield = _adapt(
        adapter, adapted, agent=evidence["blocker_agent"],
        tool="give_space", args={"fixture_id": "coffee_machine"},
    )
    nav = _adapt(
        adapter, adapted, agent=evidence["navigator_agent"],
        tool="navigate_to_fixture", args={"fixture_id": "coffee_machine"},
    )
    yielded = _execute(session, blocker_yield)
    after_space_positions = _positions(session)
    after_space_nav = _execute(session, nav)
    final_positions = _positions(session)
    session.executor._invalidate_visual_cache()
    _annotate(
        session,
        session.executor.runner._render_top_view(),
        heading="LAYOUT 50 — after blocker gives space",
        note=f"{evidence['navigator_agent']} reaches the coffee machine using the same corridor.",
    ).save(OUT / "layout50_after_give_space_navigation_top_view.jpg", quality=94)

    metadata = {
        "scene": {"layout": 50, "style": 34, "seed": 7},
        "case_id": CASE_ID,
        "blocker_agent": evidence["blocker_agent"],
        "navigator_agent": evidence["navigator_agent"],
        "initial_positions": initial_positions,
        "direct_navigation_success": bool(direct.success),
        "blocker_give_space_success": bool(yielded.success),
        "positions_after_give_space": after_space_positions,
        "navigation_after_give_space_success": bool(after_space_nav.success),
        "final_positions": final_positions,
        "fixture_geometry": evidence["geometry"],
        "resolved_counter_present": counter is not None,
        "resolved_counter_name": str(getattr(counter, "name", None)),
        "attached_exclusive_children": [str(getattr(child, "name", None)) for child in attached_children],
        "coffee_machine_name": str(getattr(coffee, "name", None)),
        "blocker_inside_corridor_predicate": bool(blocker_inside_corridor),
        "scene_fixture_states": runner.get_scene_description().get("fixtures", {}),
        "zone_definition": "Fixed 0.465 m-wide corridor extending 0.75 m outward",
    }
    (OUT / "layout50_issue_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
