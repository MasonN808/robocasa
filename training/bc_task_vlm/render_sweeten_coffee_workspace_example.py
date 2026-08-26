"""Render one concrete SweetenCoffee failure under the directional corridor."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robocasa.utils.placement import get_fixture_aabb
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.render_exclusive_workspace_top_views import (
    _front_corridor_points,
    _world_to_pixels,
)


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_directional_ab_v4"
OUT = ROOT / "training/bc_task_vlm/eval_runs/sweeten_coffee_workspace_artifact/assets"
SCENE = "layout_40_style_34_seed_99"


def _record(arm: str) -> dict:
    path = AUDIT / f"{arm}_{SCENE}" / "audit.json"
    payload = json.loads(path.read_text())
    candidates = [
        row for row in payload["records"]
        if row["task_name"] == "sweeten_coffee"
        and (row.get("evidence") or {}).get("detected")
    ]
    if not candidates:
        raise RuntimeError(f"No detected SweetenCoffee example in {path}")
    return max(
        candidates,
        key=lambda row: abs(
            (row["evidence"]["initial_positions"][int(row["evidence"]["blocker_agent"][-1])][0])
        ),
    )


def _session(trajectory: dict) -> SimSession:
    return SimSession(
        composite_task="SweetenCoffee",
        sample_trajectory=trajectory,
        layout=40,
        style=34,
        seed=99,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )


def _coffee_fixture(runner):
    matches = []
    for name, fixture in runner._fixtures.items():
        haystack = " ".join(
            [
                str(name).lower(),
                str(getattr(fixture, "name", "") or "").lower(),
                type(fixture).__name__.lower(),
            ]
        )
        if "coffee" in haystack:
            matches.append(fixture)
    if len(matches) != 1:
        available = [
            (str(name), type(fixture).__name__, str(getattr(fixture, "name", "")))
            for name, fixture in runner._fixtures.items()
        ]
        raise RuntimeError(
            f"Expected one coffee machine, found {len(matches)}; fixtures={available}"
        )
    return matches[0]


def _annotate(session: SimSession, image: np.ndarray, *, heading: str, note: str) -> Image.Image:
    runner = session.executor.runner
    fixture = _coffee_fixture(runner)
    pmin, pmax = get_fixture_aabb(fixture)
    face = runner._occupancy_grid.preferred_reachable_approach_face(fixture)
    zone_world = _front_corridor_points(
        fixture,
        np.asarray(pmin),
        np.asarray(pmax),
        0.75,
        front_face=face,
        corridor_width=0.465,
    )
    fovy = float(session.executor.env.sim.model.vis.global_.fovy)
    zone_px = _world_to_pixels(
        zone_world, runner._top_cam_config, image.shape[1], image.shape[0], fovy
    )
    result = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    polygon = [tuple(map(float, row)) for row in zone_px]
    draw.polygon(polygon, fill=(239, 68, 68, 72), outline=(220, 38, 38, 255), width=4)
    result = Image.alpha_composite(result, overlay)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.rounded_rectangle((12, 12, 555, 63), radius=7, fill=(8, 15, 30, 220))
    draw.text((23, 21), heading, fill="white", font=font)
    draw.text((23, 40), note, fill=(229, 231, 235), font=font)
    draw.rounded_rectangle((12, result.height - 40, 290, result.height - 12), radius=7, fill=(255, 255, 255, 225))
    draw.rectangle((22, result.height - 31, 39, result.height - 18), fill=(239, 68, 68, 90), outline=(220, 38, 38, 255), width=2)
    draw.text((47, result.height - 31), "Directional coffee-machine corridor", fill=(31, 41, 55), font=font)
    return result.convert("RGB")


def _positions(session: SimSession) -> list[list[float]]:
    return [
        [round(float(v), 6) for v in session.executor.runner._get_robot_position(i)]
        for i in (0, 1)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reserved_record = _record("reserved")
    trajectory = {
        "trajectory_id": reserved_record["carrier_trajectory_id"],
        "task": "sweeten_coffee",
        "initial_state": reserved_record["initial_state"],
        "steps": [],
    }

    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "0"
    before = _session(trajectory)
    before.start_trajectory(trajectory)
    before_positions = _positions(before)
    before.executor._invalidate_visual_cache()
    _annotate(
        before,
        before.executor.runner._render_top_view(),
        heading="BEFORE — original parent-counter placement",
        note="The blocking robot is outside the appliance width but still prevents access.",
    ).save(OUT / "before_top_view.jpg", quality=94)

    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    after = _session(trajectory)
    after.start_trajectory(trajectory)
    after_positions = _positions(after)
    after.executor._invalidate_visual_cache()
    _annotate(
        after,
        after.executor.runner._render_top_view(),
        heading="AFTER — fixed 46.5 cm directional corridor",
        note="Both placements remain legal, but agent_0 still blocks coffee-machine access.",
    ).save(OUT / "after_top_view.jpg", quality=94)

    evidence = reserved_record["evidence"]
    fixture = _coffee_fixture(after.executor.runner)
    pmin, pmax = get_fixture_aabb(fixture)
    blocker_idx = int(evidence["blocker_agent"][-1])
    blocker = np.asarray(evidence["initial_positions"][blocker_idx], dtype=float)[:2]
    face = after.executor.runner._occupancy_grid.preferred_reachable_approach_face(fixture)
    if face in {"neg_y", "pos_y"}:
        lateral_miss = max(float(pmin[0] - blocker[0]), 0.0, float(blocker[0] - pmax[0]))
    else:
        lateral_miss = max(float(pmin[1] - blocker[1]), 0.0, float(blocker[1] - pmax[1]))
    metadata = {
        "scene": {"layout": 40, "style": 34, "seed": 99},
        "case_id": reserved_record["case_id"],
        "trajectory_id": reserved_record["carrier_trajectory_id"],
        "before_positions": before_positions,
        "after_positions": after_positions,
        "blocker_agent": evidence["blocker_agent"],
        "navigator_agent": evidence["navigator_agent"],
        "preferred_approach_face": face,
        "coffee_machine_aabb": [[float(v) for v in pmin], [float(v) for v in pmax]],
        "blocker_lateral_miss_m": lateral_miss,
        "direct_navigation_success": evidence["direct_navigation_success"],
        "navigation_after_blocker_give_space_success": evidence["navigation_after_blocker_give_space_success"],
        "zone_definition": "Fixed 0.465 m-wide directional rectangle extending 0.75 m outward",
    }
    (OUT / "example_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
