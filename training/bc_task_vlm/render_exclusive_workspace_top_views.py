"""Render baseline and partial reserved-policy top views for one audited case."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_focused_ab_v1"
OUT = ROOT / "training/bc_task_vlm/eval_runs/exclusive_workspace_map_artifact/assets"
SCENE = "layout_11_style_14_seed_42"
CASE_ID = "3cfe34b8dc24283c"


def _record(arm: str) -> dict:
    payload = json.loads((AUDIT / f"{arm}_{SCENE}" / "audit.json").read_text())
    return next(row for row in payload["records"] if row["case_id"] == CASE_ID)


def _front_corridor_points(fixture, pmin: np.ndarray, pmax: np.ndarray, depth: float, *, front_face: str | None = None, lateral_margin: float = 0.0, corridor_width: float | None = None) -> np.ndarray:
    from robocasa.utils.placement import get_face_order

    face = front_face or get_face_order(fixture)[0]
    pmin = np.asarray(pmin, dtype=float).copy()
    pmax = np.asarray(pmax, dtype=float).copy()
    lateral_axis = 0 if face in {"neg_y", "pos_y"} else 1
    if corridor_width is not None:
        center = (pmin[lateral_axis] + pmax[lateral_axis]) / 2.0
        pmin[lateral_axis] = center - corridor_width / 2.0
        pmax[lateral_axis] = center + corridor_width / 2.0
    pmin[lateral_axis] -= lateral_margin
    pmax[lateral_axis] += lateral_margin
    if face == "neg_y":
        return np.asarray([[pmin[0], pmin[1]], [pmax[0], pmin[1]], [pmax[0], pmin[1] - depth], [pmin[0], pmin[1] - depth]])
    if face == "pos_y":
        return np.asarray([[pmin[0], pmax[1]], [pmax[0], pmax[1]], [pmax[0], pmax[1] + depth], [pmin[0], pmax[1] + depth]])
    if face == "neg_x":
        return np.asarray([[pmin[0], pmin[1]], [pmin[0], pmax[1]], [pmin[0] - depth, pmax[1]], [pmin[0] - depth, pmin[1]]])
    return np.asarray([[pmax[0], pmin[1]], [pmax[0], pmax[1]], [pmax[0] + depth, pmax[1]], [pmax[0] + depth, pmin[1]]])


def _world_to_pixels(points: np.ndarray, config: dict, width: int, height: int, fovy: float) -> np.ndarray:
    """Project XY points for the near-vertical free camera used by top_view."""
    look = np.asarray(config["lookat"], dtype=float)[:2]
    centered = np.asarray(points, dtype=float)[:, :2] - look
    azimuth = math.radians(float(config["azimuth"]))
    # MuJoCo free-camera screen right and screen up axes for an overhead view.
    right = np.asarray([math.sin(azimuth), -math.cos(azimuth)])
    up = np.asarray([math.cos(azimuth), math.sin(azimuth)])
    pixels_per_meter = height / (
        2.0 * float(config["distance"]) * math.tan(math.radians(fovy) / 2.0)
    )
    x = width / 2.0 + centered @ right * pixels_per_meter
    y = height / 2.0 - centered @ up * pixels_per_meter
    return np.column_stack([x, y])


def _annotate(session: SimSession, image: np.ndarray, *, label: str, note: str) -> Image.Image:
    runner = session.executor.runner
    fixture = runner._fixtures["toaster_oven_main_group"]
    from robocasa.utils.placement import get_fixture_aabb

    pmin, pmax = get_fixture_aabb(fixture)
    zone_world = _front_corridor_points(
        fixture,
        np.asarray(pmin),
        np.asarray(pmax),
        0.75,
        front_face=runner._occupancy_grid.preferred_reachable_approach_face(fixture),
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
    draw.rounded_rectangle((12, 12, 500, 63), radius=7, fill=(8, 15, 30, 220))
    draw.text((23, 21), label, fill="white", font=font)
    draw.text((23, 40), note, fill=(229, 231, 235), font=font)
    draw.rounded_rectangle((12, result.height - 40, 260, result.height - 12), radius=7, fill=(255, 255, 255, 225))
    draw.rectangle((22, result.height - 31, 39, result.height - 18), fill=(239, 68, 68, 90), outline=(220, 38, 38, 255), width=2)
    draw.text((47, result.height - 31), "Current exclusive-fixture zone", fill=(31, 41, 55), font=font)
    return result.convert("RGB")


def _session(trajectory: dict) -> SimSession:
    return SimSession(
        composite_task="ArrangeBreadBowl",
        sample_trajectory=trajectory,
        layout=11,
        style=14,
        seed=42,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = _record("baseline")
    trajectory = {
        "trajectory_id": baseline["carrier_trajectory_id"],
        "task": "arrange_bread_bowl",
        "initial_state": baseline["initial_state"],
        "steps": [],
    }

    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "0"
    before = _session(trajectory)
    before.start_trajectory(trajectory)
    before.executor._invalidate_visual_cache()
    before_image = before.executor.runner._render_top_view()
    _annotate(
        before,
        before_image,
        label="BEFORE — original placement",
        note="Both counter-assigned robots are placed; one occupies the toaster workspace.",
    ).save(OUT / "before_top_view.jpg", quality=94)

    os.environ["ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES"] = "1"
    after = _session(trajectory)
    failure = None
    adapter = adapted = None
    try:
        adapter, adapted = after.start_trajectory(trajectory)
    except RuntimeError as exc:
        failure = str(exc)
    # Hide the second robot only when its placement was rejected. Under a less
    # conservative policy it may have a valid pose and should remain visible.
    if failure is not None:
        after.executor.runner._set_robot_pose(1, np.asarray([100.0, 100.0]), 0.0)
        after.executor.env.sim.forward()
    after.executor._invalidate_visual_cache()
    after_image = after.executor.runner._render_top_view()
    _annotate(
        after,
        after_image,
        label="AFTER — reserved child workspace",
        note=(
            "Both robots have accepted poses outside the directional corridor."
            if failure is None else
            "Only agent_0 has an accepted pose; agent_1 placement is rejected."
        ),
    ).save(OUT / "after_top_view.jpg", quality=94)
    navigation_success = None
    positions_after_agent_1_navigation = None
    if failure is None and adapter is not None and adapted is not None:
        call = adapter._adapt_step(
            {
                "agent": "agent_1",
                "tool": "navigate_to_fixture",
                "args": {"fixture_id": "toaster_oven"},
            },
            resolved_initial_state=adapted["initial_state"],
            output_dir=None,
        )
        result = after.executor.execute(
            call["tool"],
            robot_idx=call.get("robot_idx", 0),
            **call.get("args", {}),
        )
        navigation_success = bool(result.success)
        positions_after_agent_1_navigation = [
            [float(value) for value in after.executor.runner._get_robot_position(i)]
            for i in (0, 1)
        ]
        after.executor._invalidate_visual_cache()
        navigated_image = after.executor.runner._render_top_view()
        _annotate(
            after,
            navigated_image,
            label="THIRD VIEW — agent_1 navigates to toaster oven",
            note=(
                "Navigation succeeds; agent_0 remains at its accepted counter pose."
                if navigation_success else
                "Navigation is rejected in this state."
            ),
        ).save(OUT / "after_agent_1_navigates_top_view.jpg", quality=94)
    metadata = {
        "scene": {"layout": 11, "style": 14, "seed": 42},
        "case_id": CASE_ID,
        "baseline_positions": baseline["evidence"]["initial_positions"],
        "reserved_accepted_agent_0_position": [
            float(value) for value in after.executor.runner._get_robot_position(0)
        ],
        "reserved_agent_1_position": (
            [float(value) for value in after.executor.runner._get_robot_position(1)]
            if failure is None else None
        ),
        "reserved_agent_1_error": failure,
        "agent_1_navigation_to_toaster_success": navigation_success,
        "positions_after_agent_1_navigation": positions_after_agent_1_navigation,
        "zone_definition": "Fixed 0.465 m-wide front-facing corridor extending 0.75 m outward",
        "top_camera": after.executor.runner._top_cam_config,
    }
    (OUT / "placement_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
