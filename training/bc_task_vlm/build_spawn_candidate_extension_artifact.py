"""Visualize current and proposed initial-placement candidate regions.

This is a diagnostic only. It does not change the production sampler.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robocasa.utils.placement import get_fixture_aabb
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.diagnose_layout18_workspace_failures import _world_to_pixels
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/spawn_candidate_extension_artifact"
ASSETS = OUT / "assets"
CASES = (
    {
        "label": "SetupBowls — stool1",
        "task": "SetupBowls", "run_index": 0, "layout": 15, "style": 14,
        "seed": 42, "target_symbol": "stool1", "first_symbol": "stool1",
    },
    {
        "label": "SetupBowls — stool2",
        "task": "SetupBowls", "run_index": 0, "layout": 15, "style": 14,
        "seed": 42, "target_symbol": "stool2", "first_symbol": "stool2",
    },
    {
        "label": "AlcoholServingPrep — sink",
        "task": "AlcoholServingPrep", "run_index": 12, "layout": 15,
        "style": 14, "seed": 42, "target_symbol": "sink",
        "first_symbol": "dining_table",
    },
)


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def _circle(center: np.ndarray, radius: float, count: int = 100) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, count)
    return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))


def _sample_extended_face(grid, face: str, fmin: np.ndarray, fmax: np.ndarray, standoff: float, extension: float = 0.60):
    expanded_min = np.asarray(fmin, dtype=float).copy()
    expanded_max = np.asarray(fmax, dtype=float).copy()
    parallel_axis = 0 if face in {"neg_y", "pos_y"} else 1
    expanded_min[parallel_axis] -= extension
    expanded_max[parallel_axis] += extension
    expanded = grid._sample_face(face, expanded_min, expanded_max, standoff=standoff)
    current = grid._sample_face(face, fmin, fmax, standoff=standoff)
    current_positions = [np.asarray(pos) for pos, _yaw in current]
    return [
        (pos, yaw) for pos, yaw in expanded
        if not any(float(np.linalg.norm(pos - old)) < grid._sample_spacing * 0.55 for old in current_positions)
    ]


def _draw_dot(draw: ImageDraw.ImageDraw, xy: np.ndarray, color: tuple[int, int, int, int], radius: int = 6):
    x, y = float(xy[0]), float(xy[1])
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=(15, 23, 42, 230), width=1)


def _render_case(case: dict) -> dict:
    trajectory = _trajectory(case["task"], case["run_index"])
    session = SimSession(
        composite_task=case["task"], sample_trajectory=trajectory,
        layout=case["layout"], style=case["style"], seed=case["seed"],
        gl_backend="egl", render_size=1000, map_dpi=130, map_renderer="raster",
    )
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        aliases = adapter._fixture_aliases
        target_id = aliases[case["target_symbol"]]
        first_id = aliases[case["first_symbol"]]
        runner = session.executor.runner
        grid = runner._occupancy_grid

        for index, far in enumerate((np.asarray([100.0, 100.0]), np.asarray([110.0, 110.0]))):
            runner._set_robot_pose(index, far, 0.0)
        runner.env.sim.forward()
        first_ok = bool(runner._move_robot_near_fixture(0, first_id, require_front=False))
        first_pos = runner._get_robot_position(0)[:2].copy()
        runner._set_robot_pose(1, np.asarray([110.0, 110.0]), 0.0)
        runner.env.sim.forward()

        fixture = runner._fixtures[target_id]
        fmin, fmax = get_fixture_aabb(fixture)
        fmin, fmax = np.asarray(fmin), np.asarray(fmax)
        standoffs = (grid._standoff, grid._standoff + grid.cell_size, grid._standoff + 2 * grid.cell_size)
        faces = ["neg_y", "pos_y", "neg_x", "pos_x"]
        current = []
        proposed = []
        for standoff in standoffs:
            for face in faces:
                for pos, _yaw in grid._sample_face(face, fmin, fmax, standoff=standoff):
                    standable = bool(grid.is_standable(pos))
                    separated = float(np.linalg.norm(pos - first_pos)) >= grid._MIN_ROBOT_SEPARATION
                    current.append((np.asarray(pos), standable, separated))
                for pos, _yaw in _sample_extended_face(grid, face, fmin, fmax, standoff):
                    standable = bool(grid.is_standable(pos))
                    separated = float(np.linalg.norm(pos - first_pos)) >= grid._MIN_ROBOT_SEPARATION
                    proposed.append((np.asarray(pos), standable, separated))

        image = Image.fromarray(runner._render_top_view()).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rect = np.asarray(((fmin[0], fmin[1]), (fmax[0], fmin[1]), (fmax[0], fmax[1]), (fmin[0], fmax[1])))
        draw.polygon([tuple(p) for p in _world_to_pixels(rect, session, image.size)], fill=(37, 99, 235, 42), outline=(37, 99, 235, 255), width=5)
        exclusion = _world_to_pixels(_circle(first_pos, grid._MIN_ROBOT_SEPARATION), session, image.size)
        draw.line([tuple(p) for p in exclusion], fill=(239, 68, 68, 255), width=5, joint="curve")
        first_pixel = _world_to_pixels(np.asarray([first_pos]), session, image.size)[0]
        _draw_dot(draw, first_pixel, (239, 68, 68, 255), 8)
        for pos, standable, separated in current:
            color = (22, 163, 74, 220) if standable and separated else ((239, 68, 68, 205) if standable else (100, 116, 139, 130))
            _draw_dot(draw, _world_to_pixels(np.asarray([pos]), session, image.size)[0], color, 4)
        for pos, standable, separated in proposed:
            if not standable:
                continue
            color = (6, 182, 212, 235) if separated else (245, 158, 11, 220)
            _draw_dot(draw, _world_to_pixels(np.asarray([pos]), session, image.size)[0], color, 5)
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((12, 12, 970, 93), radius=8, fill=(8, 15, 30, 225))
        font = ImageFont.load_default()
        draw.text((24, 22), case["label"], fill="white", font=font)
        draw.text((24, 43), "blue=fixture · red circle=0.40 m separation · green=current usable", fill=(226, 232, 240), font=font)
        draw.text((24, 64), "red=current too close · gray=current blocked · cyan=usable only after lateral extension · orange=extended but too close", fill=(226, 232, 240), font=font)
        path = ASSETS / (case["label"].lower().replace(" — ", "_").replace(" ", "_") + ".jpg")
        image.convert("RGB").save(path, quality=95)
        return {
            **case, "image": str(path), "target_fixture": target_id,
            "first_fixture": first_id, "first_placement_succeeded": first_ok,
            "first_robot_xy": first_pos.tolist(),
            "current_candidates": len(current),
            "current_usable": sum(s and d for _p, s, d in current),
            "extended_candidates": len(proposed),
            "extended_usable": sum(s and d for _p, s, d in proposed),
            "minimum_separation_m": float(grid._MIN_ROBOT_SEPARATION),
            "extension_each_end_m": 0.60,
            "adapted_agents": adapted["initial_state"]["agents"],
        }
    finally:
        session.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = [_render_case(case) for case in CASES]
    panels = []
    for row in rows:
        panels.append(
            f'<section><h2>{html.escape(row["label"])}</h2>'
            f'<p>First robot: <code>{html.escape(row["first_fixture"])}</code> at '
            f'<code>{html.escape(str([round(x, 3) for x in row["first_robot_xy"]]))}</code>. '
            f'Current usable candidates: <strong>{row["current_usable"]}/{row["current_candidates"]}</strong>. '
            f'Additional usable diagnostic candidates: <strong>{row["extended_usable"]}/{row["extended_candidates"]}</strong>.</p>'
            f'<figure><img src="{_data_url(Path(row["image"]))}"><figcaption>The proposed cyan points extend the same face-sampling lines by 0.60 m at each end. They do not change readiness or collision thresholds.</figcaption></figure></section>'
        )
    report = f'''<!doctype html><meta charset="utf-8"><title>Spawn candidate restrictions and extensions</title>
    <style>body{{font:16px/1.5 system-ui;background:#eef2f7;color:#172033;margin:0}}main{{max-width:1250px;margin:auto;padding:28px}}header,section{{background:#fff;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin-bottom:18px}}img{{display:block;width:100%;border-radius:9px}}figure{{margin:0}}figcaption{{color:#526079;margin-top:9px}}code{{white-space:pre-wrap}}</style>
    <main><header><h1>Current placement restrictions and proposed lateral extensions</h1><p>This is a diagnostic visualization, not an implemented policy. Gray/red/green points are candidates produced by the current fixture-AABB sampler. Cyan/orange points show what would be added by extending each sampled face laterally into the surrounding workspace. The 0.40 m robot-separation rule is unchanged.</p></header>{''.join(panels)}</main>'''
    (OUT / "report.html").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
