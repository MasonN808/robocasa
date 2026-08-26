"""Render layout-18 placement failures and trace SetUpSpiceStation grounding."""

from __future__ import annotations

from argparse import Namespace
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import robocasa.utils.occupancy_grid as occupancy_grid

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from robocasa.utils.placement import get_fixture_aabb
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/layout18_workspace_diagnostic"
ASSETS = OUT / "assets"
STYLES = (14, 28, 34, 46, 58)
_BASE_APPLIANCE_TOKENS = tuple(
    token for token in occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS
    if token not in {"stove", "stovetop", "cooktop"}
)


def _instance(task: str, run_index: int = 0):
    cfg = _config(
        Namespace(task=task, num_runs=32, run_index=run_index, location="global", temperature=0.6),
        model=FLASH_MODEL, thinking="low",
    )
    return get_task_definition(task).build_task_instance(run_index, cfg)


def _trajectory(task: str, run_index: int = 0) -> dict[str, Any]:
    instance = _instance(task, run_index)
    state = instance.initial_state
    return {
        "trajectory_id": f"layout18_{task}_{run_index}", "task": task,
        "composite_task": task, "initial_state": state,
        "grounding_map": build_grounding_map_for_task(task, state), "steps": [],
    }


def _world_to_pixels(points: np.ndarray, session: SimSession, shape) -> np.ndarray:
    config = session.executor.runner._top_cam_config
    height, width = shape[:2]
    look = np.asarray(config["lookat"], dtype=float)[:2]
    centered = np.asarray(points, dtype=float)[:, :2] - look
    azimuth = math.radians(float(config["azimuth"]))
    right = np.asarray([math.sin(azimuth), -math.cos(azimuth)])
    up = np.asarray([math.cos(azimuth), math.sin(azimuth)])
    fovy = float(session.executor.env.sim.model.vis.global_.fovy)
    ppm = height / (2 * float(config["distance"]) * math.tan(math.radians(fovy) / 2))
    return np.column_stack([width / 2 + centered @ right * ppm, height / 2 - centered @ up * ppm])


def _circle(center: np.ndarray, radius: float, samples: int = 80) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, samples)
    return np.column_stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)])


def _annotate(session: SimSession, image: np.ndarray, fixture_id: str, first: np.ndarray, hypothetical: np.ndarray | None, title: str, output_name: str, *, orange_label: str = "agent_1 pose if agent_0 were absent") -> None:
    result = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fixture = session.executor.runner._fixtures[fixture_id]
    fmin, fmax = get_fixture_aabb(fixture)
    rect = np.asarray([[fmin[0], fmin[1]], [fmax[0], fmin[1]], [fmax[0], fmax[1]], [fmin[0], fmax[1]]])
    rect_px = _world_to_pixels(rect, session, image.shape)
    draw.polygon([tuple(x) for x in rect_px], fill=(37, 99, 235, 38), outline=(37, 99, 235, 255), width=5)
    first_px = _world_to_pixels(_circle(first, 0.40), session, image.shape)
    draw.line([tuple(x) for x in first_px], fill=(239, 68, 68, 255), width=4, joint="curve")
    if hypothetical is not None:
        hyp_px = _world_to_pixels(_circle(hypothetical, 0.40), session, image.shape)
        draw.line([tuple(x) for x in hyp_px], fill=(245, 158, 11, 255), width=4, joint="curve")
        point = _world_to_pixels(np.asarray([hypothetical]), session, image.shape)[0]
        draw.ellipse((point[0]-7, point[1]-7, point[0]+7, point[1]+7), fill=(245, 158, 11, 255))
    result = Image.alpha_composite(result, overlay)
    draw = ImageDraw.Draw(result)
    draw.rounded_rectangle((12, 12, 720, 72), radius=8, fill=(8, 15, 30, 220))
    draw.text((24, 22), title, fill="white", font=ImageFont.load_default())
    draw.text((24, 43), f"blue=parent counter · red=0.40 m around agent_0 · orange={orange_label}", fill=(229,231,235), font=ImageFont.load_default())
    result.convert("RGB").save(ASSETS / output_name, quality=94)


def _save_map(session: SimSession, output_name: str) -> str:
    path = ASSETS / output_name
    session.executor.get_image(views=["map"], image_paths=[str(path)], agent_id="agent_0")
    return str(path.resolve())


def _counter_case(style: int) -> dict[str, Any]:
    task = "ArrangeTea"
    trajectory = _trajectory(task)
    session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=18, style=style, seed=42, gl_backend="egl", render_size=900, map_dpi=100, map_renderer="raster")
    adapter = TrajectoryAdapter(executor=session.executor, allow_approximate_ids=True)
    adapted = adapter.adapt(trajectory)
    error = None
    try:
        session.executor.load_initial_state(adapted["initial_state"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    concrete = adapted["initial_state"]["agents"]
    fixture_id = concrete["agent_0"]["location"]
    first = session.executor.runner._get_robot_position(0)[:2].copy()
    runner = session.executor.runner
    original = first.copy()
    runner._set_robot_pose(0, np.asarray([100.0, 100.0]), 0.0)
    fixture = runner._fixtures[fixture_id]
    hypothetical_result = runner._move_robot_grid(1, fixture, None, False)
    runner._set_robot_pose(0, original, 0.0)
    runner.env.sim.forward()
    hypothetical = hypothetical_result[0] if hypothetical_result is not None else None
    runner._set_robot_pose(1, np.asarray([100.0, 100.0]), 0.0)
    runner.env.sim.forward()
    session.executor._invalidate_visual_cache()
    image = runner._render_top_view()
    image_name = f"counter_style_{style}.jpg"
    _annotate(session, image, fixture_id, first, hypothetical, f"layout 18 · style {style} · seed 42", image_name)
    map_name = f"counter_style_{style}_map.png"
    map_path = _save_map(session, map_name)
    record = {
        "task": task, "scene": {"layout": 18, "style": style, "seed": 42},
        "symbolic_agent_locations": trajectory["initial_state"]["agents"],
        "resolved_agent_locations": concrete, "target_fixture": fixture_id,
        "first_robot_position": first.tolist(),
        "hypothetical_second_pose_without_first": None if hypothetical is None else hypothetical.tolist(),
        "hypothetical_distance_to_first": None if hypothetical is None else float(np.linalg.norm(hypothetical-first)),
        "error": error, "image": str((ASSETS / image_name).resolve()), "map": map_path,
    }
    session.close()
    return record


def _spice_case(style: int, seed: int) -> dict[str, Any]:
    task = "SetUpSpiceStation"
    trajectory = _trajectory(task)
    session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=18, style=style, seed=seed, gl_backend="egl", render_size=512, map_dpi=80, map_renderer="raster")
    adapter = TrajectoryAdapter(executor=session.executor, allow_approximate_ids=True)
    adapted = adapter.adapt(trajectory)
    agents = adapted["initial_state"]["agents"]
    fixtures = adapted["initial_state"]["fixtures"]
    result = {
        "scene": {"layout": 18, "style": style, "seed": seed},
        "symbolic_agent_locations": trajectory["initial_state"]["agents"],
        "resolved_agent_locations": agents,
        "resolved_fixture_types": {
            agent: (fixtures.get(state.get("location")) or {}).get("fixture_type")
            for agent, state in agents.items()
        },
        "same_concrete_location": agents["agent_0"].get("location") == agents["agent_1"].get("location"),
    }
    try:
        occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS = _BASE_APPLIANCE_TOKENS
        session.executor.load_initial_state(adapted["initial_state"])
        result["load_success"] = True
    except Exception as exc:
        result["load_success"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    runner = session.executor.runner
    first = runner._get_robot_position(0)[:2].copy()
    second_target = agents["agent_1"].get("location")
    scene_fixture = (session.executor.get_scene_description().get("fixtures") or {}).get(second_target, {})
    result["agent_1_target_parent"] = scene_fixture.get("parent_fixture")
    result["agent_0_position"] = first.tolist()
    if second_target in runner._fixtures:
        runner._set_robot_pose(0, np.asarray([100.0, 100.0]), 0.0)
        hypothetical = runner._move_robot_grid(
            1, runner._fixtures[second_target], None,
            session.executor._fixture_requires_front_approach(second_target),
        )
        runner._set_robot_pose(0, first, 0.0)
        runner.env.sim.forward()
        result["agent_1_pose_without_agent_0"] = (
            None if hypothetical is None else hypothetical[0].tolist()
        )
        result["distance_between_required_poses"] = (
            None if hypothetical is None else float(np.linalg.norm(hypothetical[0] - first))
        )
        runner._set_robot_pose(1, np.asarray([100.0, 100.0]), 0.0)
        runner.env.sim.forward()
        session.executor._invalidate_visual_cache()
        image_name = f"setup_spice_style_{style}_seed_{seed}.jpg"
        image = runner._render_top_view()
        parent_id = scene_fixture.get("parent_fixture")
        if parent_id not in runner._fixtures:
            parent_id = agents["agent_0"].get("location")
        _annotate(
            session, image, parent_id, first,
            None if hypothetical is None else hypothetical[0],
            f"SetUpSpiceStation · layout 18 · style {style} · seed {seed}",
            image_name, orange_label="required stovetop pose",
        )
        map_name = f"setup_spice_style_{style}_seed_{seed}_map.png"
        result["image"] = str((ASSETS / image_name).resolve())
        result["map"] = _save_map(session, map_name)
        # Proposed rule: use the same aisle-facing approach for both the
        # stovetop's exclusive corridor and navigation to the stovetop.
        occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS = (
            _BASE_APPLIANCE_TOKENS + ("stove", "stovetop", "cooktop")
        )
        session.executor.restore_baseline_state()
        try:
            session.executor.load_initial_state(adapted["initial_state"])
            result["fixed_success"] = True
            fixed_first = runner._get_robot_position(0)[:2].copy()
            fixed_second = runner._get_robot_position(1)[:2].copy()
            result["fixed_positions"] = {
                "agent_0": fixed_first.tolist(), "agent_1": fixed_second.tolist()
            }
            session.executor._invalidate_visual_cache()
            fixed_name = f"setup_spice_style_{style}_seed_{seed}_fixed.jpg"
            _annotate(
                session, runner._render_top_view(), parent_id, fixed_first,
                fixed_second,
                f"FIXED · SetUpSpiceStation · layout 18 · style {style} · seed {seed}",
                fixed_name, orange_label="actual agent_1 stovetop pose",
            )
            fixed_map_name = f"setup_spice_style_{style}_seed_{seed}_fixed_map.png"
            result["fixed_image"] = str((ASSETS / fixed_name).resolve())
            result["fixed_map"] = _save_map(session, fixed_map_name)
        except Exception as exc:
            result["fixed_success"] = False
            result["fixed_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS = _BASE_APPLIANCE_TOKENS
    session.close()
    return result


def _build_html(payload: dict[str, Any]) -> None:
    cards = []
    for row in payload["counter_cases"]:
        cards.append(f'''<section><h2>Layout 18 / style {row['scene']['style']} / seed 42</h2><div class="grid">
<figure><img src="assets/counter_style_{row['scene']['style']}.jpg"><figcaption>Top view with pose overlay</figcaption></figure>
<figure><img src="assets/counter_style_{row['scene']['style']}_map.png"><figcaption>Placement map</figcaption></figure></div><pre>{html.escape(json.dumps({k:v for k,v in row.items() if k not in {'image','map'}}, indent=2))}</pre></section>''')
    spice_cards = []
    for row in payload["setup_spice_cases"]:
        style, seed = row["scene"]["style"], row["scene"]["seed"]
        spice_cards.append(f'''<section><h2>SetUpSpiceStation / style {style} / seed {seed}</h2><div class="grid">
<figure><img src="assets/setup_spice_style_{style}_seed_{seed}.jpg"><figcaption>Top view: parent counter and conflicting stovetop pose</figcaption></figure>
<figure><img src="assets/setup_spice_style_{style}_seed_{seed}_map.png"><figcaption>Current placement map</figcaption></figure>
<figure><img src="assets/setup_spice_style_{style}_seed_{seed}_fixed.jpg"><figcaption>Corrected placement: both actual robot poses</figcaption></figure>
<figure><img src="assets/setup_spice_style_{style}_seed_{seed}_fixed_map.png"><figcaption>Corrected placement map</figcaption></figure></div><pre>{html.escape(json.dumps({k:v for k,v in row.items() if k not in {'image','map','fixed_image','fixed_map'}}, indent=2))}</pre></section>''')
    body = "".join(cards)
    (OUT / "report.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>Layout 18 workspace diagnostic</title>
<style>body{{font:16px system-ui;background:#eef2f7;color:#18202b;max-width:1400px;margin:auto;padding:28px}}section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{width:100%;border-radius:8px}}figcaption{{color:#475569;margin-top:6px}}pre{{white-space:pre-wrap;background:#111827;color:#e5e7eb;padding:14px;border-radius:8px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Layout 18 workspace failures</h1><p>Five style variants represent the 25 repeated counter failures. The blue polygon is the target counter. The red circle is the minimum separation around the successfully placed first robot. The orange circle is where the second robot would be placed if the first were absent.</p>{body}
<h1>SetUpSpiceStation stovetop failures</h1><p>Each orange pose is where agent 1 must stand to use the exclusive stovetop. It nearly coincides with agent 0's accepted parent-counter pose.</p>{''.join(spice_cards)}''')


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    payload = {
        "counter_cases": [_counter_case(style) for style in STYLES],
        "setup_spice_cases": [_spice_case(style, seed) for style in (14,34,58) for seed in (7,99)],
    }
    (OUT / "diagnostic.json").write_text(json.dumps(payload, indent=2) + "\n")
    _build_html(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
