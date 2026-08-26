"""Probe real placements using lateral candidate extensions and render evidence.

The extension is applied locally in this probe; production sampling is untouched.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robocasa.utils.placement import get_fixture_aabb
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.diagnose_layout18_workspace_failures import _world_to_pixels
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/spawn_candidate_extension_probe"
ASSETS = OUT / "assets"
CASES = (
    ("SetupBowls", 0, 15, 14, 42, "stool1", "stool1"),
    ("SetupBowls", 0, 15, 14, 42, "stool2", "stool2"),
    ("AlcoholServingPrep", 12, 15, 14, 42, "sink", "dining_table"),
)
EXTENSIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _clear(runner) -> None:
    for index in range(runner._num_robots):
        offset = float(index * 10)
        runner._set_robot_pose(index, np.asarray([100.0 + offset, 100.0 + offset]), 0.0)
    runner.env.sim.forward()


def _extended_placement(runner, robot_idx: int, fixture_id: str, extension: float):
    fixture = runner._fixtures[fixture_id]
    grid = runner._occupancy_grid
    fmin, fmax = get_fixture_aabb(fixture)
    fmin, fmax = np.asarray(fmin, dtype=float), np.asarray(fmax, dtype=float)
    others = [
        runner._get_robot_position(index)[:2].copy()
        for index in range(runner._num_robots) if index != robot_idx
    ]
    candidates = []
    for standoff in (grid._standoff, grid._standoff + grid.cell_size, grid._standoff + 2 * grid.cell_size):
        for face in ("neg_y", "pos_y", "neg_x", "pos_x"):
            expanded_min, expanded_max = fmin.copy(), fmax.copy()
            parallel_axis = 0 if face in {"neg_y", "pos_y"} else 1
            expanded_min[parallel_axis] -= extension
            expanded_max[parallel_axis] += extension
            for pos, yaw in grid._sample_face(face, expanded_min, expanded_max, standoff=standoff):
                pos = np.asarray(pos, dtype=float)
                if fmin[0] <= pos[0] <= fmax[0] and fmin[1] <= pos[1] <= fmax[1]:
                    continue
                if not grid._is_in_bounds(pos) or not grid.is_standable(pos):
                    continue
                if any(float(np.linalg.norm(pos - other)) < grid._MIN_ROBOT_SEPARATION for other in others):
                    continue
                candidates.append((float(np.linalg.norm(pos - fixture.pos[:2])), pos, yaw, face))
    if not candidates:
        return None
    _distance, pos, yaw, face = min(candidates, key=lambda row: row[0])
    runner._set_robot_pose(robot_idx, pos, yaw)
    runner.env.sim.forward()
    return {"position": pos.tolist(), "yaw": float(yaw), "face": face, "candidate_count": len(candidates)}


def _save_views(
    session: SimSession,
    tag: str,
    *,
    target_fixtures: tuple[tuple[str, str], ...] = (),
) -> dict:
    runner = session.executor.runner
    runner.env.sim.forward()
    session.executor._invalidate_visual_cache()
    top = ASSETS / f"{tag}_top.jpg"
    image = Image.fromarray(runner._render_top_view()).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = ((37, 99, 235, 255), (147, 51, 234, 255), (5, 150, 105, 255))
    for fixture_index, (fixture_id, fixture_label) in enumerate(target_fixtures):
        fixture = runner._fixtures[fixture_id]
        fmin, fmax = get_fixture_aabb(fixture)
        corners = np.asarray(((fmin[0], fmin[1]), (fmax[0], fmin[1]), (fmax[0], fmax[1]), (fmin[0], fmax[1])))
        pixels = _world_to_pixels(corners, session, image.size)
        color = colors[fixture_index % len(colors)]
        draw.polygon([tuple(point) for point in pixels], fill=(*color[:3], 35), outline=color, width=5)
        anchor = pixels[0]
        draw.text((float(anchor[0]) + 7, float(anchor[1]) + 7), fixture_label, fill=color, font=ImageFont.load_default())
    for robot_idx, color in ((0, (220, 38, 38, 255)), (1, (245, 158, 11, 255))):
        position = runner._get_robot_position(robot_idx)[:2]
        pixel = _world_to_pixels(np.asarray([position]), session, image.size)[0]
        draw.ellipse((pixel[0]-10, pixel[1]-10, pixel[0]+10, pixel[1]+10), fill=color, outline=(15, 23, 42, 255), width=2)
        draw.text((float(pixel[0]) + 13, float(pixel[1]) - 8), f"agent_{robot_idx}", fill=color, font=ImageFont.load_default())
    image = Image.alpha_composite(image, overlay)
    image.convert("RGB").save(top, quality=95)
    result = {"top": str(top)}
    for agent in ("agent_0", "agent_1"):
        # Agent cameras are cached independently of direct robot-pose edits.
        # Invalidate immediately before every capture so all three panels are
        # guaranteed to describe the same simulator state.
        runner.env.sim.forward()
        session.executor._invalidate_visual_cache()
        paths, _ = session.render_views(
            ("agentview_center",), agent_id=agent, out_dir=ASSETS,
            tag=f"{tag}_{agent}",
        )
        result[agent] = paths[0]
    return result


def _positions(runner) -> dict:
    return {
        f"agent_{index}": runner._get_robot_position(index)[:2].tolist()
        for index in range(runner._num_robots)
    }


def _probe_case(case) -> dict:
    task, run_index, layout, style, seed, target_symbol, first_symbol = case
    trajectory = _trajectory(task, run_index)
    session = SimSession(
        composite_task=task, sample_trajectory=trajectory, layout=layout,
        style=style, seed=seed, gl_backend="egl", render_size=900,
        map_dpi=120, map_renderer="raster",
    )
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapter.adapt(trajectory)
        target = adapter._fixture_aliases[target_symbol]
        first = adapter._fixture_aliases[first_symbol]
        runner = session.executor.runner

        # Historical ordering: place the broad/first target, then the disputed target.
        _clear(runner)
        first_ok = bool(runner._move_robot_near_fixture(0, first, require_front=False))
        current_ok = bool(runner._move_robot_near_fixture(1, target, require_front=False))
        current = {
            "first_ok": first_ok, "second_ok": current_ok,
            "second_ready": bool(session.executor._robot_near_fixture(1, target)),
            "positions": _positions(runner),
            "views": _save_views(
                session,
                f"{task.lower()}_{target_symbol}_current",
                target_fixtures=((target, target_symbol),),
            ),
        }

        # Same ordering, but extend only the disputed target's lateral candidate lines.
        extension_result = None
        selected_extension = None
        for extension in EXTENSIONS:
            _clear(runner)
            if not runner._move_robot_near_fixture(0, first, require_front=False):
                continue
            extension_result = _extended_placement(runner, 1, target, extension)
            if extension_result is not None:
                selected_extension = extension
                break
        extended = {
            "extension_m": selected_extension,
            "placement": extension_result,
            "second_ready": bool(extension_result is not None and session.executor._robot_near_fixture(1, target)),
            "second_valid": bool(extension_result is not None and runner._is_valid_robot_position(runner._get_robot_position(1)[:2])),
            "separation_m": float(np.linalg.norm(runner._get_robot_position(0)[:2] - runner._get_robot_position(1)[:2])) if extension_result is not None else None,
            "positions": _positions(runner),
            "views": _save_views(
                session,
                f"{task.lower()}_{target_symbol}_extended",
                target_fixtures=((target, target_symbol),),
            ),
        }

        priority = None
        if task == "AlcoholServingPrep":
            # Proposed production ordering: exclusive sink claims a pose first.
            _clear(runner)
            sink_ok = bool(runner._move_robot_near_fixture(1, target, require_front=False))
            shared_ok = bool(runner._move_robot_near_fixture(0, first, require_front=False))
            priority = {
                "sink_first_ok": sink_ok, "shared_second_ok": shared_ok,
                "sink_ready": bool(session.executor._robot_near_fixture(1, target)),
                "positions": _positions(runner),
                "separation_m": float(np.linalg.norm(runner._get_robot_position(0)[:2] - runner._get_robot_position(1)[:2])),
                "views": _save_views(
                    session,
                    "alcoholservingprep_sink_exclusive_first",
                    target_fixtures=((target, "sink"), (first, "dining workspace")),
                ),
            }
        return {
            "task": task, "target_symbol": target_symbol, "target": target,
            "first": first, "scene": {"layout": layout, "style": style, "seed": seed},
            "current": current, "extended": extended, "exclusive_first": priority,
        }
    finally:
        session.close()


def _probe_coffee() -> dict:
    task, run_index, layout, style, seed = "PrepareCoffee", 8, 50, 14, 42
    trajectory = _trajectory(task, run_index)
    session = SimSession(
        composite_task=task, sample_trajectory=trajectory, layout=layout,
        style=style, seed=seed, gl_backend="egl", render_size=900,
        map_dpi=120, map_renderer="raster",
    )
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        session.executor.restore_baseline_state()
        session.executor.load_initial_state(adapted["initial_state"])
        runner = session.executor.runner
        resolved_agents = adapted["initial_state"]["agents"]
        coffee_agent = next(
            agent_id for agent_id, state in resolved_agents.items()
            if "coffee_machine" in str(state.get("location"))
        )
        coffee_index = int(coffee_agent.rsplit("_", 1)[1])
        coffee_fixture = resolved_agents[coffee_agent]["location"]
        other_agent = "agent_1" if coffee_agent == "agent_0" else "agent_0"
        other_fixture = resolved_agents[other_agent]["location"]
        return {
            "task": task,
            "scene": {"layout": layout, "style": style, "seed": seed},
            "resolved_agents": resolved_agents,
            "positions": _positions(runner),
            "coffee_agent": coffee_agent,
            "coffee_ready": bool(session.executor._robot_near_fixture(coffee_index, coffee_fixture)),
            "other_ready": bool(session.executor._robot_near_fixture(1 - coffee_index, other_fixture)),
            "separation_m": float(np.linalg.norm(runner._get_robot_position(0)[:2] - runner._get_robot_position(1)[:2])),
            "views": _save_views(
                session,
                "preparecoffee_corrected_loader",
                target_fixtures=((coffee_fixture, "coffee machine"), (other_fixture, "cabinet parent counter")),
            ),
        }
    finally:
        session.close()


def _gallery(views: dict, prefix: str) -> str:
    labels = (("top", "Top view"), ("agent_0", "Agent 0 view"), ("agent_1", "Agent 1 view"))
    return '<div class="grid">' + "".join(
        f'<figure><img src="{_data_url(Path(views[key]))}"><figcaption>{html.escape(prefix)} — {label}</figcaption></figure>'
        for key, label in labels
    ) + "</div>"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = [_probe_case(case) for case in CASES]
    coffee = _probe_coffee()
    sections = []
    for row in rows:
        current = row["current"]
        extended = row["extended"]
        sections.append(
            f'<section><h2>{row["task"]} — {row["target_symbol"]}</h2>'
            f'<p><strong>Current sampler:</strong> second placement={current["second_ok"]}, ready={current["second_ready"]}. '
            f'<strong>Smallest tested extension that placed:</strong> {extended["extension_m"]} m per end; '
            f'ready={extended["second_ready"]}, valid={extended["second_valid"]}, separation={extended["separation_m"]:.3f} m.</p>'
            + _gallery(current["views"], "Current sampler")
            + _gallery(extended["views"], f'Extended by {extended["extension_m"]} m per end')
            + ('' if row["exclusive_first"] is None else (
                f'<h3>Exclusive target placed first</h3><p>sink placement={row["exclusive_first"]["sink_first_ok"]}, '
                f'shared placement={row["exclusive_first"]["shared_second_ok"]}, '
                f'sink ready={row["exclusive_first"]["sink_ready"]}, '
                f'separation={row["exclusive_first"]["separation_m"]:.3f} m.</p>'
                + _gallery(row["exclusive_first"]["views"], "Sink first, dining workspace second")
            )) + '</section>'
        )
    coffee_section = (
        f'<section><h2>PrepareCoffee — corrected production loader</h2>'
        f'<p>Resolved agents: <code>{html.escape(str(coffee["resolved_agents"]))}</code>. '
        f'Coffee agent ready={coffee["coffee_ready"]}; other agent ready={coffee["other_ready"]}; '
        f'separation={coffee["separation_m"]:.3f} m. Red marker=agent_0; orange marker=agent_1.</p>'
        + _gallery(coffee["views"], "Corrected loader: one synchronized state")
        + '</section>'
    )
    report = f'''<!doctype html><meta charset="utf-8"><title>Placement extension simulator probes</title>
    <style>body{{font:16px/1.5 system-ui;background:#eef2f7;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:28px}}header,section{{background:#fff;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:14px 0 24px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{color:#526079}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
    <main><header><h1>Actual simulator placements using candidate-line extensions</h1><p>Every top view and pair of agent views below is now captured from one explicitly synchronized simulator state. Top views label both robots and the relevant fixture footprints. The probe tries extensions from 0.10 through 0.60 m and reports the smallest tested value that produces a valid candidate. Production extension behavior remains unchanged.</p></header>{''.join(sections)}{coffee_section}</main>'''
    (OUT / "report.html").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps({"extension_cases": rows, "prepare_coffee": coffee}, indent=2) + "\n")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
