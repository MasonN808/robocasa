"""Render failed and clean-room initial spawn attempts for two audit cases."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/initial_spawn_conflict_artifact"
ASSETS = OUT / "assets"
CASES = (
    {"task": "AlcoholServingPrep", "run_index": 12, "layout": 15, "style": 14, "seed": 42},
    {"task": "PrepareCoffee", "run_index": 8, "layout": 50, "style": 14, "seed": 42},
)


def _save_top(session: SimSession, path: Path, title: str) -> None:
    image = Image.fromarray(session.executor.runner._render_top_view()).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((12, 12, 650, 55), radius=7, fill=(8, 15, 30))
    draw.text((24, 25), title, fill="white")
    image.save(path, quality=95)


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def _positions(session: SimSession) -> dict[str, list[float]]:
    return {
        f"agent_{index}": [float(x) for x in session.executor.runner._get_robot_position(index)]
        for index in (0, 1)
    }


def _render_case(case: dict) -> dict:
    trajectory = _trajectory(case["task"], case["run_index"])
    session = SimSession(
        composite_task=case["task"], sample_trajectory=trajectory,
        layout=case["layout"], style=case["style"], seed=case["seed"],
        gl_backend="egl", render_size=900, map_dpi=130, map_renderer="raster",
    )
    slug = case["task"].lower()
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        failure = None
        try:
            session.executor.restore_baseline_state()
            session.executor.load_initial_state(adapted["initial_state"])
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        failed_top = ASSETS / f"{slug}_current_loader_top.jpg"
        _save_top(session, failed_top, f"{case['task']} — current loader after attempt")
        current_loader_positions = _positions(session)
        failed_views = {}
        for agent in ("agent_0", "agent_1"):
            paths, views = session.render_views(
                ("agentview_center",), agent_id=agent, out_dir=ASSETS,
                tag=f"{slug}_current_loader_{agent}",
            )
            failed_views[agent] = paths[0]

        # Retry from a neutral state: neither robot's stale baseline position
        # is allowed to reserve space while the intended pair is placed.
        runner = session.executor.runner
        for index, position in enumerate((np.asarray([100.0, 100.0]), np.asarray([110.0, 110.0]))):
            runner._set_robot_pose(index, position, 0.0)
        runner.env.sim.forward()
        placements = []
        resolved_locations = {}
        for agent_id, state in adapted["initial_state"]["agents"].items():
            index = int(agent_id.rsplit("_", 1)[1])
            location = state["location"]
            resolved_locations[agent_id] = location
            require_front = session.executor._surface_fixture_prefers_front_approach(location)
            placements.append(bool(runner._move_robot_near_fixture(index, location, require_front=require_front)))
        clean_top = ASSETS / f"{slug}_clean_room_top.jpg"
        _save_top(session, clean_top, f"{case['task']} — both robots cleared, then intended placements")
        clean_views = {}
        for agent in ("agent_0", "agent_1"):
            paths, views = session.render_views(
                ("agentview_center",), agent_id=agent, out_dir=ASSETS,
                tag=f"{slug}_clean_room_{agent}",
            )
            clean_views[agent] = paths[0]
        return {
            **case, "initial_state_agents": trajectory["initial_state"]["agents"],
            "resolved_locations": resolved_locations,
            "current_loader_error": failure,
            "current_loader_positions": current_loader_positions,
            "clean_room_placements": placements,
            "clean_room_positions": _positions(session),
            "current_top": str(failed_top), "clean_top": str(clean_top),
            "current_views": failed_views, "clean_views": clean_views,
        }
    finally:
        session.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = [_render_case(case) for case in CASES]
    panels = []
    for row in rows:
        images = [
            (row["current_top"], "Current loader: partial state after rejection"),
            (row["clean_top"], "Clean-room retry: clear both robots before placement"),
        ]
        for phase in ("current", "clean"):
            for agent in ("agent_0", "agent_1"):
                images.append((row[f"{phase}_views"][agent], f"{phase.title()} {agent} view"))
        figures = "".join(
            f'<figure><img src="{_data_url(Path(path))}"><figcaption>{html.escape(label)}</figcaption></figure>'
            for path, label in images
        )
        panels.append(
            f'<section><h2>{row["task"]}</h2><p><code>{html.escape(str(row["initial_state_agents"]))}</code></p>'
            f'<p>Current error: <code>{html.escape(str(row["current_loader_error"]))}</code><br>'
            f'Clean-room placement results: <strong>{row["clean_room_placements"]}</strong></p>'
            f'<div class="grid">{figures}</div></section>'
        )
    report = f'''<!doctype html><meta charset="utf-8"><title>Initial spawn conflicts</title>
    <style>body{{font:16px/1.5 system-ui;background:#eef2f7;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:28px}}section,header{{background:white;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{color:#526079}}code{{white-space:pre-wrap}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
    <main><header><h1>AlcoholServingPrep and PrepareCoffee initial placement inspection</h1><p>Each task shows the existing state-loading attempt and a diagnostic retry that first moves both robots away, then places both intended locations. Images are embedded in this HTML.</p></header>{''.join(panels)}</main>'''
    (OUT / "report.html").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
