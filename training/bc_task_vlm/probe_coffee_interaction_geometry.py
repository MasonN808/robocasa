"""Probe coffee-machine working faces from dispenser and button geometry."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from robocasa.utils.placement import (
    get_face_order,
    get_front_alignment_metrics,
    infer_front_face_from_target,
)
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.probe_spawn_candidate_extensions import (
    _clear,
    _positions,
    _save_views,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/coffee_interaction_geometry_artifact"
ASSETS = OUT / "assets"


def _url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _world_site(session: SimSession, name: str) -> np.ndarray:
    site_id = session.executor.env.sim.model.site_name2id(name)
    return np.asarray(session.executor.env.sim.data.site_xpos[site_id][:2], dtype=float)


def _world_geom(session: SimSession, name: str) -> np.ndarray:
    geom_id = session.executor.env.sim.model.geom_name2id(name)
    return np.asarray(session.executor.env.sim.data.geom_xpos[geom_id][:2], dtype=float)


def _gallery(title: str, views: dict, note: str) -> str:
    return f'<h3>{html.escape(title)}</h3><p>{html.escape(note)}</p><div class="grid">' + "".join(
        f'<figure><img src="{_url(views[key])}"><figcaption>{label}</figcaption></figure>'
        for key, label in (("top", "Top view"), ("agent_0", "Agent 0 view"), ("agent_1", "Agent 1 view"))
    ) + "</div>"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    task, run_index, layout, style, seed = "PrepareCoffee", 8, 50, 14, 42
    trajectory = _trajectory(task, run_index)
    session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=layout, style=style, seed=seed, gl_backend="egl", render_size=900, map_dpi=120, map_renderer="raster")
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        agents = adapted["initial_state"]["agents"]
        coffee_agent = next(agent for agent, state in agents.items() if "coffee_machine" in state["location"])
        counter_agent = "agent_1" if coffee_agent == "agent_0" else "agent_0"
        coffee_idx, counter_idx = int(coffee_agent[-1]), int(counter_agent[-1])
        coffee_id, counter_id = agents[coffee_agent]["location"], agents[counter_agent]["location"]
        runner = session.executor.runner
        fixture = runner._fixtures[coffee_id]
        prefix = fixture.naming_prefix
        pour_xy = _world_site(session, f"{prefix}receptacle_place_site")
        button_names = [f"{prefix}{name}" for name in fixture._start_button_names]
        button_positions = [_world_geom(session, name) for name in button_names]
        button_xy = np.mean(np.stack(button_positions), axis=0)
        combined_xy = np.mean(np.stack((pour_xy, button_xy)), axis=0)
        from robocasa.utils.placement import get_fixture_aabb
        fmin, fmax = get_fixture_aabb(fixture)
        targets = {
            "occupancy": None,
            "pour_site": pour_xy,
            "start_button": button_xy,
            "combined": combined_xy,
        }
        records = []
        for label, target in targets.items():
            _clear(runner)
            if target is None:
                coffee_ok = bool(runner._move_robot_near_fixture(coffee_idx, coffee_id, require_front=False))
                face = runner._occupancy_grid.preferred_reachable_approach_face(fixture)
            else:
                coffee_ok = bool(runner._move_robot_near_fixture(coffee_idx, coffee_id, ref_pos_override=target, require_front=True))
                face = infer_front_face_from_target(fmin, fmax, target)
            counter_ok = bool(runner._move_robot_near_fixture(counter_idx, counter_id, require_front=False))
            position = runner._get_robot_position(coffee_idx)[:2].copy()
            metrics = get_front_alignment_metrics(fixture, position, target_xy=target, front_face=face)
            records.append({
                "label": label, "target_xy": None if target is None else target.tolist(),
                "face": face, "coffee_placement": coffee_ok, "counter_placement": counter_ok,
                "positions": _positions(runner), "metrics": metrics,
                "coffee_ready_under_candidate_face": bool(
                    coffee_ok and metrics is not None and metrics["on_front_face"] and metrics["within_span"]
                ),
                "separation_m": float(np.linalg.norm(runner._get_robot_position(0)[:2] - runner._get_robot_position(1)[:2])) if coffee_ok and counter_ok else None,
                "views": _save_views(session, f"coffee_geometry_{label}", target_fixtures=((coffee_id, "coffee machine"), (counter_id, "parent counter"))),
            })
        payload = {
            "scene": {"layout": layout, "style": style, "seed": seed},
            "agents": agents, "coffee_fixture": coffee_id, "counter_fixture": counter_id,
            "pour_site_xy": pour_xy.tolist(), "button_xy": button_xy.tolist(),
            "combined_xy": combined_xy.tolist(), "asset_rotation_face": get_face_order(fixture)[0],
            "records": records,
        }
        sections = []
        explanations = {
            "occupancy": "Current reachable-side inference; it ignores dispenser and button geometry.",
            "pour_site": "Front face inferred from the mug placement site beneath the dispenser.",
            "start_button": "Front face inferred from the mean world position of the start-button geometry.",
            "combined": "Front face inferred jointly from dispenser and button geometry.",
        }
        for record in records:
            note = (
                f"{explanations[record['label']]} Face={record['face']}; "
                f"coffee placement={record['coffee_placement']}; counter placement={record['counter_placement']}; "
                f"candidate-face ready={record['coffee_ready_under_candidate_face']}; separation={record['separation_m']}."
            )
            sections.append(_gallery(record["label"].replace("_", " ").title(), record["views"], note))
        report = f'''<!doctype html><meta charset="utf-8"><title>Coffee interaction geometry</title><style>body{{font:16px/1.5 system-ui;background:#eef2f7;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:28px}}header,section{{background:white;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{color:#526079}}code,pre{{white-space:pre-wrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style><main><header><h1>PrepareCoffee working-face probe</h1><p>The requested state remains fixed: {coffee_agent} at <code>{coffee_id}</code>, {counter_agent} at <code>{counter_id}</code>. Only the rule used to choose the coffee-machine working face changes. Every group is one synchronized state.</p></header><section>{''.join(sections)}<details><summary>Exact geometry and metrics</summary><pre>{html.escape(json.dumps(payload,indent=2,default=str))}</pre></details></section></main>'''
        (OUT / "report.html").write_text(report)
        (OUT / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(OUT / "report.html")
    finally:
        session.close()


if __name__ == "__main__":
    main()
