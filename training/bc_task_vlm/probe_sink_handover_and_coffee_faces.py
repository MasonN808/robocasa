"""Clarify sink handover and compare coffee-machine approach definitions."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from robocasa.utils.placement import get_face_order, get_front_alignment_metrics
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.probe_spawn_candidate_extensions import (
    _clear,
    _positions,
    _save_views,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/sink_handover_coffee_face_artifact"
ASSETS = OUT / "assets"


def _url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _gallery(title: str, views: dict) -> str:
    return f'<h3>{html.escape(title)}</h3><div class="grid">' + "".join(
        f'<figure><img src="{_url(views[key])}"><figcaption>{label}</figcaption></figure>'
        for key, label in (("top", "Top view"), ("agent_0", "Agent 0 view"), ("agent_1", "Agent 1 view"))
    ) + "</div>"


def _alcohol() -> dict:
    task, run_index, layout, style, seed = "AlcoholServingPrep", 12, 15, 14, 42
    trajectory = _trajectory(task, run_index)
    session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=layout, style=style, seed=seed, gl_backend="egl", render_size=900, map_dpi=120, map_renderer="raster")
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        session.executor.restore_baseline_state()
        session.executor.load_initial_state(adapted["initial_state"])
        agents = adapted["initial_state"]["agents"]
        holder = next(agent for agent, state in agents.items() if "sink" in state["location"])
        waiter = "agent_1" if holder == "agent_0" else "agent_0"
        holder_idx, waiter_idx = int(holder[-1]), int(waiter[-1])
        sink = agents[holder]["location"]
        dining = agents[waiter]["location"]
        before = {"positions": _positions(session.executor.runner), "views": _save_views(session, "alcohol_before_handover", target_fixtures=((sink, "sink"), (dining, "dining workspace")))}
        yield_result = session.executor.give_space(sink, robot_idx=holder_idx)
        after_yield = {"success": yield_result.success, "positions": _positions(session.executor.runner), "views": _save_views(session, "alcohol_after_holder_yields", target_fixtures=((sink, "sink"), (dining, "dining workspace")))}
        enter_result = session.executor.navigate_to_fixture(sink, robot_idx=waiter_idx)
        after_enter = {
            "success": enter_result.success,
            "waiter_ready_at_sink": bool(session.executor._robot_near_fixture(waiter_idx, sink)),
            "positions": _positions(session.executor.runner),
            "separation_m": float(np.linalg.norm(session.executor.runner._get_robot_position(0)[:2] - session.executor.runner._get_robot_position(1)[:2])),
            "views": _save_views(session, "alcohol_after_partner_enters", target_fixtures=((sink, "sink"), (dining, "dining workspace"))),
        }
        return {"holder": holder, "waiter": waiter, "sink": sink, "dining": dining, "before": before, "after_yield": after_yield, "after_enter": after_enter}
    finally:
        session.close()


def _alignment(session: SimSession, robot_idx: int, fixture_id: str) -> dict:
    runner = session.executor.runner
    fixture = runner._fixtures[fixture_id]
    pos = runner._get_robot_position(robot_idx)[:2]
    target = runner._get_fixture_front_target_xy(fixture_id)
    occupancy_face = runner._occupancy_grid.preferred_reachable_approach_face(fixture)
    asset_face = get_face_order(fixture, front_target_xy=target)[0]
    return {
        "position": pos.tolist(),
        "occupancy_face": occupancy_face,
        "asset_rotation_face": asset_face,
        "occupancy_metrics": get_front_alignment_metrics(fixture, pos, target_xy=target, front_face=occupancy_face),
        "asset_rotation_metrics": get_front_alignment_metrics(fixture, pos, target_xy=target),
    }


def _coffee() -> dict:
    task, run_index, layout, style, seed = "PrepareCoffee", 8, 50, 14, 42
    trajectory = _trajectory(task, run_index)
    session = SimSession(composite_task=task, sample_trajectory=trajectory, layout=layout, style=style, seed=seed, gl_backend="egl", render_size=900, map_dpi=120, map_renderer="raster")
    try:
        adapter = TrajectoryAdapter(session.executor, allow_approximate_ids=True)
        adapted = adapter.adapt(trajectory)
        agents = adapted["initial_state"]["agents"]
        coffee_agent = next(agent for agent, state in agents.items() if "coffee_machine" in state["location"])
        other_agent = "agent_1" if coffee_agent == "agent_0" else "agent_0"
        coffee_idx, other_idx = int(coffee_agent[-1]), int(other_agent[-1])
        coffee = agents[coffee_agent]["location"]
        counter = agents[other_agent]["location"]
        runner = session.executor.runner

        _clear(runner)
        occupancy_coffee_ok = bool(runner._move_robot_near_fixture(coffee_idx, coffee, require_front=False))
        occupancy_counter_ok = bool(runner._move_robot_near_fixture(other_idx, counter, require_front=False))
        occupancy = {
            "coffee_placement": occupancy_coffee_ok, "counter_placement": occupancy_counter_ok,
            "alignment": _alignment(session, coffee_idx, coffee), "positions": _positions(runner),
            "views": _save_views(session, "coffee_occupancy_face", target_fixtures=((coffee, "coffee machine"), (counter, "parent counter"))),
        }

        _clear(runner)
        strict_coffee_ok = bool(runner._move_robot_near_fixture(coffee_idx, coffee, require_front=True))
        strict_counter_ok = bool(runner._move_robot_near_fixture(other_idx, counter, require_front=False))
        strict = {
            "coffee_placement": strict_coffee_ok, "counter_placement": strict_counter_ok,
            "alignment": _alignment(session, coffee_idx, coffee), "positions": _positions(runner),
            "views": _save_views(session, "coffee_fixture_face", target_fixtures=((coffee, "coffee machine"), (counter, "parent counter"))),
        }
        return {"coffee_agent": coffee_agent, "other_agent": other_agent, "coffee": coffee, "counter": counter, "occupancy": occupancy, "fixture_face": strict}
    finally:
        session.close()


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    alcohol, coffee = _alcohol(), _coffee()
    report = f'''<!doctype html><meta charset="utf-8"><title>Sink handover and coffee approach</title><style>body{{font:16px/1.5 system-ui;background:#eef2f7;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:28px}}header,section{{background:white;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;border-radius:8px}}figcaption{{color:#526079}}code{{white-space:pre-wrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style><main>
    <header><h1>Clarifying the sink handover and coffee-machine pose</h1><p>Each three-panel group is one synchronized simulator state. Red=agent_0; orange=agent_1.</p></header>
    <section><h2>AlcoholServingPrep handover</h2><p>The initial sink holder is <strong>{alcohol['holder']}</strong>; the agent initially at the dining workspace is <strong>{alcohol['waiter']}</strong>. The third state is the actual question: after the holder gives space, can the other agent enter and become ready at the sink? <strong>{alcohol['after_enter']['success']} / ready={alcohol['after_enter']['waiter_ready_at_sink']}</strong>, separation={alcohol['after_enter']['separation_m']:.3f} m.</p>{_gallery('1. Initial exclusive-first state', alcohol['before']['views'])}{_gallery('2. Sink holder gives space', alcohol['after_yield']['views'])}{_gallery('3. Other agent navigates from dining workspace to sink', alcohol['after_enter']['views'])}</section>
    <section><h2>PrepareCoffee approach-face comparison</h2><p>The requested configuration is {coffee['coffee_agent']} at <code>{coffee['coffee']}</code> and {coffee['other_agent']} at <code>{coffee['counter']}</code>. The first branch is the current occupancy-derived side. The second requests the asset-rotation-derived face. The generic resolver finds no handle/front target on this coffee-machine asset, so the second branch is <em>not</em> evidence of the dispenser-facing side; it is included to show why asset rotation is not an adequate replacement. This comparison is diagnostic and does not change production placement.</p>{_gallery('Current occupancy-derived approach', coffee['occupancy']['views'])}{_gallery('Asset-rotation-derived approach (placement fails)', coffee['fixture_face']['views'])}<details><summary>Coordinates and alignment metrics</summary><pre>{html.escape(json.dumps({'occupancy':coffee['occupancy'],'asset_rotation':coffee['fixture_face']},indent=2,default=str))}</pre></details></section></main>'''
    (OUT / "report.html").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps({"alcohol": alcohol, "coffee": coffee}, indent=2) + "\n")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
