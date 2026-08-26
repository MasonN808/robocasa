"""Probe navigation to a child cabinet when both agents start at its parent."""

from __future__ import annotations

import json
from pathlib import Path

from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.probe_parent_access_occupied import (
    ROOT,
    SCENE,
    SCENARIOS,
    _annotate,
    _positions,
    _trajectory,
)


OUT = ROOT / "training/bc_task_vlm/eval_runs/both_counter_to_cabinet_probe"
ASSETS = OUT / "assets"
CASES = [
    case for case in SCENARIOS
    if case["task"] in {"GatherMarinadeIngredients", "SpicyMarinade"}
]


def _render(session: SimSession, path: Path, title: str) -> None:
    session.executor._invalidate_visual_cache()
    _annotate(
        session.executor.runner._render_top_view(),
        title,
        "Both agents started at the parent counter; active agent navigates to cabinet.",
    ).save(path, quality=94)


def _run(case: dict, *, silent_clear: bool) -> dict:
    mode = "silent_clear_enabled" if silent_clear else "silent_clear_disabled"
    trajectory = _trajectory(case)
    for state in trajectory["initial_state"]["agents"].values():
        state["location"] = case["parent"]
    session = SimSession(
        composite_task=case["task"], sample_trajectory=trajectory, **SCENE,
        gl_backend="egl", render_size=768, map_dpi=150, map_renderer="raster",
    )
    record = {"task": case["task"], "mode": mode, "actor": case["actor"]}
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        if not silent_clear:
            session.executor._clear_fixture_blockers = lambda *_args, **_kwargs: False
        record["initial_positions"] = _positions(session)
        _render(session, ASSETS / f'{case["name"]}_{mode}_initial.jpg', f'{case["task"]}: initial')
        paths, views = session.render_views(
            ("agentview_center", "wrist"), agent_id=case["actor"], out_dir=ASSETS,
            tag=f'{case["name"]}_{mode}_initial',
        )
        record["active_views_initial"] = dict(zip(views, paths))
        call = adapter._adapt_step(
            {"agent": case["actor"], "tool": "navigate_to_fixture", "args": {"fixture_id": case["child"]}},
            resolved_initial_state=adapted["initial_state"], output_dir=None,
        )
        result = session.executor.execute(call["tool"], robot_idx=call["robot_idx"], **call["args"])
        record["navigation_success"] = bool(result.success)
        record["navigation_details"] = result.details
        record["final_positions"] = _positions(session)
        _render(session, ASSETS / f'{case["name"]}_{mode}_after_navigation.jpg', f'{case["task"]}: after navigation')
        paths, views = session.render_views(
            ("agentview_center", "wrist"), agent_id=case["actor"], out_dir=ASSETS,
            tag=f'{case["name"]}_{mode}_after_navigation',
        )
        record["active_views_after_navigation"] = dict(zip(views, paths))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = [_run(case, silent_clear=mode) for case in CASES for mode in (True, False)]
    (OUT / "probe_results.json").write_text(json.dumps({"scene": SCENE, "records": records}, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
