"""Behaviorally audit agents that block exclusive fixtures on a parent surface.

This is evidence collection only. It does not change prompts, FSM state, or
trajectory data. A case is detected when navigation fails with the suspected
blocker present, succeeds after that blocker gives space, and still fails when
only the attempted navigator moves away.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401  # registers RoboCasa task classes with robosuite
from data_generation.task_level.tasks.shared.concurrent_fsm import is_exclusive_fixture
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata
from robosuite.environments.base import REGISTERED_ENVS


def _hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _adapt(adapter, adapted, *, agent: str, tool: str, args: dict[str, Any]):
    return adapter._adapt_step(
        {"agent": agent, "tool": tool, "args": args},
        resolved_initial_state=adapted["initial_state"],
        output_dir=None,
    )


def _positions(session: SimSession) -> list[list[float]]:
    return [
        [round(float(x), 6) for x in session.executor.runner._get_robot_position(i)]
        for i in (0, 1)
    ]


def _execute(session: SimSession, call: dict[str, Any]):
    return session.executor.execute(
        call["tool"], robot_idx=call.get("robot_idx", 0), **call.get("args", {})
    )


def _probe(
    session: SimSession,
    trajectory: dict[str, Any],
    *,
    blocker_idx: int,
    fixture_id: str,
    parent_id: str,
) -> dict[str, Any]:
    navigator_idx = 1 - blocker_idx

    def reset_calls():
        adapter, adapted = session.start_trajectory(trajectory)
        nav = _adapt(
            adapter, adapted, agent=f"agent_{navigator_idx}",
            tool="navigate_to_fixture", args={"fixture_id": fixture_id},
        )
        blocker_yield = _adapt(
            adapter, adapted, agent=f"agent_{blocker_idx}",
            tool="give_space", args={"fixture_id": fixture_id},
        )
        navigator_yield = _adapt(
            adapter, adapted, agent=f"agent_{navigator_idx}",
            tool="give_space", args={"fixture_id": parent_id},
        )
        blocker_nav = _adapt(
            adapter, adapted, agent=f"agent_{blocker_idx}",
            tool="navigate_to_fixture", args={"fixture_id": fixture_id},
        )
        return nav, blocker_yield, navigator_yield, blocker_nav

    nav, _blocker_yield, _navigator_yield, _blocker_nav = reset_calls()
    initial_positions = _positions(session)
    direct = _execute(session, nav)

    nav, blocker_yield, _navigator_yield, _blocker_nav = reset_calls()
    blocker_space = _execute(session, blocker_yield)
    after_blocker_space = _positions(session)
    after_blocker = _execute(session, nav)

    nav, _blocker_yield, navigator_yield, _blocker_nav = reset_calls()
    navigator_space = _execute(session, navigator_yield)
    after_navigator_space = _positions(session)
    after_navigator = _execute(session, nav)

    _nav, _blocker_yield, _navigator_yield, blocker_nav = reset_calls()
    blocker_direct = _execute(session, blocker_nav)
    positions_after_blocker_navigation = _positions(session)

    resolved_fixture_id = blocker_nav.get("args", {}).get("fixture_id")
    geometry = {}
    resolved_fixture = session.executor.runner._fixtures.get(resolved_fixture_id)
    if resolved_fixture is not None:
        from robocasa.utils.placement import get_fixture_aabb

        aabb = get_fixture_aabb(resolved_fixture)
        if aabb is not None:
            geometry["fixture_aabb"] = [
                [float(value) for value in aabb[0]],
                [float(value) for value in aabb[1]],
            ]
        geometry["preferred_reachable_approach_face"] = (
            session.executor.runner._occupancy_grid.preferred_reachable_approach_face(
                resolved_fixture
            )
        )

    detected = (
        not direct.success
        and blocker_space.success
        and after_blocker.success
        and navigator_space.success
        and not after_navigator.success
    )
    return {
        "detected": detected,
        "blocker_agent": f"agent_{blocker_idx}",
        "navigator_agent": f"agent_{navigator_idx}",
        "fixture_id": fixture_id,
        "parent_fixture_id": parent_id,
        "initial_positions": initial_positions,
        "direct_navigation_success": bool(direct.success),
        "blocker_give_space_success": bool(blocker_space.success),
        "positions_after_blocker_give_space": after_blocker_space,
        "navigation_after_blocker_give_space_success": bool(after_blocker.success),
        "navigator_give_space_success": bool(navigator_space.success),
        "positions_after_navigator_give_space": after_navigator_space,
        "navigation_after_navigator_give_space_success": bool(after_navigator.success),
        "blocker_navigation_success": bool(blocker_direct.success),
        "positions_after_blocker_navigation": positions_after_blocker_navigation,
        "resolved_fixture_id": resolved_fixture_id,
        "geometry": geometry,
    }


def _save_views(
    session: SimSession, output: Path, case_id: str, *, stage: str
) -> dict[str, str]:
    case_dir = output / "images" / case_id / stage
    case_dir.mkdir(parents=True, exist_ok=True)
    result = session.executor.get_image(
        views=["top_view", "room_view", "map"],
        image_paths=[
            str(case_dir / "top_view.jpg"),
            str(case_dir / "room_view.jpg"),
            str(case_dir / "map.png"),
        ],
        agent_id="agent_0",
    )
    paths = result.details.get("image_paths") or []
    return {
        name: str(Path(path).resolve())
        for name, path in zip(("top_view", "room_view", "map"), paths)
    }


def audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_dirs = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if args.tasks:
        wanted = set(args.tasks.split(","))
        task_dirs = [path for path in task_dirs if path.name in wanted]
    for task_dir in task_dirs:
        metadata = get_task_metadata(task_dir.name)
        task_class = REGISTERED_ENVS[metadata.composite_task]
        # Match Kitchen's own scene-sampling contract. Unsupported scenes are
        # not failed probes: RoboCasa removes them before sampling a scene.
        if args.layout in task_class.EXCLUDE_LAYOUTS or args.style in task_class.EXCLUDE_STYLES:
            continue
        unique: dict[str, tuple[Path, dict[str, Any]]] = {}
        for path in sorted(task_dir.glob("traj_*/original_trajectory.json")):
            trajectory = json.loads(path.read_text(encoding="utf-8"))
            signature = _hash(trajectory.get("initial_state"))
            unique.setdefault(signature, (path, trajectory))
        if not unique:
            continue
        session: SimSession | None = None
        try:
            sample = next(iter(unique.values()))[1]
            session = SimSession(
                composite_task=metadata.composite_task,
                sample_trajectory=sample,
                layout=args.layout, style=args.style, seed=args.seed,
                gl_backend=args.gl_backend, render_size=args.render_size,
                map_dpi=args.map_dpi, map_renderer="raster",
            )
            for state_signature, (path, trajectory) in unique.items():
                initial = trajectory["initial_state"]
                fixtures = initial.get("fixtures") or {}
                agents = initial.get("agents") or {}
                candidates = []
                for fixture_id, fixture in fixtures.items():
                    if not isinstance(fixture, dict) or not is_exclusive_fixture(fixture_id, initial):
                        continue
                    parent_id = fixture.get("parent_fixture")
                    if not isinstance(parent_id, str):
                        continue
                    for blocker_idx in (0, 1):
                        if (agents.get(f"agent_{blocker_idx}") or {}).get("location") == parent_id:
                            candidates.append((blocker_idx, fixture_id, parent_id))
                for blocker_idx, fixture_id, parent_id in candidates:
                    try:
                        evidence = _probe(
                            session, trajectory, blocker_idx=blocker_idx,
                            fixture_id=fixture_id, parent_id=parent_id,
                        )
                        error = None
                    except Exception as exc:
                        evidence = {"detected": False}
                        error = f"{type(exc).__name__}: {exc}"
                    case_id = _hash(
                        [task_dir.name, state_signature, blocker_idx, fixture_id]
                    )[:16]
                    views = {}
                    if evidence.get("detected") and not args.no_save_views:
                        session.start_trajectory(trajectory)
                        initial_views = _save_views(
                            session, args.output_dir, case_id, stage="initial"
                        )
                        adapter, adapted = session.start_trajectory(trajectory)
                        blocker_nav = _adapt(
                            adapter,
                            adapted,
                            agent=evidence["blocker_agent"],
                            tool="navigate_to_fixture",
                            args={"fixture_id": fixture_id},
                        )
                        blocker_nav_result = _execute(session, blocker_nav)
                        evidence["blocker_navigation_for_views_success"] = bool(
                            blocker_nav_result.success
                        )
                        navigated_views = _save_views(
                            session,
                            args.output_dir,
                            case_id,
                            stage="after_blocker_navigates",
                        )
                        views = {
                            "initial": initial_views,
                            "after_blocker_navigates": navigated_views,
                        }
                    records.append({
                        "case_id": case_id,
                        "task_name": task_dir.name,
                        "composite_task": metadata.composite_task,
                        "state_signature": state_signature,
                        "carrier_trajectory_id": path.parent.name,
                        "initial_state": initial,
                        "evidence": evidence,
                        "views": views,
                        "error": error,
                        "manual_decision": None,
                    })
        except Exception as exc:
            records.append({
                "case_id": _hash(
                    [task_dir.name, args.layout, args.style, args.seed, "session_error"]
                )[:16],
                "task_name": task_dir.name,
                "composite_task": metadata.composite_task,
                "state_signature": None,
                "carrier_trajectory_id": None,
                "initial_state": None,
                "evidence": {"detected": False},
                "views": {},
                "error": f"session initialization failed: {type(exc).__name__}: {exc}",
                "manual_decision": None,
            })
        finally:
            if session is not None:
                session.close()
    return records


def _artifact(records: list[dict[str, Any]], output_dir: Path) -> None:
    detected = [row for row in records if row["evidence"].get("detected")]
    cards = []
    for row in detected:
        ev = row["evidence"]
        image_sections = []
        for stage, stage_views in row["views"].items():
            images = "".join(
                f'<figure><img src="{html.escape(str(Path(path).relative_to(output_dir.resolve())))}"><figcaption>{name}</figcaption></figure>'
                for name, path in stage_views.items()
            )
            title = (
                "Initial positions"
                if stage == "initial"
                else "After the proposed blocker explicitly navigates to the fixture"
            )
            image_sections.append(
                f'<h3>{html.escape(title)}</h3><div class="images">{images}</div>'
            )
        cards.append(f"""
<section><h2>{html.escape(row['task_name'])}: {html.escape(ev['fixture_id'])}</h2>
<p><b>Proposed blocker:</b> {ev['blocker_agent']} · <b>Broad location:</b> {html.escape(ev['parent_fixture_id'])} · <b>Case:</b> <code>{row['case_id']}</code></p>
{''.join(image_sections)}
<table><tr><th>Direct navigation</th><td>{ev['direct_navigation_success']}</td></tr>
<tr><th>After blocker gives space</th><td>{ev['navigation_after_blocker_give_space_success']}</td></tr>
<tr><th>After navigator alone moves</th><td>{ev['navigation_after_navigator_give_space_success']}</td></tr>
<tr><th>Blocker can navigate to fixture</th><td>{ev['blocker_navigation_success']}</td></tr></table>
<details><summary>Initial state and coordinates</summary><pre>{html.escape(json.dumps({'initial_state':row['initial_state'],'evidence':ev},indent=2))}</pre></details>
<p class="decision">Manual decision: ☐ confirm blocks_access_to &nbsp; ☐ false positive</p></section>""")
    report = f"""<!doctype html><html><head><meta charset="utf-8"><title>Access blocking audit</title><style>
body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#17202a}}section{{background:white;padding:20px;margin:18px 0;border:1px solid #dfe5ec;border-radius:12px}}.images{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}img{{width:100%;max-height:360px;object-fit:contain;background:#eee}}figure{{margin:0}}table{{border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}pre{{overflow:auto}}.decision{{padding:12px;background:#fff6d8}}code{{word-break:break-all}}
</style></head><body><h1>Colocated access-blocking audit</h1><p>{len(detected)} behaviorally detected cases from {len(records)} probes. No state, prompt, or FSM changes have been applied; every case requires manual review.</p>{''.join(cards) or '<p>No cases detected.</p>'}</body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gl-backend", default="egl")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--map-dpi", type=int, default=60)
    parser.add_argument(
        "--no-save-views", action="store_true",
        help="Skip images for broad prevalence sweeps; retain structured evidence.",
    )
    parser.add_argument("--trial", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = audit(args)
    (args.output_dir / "audit.json").write_text(
        json.dumps({
            "schema_version": 1,
            "method": "parent_fixture navigation/give-space counterfactual",
            "scene": {
                "layout": args.layout, "style": args.style, "seed": args.seed,
                "trial": args.trial,
            },
            "num_probes": len(records),
            "num_detected": sum(row["evidence"].get("detected", False) for row in records),
            "records": records,
        }, indent=2) + "\n", encoding="utf-8",
    )
    _artifact(records, args.output_dir)
    print(f"probes={len(records)} detected={sum(r['evidence'].get('detected', False) for r in records)} output={args.output_dir}")


if __name__ == "__main__":
    main()
