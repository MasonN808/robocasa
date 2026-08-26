"""Audit symbolic cabinet parents against simulator-derived scene parents."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401  # register RoboCasa task classes
from robosuite.environments.base import REGISTERED_ENVS
from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from data_generation.task_level.tasks.specs import load_verified_task_specs
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/cabinet_parent_relationship_audit"


def _trajectory(task: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": f"audit_{task}", "task": task, "composite_task": task,
        "initial_state": state,
        "grounding_map": build_grounding_map_for_task(task, state), "steps": [],
    }


def _resolve_fixture(adapter: Any, adapted: dict[str, Any], symbol: str) -> str:
    return adapter._adapt_step(
        {"agent": "agent_0", "tool": "navigate_to_fixture", "args": {"fixture_id": symbol}},
        resolved_initial_state=adapted["initial_state"], output_dir=None,
    )["args"]["fixture_id"]


def _audit_task(task: str, scene: dict[str, int]) -> list[dict[str, Any]]:
    config = _config(
        Namespace(task=task, num_runs=32, run_index=0, location="global", temperature=0.6),
        model=FLASH_MODEL, thinking="low",
    )
    state = get_task_definition(task).build_task_instance(0, config).initial_state
    trajectory = _trajectory(task, state)
    session = SimSession(
        composite_task=task, sample_trajectory=trajectory, **scene,
        gl_backend="egl", render_size=256, map_dpi=80, map_renderer="raster",
    )
    records: list[dict[str, Any]] = []
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        scene_fixtures = session.executor.get_scene_description().get("fixtures") or {}
        fixtures = adapted["initial_state"].get("fixtures") or {}
        resolved_by_symbol: dict[str, str] = {}
        for symbol in fixtures:
            try:
                resolved_by_symbol[symbol] = _resolve_fixture(adapter, adapted, symbol)
            except Exception:
                continue
        for cabinet_symbol, cabinet_state in fixtures.items():
            if "cabinet" not in str(cabinet_state.get("fixture_type") or "").lower():
                continue
            parent_symbol = cabinet_state.get("parent_fixture")
            concrete_cabinet = resolved_by_symbol.get(cabinet_symbol)
            scene_parent = (
                scene_fixtures.get(concrete_cabinet, {}).get("parent_fixture")
                if isinstance(concrete_cabinet, str) else None
            )
            resolved_parent = (
                resolved_by_symbol.get(parent_symbol)
                if isinstance(parent_symbol, str) else None
            )
            equivalent_symbols = sorted(
                symbol for symbol, concrete in resolved_by_symbol.items()
                if symbol not in {cabinet_symbol, parent_symbol} and concrete == scene_parent
            )
            records.append({
                "task": task,
                "cabinet_symbol": cabinet_symbol,
                "parent_symbol": parent_symbol,
                "parent_is_fallback": parent_symbol == "cabinet_parent_counter",
                "concrete_cabinet": concrete_cabinet,
                "simulator_parent": scene_parent,
                "resolved_symbolic_parent": resolved_parent,
                "matches": isinstance(scene_parent, str) and resolved_parent == scene_parent,
                "other_symbols_for_same_parent": equivalent_symbols,
            })
    finally:
        session.close()
    return records


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    scene = {"layout": args.layout, "style": args.style, "seed": args.seed}
    tasks = []
    for spec in load_verified_task_specs():
        fixtures = spec.initial_state.get("fixtures") or {}
        if any("cabinet" in str(state.get("fixture_type") or "").lower() for state in fixtures.values()):
            tasks.append(spec.composite_task)
    records = []
    errors = []
    skipped = []
    for task in tasks:
        task_class = REGISTERED_ENVS[get_task_metadata(task).composite_task]
        if args.layout in task_class.EXCLUDE_LAYOUTS or args.style in task_class.EXCLUDE_STYLES:
            skipped.append(task)
            continue
        try:
            records.extend(_audit_task(task, scene))
        except Exception as exc:
            errors.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": scene,
        "task_count": len(tasks),
        "relationship_count": len(records),
        "matching_count": sum(record["matches"] for record in records),
        "mismatch_count": sum(not record["matches"] for record in records),
        "duplicate_fallback_count": sum(
            record["parent_is_fallback"] and bool(record["other_symbols_for_same_parent"])
            for record in records
        ),
        "error_count": len(errors),
        "unsupported_task_count": len(skipped),
    }
    payload = {"summary": summary, "records": records, "errors": errors, "unsupported_tasks": skipped}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors and not summary["mismatch_count"] else 1)


if __name__ == "__main__":
    main()
