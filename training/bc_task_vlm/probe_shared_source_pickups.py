"""Probe whether two agents can pick distinct objects from one movable source.

Live evaluation admits an atomic batch against one pre-tick state, then commits
simulator calls in deterministic order because a MuJoCo environment is not
thread-safe.  This probe therefore executes both possible commit orders from
fresh identical states and requires both calls to succeed in both orders.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/shared_source_pickup_probe_artifact"
ASSETS = OUT / "assets"
SPEC_DIR = ROOT / "data_generation/task_level/tasks/specs/verified"
SCENE = {"layout": 11, "style": 34}
SEEDS = (42, 43, 44)

CASES = {
    "GarnishCake": {
        "spec": SPEC_DIR / "garnishcake.json",
        "source": "fruit_plate",
        "objects": ("cherry1", "strawberry1"),
    },
    "MeatSkewerAssembly": {
        "spec": SPEC_DIR / "meatskewerassembly.json",
        "source": "skewer_plate",
        "objects": ("skewer1", "skewer2"),
    },
}


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "details": deepcopy(getattr(result, "details", None) or {}),
    }


def _held(executor: Any) -> dict[str, str | None]:
    held = getattr(executor, "_held_objects", {})
    return {f"agent_{idx}": held.get(idx) for idx in (0, 1)}


def _annotate(image: Any, *, heading: str, note: str) -> Image.Image:
    result = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.rounded_rectangle((12, 12, result.width - 12, 67), radius=7, fill=(8, 15, 30))
    draw.text((24, 23), heading, fill="white", font=font)
    draw.text((24, 44), note, fill=(226, 232, 240), font=font)
    return result


def _render(session: SimSession, path: Path, *, heading: str, note: str) -> None:
    session.executor._invalidate_visual_cache()
    image = session.executor.runner._render_top_view()
    _annotate(image, heading=heading, note=note).save(path, quality=94)


def _run_order(
    *,
    task: str,
    spec: dict[str, Any],
    source: str,
    objects: tuple[str, str],
    seed: int,
    order: tuple[int, int],
) -> dict[str, Any]:
    trajectory = {
        "trajectory_id": f"shared_source_probe_{task}_{seed}_{order[0]}{order[1]}",
        "task": task,
        "initial_state": deepcopy(spec["initial_state"]),
        "steps": [],
    }
    session = SimSession(
        composite_task=task,
        sample_trajectory=trajectory,
        layout=SCENE["layout"],
        style=SCENE["style"],
        seed=seed,
        gl_backend="egl",
        render_size=768,
        map_dpi=150,
        map_renderer="raster",
    )
    record: dict[str, Any] = {
        "task": task,
        "source": source,
        "seed": seed,
        "layout": SCENE["layout"],
        "style": SCENE["style"],
        "commit_order": [f"agent_{idx}" for idx in order],
        "calls": [],
    }
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        record["resolved_objects"] = {
            object_id: adapted["initial_state"]["objects"][object_id]
            for object_id in objects
        }
        if seed == SEEDS[0]:
            prefix = f"{task.lower()}_order_{order[0]}{order[1]}"
            _render(
                session,
                ASSETS / f"{prefix}_before.jpg",
                heading=f"{task}: before the atomic pickup tick",
                note=f"Both agents start at the shared source {source}.",
            )
        for position, agent_idx in enumerate(order, start=1):
            object_id = objects[agent_idx]
            symbolic = {
                "step": 0,
                "agent": f"agent_{agent_idx}",
                "tool": "pick_up_object",
                "args": {"object_id": object_id, "source_id": source},
            }
            adapted_call = adapter._adapt_step(
                symbolic,
                resolved_initial_state=adapted["initial_state"],
                output_dir=None,
            )
            result = session.executor.execute(
                adapted_call["tool"],
                robot_idx=adapted_call.get("robot_idx", 0),
                **adapted_call.get("args", {}),
            )
            call_record = {
                "atomic_tick": 0,
                "symbolic_call": symbolic,
                "adapted_call": adapted_call,
                "result": _result_payload(result),
                "held_after_call": _held(session.executor),
            }
            record["calls"].append(call_record)
            if seed == SEEDS[0]:
                prefix = f"{task.lower()}_order_{order[0]}{order[1]}"
                _render(
                    session,
                    ASSETS / f"{prefix}_after_{position}.jpg",
                    heading=f"{task}: after commit {position} of the same atomic tick",
                    note=(
                        f"agent_{agent_idx} pickup success={bool(result.success)}; "
                        f"held={_held(session.executor)}"
                    ),
                )
        record["final_held_objects"] = _held(session.executor)
        if seed == SEEDS[0]:
            prefix = f"{task.lower()}_order_{order[0]}{order[1]}"
            for agent_idx in (0, 1):
                paths, views = session.render_views(
                    ("agentview_center", "wrist"),
                    agent_id=f"agent_{agent_idx}",
                    out_dir=ASSETS,
                    tag=f"{prefix}_agent_{agent_idx}_after_both",
                )
                record.setdefault("post_pickup_close_views", {})[
                    f"agent_{agent_idx}"
                ] = dict(zip(views, paths))
        record["passed"] = (
            all(call["result"]["success"] for call in record["calls"])
            and record["final_held_objects"]
            == {"agent_0": record["calls"][order.index(0)]["adapted_call"]["args"]["object_id"],
                "agent_1": record["calls"][order.index(1)]["adapted_call"]["args"]["object_id"]}
        )
    except Exception as exc:
        record["passed"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return record


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task, case in CASES.items():
        spec = json.loads(case["spec"].read_text(encoding="utf-8"))
        for seed in SEEDS:
            for order in ((0, 1), (1, 0)):
                records.append(
                    _run_order(
                        task=task,
                        spec=spec,
                        source=case["source"],
                        objects=case["objects"],
                        seed=seed,
                        order=order,
                    )
                )
    payload = {
        "probe_contract": "same_pre_tick_state_both_serial_commit_orders_v1",
        "scene": {**SCENE, "seeds": list(SEEDS)},
        "records": records,
        "summary": {
            "runs": len(records),
            "passed": sum(bool(row.get("passed")) for row in records),
            "all_passed": all(bool(row.get("passed")) for row in records),
        },
    }
    (OUT / "probe_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    if not payload["summary"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
