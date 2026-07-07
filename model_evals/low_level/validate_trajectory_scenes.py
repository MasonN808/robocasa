"""Validate one symbolic trajectory across multiple RoboCasa scenes.

This is a dry synthetic validation utility: it does not call a VLA backend. For
each requested scene it creates a SimToolExecutor, adapts the symbolic
trajectory, loads the symbolic initial state, and optionally executes the full
adapted synthetic tool sequence.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys
import traceback
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_json", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Explicit scenes as layout:style:seed, e.g. 11:34:42 27:34:43.",
    )
    parser.add_argument("--layouts", type=int, nargs="+", default=[11, 27, 35])
    parser.add_argument("--styles", type=int, nargs="+", default=[34])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--execute", action="store_true", help="Execute the full adapted synthetic tool sequence after loading initial state.")
    parser.add_argument("--stop_on_first_failure", action="store_true")
    parser.add_argument("--render_width", type=int, default=128)
    parser.add_argument("--render_height", type=int, default=128)
    parser.add_argument("--gl_backend", default="osmesa")
    return parser.parse_args()


def _parse_scene(raw: str) -> tuple[int, int, int]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Scene must be layout:style:seed, got {raw!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _scene_list(args) -> list[tuple[int, int, int]]:
    if args.scenes:
        scenes = [_parse_scene(raw) for raw in args.scenes]
    else:
        scenes = list(itertools.product(args.layouts, args.styles, args.seeds))
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    return scenes


def _agent_to_robot_idx(agent) -> int:
    text = str(agent or "agent_0")
    if text.startswith("agent_"):
        return int(text.replace("agent_", ""))
    return int(text)


def _execute_tool_sequence(executor: SimToolExecutor, adapted: dict[str, Any], original_steps: list[dict[str, Any]]) -> dict[str, Any]:
    executed = []
    for step_pos, step in enumerate(adapted.get("tool_calls", [])):
        original_step = original_steps[step_pos] if step_pos < len(original_steps) else step
        tool = step.get("tool")
        robot_idx = int(step.get("robot_idx", _agent_to_robot_idx(original_step.get("agent", "agent_0"))))
        args = dict(step.get("args") or {})
        try:
            result = executor.execute(tool, robot_idx=robot_idx, **args)
            record = {
                "step_pos": step_pos,
                "step_index": original_step.get("step", step_pos),
                "tool": tool,
                "robot_idx": robot_idx,
                "success": bool(result.success),
                "details": result.details,
            }
        except Exception as exc:
            record = {
                "step_pos": step_pos,
                "step_index": original_step.get("step", step_pos),
                "tool": tool,
                "robot_idx": robot_idx,
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        executed.append(record)
        if not record["success"]:
            return {
                "executed_steps": executed,
                "execution_success": False,
                "first_failure": record,
            }
    return {"executed_steps": executed, "execution_success": True, "first_failure": None}


def _validate_scene(
    trajectory: dict[str, Any],
    task_name: str,
    scene: tuple[int, int, int],
    scene_dir: Path,
    args,
) -> dict[str, Any]:
    layout, style, seed = scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task": task_name,
        "trajectory_id": trajectory.get("trajectory_id"),
        "layout": layout,
        "style": style,
        "seed": seed,
        "adapt_success": False,
        "load_success": False,
        "execution_success": None,
    }
    executor = None
    try:
        executor = SimToolExecutor(
            task_name=task_name,
            robots=args.robots,
            layout=layout,
            style=style,
            seed=seed,
            render_width=args.render_width,
            render_height=args.render_height,
            gl_backend=args.gl_backend,
        )
        adapter = TrajectoryAdapter(executor=executor)
        adapted = adapter.adapt(trajectory, output_dir=scene_dir)
        result["adapt_success"] = True
        result["num_tool_calls"] = len(adapted.get("tool_calls", []))
        result["resolution_log"] = adapted.get("resolution_log", [])
        (scene_dir / "adapted_trajectory.json").write_text(json.dumps(adapted, indent=2, default=str))

        load_result = executor.load_initial_state(adapted.get("initial_state"))
        result["load_success"] = True
        result["initial_state_load"] = load_result
        (scene_dir / "initial_state_load.json").write_text(json.dumps(load_result, indent=2, default=str))

        if args.execute:
            execution = _execute_tool_sequence(executor, adapted, trajectory.get("steps", []))
            result.update(execution)
        return result
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result
    finally:
        if executor is not None:
            executor.close()


def main():
    args = parse_args()
    traj_path = Path(args.trajectory_json)
    trajectory = json.loads(traj_path.read_text())
    task_name = args.task or trajectory.get("composite_task")
    if not task_name:
        raise ValueError("Task name must be provided with --task or trajectory.composite_task")

    scenes = _scene_list(args)
    run_dir = Path(args.log_dir) / task_name / trajectory.get("trajectory_id", traj_path.stem) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (run_dir / "scenes.json").write_text(json.dumps([{"layout": l, "style": s, "seed": sd} for l, s, sd in scenes], indent=2))

    results = []
    for layout, style, seed in scenes:
        label = f"L{layout}_S{style}_sd{seed}"
        scene_result = _validate_scene(trajectory, task_name, (layout, style, seed), run_dir / label, args)
        results.append(scene_result)
        print(json.dumps({k: v for k, v in scene_result.items() if k not in {"resolution_log", "executed_steps", "traceback", "initial_state_load"}}, indent=2))
        if args.stop_on_first_failure and not (
            scene_result.get("adapt_success")
            and scene_result.get("load_success")
            and (scene_result.get("execution_success") in {True, None})
        ):
            break

    summary = {
        "task": task_name,
        "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
        "trajectory_path": str(traj_path),
        "execute": args.execute,
        "num_scenes": len(results),
        "num_adapt_success": sum(1 for r in results if r.get("adapt_success")),
        "num_load_success": sum(1 for r in results if r.get("load_success")),
        "num_execution_success": sum(1 for r in results if r.get("execution_success") is True),
        "results": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Log dir: {run_dir}")


if __name__ == "__main__":
    main()
