"""Collect low-level teleop segments from one synthetic trajectory.

This is a line-mode collector: enter short commands per step, then decide save,
discard, retry, or quit for each physical tool segment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import traceback

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from model_evals.low_level.eval_trajectory_tools import (
    _agent_to_robot_idx,
    _close_gripper_action_for_robot,
    _complete_scene_metadata,
    _env_action_for_robot,
    _execute_synthetic_step,
    _prepare_held_objects_for_probe,
    _resolve_scene,
)
from model_evals.low_level.observations import build_live_vla_obs, step_executor_env
from model_evals.low_level.predicates import evaluate_predicate
from model_evals.low_level.state_snapshot import capture_executor_state, restore_executor_state
from model_evals.low_level.tool_categories import expand_tool_filter, support_tier
from model_evals.low_level.tool_prompts import prompt_for_tool_call
from model_evals.low_level_data.recorder import SegmentWriter
from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_json", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scene_sampling", choices=["official", "trajectory"], default="official")
    parser.add_argument("--split", choices=["pretrain", "target", "all"], default="pretrain")
    parser.add_argument("--scene_seed", type=int, default=7)
    parser.add_argument("--scene_episode_index", type=int, default=0)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--segments_dir", default=None, help="Defaults to <log_dir>/segments")
    parser.add_argument("--tools", nargs="+", default=["all_physical"])
    parser.add_argument("--skip_tools", nargs="+", default=[])
    parser.add_argument("--max_segments", type=int, default=None)
    parser.add_argument("--max_steps_per_segment", type=int, default=240)
    parser.add_argument("--prompt_override", default=None)
    parser.add_argument("--render_width", type=int, default=256)
    parser.add_argument("--render_height", type=int, default=256)
    parser.add_argument("--gl_backend", default="osmesa")
    parser.add_argument("--step_scale", type=float, default=0.15)
    return parser.parse_args()


def _command_to_action(command: str, scale: float) -> np.ndarray | None:
    command = command.strip().lower()
    if not command:
        return np.zeros(12, dtype=np.float32)
    if command.startswith("raw "):
        values = [float(v) for v in command.split()[1:]]
        if len(values) != 12:
            raise ValueError("raw action must have 12 values")
        return np.asarray(values, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    for ch in command:
        if ch == "w":
            action[0] += scale
        elif ch == "s":
            action[0] -= scale
        elif ch == "a":
            action[1] += scale
        elif ch == "d":
            action[1] -= scale
        elif ch == "e":
            action[2] += scale
        elif ch == "q":
            action[2] -= scale
        elif ch == "j":
            action[3] += scale
        elif ch == "l":
            action[3] -= scale
        elif ch == "i":
            action[4] += scale
        elif ch == "k":
            action[4] -= scale
        elif ch == "u":
            action[5] += scale
        elif ch == "o":
            action[5] -= scale
        elif ch == "g":
            action[6] = 1.0
        elif ch == "h":
            action[6] = -1.0
        elif ch == "x":
            pass
        else:
            raise ValueError(f"unknown teleop command character: {ch}")
    return action


def _run_operator_segment(executor, step, prompt: str, robot_idx: int, max_steps: int, scale: float):
    observations = []
    actions = []
    print("\nTeleop commands: wasd/qe translate, ijkl/uo rotate, g close, h open, x/no input zero")
    print("Use 'raw <12 floats>' for exact action, 'done' to finish, 'abort' to discard immediately.")
    print(f"Prompt: {prompt}")
    aborted = False
    for t in range(max_steps):
        obs = build_live_vla_obs(executor, prompt, robot_idx=robot_idx)
        predicate = evaluate_predicate(executor, step, robot_idx)
        print(f"step {t:03d} predicate_success={predicate.success} reason={predicate.reason}")
        command = input("teleop> ").strip()
        if command == "done":
            observations.append(obs)
            break
        if command == "abort":
            aborted = True
            observations.append(obs)
            break
        try:
            model_action = _command_to_action(command, scale)
        except Exception as exc:
            print(f"Invalid command: {exc}")
            continue
        env_action = _env_action_for_robot(executor, robot_idx, model_action)
        observations.append(obs)
        step_executor_env(executor, env_action)
        actions.append(model_action)
    final_predicate = evaluate_predicate(executor, step, robot_idx)
    return {
        "aborted": aborted,
        "success": bool(final_predicate.success),
        "predicate": final_predicate.to_dict(),
        "observations": observations,
        "actions": np.asarray(actions, dtype=np.float32),
        "num_env_steps": len(actions),
    }


def _operator_decision() -> str:
    while True:
        decision = input("[s] save, [d] discard, [r] retry, [q] quit > ").strip().lower()
        if decision in {"s", "d", "r", "q"}:
            return decision
        print("Please enter s, d, r, or q.")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    traj_path = Path(args.trajectory_json)
    trajectory = json.loads(traj_path.read_text())
    task_name = args.task or trajectory.get("composite_task")
    if not task_name:
        raise ValueError("Task name must be provided with --task or trajectory.composite_task")

    selected_tools = expand_tool_filter(args.tools) - set(args.skip_tools or [])
    layout, style, seed, split, scene_metadata = _resolve_scene(args, trajectory)
    run_root = Path(args.log_dir) / task_name / trajectory.get("trajectory_id", traj_path.stem) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    segments_root = Path(args.segments_dir) if args.segments_dir else Path(args.log_dir) / "segments"
    run_root.mkdir(parents=True, exist_ok=True)
    writer = SegmentWriter(segments_root)

    executor = SimToolExecutor(
        task_name=task_name,
        robots=args.robots,
        layout=layout,
        style=style,
        seed=seed,
        split=split,
        render_width=args.render_width,
        render_height=args.render_height,
        gl_backend=args.gl_backend,
    )
    scene_metadata = _complete_scene_metadata(executor, scene_metadata, args.scene_episode_index)
    results = []
    probed = 0
    try:
        adapted = TrajectoryAdapter(executor=executor).adapt(trajectory, output_dir=run_root)
        load_result = executor.load_initial_state(adapted.get("initial_state"))
        (run_root / "input_trajectory.json").write_text(json.dumps(trajectory, indent=2))
        (run_root / "adapted_trajectory.json").write_text(json.dumps(adapted, indent=2, default=str))
        (run_root / "initial_state_load.json").write_text(json.dumps(load_result, indent=2, default=str))
        (run_root / "scene_metadata.json").write_text(json.dumps(scene_metadata, indent=2, default=str))

        original_steps = trajectory.get("steps", [])
        adapted_steps = adapted.get("tool_calls", [])
        for step_pos, step in enumerate(adapted_steps):
            original_step = original_steps[step_pos] if step_pos < len(original_steps) else step
            tool = step.get("tool")
            if tool not in selected_tools:
                continue
            if args.max_segments is not None and probed >= args.max_segments:
                break
            step_index = int(original_step.get("step", step_pos))
            robot_idx = int(step.get("robot_idx", _agent_to_robot_idx(original_step.get("agent", "agent_0"))))
            prompt = args.prompt_override or prompt_for_tool_call(original_step)
            agent_label = original_step.get("agent", f"agent_{robot_idx}")
            probe_name = f"step_{step_index:03d}_{tool}_{agent_label}"

            while True:
                _prepare_held_objects_for_probe(executor)
                snapshot = capture_executor_state(executor)
                try:
                    result = _run_operator_segment(
                        executor,
                        step,
                        prompt,
                        robot_idx,
                        max_steps=args.max_steps_per_segment,
                        scale=args.step_scale,
                    )
                    decision = "d" if result["aborted"] else _operator_decision()
                finally:
                    restore_executor_state(executor, snapshot)
                if decision == "r":
                    continue
                if decision == "q":
                    raise KeyboardInterrupt
                accepted = decision == "s"
                metadata = {
                    "source": "teleop",
                    "backend": "human",
                    "task": task_name,
                    "trajectory_path": str(traj_path),
                    "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
                    "step_index": step_index,
                    "tool": tool,
                    "support_tier": support_tier(tool),
                    "agent": original_step.get("agent"),
                    "robot_idx": robot_idx,
                    "prompt": prompt,
                    "scene": scene_metadata,
                    "args": original_step.get("args") or {},
                    "adapted_args": step.get("args") or {},
                    "accepted_by_operator": accepted,
                    "predicate_success": bool(result["success"]),
                    "predicate": result["predicate"],
                }
                segment_path = None
                if result["observations"]:
                    segment_path = writer.write(
                        metadata=metadata,
                        observations=result["observations"],
                        actions_hdf5=result["actions"],
                        accepted=accepted,
                        extra={"run_root": str(run_root), "probe_name": probe_name},
                    )
                metadata.update(
                    {
                        "accepted": accepted,
                        "success": bool(result["success"]),
                        "segment_path": str(segment_path) if segment_path else None,
                        "num_env_steps": int(result["num_env_steps"]),
                    }
                )
                results.append(metadata)
                break

            try:
                synthetic_result = _execute_synthetic_step(executor, step, robot_idx)
                if not synthetic_result.success:
                    logger.warning("Synthetic advance reported failure for %s", probe_name)
            except Exception:
                logger.exception("Synthetic advancement failed for %s", probe_name)
            probed += 1

        summary = {
            "task": task_name,
            "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
            "scene": scene_metadata,
            "segments_root": str(segments_root),
            "num_segments": len(results),
            "num_accepted": sum(1 for r in results if r["accepted"]),
            "segments": results,
        }
        (run_root / "teleop_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2))
        print(f"Run dir: {run_root}")
        print(f"Segments dir: {segments_root}")
    finally:
        executor.close()


if __name__ == "__main__":
    main()
