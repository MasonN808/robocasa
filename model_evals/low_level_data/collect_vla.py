"""Collect low-level VLA rollout segments from one synthetic trajectory."""

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
    _complete_scene_metadata,
    _execute_synthetic_step,
    _load_adapter_and_client,
    _prepare_held_objects_for_probe,
    _resolve_scene,
    run_vla_probe,
)
from model_evals.low_level.state_snapshot import capture_executor_state, restore_executor_state
from model_evals.low_level.tool_categories import expand_tool_filter, support_tier
from model_evals.low_level.tool_prompts import prompt_for_tool_call
from model_evals.low_level_data.recorder import SegmentWriter
from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["dry", "gwp", "pi05", "rldx", "gr00t_n1_5"], default="dry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
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
    parser.add_argument("--max_steps_per_probe", type=int, default=240)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--cameras", default="default")
    parser.add_argument("--prompt_override", default=None)
    parser.add_argument("--render_width", type=int, default=256)
    parser.add_argument("--render_height", type=int, default=256)
    parser.add_argument("--gl_backend", default="osmesa")
    parser.add_argument("--save_policy", choices=["all", "success_only", "manual_review"], default="success_only")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_success", action="store_true")
    return parser.parse_args()


def _accepted(save_policy: str, success: bool) -> bool:
    if save_policy == "all":
        return True
    if save_policy == "success_only":
        return bool(success)
    return False


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.dry_run:
        args.backend = "dry"

    traj_path = Path(args.trajectory_json)
    trajectory = json.loads(traj_path.read_text())
    task_name = args.task or trajectory.get("composite_task")
    if not task_name:
        raise ValueError("Task name must be provided with --task or trajectory.composite_task")

    selected_tools = expand_tool_filter(args.tools) - set(args.skip_tools or [])
    layout, style, seed, split, scene_metadata = _resolve_scene(args, trajectory)
    adapter, client = _load_adapter_and_client(args.backend, args.host, args.port)
    if adapter is not None and getattr(adapter, "on_episode_start", None) is not None:
        adapter.on_episode_start(task_name, args.scene_episode_index)

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
            logger.info("Collecting %s with prompt: %s", probe_name, prompt)

            held_objects_before_probe = _prepare_held_objects_for_probe(executor)
            snapshot = capture_executor_state(executor)
            try:
                probe_result = run_vla_probe(
                    executor=executor,
                    step=step,
                    prompt=prompt,
                    adapter=adapter,
                    client=client,
                    backend=args.backend,
                    robot_idx=robot_idx,
                    max_steps=args.max_steps_per_probe,
                    replan_steps=args.replan_steps,
                    expected_action_chunk=args.action_chunk,
                    cameras=args.cameras,
                    stop_on_success=args.stop_on_success,
                    record_observations=True,
                )
            except Exception as exc:
                traceback.print_exc()
                probe_result = {
                    "success": False,
                    "predicate": {"success": False, "metrics": {}, "reason": f"probe error: {exc}"},
                    "initial_predicate": {"success": False, "metrics": {}, "reason": f"probe error: {exc}"},
                    "first_success_step": None,
                    "num_env_steps": 0,
                    "actions": np.zeros((0, 12), dtype=np.float32),
                    "frames_by_label": {},
                    "observations": [],
                }
            finally:
                restore_executor_state(executor, snapshot)

            synthetic_success = False
            synthetic_details = {}
            try:
                synthetic_result = _execute_synthetic_step(executor, step, robot_idx)
                synthetic_success = bool(synthetic_result.success)
                synthetic_details = synthetic_result.details
            except Exception as exc:
                synthetic_details = {"error": str(exc), "traceback": traceback.format_exc()}
                logger.exception("Synthetic advancement failed for %s", probe_name)

            success = bool(probe_result["success"])
            accepted = _accepted(args.save_policy, success)
            metadata = {
                "source": "vla",
                "backend": args.backend,
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
                "held_objects_before_probe": held_objects_before_probe,
                "predicate_success": success,
                "predicate": probe_result.get("predicate"),
                "initial_predicate": probe_result.get("initial_predicate"),
                "first_success_step": probe_result.get("first_success_step"),
                "synthetic_advance_success": synthetic_success,
                "synthetic_advance_details": synthetic_details,
                "save_policy": args.save_policy,
            }
            segment_path = None
            if probe_result.get("observations"):
                segment_path = writer.write(
                    metadata=metadata,
                    observations=probe_result["observations"],
                    actions_hdf5=probe_result["actions"],
                    accepted=accepted,
                    segment_name=None,
                    extra={"run_root": str(run_root)},
                )

            result = dict(metadata)
            result.update(
                {
                    "success": success,
                    "accepted": accepted,
                    "segment_path": str(segment_path) if segment_path else None,
                    "num_env_steps": int(probe_result.get("num_env_steps", 0)),
                    "actions_shape": list(np.asarray(probe_result.get("actions", [])).shape),
                }
            )
            results.append(result)
            probed += 1

        summary = {
            "task": task_name,
            "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
            "backend": args.backend,
            "scene": scene_metadata,
            "save_policy": args.save_policy,
            "segments_root": str(segments_root),
            "num_segments": len(results),
            "num_successes": sum(1 for r in results if r["success"]),
            "num_accepted": sum(1 for r in results if r["accepted"]),
            "segments": results,
        }
        (run_root / "collection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2, default=str))
        print(f"Run dir: {run_root}")
        print(f"Segments dir: {segments_root}")
    finally:
        if client is not None:
            client.close()
        executor.close()


if __name__ == "__main__":
    main()
