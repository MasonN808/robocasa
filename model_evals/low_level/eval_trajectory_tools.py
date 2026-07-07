"""Probe pretrained VLA backends on physical tool calls inside synthetic trajectories."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import traceback

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio
import numpy as np

from model_evals.common.cameras import resolve_record_cameras, capture_frame
from model_evals.common.robocasa_eval_client import PolicyAdapter
from model_evals.common.torch_zmq_client import TorchZmqPolicyClient
from model_evals.common.websocket import WebsocketClient
from model_evals.common.zmq_client import ZmqPolicyClient
from model_evals.low_level.observations import build_live_vla_obs, step_executor_env
from model_evals.low_level.predicates import evaluate_predicate
from model_evals.low_level.state_snapshot import capture_executor_state, restore_executor_state
from model_evals.low_level.tool_categories import expand_tool_filter, support_tier
from model_evals.low_level.tool_prompts import prompt_for_tool_call
from robocasa.utils.env_utils import convert_action
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
    parser.add_argument("--scene_sampling", choices=["official", "trajectory"], default="official", help="Scene selection mode. official samples layout/style like RoboCasa eval split resets; trajectory uses trajectory scene_parameters or fallbacks.")
    parser.add_argument("--split", choices=["pretrain", "target", "all"], default="pretrain", help="Scene split for --scene_sampling official.")
    parser.add_argument("--scene_seed", type=int, default=7, help="RNG seed for official-style scene sampling, matching the official eval default.")
    parser.add_argument("--scene_episode_index", type=int, default=0, help="Per-task episode/reset index for official-style scene sampling.")
    parser.add_argument("--log_dir", required=True)
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
    parser.add_argument("--dry_run", action="store_true", help="Alias for --backend dry.")
    parser.add_argument("--stop_on_success", action="store_true", help="Stop a probe as soon as its predicate becomes true. By default, probes run to the full horizon and success is checked at the end.")
    return parser.parse_args()


def _resolve_scene_param(payload: dict, explicit, name: str, fallback: int):
    if explicit is not None:
        return explicit
    scene_params = payload.get("scene_parameters") if isinstance(payload, dict) else None
    if isinstance(scene_params, dict) and scene_params.get(name) is not None:
        return int(scene_params[name])
    if isinstance(payload, dict) and payload.get(name) is not None:
        return int(payload[name])
    return fallback


def _resolve_scene(args, trajectory: dict) -> tuple[int | None, int | None, int, str | None, dict]:
    if args.layout is not None and args.style is not None:
        seed = args.seed if args.seed is not None else _resolve_scene_param(trajectory, None, "seed", args.scene_seed)
        metadata = {
            "sampling": "explicit",
            "layout": int(args.layout),
            "style": int(args.style),
            "seed": int(seed),
        }
        return int(args.layout), int(args.style), int(seed), None, metadata

    if args.scene_sampling == "official":
        seed = args.seed if args.seed is not None else args.scene_seed
        metadata = {
            "sampling": "official_persistent_reset",
            "split": args.split,
            "scene_seed": int(args.scene_seed),
            "episode_index": int(args.scene_episode_index),
            "seed": int(seed),
        }
        return args.layout, args.style, int(seed), args.split, metadata

    layout = _resolve_scene_param(trajectory, args.layout, "layout", 11)
    style = _resolve_scene_param(trajectory, args.style, "style", 34)
    seed = _resolve_scene_param(trajectory, args.seed, "seed", 42)
    metadata = {
        "sampling": "trajectory",
        "layout": int(layout),
        "style": int(style),
        "seed": int(seed),
    }
    return int(layout), int(style), int(seed), None, metadata


def _complete_scene_metadata(executor, metadata: dict, episode_index: int | None = None) -> dict:
    completed = dict(metadata)
    completed["layout"] = int(getattr(executor.env, "layout_id"))
    completed["style"] = int(getattr(executor.env, "style_id"))
    completed["seed"] = int(getattr(executor.env, "seed", completed.get("seed", 0)))
    if episode_index is not None:
        completed["episode_index"] = int(episode_index)
    return completed

def _agent_to_robot_idx(agent) -> int:
    text = str(agent or "agent_0")
    if text.startswith("agent_"):
        return int(text.replace("agent_", ""))
    return int(text)


def _load_adapter_and_client(backend: str, host: str, port: int | None):
    if backend == "dry":
        return None, None
    if backend == "pi05":
        from model_evals.backends.pi05.inference_client import PI05_ADAPTER

        return PI05_ADAPTER, WebsocketClient(host, port or 8000)
    if backend == "gwp":
        from model_evals.backends.gwp.inference_client import GWP_ADAPTER

        return GWP_ADAPTER, WebsocketClient(host, port or 16055)
    if backend == "rldx":
        from model_evals.backends.rldx.inference_client import RLDX_ADAPTER

        return RLDX_ADAPTER, ZmqPolicyClient(host, port or 5555)
    if backend == "gr00t_n1_5":
        from model_evals.backends.gr00t_n1_5.inference_client import GR00T_N15_ADAPTER

        return GR00T_N15_ADAPTER, TorchZmqPolicyClient(host, port or 5556)
    raise ValueError(f"Unsupported backend {backend}")


def _action_dict_from_adapter_action(action: np.ndarray) -> dict:
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

    action_dict = convert_action(np.asarray(action, dtype=np.float32))
    return PandaOmronKeyConverter.unmap_action(action_dict)


def _env_action_for_robot(executor, robot_idx: int, model_action: np.ndarray) -> np.ndarray:
    """Map a model-standard robot0 action onto the selected raw robosuite robot slice."""
    raw_model_action = _action_dict_from_adapter_action(model_action)
    low, _high = executor.env.action_spec
    env_action = np.zeros_like(low, dtype=np.float32)

    offset = 0
    for idx, robot in enumerate(executor.env.robots):
        cc = robot.composite_controller
        robot_dim = int(cc.action_limits[0].shape[0])
        if idx != robot_idx:
            offset += robot_dim
            continue
        local_action = np.zeros(robot_dim, dtype=np.float32)
        for part_name, _controller in cc.part_controllers.items():
            start_idx, end_idx = cc._action_split_indexes[part_name]
            source_key = f"robot0_{part_name}"
            value = raw_model_action.get(source_key)
            if value is None:
                value = np.zeros(end_idx - start_idx, dtype=np.float32)
            local_action[start_idx:end_idx] = np.asarray(value, dtype=np.float32)
        if f"robot0_base_mode" in raw_model_action:
            local_action[-1] = float(np.asarray(raw_model_action["robot0_base_mode"]).reshape(-1)[0])
        env_action[offset : offset + robot_dim] = local_action
        break
    return env_action


def _close_gripper_action_for_robot(executor, robot_idx: int) -> np.ndarray:
    low, _high = executor.env.action_spec
    env_action = np.zeros_like(low, dtype=np.float32)
    offset = 0
    for idx, robot in enumerate(executor.env.robots):
        robot_dim = int(robot.composite_controller.action_limits[0].shape[0])
        if idx == robot_idx:
            action_dict = {"right_gripper": np.array([1.0], dtype=np.float32)}
            env_action[offset : offset + robot_dim] = robot.create_action_vector(action_dict)
            break
        offset += robot_dim
    return env_action


def _prepare_held_objects_for_probe(executor, settle_steps: int = 8) -> dict:
    """Stabilize synthetic-held objects before handing control to a VLA."""
    held_objects = dict(getattr(executor, "_held_objects", {}) or {})
    if not held_objects:
        return {}

    for robot_idx in held_objects:
        executor._sync_held_object(int(robot_idx))
        close_action = _close_gripper_action_for_robot(executor, int(robot_idx))
        for _ in range(settle_steps):
            step_executor_env(executor, close_action)
            executor._sync_held_object(int(robot_idx))
    executor.env.sim.forward()
    return {f"robot{idx}": object_id for idx, object_id in sorted(held_objects.items())}


def _first_available_sim_camera(env, names: list[str]):
    model = getattr(getattr(env, "sim", None), "model", None)
    available = set(getattr(model, "camera_names", ()) or ())
    for name in names:
        if name in available:
            return ("sim", name, name)
    return None


def _resolve_probe_cameras(executor, obs: dict, cameras: str, acting_robot_idx: int):
    if cameras not in {"", "default"}:
        return resolve_record_cameras(executor.env, obs, cameras)

    other_robot_idx = 1 - acting_robot_idx if len(executor.env.robots) > 1 else None
    entries = []
    if other_robot_idx is not None:
        other_default = _first_available_sim_camera(
            executor.env,
            [
                f"robot{other_robot_idx}_agentview_center",
                f"robot{other_robot_idx}_robotview",
                f"robot{other_robot_idx}_agentview_left",
                f"robot{other_robot_idx}_agentview_right",
            ],
        )
        if other_default is not None:
            entries.append(other_default)

    acting_default = _first_available_sim_camera(
        executor.env,
        [
            f"robot{acting_robot_idx}_agentview_center",
            f"robot{acting_robot_idx}_robotview",
            f"robot{acting_robot_idx}_agentview_left",
            f"robot{acting_robot_idx}_agentview_right",
        ],
    )
    acting_wrist = _first_available_sim_camera(
        executor.env,
        [f"robot{acting_robot_idx}_eye_in_hand"],
    )
    for entry in (acting_default, acting_wrist):
        if entry is not None:
            entries.append(entry)

    seen = set()
    unique = []
    for entry in entries:
        if entry[2] in seen:
            continue
        seen.add(entry[2])
        unique.append(entry)
    return unique or resolve_record_cameras(executor.env, obs, cameras)


def _record_frame(executor, obs: dict, entries, frames_by_label):
    # capture_frame expects a Gym wrapper with .env for sim cameras. A tiny shim
    # keeps the existing utility reusable for executor.env.
    class _Shim:
        def __init__(self, env):
            self.env = env

        def render(self):
            return next(iter(executor.render().values()))

    shim = _Shim(executor.env)
    for entry in entries:
        frame = capture_frame(shim, obs, entry)
        if frame is not None:
            frames_by_label[entry[2]].append(frame)


def run_vla_probe(
    executor,
    step: dict,
    prompt: str,
    adapter: PolicyAdapter,
    client,
    backend: str,
    robot_idx: int,
    max_steps: int,
    replan_steps: int,
    expected_action_chunk: int | None,
    cameras: str,
    stop_on_success: bool = False,
    record_observations: bool = False,
):
    obs = build_live_vla_obs(executor, prompt, robot_idx=robot_idx)
    camera_entries = _resolve_probe_cameras(executor, obs, cameras, robot_idx)
    frames_by_label = {label: [] for _kind, _key, label in camera_entries}
    observations = [obs] if record_observations else []
    actions = []
    action_plan = deque()
    chunk_checked = False

    predicate = evaluate_predicate(executor, step, robot_idx)
    initial_predicate = predicate
    first_success_step = 0 if predicate.success else None
    _record_frame(executor, obs, camera_entries, frames_by_label)
    if predicate.success and stop_on_success:
        return {
            "success": True,
            "predicate": predicate.to_dict(),
            "initial_predicate": initial_predicate.to_dict(),
            "first_success_step": first_success_step,
            "num_env_steps": 0,
            "actions": np.zeros((0, 12), dtype=np.float32),
            "frames_by_label": frames_by_label,
            "observations": observations,
        }

    if backend == "dry":
        return {
            "success": bool(predicate.success),
            "predicate": predicate.to_dict(),
            "initial_predicate": initial_predicate.to_dict(),
            "first_success_step": first_success_step,
            "num_env_steps": 0,
            "actions": np.zeros((0, 12), dtype=np.float32),
            "frames_by_label": frames_by_label,
            "observations": observations,
        }

    for t in range(max_steps):
        if not action_plan:
            request = adapter.build_request(obs, prompt)
            result = client.infer(request)
            action_chunk = adapter.extract_actions(result)
            if expected_action_chunk is not None and not chunk_checked:
                if len(action_chunk) != expected_action_chunk:
                    raise RuntimeError(
                        f"Server returned {len(action_chunk)} actions; expected {expected_action_chunk}"
                    )
                chunk_checked = True
            if len(action_chunk) < replan_steps:
                raise RuntimeError(
                    f"Need {replan_steps} replan steps but backend returned {len(action_chunk)} actions"
                )
            action_plan.extend(action_chunk[:replan_steps])

        model_action = np.asarray(action_plan.popleft(), dtype=np.float32)
        post_action = adapter.postprocess_action(model_action)
        env_action = _env_action_for_robot(executor, robot_idx, post_action)
        step_executor_env(executor, env_action)
        actions.append(post_action)
        obs = build_live_vla_obs(executor, prompt, robot_idx=robot_idx)
        if record_observations:
            observations.append(obs)

        if t % 2 == 0:
            _record_frame(executor, obs, camera_entries, frames_by_label)

        predicate = evaluate_predicate(executor, step, robot_idx)
        if predicate.success and first_success_step is None:
            first_success_step = t + 1
        if predicate.success and stop_on_success:
            _record_frame(executor, obs, camera_entries, frames_by_label)
            break

    final_predicate = evaluate_predicate(executor, step, robot_idx)
    _record_frame(executor, obs, camera_entries, frames_by_label)
    return {
        "success": bool(final_predicate.success),
        "predicate": final_predicate.to_dict(),
        "initial_predicate": initial_predicate.to_dict(),
        "first_success_step": first_success_step,
        "num_env_steps": len(actions),
        "actions": np.asarray(actions, dtype=np.float32),
        "frames_by_label": frames_by_label,
        "observations": observations,
    }


def _write_probe_outputs(probe_dir: Path, result: dict, stats: dict):
    probe_dir.mkdir(parents=True, exist_ok=True)
    np.save(probe_dir / "actions.npy", result["actions"])
    for label, frames in result["frames_by_label"].items():
        if not frames:
            continue
        imageio.mimwrite(probe_dir / f"video_{label}.mp4", [np.asarray(f) for f in frames], fps=20)
    with open(probe_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def _execute_synthetic_step(executor, step: dict, robot_idx: int):
    tool = step["tool"]
    args = dict(step.get("args") or {})
    return executor.execute(tool, robot_idx=robot_idx, **args)


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
        adapter.on_episode_start(task_name, 0)

    log_root = Path(args.log_dir) / task_name / trajectory.get("trajectory_id", traj_path.stem) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_root.mkdir(parents=True, exist_ok=True)

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
    layout = scene_metadata["layout"]
    style = scene_metadata["style"]
    seed = scene_metadata["seed"]

    results = []
    probed = 0
    try:
        adapted = TrajectoryAdapter(executor=executor).adapt(trajectory, output_dir=log_root)
        load_result = executor.load_initial_state(adapted.get("initial_state"))
        (log_root / "input_trajectory.json").write_text(json.dumps(trajectory, indent=2))
        (log_root / "adapted_trajectory.json").write_text(json.dumps(adapted, indent=2, default=str))
        (log_root / "initial_state_load.json").write_text(json.dumps(load_result, indent=2, default=str))
        (log_root / "scene_metadata.json").write_text(json.dumps(scene_metadata, indent=2, default=str))

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
            probe_dir = log_root / probe_name

            logger.info("Probing %s with prompt: %s", probe_name, prompt)
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
                synthetic_details = {"error": str(exc)}
                logger.exception("Synthetic advancement failed for %s", probe_name)

            stats = {
                "task": task_name,
                "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
                "trajectory_path": str(traj_path),
                "scene": scene_metadata,
                "layout": layout,
                "style": style,
                "seed": seed,
                "step_index": step_index,
                "agent": original_step.get("agent"),
                "robot_idx": robot_idx,
                "tool": tool,
                "support_tier": support_tier(tool),
                "args": original_step.get("args") or {},
                "adapted_args": step.get("args") or {},
                "prompt": prompt,
                "backend": args.backend,
                "max_steps_per_probe": args.max_steps_per_probe,
                "replan_steps": args.replan_steps,
                "held_objects_before_probe": held_objects_before_probe,
                "success": bool(probe_result["success"]),
                "predicate": probe_result["predicate"],
                "initial_predicate": probe_result.get("initial_predicate"),
                "first_success_step": probe_result.get("first_success_step"),
                "stop_on_success": args.stop_on_success,
                "num_env_steps": int(probe_result["num_env_steps"]),
                "actions_shape": list(np.asarray(probe_result["actions"]).shape),
                "synthetic_advance_success": synthetic_success,
                "synthetic_advance_details": synthetic_details,
            }
            _write_probe_outputs(probe_dir, probe_result, stats)
            results.append(stats)
            probed += 1

        summary = {
            "task": task_name,
            "trajectory_id": trajectory.get("trajectory_id", traj_path.stem),
            "backend": args.backend,
            "scene": scene_metadata,
            "layout": layout,
            "style": style,
            "seed": seed,
            "num_segments": len(results),
            "num_successes": sum(1 for r in results if r["success"]),
            "success_rate": (sum(1 for r in results if r["success"]) / len(results)) if results else 0.0,
            "segments": results,
        }
        (log_root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "segments"}, indent=2))
        print(f"Log dir: {log_root}")
    finally:
        if client is not None:
            client.close()
        executor.close()


if __name__ == "__main__":
    main()
