"""Build live VLA observations from a SimToolExecutor environment."""

from __future__ import annotations

import numpy as np


def _raw_obs(executor) -> dict:
    env = executor.env
    if getattr(env, "viewer_get_obs", False) and getattr(env, "viewer", None) is not None:
        return env.viewer._get_observations(force_update=True)
    return env._get_observations(force_update=True)


def _state_from_raw(raw_obs: dict, robot_idx: int) -> dict[str, np.ndarray]:
    prefix = f"robot{robot_idx}_"
    return {
        "state.gripper_qpos": np.asarray(raw_obs[f"{prefix}gripper_qpos"], dtype=np.float32),
        "state.base_position": np.asarray(raw_obs[f"{prefix}base_pos"], dtype=np.float32),
        "state.base_rotation": np.asarray(raw_obs[f"{prefix}base_quat"], dtype=np.float32),
        "state.end_effector_position_relative": np.asarray(
            raw_obs[f"{prefix}base_to_eef_pos"], dtype=np.float32
        ),
        "state.end_effector_rotation_relative": np.asarray(
            raw_obs[f"{prefix}base_to_eef_quat"], dtype=np.float32
        ),
    }


def build_live_vla_obs(executor, prompt: str, robot_idx: int = 0) -> dict:
    """Return the same observation keys used by RoboCasaGymEnv, from live executor state."""
    raw_obs = _raw_obs(executor)
    invalidate = getattr(executor, "_invalidate_visual_cache", None)
    if callable(invalidate):
        invalidate()

    obs = _state_from_raw(raw_obs, robot_idx)
    obs.update(
        {
            "video.robot0_agentview_left": np.ascontiguousarray(
                executor._render_camera(f"robot{robot_idx}_agentview_left"), dtype=np.uint8
            ),
            "video.robot0_agentview_right": np.ascontiguousarray(
                executor._render_camera(f"robot{robot_idx}_agentview_right"), dtype=np.uint8
            ),
            "video.robot0_eye_in_hand": np.ascontiguousarray(
                executor._render_camera(f"robot{robot_idx}_eye_in_hand"), dtype=np.uint8
            ),
            "annotation.human.task_description": str(prompt),
        }
    )
    return obs


def step_executor_env(executor, action: np.ndarray):
    """Step the raw executor env with a flat RoboCasa action vector."""
    result = executor.env.step(np.asarray(action, dtype=np.float32))
    invalidate = getattr(executor, "_invalidate_visual_cache", None)
    if callable(invalidate):
        invalidate()
    return result
