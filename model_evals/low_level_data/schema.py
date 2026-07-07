"""Shared schema helpers for low-level VLA data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


HDF5_ACTION_KEYS = (
    ("end_effector_position", 0, 3),
    ("end_effector_rotation", 3, 6),
    ("gripper_close", 6, 7),
    ("base_motion", 7, 11),
    ("control_mode", 11, 12),
)

LEROBOT_ACTION_KEYS = (
    ("base_motion", 0, 4),
    ("control_mode", 4, 5),
    ("end_effector_position", 5, 8),
    ("end_effector_rotation", 8, 11),
    ("gripper_close", 11, 12),
)

STATE_KEYS = (
    ("base_position", 0, 3),
    ("base_rotation", 3, 7),
    ("end_effector_position_relative", 7, 10),
    ("end_effector_rotation_relative", 10, 14),
    ("gripper_qpos", 14, 16),
)

STATE_EXPORT_ORDER = (
    "base_position",
    "base_rotation",
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "gripper_qpos",
)

CAMERA_KEYS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def hdf5_action_to_lerobot(action: np.ndarray) -> np.ndarray:
    """Convert RoboCasa env/HDF5 action order to PandaOmron LeRobot order."""
    action = np.asarray(action)
    return np.concatenate(
        [
            action[..., 7:11],
            action[..., 11:12],
            action[..., 0:3],
            action[..., 3:6],
            action[..., 6:7],
        ],
        axis=-1,
    )


def lerobot_action_to_hdf5(action: np.ndarray) -> np.ndarray:
    """Convert PandaOmron LeRobot action order to RoboCasa env/HDF5 order."""
    action = np.asarray(action)
    return np.concatenate(
        [
            action[..., 5:8],
            action[..., 8:11],
            action[..., 11:12],
            action[..., 0:4],
            action[..., 4:5],
        ],
        axis=-1,
    )


def state_dict_to_vector(obs: dict[str, Any]) -> np.ndarray:
    """Flatten canonical `state.*` keys into RoboCasa PandaOmron state order."""
    return np.concatenate(
        [
            np.asarray(obs[f"state.{key}"], dtype=np.float32).reshape(-1)
            for key in STATE_EXPORT_ORDER
        ],
        axis=0,
    )


@dataclass(frozen=True)
class SegmentMetadata:
    source: str
    task: str
    trajectory_id: str
    step_index: int
    tool: str
    agent: str | None
    robot_idx: int
    prompt: str
    backend: str | None = None
    trajectory_path: str | None = None
    scene: dict[str, Any] = field(default_factory=dict)
    accepted_by_operator: bool | None = None
    predicate_success: bool | None = None
    predicate: dict[str, Any] | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "backend": self.backend,
            "task": self.task,
            "trajectory_path": self.trajectory_path,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "tool": self.tool,
            "agent": self.agent,
            "robot_idx": self.robot_idx,
            "prompt": self.prompt,
            "scene": self.scene,
            "accepted_by_operator": self.accepted_by_operator,
            "predicate_success": self.predicate_success,
            "predicate": self.predicate,
            "notes": self.notes,
        }

