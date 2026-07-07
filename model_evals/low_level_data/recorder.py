"""Raw segment writer for low-level VLA data collection."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from model_evals.low_level_data.schema import (
    CAMERA_KEYS,
    hdf5_action_to_lerobot,
    state_dict_to_vector,
)


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _next_segment_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in root.glob("segment_*") if path.is_dir())
    next_idx = 0
    if existing:
        next_idx = max(int(path.name.split("_")[-1]) for path in existing) + 1
    return root / f"segment_{next_idx:06d}"


class SegmentWriter:
    """Writes canonical raw segments before model-specific export."""

    def __init__(self, root: str | Path, fps: int = 20):
        self.root = Path(root)
        self.fps = int(fps)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        metadata: dict[str, Any],
        observations: list[dict[str, Any]],
        actions_hdf5: np.ndarray,
        accepted: bool,
        segment_name: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        segment_dir = self.root / segment_name if segment_name else _next_segment_dir(self.root)
        segment_dir.mkdir(parents=True, exist_ok=False)
        arrays_dir = segment_dir / "arrays"
        videos_dir = segment_dir / "videos"
        arrays_dir.mkdir()
        videos_dir.mkdir()

        actions_hdf5 = np.asarray(actions_hdf5, dtype=np.float32).reshape((-1, 12))
        actions_lerobot = hdf5_action_to_lerobot(actions_hdf5)
        states = np.asarray([state_dict_to_vector(obs) for obs in observations], dtype=np.float32)

        np.save(arrays_dir / "actions_hdf5_order.npy", actions_hdf5)
        np.save(arrays_dir / "actions_lerobot_order.npy", actions_lerobot)
        np.save(arrays_dir / "states.npy", states)

        for camera in CAMERA_KEYS:
            obs_key = f"video.{camera}"
            frames = [np.asarray(obs[obs_key], dtype=np.uint8) for obs in observations if obs_key in obs]
            if frames:
                imageio.mimwrite(videos_dir / f"{camera}.mp4", frames, fps=self.fps)
                np.save(arrays_dir / f"{camera}.npy", np.asarray(frames, dtype=np.uint8))

        payload = dict(metadata)
        payload.update(
            {
                "accepted": bool(accepted),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "fps": self.fps,
                "num_observations": len(observations),
                "num_actions": int(actions_hdf5.shape[0]),
                "segment_dir": str(segment_dir),
            }
        )
        if extra:
            payload["extra"] = extra
        (segment_dir / "metadata.json").write_text(json.dumps(payload, indent=2, default=_json_default))
        return segment_dir


def iter_segment_dirs(root: str | Path, accepted_only: bool = True):
    root = Path(root)
    for metadata_path in sorted(root.glob("**/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if accepted_only and not metadata.get("accepted", False):
            continue
        yield metadata_path.parent, metadata

