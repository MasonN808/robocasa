"""Export raw low-level segments to canonical RoboCasa LeRobot format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import imageio
import numpy as np
import pandas as pd

from model_evals.low_level_data.recorder import iter_segment_dirs
from model_evals.low_level_data.schema import CAMERA_KEYS

MODALITY_TEMPLATE = {
    "state": {
        "base_position": {"original_key": "observation.state", "start": 0, "end": 3},
        "base_rotation": {"original_key": "observation.state", "start": 3, "end": 7},
        "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
        "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
        "gripper_qpos": {"original_key": "observation.state", "start": 14, "end": 16},
    },
    "action": {
        "base_motion": {"original_key": "action", "start": 0, "end": 4},
        "control_mode": {"original_key": "action", "start": 4, "end": 5},
        "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
        "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
        "gripper_close": {"original_key": "action", "start": 11, "end": 12},
    },
    "video": {
        "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
        "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
        "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"},
    },
    "annotation": {
        "human.task_description": {"original_key": "annotation.human.task_description"},
    },
}


def _to_records(values: np.ndarray) -> list[list[float]]:
    return [np.asarray(v).astype(float).tolist() for v in values]


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        values = np.zeros((1, 1), dtype=np.float32)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_video(src_segment: Path, dst_dataset: Path, episode_index: int, camera: str, fps: int, frames_count: int) -> dict:
    original_key = f"observation.images.{camera}"
    dst_dir = dst_dataset / "videos" / "chunk-000" / original_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"episode_{episode_index:06d}.mp4"
    src_mp4 = src_segment / "videos" / f"{camera}.mp4"
    if src_mp4.exists():
        shutil.copyfile(src_mp4, dst_path)
    else:
        arr_path = src_segment / "arrays" / f"{camera}.npy"
        if not arr_path.exists():
            return {"path": None, "shape": [256, 256, 3]}
        frames = np.load(arr_path)
        imageio.mimwrite(dst_path, [np.asarray(frame, dtype=np.uint8) for frame in frames[:frames_count]], fps=fps)
    try:
        frame = imageio.v3.imread(dst_path, index=0)
        shape = list(frame.shape)
    except Exception:
        shape = [256, 256, 3]
    return {"path": str(dst_path), "shape": shape}


def export_lerobot(segments_dir: str | Path, output_dir: str | Path, *, accepted_only: bool = True) -> Path:
    segments = list(iter_segment_dirs(segments_dir, accepted_only=accepted_only))
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    (output / "data" / "chunk-000").mkdir(parents=True)
    (output / "meta").mkdir(parents=True)

    episodes = []
    tasks = []
    all_states = []
    all_actions = []
    fps = 20
    feature_shapes = {camera: [256, 256, 3] for camera in CAMERA_KEYS}

    for episode_index, (segment_dir, metadata) in enumerate(segments):
        states = np.load(segment_dir / "arrays" / "states.npy").astype(np.float32)
        actions = np.load(segment_dir / "arrays" / "actions_lerobot_order.npy").astype(np.float32)
        horizon = min(len(states), len(actions))
        if horizon <= 0:
            continue
        states = states[:horizon]
        actions = actions[:horizon]
        fps = int(metadata.get("fps", fps) or fps)
        prompt = str(metadata.get("prompt") or metadata.get("task") or "")
        task_index = len(tasks)
        tasks.append({"task_index": task_index, "task": prompt})
        timestamps = np.arange(horizon, dtype=np.float32) / float(fps)
        df = pd.DataFrame(
            {
                "observation.state": _to_records(states),
                "action": _to_records(actions),
                "annotation.human.task_description": [task_index] * horizon,
                "episode_index": [episode_index] * horizon,
                "frame_index": list(range(horizon)),
                "timestamp": timestamps.tolist(),
                "next.done": [False] * max(horizon - 1, 0) + [True],
                "task_index": [task_index] * horizon,
            }
        )
        df.to_parquet(output / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet", index=False)
        for camera in CAMERA_KEYS:
            info = _write_video(segment_dir, output, episode_index, camera, fps, horizon)
            feature_shapes[camera] = info["shape"] or feature_shapes[camera]
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt],
                "length": horizon,
                "metadata": metadata,
            }
        )
        all_states.append(states)
        all_actions.append(actions)

    if not episodes:
        raise RuntimeError(f"No exportable segments found under {segments_dir}")

    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [12], "names": ["action"]},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": ["task"]},
    }
    for camera in CAMERA_KEYS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": feature_shapes[camera],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": fps},
        }

    info = {
        "codebase_version": "robocasa_low_level_data",
        "fps": fps,
        "robot_type": "panda_omron",
        "total_episodes": len(episodes),
        "total_frames": sum(ep["length"] for ep in episodes),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (output / "meta" / "episodes.jsonl").open("w") as f:
        for episode in episodes:
            f.write(json.dumps(episode, default=str) + "\n")
    with (output / "meta" / "tasks.jsonl").open("w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    (output / "meta" / "modality.json").write_text(json.dumps(MODALITY_TEMPLATE, indent=2))
    stats = {
        "observation.state": _stats(np.concatenate(all_states, axis=0)),
        "action": _stats(np.concatenate(all_actions, axis=0)),
    }
    (output / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_rejected", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    path = export_lerobot(args.segments_dir, args.output_dir, accepted_only=not args.include_rejected)
    print(f"Exported LeRobot dataset: {path}")


if __name__ == "__main__":
    main()
