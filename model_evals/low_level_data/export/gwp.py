"""Export raw segments to GWP runtime-schema format.

The local GWP code in this repo is inference-only. This exporter writes the
schema and norm stats observed by that runtime; final training compatibility
still depends on the upstream GWP training script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import imageio
import numpy as np

from model_evals.low_level_data.recorder import iter_segment_dirs


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def export_gwp(segments_dir: str | Path, output_dir: str | Path, *, accepted_only: bool = True) -> Path:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)
    manifest = []
    all_states = []
    all_actions = []
    for episode_index, (segment_dir, metadata) in enumerate(iter_segment_dirs(segments_dir, accepted_only=accepted_only)):
        states = np.load(segment_dir / "arrays" / "states.npy").astype(np.float32)
        actions = np.load(segment_dir / "arrays" / "actions_lerobot_order.npy").astype(np.float32)
        horizon = min(len(states), len(actions))
        if horizon <= 0:
            continue
        ep_dir = output / "episodes" / f"episode_{episode_index:06d}"
        ep_dir.mkdir(parents=True)
        np.save(ep_dir / "observation_state.npy", states[:horizon])
        np.save(ep_dir / "action.npy", actions[:horizon])
        (ep_dir / "prompt.txt").write_text(str(metadata.get("prompt") or ""))
        (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        for src_name, dst_name in (
            ("robot0_agentview_left", "observation_image"),
            ("robot0_eye_in_hand", "observation_wrist_image"),
            ("robot0_agentview_right", "observation_right_image"),
        ):
            src = segment_dir / "videos" / f"{src_name}.mp4"
            if src.exists():
                shutil.copyfile(src, ep_dir / f"{dst_name}.mp4")
            arr = segment_dir / "arrays" / f"{src_name}.npy"
            if arr.exists():
                frames = np.load(arr)[:horizon]
                imageio.mimwrite(ep_dir / f"{dst_name}.mp4", [np.asarray(f, dtype=np.uint8) for f in frames], fps=int(metadata.get("fps", 20)))
        manifest.append(
            {
                "episode_index": episode_index,
                "path": str(ep_dir),
                "length": horizon,
                "prompt": metadata.get("prompt"),
                "source_segment": str(segment_dir),
            }
        )
        all_states.append(states[:horizon])
        all_actions.append(actions[:horizon])
    if not manifest:
        raise RuntimeError(f"No exportable segments found under {segments_dir}")
    norm_stats = {
        "norm_stats": {
            "observation.state": _stats(np.concatenate(all_states, axis=0)),
            "action": _stats(np.concatenate(all_actions, axis=0)),
        }
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "norm_stats_delta.json").write_text(json.dumps(norm_stats, indent=2))
    (output / "README.md").write_text(
        "# GWP Runtime-Schema Export\n\n"
        "This directory matches the observation/action schema used by the local GWP inference server. "
        "The repo currently does not include GWP training code, so validate this package against the upstream trainer before launching fine-tuning.\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_rejected", action="store_true")
    args = parser.parse_args()
    path = export_gwp(args.segments_dir, args.output_dir, accepted_only=not args.include_rejected)
    print(f"Exported GWP runtime-schema dataset: {path}")


if __name__ == "__main__":
    main()
