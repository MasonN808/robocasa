#!/usr/bin/env python
"""Builds a trajectory-count-capped training root from a larger raw dataset.

Used to match a new data-generation source's training-set scale to v1's
actual training data (robocasa_local_train_subset_21: 47 tasks x 21
traj/task), so a v1-vs-v1.5 ablation on the new source isn't confounded by
having a larger training pool. Only emits the *train* tasks (the held-out
tasks are never used for training); trajectories are selected by sorted
`traj_XXXXXX` directory name (the first N), which is deterministic and
independent of any download/staging order.

Output entries are symlinks to the source directories (not copies), so this
is cheap regardless of dataset size.

Usage:
  python cap_trajectories_per_task.py \
    --source-root <raw dataset root> \
    --output-root <capped training root> \
    --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
    --trajectories-per-task 21
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trajectories-per-task", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    train_tasks: list[str] = selection["train_tasks"]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for task_name in train_tasks:
        source_task_dir = args.source_root / task_name
        if not source_task_dir.is_dir():
            raise SystemExit(f"Missing task directory in source: {source_task_dir}")

        trajectory_dirs = sorted(
            p for p in source_task_dir.iterdir() if p.is_dir() and p.name.startswith("traj_")
        )
        if len(trajectory_dirs) < args.trajectories_per_task:
            raise SystemExit(
                f"{task_name} has only {len(trajectory_dirs)} trajectories, "
                f"need {args.trajectories_per_task}"
            )
        selected = trajectory_dirs[: args.trajectories_per_task]

        output_task_dir = args.output_root / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        for trajectory_dir in selected:
            link_path = output_task_dir / trajectory_dir.name
            if link_path.exists() or link_path.is_symlink():
                continue
            link_path.symlink_to(trajectory_dir.resolve())

        print(f"{task_name}: linked {len(selected)}/{len(trajectory_dirs)} trajectories")

    print(f"Done. Capped training root at {args.output_root}")


if __name__ == "__main__":
    main()
