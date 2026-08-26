#!/usr/bin/env python3
"""Build comparable tick100 live-sim manifests from the exact SFT split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--known-task-trajectories", type=int, default=10)
    parser.add_argument("--heldout-task-trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    config = json.loads(
        (args.preprocessed_dir / "preprocess_config.json").read_text(encoding="utf-8")
    )
    train_tasks = list(config["train_tasks"])
    val_ids = config["val_trajectory_ids_by_task"]
    dataset_tasks = sorted(
        path.name for path in args.dataset_root.iterdir() if path.is_dir()
    )
    heldout_tasks = sorted(set(dataset_tasks) - set(train_tasks))
    if len(train_tasks) != 47 or len(heldout_tasks) != 6:
        raise ValueError(
            f"Expected 47 train and 6 held-out tasks; got {len(train_tasks)} and {len(heldout_tasks)}"
        )

    rng = random.Random(args.seed)
    known: dict[str, list[str]] = {}
    for task in sorted(train_tasks):
        pool = sorted(val_ids[task])
        if len(pool) < args.known_task_trajectories:
            raise ValueError(f"{task}: only {len(pool)} held-out trajectories")
        # With 100 trajectories/task and a 90/10 training split, this selects
        # the complete validation pool when the requested count is ten.
        known[task] = sorted(rng.sample(pool, args.known_task_trajectories))

    unseen: dict[str, list[str]] = {}
    for task in heldout_tasks:
        pool = sorted(
            path.name
            for path in (args.dataset_root / task).iterdir()
            if path.is_dir() and path.name.startswith("traj_")
        )
        if len(pool) < args.heldout_task_trajectories:
            raise ValueError(f"{task}: only {len(pool)} trajectories")
        unseen[task] = sorted(rng.sample(pool, args.heldout_task_trajectories))

    # Keep the compact format accepted by live_sim_eval and used by tick30.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "eval_manifest_heldout_trajectories.json": known,
        "eval_manifest_heldout_tasks.json": unseen,
    }
    for filename, ids_by_task in payloads.items():
        payload = {"trajectory_ids_by_task": ids_by_task}
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{filename}: tasks={len(ids_by_task)} "
            f"trajectories={sum(map(len, ids_by_task.values()))}"
        )
    print(f"held-out tasks: {heldout_tasks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
