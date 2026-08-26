#!/usr/bin/env python3
"""Hardlink a seeded per-task subset of a fully rendered trajectory corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

from data_generation.task_level.tasks.shared.validation_contract import (
    require_current_validation,
)


def task_seed(seed: int, task: str) -> int:
    return seed ^ int.from_bytes(hashlib.sha256(task.encode()).digest()[:8], "big")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-task", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    tasks = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if len(tasks) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {len(tasks)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_root": str(args.input_root),
        "per_task": args.per_task,
        "seed": args.seed,
        "tasks": {},
    }
    for task_dir in tasks:
        candidates = sorted(
            path for path in task_dir.iterdir()
            if path.is_dir() and path.name.startswith("traj_")
        )
        for candidate in candidates:
            source = candidate / "original_trajectory.json"
            require_current_validation(
                json.loads(source.read_text(encoding="utf-8")), source=str(source)
            )
        if len(candidates) != 100:
            raise ValueError(f"{task_dir.name}: expected 100 renders, found {len(candidates)}")
        chosen = sorted(
            random.Random(task_seed(args.seed, task_dir.name)).sample(
                candidates, args.per_task
            ),
            key=lambda path: path.name,
        )
        out_task = args.output_root / task_dir.name
        out_task.mkdir()
        for source in chosen:
            shutil.copytree(source, out_task / source.name, copy_function=os.link)
        manifest["tasks"][task_dir.name] = [path.name for path in chosen]
        print(f"{task_dir.name}: selected {len(chosen)}/100")

    (args.output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    count = sum(len(values) for values in manifest["tasks"].values())
    if count != args.expected_tasks * args.per_task:
        raise ValueError(f"Expected {args.expected_tasks * args.per_task}, got {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
