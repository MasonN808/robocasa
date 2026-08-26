#!/usr/bin/env python3
"""Select a deterministic fixed-size subset from each raw generation task."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.task_level.generation.image.processor import (
    revalidate_tick_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-task", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Replay every candidate through the current canonical concurrent FSM.",
    )
    parser.add_argument(
        "--preserve-selection-root",
        type=Path,
        help="Keep valid IDs from this prior selection and replace only failures.",
    )
    args = parser.parse_args()

    prior_tasks: dict[str, list[str]] = {}
    if args.preserve_selection_root is not None:
        prior_manifest_path = args.preserve_selection_root / "selection_manifest.json"
        prior_manifest = json.loads(prior_manifest_path.read_text())
        prior_tasks = prior_manifest.get("tasks") or {}

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    task_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if len(task_dirs) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {len(task_dirs)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_root": str(args.input_root),
        "per_task": args.per_task,
        "seed": args.seed,
        "revalidated": args.revalidate,
        "tasks": {},
        "validation": {},
        "replacements": {},
    }
    for task_dir in task_dirs:
        summary_path = task_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        files = list(summary.get("trajectory_files") or [])
        input_candidate_count = len(files)
        payloads = {}
        validation_errors: dict[str, dict[str, str | None]] = {}
        if args.revalidate:
            valid_files = []
            for entry in files:
                source_path = task_dir / entry["path"]
                payload = json.loads(source_path.read_text())
                revalidate_tick_trajectory(payload)
                payloads[entry["trajectory_id"]] = payload
                verdict = payload.get("validation") or {}
                if verdict.get("is_valid"):
                    valid_files.append(entry)
                else:
                    validation_errors[entry["trajectory_id"]] = {
                        "error_type": verdict.get("error_type"),
                        "error": verdict.get("error"),
                    }
            files = valid_files
        if len(files) < args.per_task:
            raise ValueError(
                f"{task_dir.name}: only {len(files)} currently valid; "
                f"need {args.per_task}"
            )
        task_seed = args.seed ^ int.from_bytes(
            hashlib.sha256(task_dir.name.encode()).digest()[:8], "big"
        )
        by_id = {entry["trajectory_id"]: entry for entry in files}
        kept_ids = [
            trajectory_id
            for trajectory_id in prior_tasks.get(task_dir.name, [])
            if trajectory_id in by_id
        ]
        if len(kept_ids) > args.per_task:
            kept_ids = kept_ids[:args.per_task]
        candidates = [entry for entry in files if entry["trajectory_id"] not in kept_ids]
        needed = args.per_task - len(kept_ids)
        replacements = random.Random(task_seed).sample(candidates, needed)
        chosen = sorted(
            [by_id[trajectory_id] for trajectory_id in kept_ids] + replacements,
            key=lambda x: x["trajectory_id"],
        )
        out_task = args.output_root / task_dir.name
        out_traj = out_task / "trajectories"
        out_traj.mkdir(parents=True)
        for entry in chosen:
            if args.revalidate:
                (out_task / entry["path"]).write_text(
                    json.dumps(payloads[entry["trajectory_id"]], indent=2) + "\n"
                )
            else:
                shutil.copy2(task_dir / entry["path"], out_task / entry["path"])
        selected_indices = sorted(int(entry["trajectory_id"].rsplit("_", 1)[1]) for entry in chosen)
        summary.update(
            num_runs=args.per_task,
            num_trajectories=args.per_task,
            completed_trajectories=args.per_task,
            invalid_trajectories=0,
            successful_trajectory_fraction=1.0,
            completed_run_indices=selected_indices,
            failed_run_indices=[],
            pending_run_indices=[],
            is_complete=True,
            trajectory_files=chosen,
        )
        summary.pop("cost_summary", None)
        (out_task / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["tasks"][task_dir.name] = [entry["trajectory_id"] for entry in chosen]
        manifest["replacements"][task_dir.name] = {
            "kept": sorted(kept_ids),
            "removed": sorted(
                set(prior_tasks.get(task_dir.name, [])) - set(kept_ids)
            ),
            "added": sorted(entry["trajectory_id"] for entry in replacements),
        }
        manifest["validation"][task_dir.name] = {
            "input_candidates": input_candidate_count,
            "valid_candidates": len(files),
            "invalid_candidates": len(validation_errors),
            "errors": validation_errors,
        }
        print(
            f"{task_dir.name}: selected {args.per_task}/{len(files)} valid "
            f"({len(validation_errors)} rejected by current FSM)"
        )
    (args.output_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
