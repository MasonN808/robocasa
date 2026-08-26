#!/usr/bin/env python3
"""Reuse unchanged renders and stage only replacements for simulator replay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, copy_function=os.link)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--prior-render-root", type=Path, required=True)
    parser.add_argument("--output-render-root", type=Path, required=True)
    parser.add_argument("--affected-input-root", type=Path, required=True)
    parser.add_argument("--force-rerender-task", action="append", default=[])
    args = parser.parse_args()

    for target in (args.output_render_root, args.affected_input_root):
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Target must be absent or empty: {target}")
        target.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (args.selection_root / "selection_manifest.json").read_text()
    )
    forced = set(args.force_rerender_task)
    total_reused = 0
    total_affected = 0
    staged: dict[str, list[str]] = {}

    for task, selected_ids in sorted(manifest["tasks"].items()):
        added = set((manifest.get("replacements", {}).get(task) or {}).get("added", []))
        affected = set(selected_ids) if task in forced else added
        reused = set(selected_ids) - affected

        for trajectory_id in sorted(reused):
            link_tree(
                args.prior_render_root / task / trajectory_id,
                args.output_render_root / task / trajectory_id,
            )
        total_reused += len(reused)

        if not affected:
            continue
        source_summary = json.loads(
            (args.processed_root / task / "summary.json").read_text()
        )
        entries = [
            entry
            for entry in source_summary.get("trajectory_files", [])
            if entry["trajectory_id"] in affected
        ]
        if {entry["trajectory_id"] for entry in entries} != affected:
            raise ValueError(f"{task}: affected IDs missing from processed summary")
        task_out = args.affected_input_root / task
        (task_out / "trajectories").mkdir(parents=True)
        for entry in entries:
            shutil.copy2(
                args.processed_root / task / entry["path"],
                task_out / entry["path"],
            )
        source_summary.update(
            num_runs=len(entries),
            num_trajectories=len(entries),
            completed_trajectories=len(entries),
            invalid_trajectories=0,
            trajectory_files=entries,
        )
        (task_out / "summary.json").write_text(
            json.dumps(source_summary, indent=2) + "\n"
        )
        staged[task] = sorted(affected)
        total_affected += len(affected)

    report = {
        "reused": total_reused,
        "affected": total_affected,
        "total": total_reused + total_affected,
        "affected_by_task": staged,
        "force_rerender_tasks": sorted(forced),
    }
    (args.affected_input_root / "incremental_render_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if report["total"] != 5300:
        raise ValueError(f"Expected 5300 selected trajectories, got {report['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
