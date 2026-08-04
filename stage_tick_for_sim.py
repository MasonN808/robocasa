#!/usr/bin/env python
"""Stage generated tick trajectories into the layout live_sim_eval expects.

Generation writes <out>/<task>/trajectories/traj_N.json; the evaluator wants
<root>/<task>/<traj_id>/original_trajectory.json plus a manifest keyed by task.
This only copies and indexes -- it does not alter a single step, so what runs in
the simulator is exactly what the concurrent validator accepted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SRC = Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_enforced")
DST = Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_sim_check")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--dst", type=Path, default=DST)
    args = parser.parse_args()

    if args.dst.exists():
        shutil.rmtree(args.dst)
    root = args.dst / "data"
    ids_by_task: dict[str, list[str]] = {}

    for task_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        for path in sorted((task_dir / "trajectories").glob("*.json")):
            record = json.loads(path.read_text())
            if not (record.get("validation") or {}).get("is_valid"):
                continue  # only what the gate accepted
            traj_id = path.stem
            out = root / task_dir.name / traj_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "original_trajectory.json").write_text(json.dumps(record, indent=2))
            ids_by_task.setdefault(task_dir.name, []).append(traj_id)

    manifest = {
        "split": "tick_enforced",
        "dataset_root": str(root),
        "trajectory_ids_by_task": ids_by_task,
    }
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(len(v) for v in ids_by_task.values())
    print(f"staged {total} valid trajectories across {len(ids_by_task)} tasks")
    for task, ids in ids_by_task.items():
        print(f"  {task}: {len(ids)}")
    print(f"manifest: {args.dst / 'manifest.json'}")
    print(f"dataset-root: {root}")


if __name__ == "__main__":
    main()
