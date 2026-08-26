"""Inventory supervision affected by canonical cabinet-parent workspaces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    affected = []
    trajectory_count = 0
    for path in sorted(args.dataset_root.glob("**/*.json")):
        try:
            trajectory = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trajectory, dict) or not isinstance(
            trajectory.get("steps"), list
        ):
            continue
        trajectory_count += 1
        fixtures = (trajectory.get("initial_state") or {}).get("fixtures") or {}
        cabinets = {
            fixture_id
            for fixture_id, state in fixtures.items()
            if "cabinet" in str((state or {}).get("fixture_type") or "").lower()
        }
        counts: Counter[str] = Counter()
        for step in trajectory["steps"]:
            tool = step.get("tool")
            step_args = step.get("args") or {}
            if tool == "navigate_to_fixture" and step_args.get("fixture_id") in cabinets:
                counts["cabinet_navigation"] += 1
            elif tool == "give_space" and step_args.get("fixture_id") in cabinets:
                counts["cabinet_give_space"] += 1
            elif tool == "wait_for_signal" and step_args.get("about") in cabinets:
                counts["cabinet_wait"] += 1
        if not counts:
            continue
        task = str(trajectory.get("task") or trajectory.get("composite_task"))
        task_counts[task] += 1
        tool_counts.update(counts)
        affected.append(
            {
                "path": str(path),
                "task": task,
                "counts": dict(counts),
                "recommended_action": (
                    "regenerate" if counts["cabinet_give_space"] or counts["cabinet_wait"]
                    else "safe_navigation_canonicalization"
                ),
            }
        )
    payload = {
        "dataset_root": str(args.dataset_root),
        "trajectory_count": trajectory_count,
        "affected_trajectory_count": len(affected),
        "affected_task_count": len(task_counts),
        "tool_counts": dict(tool_counts),
        "task_counts": dict(sorted(task_counts.items())),
        "trajectories": affected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "trajectories"}, indent=2))


if __name__ == "__main__":
    main()
