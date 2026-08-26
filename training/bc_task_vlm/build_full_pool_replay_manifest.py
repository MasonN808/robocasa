"""Build a deterministic live-sim oracle manifest containing every trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _configuration_signature(
    trajectory: dict,
    *,
    layout: int,
    style: int,
    environment_seed: int,
) -> str:
    """Hash the model-visible physical configuration, excluding expert actions."""

    payload = {
        "initial_state": trajectory.get("initial_state"),
        "coordinator_id": trajectory.get("coordinator_id"),
        "scene": {
            "layout": layout,
            "style": style,
            "seed": environment_seed,
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deduplicate-configurations", action="store_true")
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--environment-seed", type=int, default=42)
    args = parser.parse_args()
    mapping = {}
    configuration_ledger = {}
    for task_dir in sorted(path for path in args.dataset_root.iterdir() if path.is_dir()):
        paths = sorted(task_dir.glob("traj_*/original_trajectory.json"))
        ids = [path.parent.name for path in paths]
        if args.deduplicate_configurations:
            groups = {}
            for path in paths:
                trajectory = json.loads(path.read_text(encoding="utf-8"))
                signature = _configuration_signature(
                    trajectory,
                    layout=args.layout,
                    style=args.style,
                    environment_seed=args.environment_seed,
                )
                groups.setdefault(signature, []).append(path.parent.name)
            ids = [sorted(groups[signature])[0] for signature in sorted(groups)]
            configuration_ledger[task_dir.name] = [
                {
                    "configuration_signature": signature,
                    "selected_carrier": sorted(groups[signature])[0],
                    "alternate_carriers": sorted(groups[signature])[1:],
                    "num_carriers": len(groups[signature]),
                }
                for signature in sorted(groups)
            ]
        if ids:
            mapping[task_dir.name] = ids
    payload = {
        "schema_version": 2,
        "dataset_root": str(args.dataset_root.resolve()),
        "trajectory_ids_by_task": mapping,
        "num_tasks": len(mapping),
        "num_trajectories": sum(map(len, mapping.values())),
        "configuration_deduplicated": args.deduplicate_configurations,
        "scene": {
            "layout": args.layout,
            "style": args.style,
            "seed": args.environment_seed,
        },
    }
    if args.deduplicate_configurations:
        payload["configuration_ledger"] = configuration_ledger
        payload["num_source_trajectories"] = sum(
            row["num_carriers"]
            for rows in configuration_ledger.values()
            for row in rows
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tasks": payload["num_tasks"], "trajectories": payload["num_trajectories"]}))


if __name__ == "__main__":
    main()
