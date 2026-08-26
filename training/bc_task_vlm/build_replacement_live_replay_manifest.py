"""Verify replacement renders and build a bounded live-sim replay manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_generation.task_level.tasks.shared.validation_contract import (
    require_current_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=2)
    parser.add_argument("--controls-per-task", type=int, default=1)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--environment-seed", type=int, default=42)
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    replay_ids: dict[str, list[str]] = {}
    replacement_sample: dict[str, list[str]] = {}
    control_sample: dict[str, list[str]] = {}
    verified = 0
    failed_steps: list[dict[str, object]] = []
    for task, replacement in sorted(selection.get("replacements", {}).items()):
        added = sorted((replacement or {}).get("added") or [])
        if not added:
            continue
        sampled_replacements = added[: args.per_task]
        selected = sorted(selection.get("tasks", {}).get(task) or [])
        controls = [trajectory_id for trajectory_id in selected if trajectory_id not in added]
        controls.sort(
            key=lambda trajectory_id: hashlib.sha256(
                f"42:{task}:{trajectory_id}".encode("utf-8")
            ).hexdigest()
        )
        sampled_controls = controls[: args.controls_per_task]
        replacement_sample[task] = sampled_replacements
        control_sample[task] = sampled_controls
        replay_ids[task] = sampled_replacements + sampled_controls
        for trajectory_id in added:
            run_dir = args.render_root / task / trajectory_id
            original_path = run_dir / "original_trajectory.json"
            original = json.loads(original_path.read_text(encoding="utf-8"))
            require_current_validation(original, source=str(original_path))
            metadata_path = run_dir / "trajectory_execution_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for step in metadata.get("steps", []):
                if step.get("success") is not True:
                    failed_steps.append(
                        {
                            "task": task,
                            "trajectory_id": trajectory_id,
                            "step_index": step.get("step_index"),
                            "tool": step.get("tool"),
                            "success": step.get("success"),
                        }
                    )
            verified += 1
    if failed_steps:
        raise ValueError(f"Replacement render steps failed: {failed_steps[:10]}")

    payload = {
        "schema_version": 2,
        "dataset_root": str(args.render_root.resolve()),
        "trajectory_ids_by_task": replay_ids,
        "replacement_sample_by_task": replacement_sample,
        "control_sample_by_task": control_sample,
        "num_tasks": len(replay_ids),
        "num_trajectories": sum(map(len, replay_ids.values())),
        "num_replacement_renders_verified": verified,
        "render_step_failures": 0,
        "selection_rule": (
            f"first {args.per_task} sorted replacement IDs plus "
            f"{args.controls_per_task} deterministic unchanged control per affected task"
        ),
        "scene": {
            "layout": args.layout,
            "style": args.style,
            "seed": args.environment_seed,
        },
        "purpose": "transactional replacement render/live-sim agreement regression",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
