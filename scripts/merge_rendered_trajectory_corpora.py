#!/usr/bin/env python3
"""Merge rendered trajectory roots with collision-safe sequential IDs.

JSON payloads and image subdirectories are rewritten to the new trajectory ID;
large image files are hard-linked when possible so the merged training root does
not duplicate the rendered corpora on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _json_rewrite(value: Any, old_id: str, new_id: str) -> Any:
    if isinstance(value, dict):
        return {key: _json_rewrite(item, old_id, new_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_rewrite(item, old_id, new_id) for item in value]
    if isinstance(value, str):
        return value.replace(old_id, new_id)
    return value


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_trajectory(source: Path, destination: Path, new_id: str) -> None:
    old_id = source.name
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        rewritten_parts = tuple(new_id if part == old_id else part for part in relative.parts)
        target = destination.joinpath(*rewritten_parts)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = _json_rewrite(payload, old_id, new_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            _link_or_copy(path, target)


def _trajectory_signature(trajectory_dir: Path) -> str:
    payload = json.loads((trajectory_dir / "original_trajectory.json").read_text())
    trajectory_id = str(payload.get("trajectory_id", trajectory_dir.name))
    payload = _json_rewrite(payload, trajectory_id, "<TRAJECTORY_ID>")
    normalized = {
        "task": payload.get("task"),
        "initial_state": payload.get("initial_state"),
        "steps": payload.get("steps"),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_dirs(task_root: Path) -> list[Path]:
    return sorted(
        path
        for path in task_root.iterdir()
        if path.is_dir() and path.name.startswith("traj_")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-per-task", type=int, default=100)
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_tasks = {path.name for path in args.base_root.iterdir() if path.is_dir()}
    supplement_tasks = {
        path.name for path in args.supplement_root.iterdir() if path.is_dir()
    }
    if base_tasks != supplement_tasks:
        raise ValueError(
            f"Task roots differ: base_only={sorted(base_tasks - supplement_tasks)} "
            f"supplement_only={sorted(supplement_tasks - base_tasks)}"
        )

    manifest: dict[str, Any] = {
        "base_root": str(args.base_root),
        "supplement_root": str(args.supplement_root),
        "expected_per_task": args.expected_per_task,
        "tasks": {},
    }
    total = 0
    for task_name in sorted(base_tasks):
        base = _trajectory_dirs(args.base_root / task_name)
        supplement = _trajectory_dirs(args.supplement_root / task_name)
        sources = [("base", path) for path in base] + [
            ("supplement", path) for path in supplement
        ]
        if len(sources) != args.expected_per_task:
            raise ValueError(
                f"{task_name}: expected {args.expected_per_task}, "
                f"found base={len(base)} supplement={len(supplement)}"
            )
        signatures: dict[str, Path] = {}
        for _, path in sources:
            signature = _trajectory_signature(path)
            if signature in signatures:
                raise ValueError(
                    f"Exact duplicate in {task_name}: {signatures[signature]} and {path}"
                )
            signatures[signature] = path

        task_output = args.output_root / task_name
        task_output.mkdir(parents=True)
        records = []
        for index, (source_name, source_path) in enumerate(sources):
            new_id = f"traj_{index:06d}"
            _copy_trajectory(source_path, task_output / new_id, new_id)
            records.append(
                {
                    "trajectory_id": new_id,
                    "source": source_name,
                    "source_path": str(source_path),
                    "signature": _trajectory_signature(task_output / new_id),
                }
            )
        manifest["tasks"][task_name] = {
            "base_count": len(base),
            "supplement_count": len(supplement),
            "total_count": len(records),
            "trajectories": records,
        }
        total += len(records)
        print(f"{task_name}: {len(base)} + {len(supplement)} = {len(records)}")

    manifest["total_trajectories"] = total
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"merged tasks={len(base_tasks)} trajectories={total} -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
