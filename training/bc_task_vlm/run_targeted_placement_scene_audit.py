"""Run one indexed scene from the targeted two-task placement regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MODES = {
    "initialization": (
        ROOT / "training/bc_task_vlm/eval_runs/certified_scene_compatibility_v1",
        ROOT / "training/bc_task_vlm/eval_runs/placement_center_targeted_initialization_v1",
        ROOT / "training/bc_task_vlm/audit_certified_scene_compatibility.py",
    ),
    "workspace": (
        ROOT / "training/bc_task_vlm/eval_runs/task_scene_shared_workspace_v1",
        ROOT / "training/bc_task_vlm/eval_runs/placement_center_targeted_workspace_v1",
        ROOT / "training/bc_task_vlm/audit_task_scene_shared_workspaces.py",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source_dir, output_dir, audit_script = MODES[args.mode]
    if args.output_dir is not None:
        output_dir = args.output_dir
    sources = sorted(source_dir.glob("*.json"))
    if not 0 <= args.index < len(sources):
        raise IndexError(f"index {args.index} outside 0..{len(sources)-1}")
    payload = json.loads(sources[args.index].read_text())
    scene = payload["scene"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / sources[args.index].name
    command = [
        sys.executable, str(audit_script),
        "--layout", str(scene["layout"]),
        "--style", str(scene["style"]),
        "--seed", str(scene["seed"]),
        "--task", "GarnishCake",
        "--task", "MeatSkewerAssembly",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
