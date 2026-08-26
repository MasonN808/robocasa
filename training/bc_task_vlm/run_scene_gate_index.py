"""Run one indexed certified-scene audit shard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.task_level.scene_sampling import candidate_scenes_v1


MODES = {
    "focused_behavior": (
        ROOT / "training/bc_task_vlm/audit_repaired_task_behaviors.py",
        ROOT / "training/bc_task_vlm/eval_runs/repaired_task_behavior_all_scenes_v1",
        (),
    ),
    "focused_initialization": (
        ROOT / "training/bc_task_vlm/audit_certified_scene_compatibility.py",
        ROOT / "training/bc_task_vlm/eval_runs/repaired_tasks_initialization_all_scenes_v1",
        ("--task", "SetupBowls", "--task", "AlcoholServingPrep", "--task", "PrepareCoffee"),
    ),
    "focused_workspace": (
        ROOT / "training/bc_task_vlm/audit_task_scene_shared_workspaces.py",
        ROOT / "training/bc_task_vlm/eval_runs/repaired_tasks_workspace_all_scenes_v1",
        ("--task", "SetupBowls", "--task", "AlcoholServingPrep", "--task", "PrepareCoffee"),
    ),
    "full_initialization": (
        ROOT / "training/bc_task_vlm/audit_certified_scene_compatibility.py",
        ROOT / "training/bc_task_vlm/eval_runs/certified_scene_compatibility_v2_repaired_placement",
        (),
    ),
    "full_workspace": (
        ROOT / "training/bc_task_vlm/audit_task_scene_shared_workspaces.py",
        ROOT / "training/bc_task_vlm/eval_runs/task_scene_shared_workspace_v2_repaired_placement",
        (),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    scenes = candidate_scenes_v1()
    if not 0 <= args.index < len(scenes):
        raise IndexError(f"index {args.index} outside 0..{len(scenes)-1}")
    scene = scenes[args.index]
    script, output_dir, extra = MODES[args.mode]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"layout_{scene['layout']}_style_{scene['style']}_seed_{scene['seed']}.json"
    )
    command = [
        sys.executable,
        str(script),
        "--layout", str(scene["layout"]),
        "--style", str(scene["style"]),
        "--seed", str(scene["seed"]),
        "--output", str(output),
        *extra,
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(ROOT)
    )
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
