#!/usr/bin/env python3
"""Inventory native RoboCasa success checks beside verified symbolic FSM specs.

This is intentionally a static evidence collector, not an equivalence prover.
It preserves the native ``_check_success`` source and the complete symbolic
goal/precondition payload so semantic classifications remain inspectable.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECS = ROOT / "data_generation/task_level/tasks/specs/verified"


def _native_success_source(module_name: str) -> tuple[str, str]:
    module_path = ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
    tree = ast.parse(module_path.read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_check_success"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one _check_success in {module_path}, found {len(matches)}"
        )
    source = module_path.read_text()
    lines = source.splitlines()
    node = matches[0]
    return str(module_path.relative_to(ROOT)), "\n".join(
        lines[node.lineno - 1 : node.end_lineno]
    )


def collect(spec_dir: Path) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(spec_dir.glob("*.json")):
        spec = json.loads(path.read_text())
        native_path, native_source = _native_success_source(
            spec["source_python_module"]
        )
        tasks.append(
            {
                "task": spec["composite_task"],
                "spec_path": str(path.relative_to(ROOT)),
                "native_path": native_path,
                "task_goal": spec["task_goal"],
                "initial_objects": sorted(spec["initial_state"].get("objects", {})),
                "goal_conditions": spec["goal_conditions"],
                "task_preconditions": spec["task_preconditions"],
                "task_effects": spec["task_effects"],
                "native_check_success_source": native_source,
            }
        )
    return {"schema_version": 1, "task_count": len(tasks), "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = collect(args.spec_dir)
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
