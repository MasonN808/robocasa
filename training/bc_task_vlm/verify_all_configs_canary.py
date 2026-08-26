"""Gate full generation on complete, balanced, current-contract canary output."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from data_generation.task_level.generation.image.processor import revalidate_tick_trajectory
from data_generation.task_level.tasks.shared.validation_contract import current_validation_error
from data_generation.task_level.subatomic_tool_specs import build_model_tool_specs
from training.bc_task_vlm.schema_utils import validate_single_step_payload
from data_generation.task_level.tasks.specs import load_verified_task_specs
from data_generation.task_level.tasks.shared.workspace_semantics import (
    canonical_agent_workspace,
)


def _hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _configuration(
    trajectory: dict[str, Any], canonical_initial_state: dict[str, Any]
) -> dict[str, Any]:
    state = trajectory.get("initial_state") or {}
    return {
        "coordinator_id": trajectory.get("coordinator_id"),
        "agent_locations": {
            agent: canonical_agent_workspace(
                canonical_initial_state, value.get("location")
            )
            for agent, value in sorted((state.get("agents") or {}).items())
        },
        "access_states": {
            f"{fixture_id}.{part_id}": part.get("state")
            for fixture_id, fixture in sorted((state.get("fixtures") or {}).items())
            for part_id, part in sorted((fixture.get("parts") or {}).items())
            if part.get("state") in {"open", "closed"}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    global_specs = build_model_tool_specs(include_get_image=True)
    specs = {
        spec.composite_task: spec for spec in load_verified_task_specs()
    }
    failures = []
    task_reports = []
    for task in manifest["tasks"]:
        task_name = task["task"]
        expected = {
            index: _hash(run["configuration"])
            for index, run in enumerate(task["runs"][: task["possible_joint_configurations"]])
        }
        task_dir = args.canary_root / task_name
        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            failures.append({"task": task_name, "error": "missing_summary"})
            continue
        summary = json.loads(summary_path.read_text())
        completed = set(summary.get("completed_run_indices") or [])
        if completed != set(expected):
            failures.append({
                "task": task_name,
                "error": "incomplete_run_indices",
                "missing": sorted(set(expected) - completed),
                "unexpected": sorted(completed - set(expected)),
            })
        observed = Counter()
        valid = 0
        for path in sorted((task_dir / "trajectories").glob("traj_*.json")):
            trajectory = json.loads(path.read_text())
            run_index = (trajectory.get("sampling_metadata") or {}).get("run_index")
            signature = _hash(
                _configuration(trajectory, specs[task_name].initial_state)
            )
            observed[signature] += 1
            if expected.get(run_index) != signature:
                failures.append({"task": task_name, "run_index": run_index, "error": "configuration_mismatch"})
            for index, step in enumerate(trajectory.get("steps") or []):
                try:
                    validate_single_step_payload(
                        {"steps": [{"step": int(step.get("step", index)), "agent": step.get("agent"), "tool": step.get("tool"), "args": step.get("args") or {}}]},
                        agent_ids=("agent_0", "agent_1"),
                        allowed_tool_specs=global_specs,
                    )
                except (TypeError, ValueError) as exc:
                    failures.append({"task": task_name, "run_index": run_index, "error": "global_schema", "detail": str(exc)})
            replay = deepcopy(trajectory)
            revalidate_tick_trajectory(replay)
            error = current_validation_error(replay)
            if error is None:
                valid += 1
            else:
                failures.append({"task": task_name, "run_index": run_index, "error": "current_fsm", "detail": error})
        task_reports.append({
            "task": task_name,
            "expected_configurations": len(expected),
            "observed_unique_configurations": len(observed),
            "valid_trajectories": valid,
            "complete": completed == set(expected) and valid == len(expected) and len(observed) == len(expected),
        })
    report = {
        "gate_passed": not failures and all(row["complete"] for row in task_reports),
        "expected_tasks": len(manifest["tasks"]),
        "completed_tasks": sum(row["complete"] for row in task_reports),
        "expected_configurations": sum(row["expected_configurations"] for row in task_reports),
        "valid_trajectories": sum(row["valid_trajectories"] for row in task_reports),
        "failures": failures,
        "tasks": task_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"failures", "tasks"}}, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
