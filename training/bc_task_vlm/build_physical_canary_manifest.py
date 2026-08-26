"""Select exactly one balanced-coordinator row per canonical physical state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _physical(configuration: dict[str, Any]) -> str:
    return json.dumps(
        {
            "agent_locations": configuration.get("agent_locations") or {},
            "access_states": configuration.get("access_states") or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build(source: dict[str, Any]) -> dict[str, Any]:
    selected_tasks = []
    coordinator_counts = {"agent_0": 0, "agent_1": 0}
    selection_index = 0
    for task in source["tasks"]:
        unique_joint = task["runs"][: task["possible_joint_configurations"]]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in unique_joint:
            groups.setdefault(_physical(row["configuration"]), []).append(row)
        selected = []
        for physical_signature in sorted(groups):
            desired = f"agent_{selection_index % 2}"
            matches = [
                row for row in groups[physical_signature]
                if row["configuration"].get("coordinator_id") == desired
            ]
            if not matches:
                raise ValueError(
                    f"{task['task']} physical state lacks coordinator {desired}"
                )
            row = matches[0]
            selected.append(
                {
                    "task": task["task"],
                    "run_index": row["run_index"],
                    "num_runs": source["num_runs_per_task"],
                    "physical_configuration_signature": physical_signature,
                    "configuration": row["configuration"],
                }
            )
            coordinator_counts[desired] += 1
            selection_index += 1
        selected_tasks.append(
            {
                "task": task["task"],
                "physical_configuration_count": len(selected),
                "configurations": selected,
            }
        )
    rows = [row for task in selected_tasks for row in task["configurations"]]
    return {
        "schema_version": 1,
        "manifest_type": "physical_configuration_generation_canary",
        "source_contract": source.get("contract"),
        "source_num_runs_per_task": source["num_runs_per_task"],
        "configuration_count": len(rows),
        "coordinator_counts": coordinator_counts,
        "tasks": selected_tasks,
        "configurations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("configuration_count", "coordinator_counts")}, indent=2))


if __name__ == "__main__":
    main()
