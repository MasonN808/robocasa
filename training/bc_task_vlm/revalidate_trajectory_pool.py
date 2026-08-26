"""Revalidate a rendered trajectory pool without modifying source files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from data_generation.task_level.generation.image.processor import (
    revalidate_tick_trajectory,
)
from data_generation.task_level.tasks.shared.validation_contract import (
    VALIDATOR_CONTRACT_VERSION,
    current_validation_error,
)
from data_generation.task_level.subatomic_tool_specs import build_model_tool_specs
from training.bc_task_vlm.schema_utils import validate_single_step_payload


def _classify(validation: dict[str, Any]) -> str:
    error_type = str(validation.get("error_type") or "unknown")
    error = str(validation.get("error") or "").lower()
    if "resource" in error_type.lower() or "conflict" in error:
        return "resource_conflict"
    if "wait" in error_type.lower() or "release" in error:
        return "wait_release_protocol"
    if "opening" in error or "coordinator" in error or "confirm" in error:
        return "opening_protocol"
    if "goal" in error_type.lower() or "goal" in error:
        return "goal_or_termination"
    if "unsupported" in error_type.lower() or "argument" in error:
        return "tool_or_argument_schema"
    return error_type


def _audit_one(path: Path) -> dict[str, Any]:
    original = json.loads(path.read_text(encoding="utf-8"))
    old_validation = original.get("validation")
    old_validation = old_validation if isinstance(old_validation, dict) else {}
    updated = deepcopy(original)
    structural_errors = []
    global_specs = build_model_tool_specs(include_get_image=True)
    for index, step in enumerate(original.get("steps") or []):
        try:
            validate_single_step_payload(
                {"steps": [{
                    "step": int(step.get("step", index)),
                    "agent": step.get("agent"),
                    "tool": step.get("tool"),
                    "args": step.get("args") or {},
                }]},
                agent_ids=("agent_0", "agent_1"),
                allowed_tool_specs=global_specs,
            )
        except (TypeError, ValueError) as exc:
            structural_errors.append({"step": index, "error": str(exc)})
    revalidate_tick_trajectory(updated)
    new_validation = updated["validation"]
    old_valid = old_validation.get("is_valid") is True
    new_valid = current_validation_error(updated) is None
    return {
        "task": path.parent.parent.name,
        "trajectory_id": original.get("trajectory_id", path.parent.name),
        "path": str(path),
        "old_is_valid": old_validation.get("is_valid"),
        "old_contract_version": old_validation.get("validator_contract_version"),
        "new_is_valid": new_validation.get("is_valid"),
        "new_contract_version": new_validation.get("validator_contract_version"),
        "transition": (
            "valid_to_valid" if old_valid and new_valid else
            "valid_to_invalid" if old_valid else
            "invalid_to_valid" if new_valid else
            "invalid_to_invalid"
        ),
        "root_cause": None if new_valid else _classify(new_validation),
        "error_type": new_validation.get("error_type"),
        "error": new_validation.get("error"),
        "step": new_validation.get("step"),
        "global_schema_valid": not structural_errors,
        "global_schema_errors": structural_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(args.dataset_root.glob("*/traj_*/original_trajectory.json"))
    if not paths:
        paths = sorted(args.dataset_root.glob("*/trajectories/traj_*.json"))
    if not paths:
        raise ValueError(f"No trajectories found beneath {args.dataset_root}")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(_audit_one, paths))

    transition_counts = Counter(record["transition"] for record in records)
    failure_causes = Counter(
        record["root_cause"] for record in records if record["new_is_valid"] is not True
    )
    failures_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        if record["new_is_valid"] is not True:
            failures_by_task[record["task"]][str(record["root_cause"])] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "trajectory_revalidation.jsonl"
    ledger_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "trajectory_count": len(records),
        "transition_counts": dict(sorted(transition_counts.items())),
        "current_valid_count": sum(record["new_is_valid"] is True for record in records),
        "current_invalid_count": sum(record["new_is_valid"] is not True for record in records),
        "global_schema_valid_count": sum(record["global_schema_valid"] for record in records),
        "global_schema_invalid_count": sum(not record["global_schema_valid"] for record in records),
        "failure_causes": dict(sorted(failure_causes.items())),
        "failures_by_task": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(failures_by_task.items())
        },
        "ledger": ledger_path.name,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
