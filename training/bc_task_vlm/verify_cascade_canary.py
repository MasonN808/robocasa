"""Revalidate and summarize the all-configuration cascade canary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from data_generation.task_level.generation.raw.config import RuntimeConfig
from data_generation.task_level.generation.raw.runtime_support import _validate_candidate
from data_generation.task_level.tasks import get_task_definition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--canary-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.canary_root)
    failures = []
    stages: Counter[str] = Counter()
    total = 0
    if manifest.get("manifest_type") == "physical_configuration_generation_canary":
        rows = list(manifest["configurations"])
    else:
        rows = [
            {
                "task": task_entry["task"],
                "run_index": run_index,
                "num_runs": task_entry["possible_joint_configurations"],
            }
            for task_entry in manifest["tasks"]
            for run_index in range(task_entry["possible_joint_configurations"])
        ]
    for row in rows:
        task = row["task"]
        run_index = int(row["run_index"])
        count = int(row["num_runs"])
        definition = get_task_definition(task)
        config = RuntimeConfig(
            composite_task=task,
            num_runs=count,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project=None,
            location="global",
            temperature=0.6,
            max_workers=1,
            max_retries=1,
            random_start_location=True,
            random_access_state=True,
            partition_policy="none",
            tick_format=True,
            sampling="structured_random",
            thinking_level="low",
            prompt_style="simplified_v3",
        )
        total += 1
        path = root / task / f"run_{run_index:06d}.json"
        if not path.exists():
            failures.append({"task": task, "run_index": run_index, "error": "missing"})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("task") != task or result.get("run_index") != run_index:
            failures.append({"task": task, "run_index": run_index, "error": "result_identity_mismatch"})
            continue
        raw_candidate = result.get("accepted_raw_candidate")
        if not isinstance(raw_candidate, dict):
            failures.append({"task": task, "run_index": run_index, "error": "missing_raw_candidate"})
            continue
        instance = definition.build_task_instance(run_index, config)
        if "configuration" in row:
            expected = row["configuration"]
            observed = {
                "coordinator_id": instance.coordinator_id,
                "agent_locations": {
                    agent: state.get("location")
                    for agent, state in sorted(instance.initial_state.get("agents", {}).items())
                },
                "access_states": {
                    f"{fixture_id}.{part_id}": part.get("state")
                    for fixture_id, fixture in sorted(instance.initial_state.get("fixtures", {}).items())
                    for part_id, part in sorted((fixture.get("parts") or {}).items())
                    if part.get("state") in {"open", "closed"}
                },
            }
            if observed != expected:
                failures.append({"task": task, "run_index": run_index, "error": "configuration_reconstruction_mismatch", "expected": expected, "observed": observed})
                continue
        validator = definition.validator_factory(instance)
        validation, _ = _validate_candidate(
            raw_candidate,
            validator,
            enforce_validation=False,
        )
        if validation.get("is_valid") is not True:
            failures.append(
                {
                    "task": task,
                    "run_index": run_index,
                    "error": validation.get("error"),
                    "error_type": validation.get("error_type"),
                }
            )
            continue
        stages[str(result.get("accepted_stage"))] += 1
    report = {
        "expected_configurations": total,
        "valid_configurations": total - len(failures),
        "gate_passed": not failures,
        "accepted_stage_counts": dict(sorted(stages.items())),
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
