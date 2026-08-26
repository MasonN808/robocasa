"""Preflight balanced initial configurations without generating trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from data_generation.task_level.generation.raw.config import RuntimeConfig
from data_generation.task_level.tasks.shared.instances import (
    count_balanced_initial_configurations,
    eligible_access_state_parts,
)
from data_generation.task_level.tasks.specs import load_all_task_specs
from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY
from data_generation.task_level.tasks.shared.workspace_semantics import (
    canonical_agent_workspace,
)
from data_generation.utils import stable_json_sha256


def _signature(initial_state: dict[str, Any], coordinator_id: str | None) -> dict[str, Any]:
    return {
        "coordinator_id": coordinator_id,
        "agent_locations": {
            agent_id: canonical_agent_workspace(initial_state, state.get("location"))
            for agent_id, state in sorted((initial_state.get("agents") or {}).items())
        },
        "access_states": {
            f"{fixture_id}.{part_id}": part.get("state")
            for fixture_id, fixture in sorted(
                (initial_state.get("fixtures") or {}).items()
            )
            for part_id, part in sorted((fixture.get("parts") or {}).items())
            if part.get("state") in {"open", "closed"}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num-runs must be positive")

    runtime_config = RuntimeConfig(
        composite_task=None,
        num_runs=args.num_runs,
        model="gemini-3-flash-preview",
        sdk="google-genai",
        project=None,
        location="global",
        temperature=0.6,
        max_workers=1,
        max_retries=10,
        sampling="structured_random",
        random_start_location=True,
        random_access_state=True,
        tick_format=True,
        prompt_style="simplified_v3",
        retry_feedback_style="targeted",
        partition_policy="none",
    )
    tasks = []
    for spec in load_all_task_specs():
        definition = SPEC_TASK_REGISTRY[spec.composite_task]
        signatures = []
        for run_index in range(args.num_runs):
            instance = definition.build_task_instance(run_index, runtime_config)
            signatures.append(_signature(instance.initial_state, instance.coordinator_id))
        signature_keys = [stable_json_sha256(value) for value in signatures]
        counts = Counter(signature_keys)
        expected_count = len(spec.agent_ids) * count_balanced_initial_configurations(
            composite_task=spec.composite_task,
            agent_ids=spec.agent_ids,
            initial_state=spec.initial_state,
            allowed_tool_specs=spec.allowed_tool_specs,
        )
        tasks.append(
            {
                "task": spec.composite_task,
                "eligible_access_parts": [
                    {"fixture_id": fixture_id, "part_id": part_id}
                    for fixture_id, part_id in eligible_access_state_parts(
                        initial_state=spec.initial_state,
                        allowed_tool_specs=spec.allowed_tool_specs,
                    )
                ],
                "possible_joint_configurations": expected_count,
                "sampled_unique_configurations": len(counts),
                "minimum_repeats": min(counts.values()),
                "maximum_repeats": max(counts.values()),
                "balanced": max(counts.values()) - min(counts.values()) <= 1,
                "runs": [
                    {"run_index": index, "configuration": signature}
                    for index, signature in enumerate(signatures)
                ],
            }
        )
    payload = {
        "contract": "balanced_joint_initial_configs_v1",
        "num_runs_per_task": args.num_runs,
        "random_start_location": True,
        "random_access_state": True,
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
