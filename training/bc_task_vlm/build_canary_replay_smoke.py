"""Materialize exact accepted canary trajectories for concurrent live replay."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from data_generation.task_level.generation.raw.config import RuntimeConfig
from data_generation.task_level.scene_sampling import (
    physical_configuration,
    physical_configuration_signature,
)
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.task_registry import get_task_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args()
    wanted = [value.strip() for value in args.tasks.split(",") if value.strip()]
    source = json.loads(args.canary_manifest.read_text())
    rows_by_task = {}
    for row in source["configurations"]:
        rows_by_task.setdefault(row["task"], []).append(row)
    episodes = {}
    for composite in wanted:
        row = sorted(rows_by_task[composite], key=lambda value: value["run_index"])[0]
        result_path = (
            args.canary_root / composite / f"run_{row['run_index']:06d}.json"
        )
        result = json.loads(result_path.read_text())
        if result.get("is_valid") is not True:
            raise ValueError(f"canary result is not valid: {result_path}")
        config = RuntimeConfig(
            composite_task=composite,
            num_runs=row["num_runs"],
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project=None,
            location="global",
            temperature=0.6,
            max_workers=1,
            max_retries=1,
            random_start_location=True,
            random_access_state=True,
            sampling="structured_random",
            thinking_level="low",
            tick_format=True,
            prompt_style="simplified_v3",
            partition_policy="none",
        )
        definition = get_task_definition(composite)
        instance = definition.build_task_instance(row["run_index"], config)
        candidate = deepcopy(result["accepted_candidate"])
        trajectory_id = f"canary_{row['run_index']:06d}"
        record = definition.build_trajectory_record(
            candidate=candidate,
            validation=deepcopy(result["final_validation"]),
            trajectory_id=trajectory_id,
            generation_usage={},
            task_instance=instance,
        )
        record["tick_rows"] = deepcopy(candidate["tick_rows"])
        record["coordinator_id"] = instance.coordinator_id
        record["physical_configuration"] = physical_configuration(
            row["configuration"]
        )
        record["physical_configuration_signature"] = physical_configuration_signature(
            row["configuration"]
        )
        dataset_name = get_task_metadata(composite).dataset_name
        destination = args.dataset_root / dataset_name / trajectory_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "original_trajectory.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        episodes.setdefault(dataset_name, []).append(
            {
                "episode_id": f"canary-replay:{dataset_name}:{trajectory_id}",
                "episode_rank": 0,
                "task_name": dataset_name,
                "trajectory_id": trajectory_id,
                "coordinator_id": instance.coordinator_id,
                "configuration_signature": row.get("physical_configuration_signature"),
                "physical_configuration_signature": record[
                    "physical_configuration_signature"
                ],
            }
        )
    manifest = {
        "manifest_type": "fixed_live_sim",
        "frozen": False,
        "contract": {
            "partial_history": True,
            "partial_step_index_mode": "none",
        },
        "splits": {"train_task_types": episodes, "heldout_task_types": {}},
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tasks": len(episodes), "episodes": len(wanted)}, indent=2))


if __name__ == "__main__":
    main()
