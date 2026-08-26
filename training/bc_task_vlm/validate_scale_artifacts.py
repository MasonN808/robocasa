"""Validate nested scale artifacts before allocating training GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-pattern", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    previous: dict[str, set[str]] | None = None
    report = {}
    for scale in (30, 60, 90, 120, 150):
        root = Path(args.artifact_pattern.format(scale=scale))
        dataset = load_from_disk(str(root / "dataset"))
        config = json.loads((root / "preprocess_config.json").read_text())
        selection = json.loads(
            (args.manifest_dir / f"selection_{scale}.json").read_text()
        )["train_trajectory_ids_by_task"]
        selected = {task: set(ids) for task, ids in selection.items()}
        observed: dict[str, set[str]] = {task: set() for task in selected}
        for task, trajectory_id in zip(
            dataset["train"]["task_name"],
            dataset["train"]["trajectory_id"],
            strict=True,
        ):
            if task not in observed or trajectory_id not in selected[task]:
                raise ValueError(f"Unexpected row {task}/{trajectory_id} in scale {scale}")
            observed[task].add(trajectory_id)
        if observed != selected:
            raise ValueError(f"Scale {scale} artifact does not cover its manifest")
        if previous is not None and any(
            not previous[task].issubset(selected[task]) for task in previous
        ):
            raise ValueError(f"Scale {scale} is not nested")
        if len(dataset["validation"]) != 0:
            raise ValueError(f"Scale {scale} unexpectedly contains validation rows")
        if config.get("partial_step_index_mode") != "none":
            raise ValueError(f"Scale {scale} uses step indexing")
        if config.get("prompt_contract_version") is None:
            raise ValueError(f"Scale {scale} lacks prompt contract metadata")
        report[str(scale)] = {
            "samples": len(dataset["train"]),
            "tasks": len(selected),
            "trajectories": sum(len(ids) for ids in selected.values()),
        }
        previous = selected
    print(json.dumps({"valid": True, "scales": report}, indent=2))


if __name__ == "__main__":
    main()
