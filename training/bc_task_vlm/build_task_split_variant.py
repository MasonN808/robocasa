"""Build a reproducible reduced-training task split and matching fixed cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--base-cohort", type=Path, required=True)
    parser.add_argument("--held-out-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.base_split.read_text(encoding="utf-8"))
    selection = json.loads(args.base_selection.read_text(encoding="utf-8"))
    cohort = json.loads(args.base_cohort.read_text(encoding="utf-8"))
    train = sorted(split["train_tasks"])
    held = sorted(split["held_out_tasks"])
    additional_count = args.held_out_count - len(held)
    if additional_count < 0 or additional_count > len(train):
        raise ValueError("held-out count is incompatible with the base split")
    moved = sorted(random.Random(args.seed).sample(train, additional_count))
    new_train = sorted(set(train) - set(moved))
    new_held = sorted(set(held) | set(moved))

    split_out = {
        "schema_version": 1,
        "name": f"seeded_{len(new_train)}_train_{len(new_held)}_heldout",
        "train_tasks": new_train,
        "held_out_tasks": new_held,
        "source": str(args.base_split.resolve()),
        "selection_seed": args.seed,
        "moved_from_training_to_held_out": moved,
    }
    split_path = args.output_dir / f"{len(new_train)}_train_{len(new_held)}_heldout.json"
    _write_json(split_path, split_out)
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()

    selection_out = dict(selection)
    selection_out["name"] = f"nested_{selection['trajectories_per_task']}_per_task_{len(new_train)}train_seed_{args.seed}"
    selection_out["task_split_manifest"] = str(split_path.resolve())
    selection_out["task_split_sha256"] = split_sha
    selection_out["train_trajectory_ids_by_task"] = {
        task: selection["train_trajectory_ids_by_task"][task] for task in new_train
    }
    selection_path = args.output_dir / f"selection_{selection['trajectories_per_task']}.json"
    _write_json(selection_path, selection_out)

    old_configs = cohort["configurations"]
    all_configs = {**old_configs["train_task_types"], **old_configs["heldout_task_types"]}
    missing = sorted((set(new_train) | set(new_held)) - set(all_configs))
    if missing:
        raise ValueError(f"cohort is missing tasks: {missing}")
    cohort_out = dict(cohort)
    cohort_out["source_task_split"] = str(split_path.resolve())
    cohort_out["task_split_seed"] = args.seed
    cohort_out["configurations"] = {
        "train_task_types": {task: all_configs[task] for task in new_train},
        "heldout_task_types": {task: all_configs[task] for task in new_held},
    }
    cohort_path = args.output_dir / "fixed_live_sim_cohort.json"
    _write_json(cohort_path, cohort_out)
    print(json.dumps({
        "train_tasks": len(new_train),
        "held_out_tasks": len(new_held),
        "moved": moved,
        "split": str(split_path),
        "selection": str(selection_path),
        "cohort": str(cohort_path),
    }, indent=2))


if __name__ == "__main__":
    main()
