"""Builds fixed eval manifests for the SFT-necessity model comparison.

Emits three manifests under --output-dir, all drawing from one dataset root:

- eval_manifest_heldout_trajectories.json — steps from complete trajectories
  inside the exact training validation holdout (same selection function and
  seed as training) of the train tasks.
- eval_manifest_heldout_tasks.json — steps from complete trajectories of the
  held-out tasks (never seen in SFT).
- eval_manifest_pilot.json — a small disjoint-trajectory subset of the
  held-out-trajectories pool for decoding-config ablations.

Every eval backend consumes these manifests verbatim, so all models are
scored on identical samples.

Usage:
    python -m training.bc_task_vlm.build_eval_manifest \
        --dataset-root data_generation/task_level/data/image/experiment_52 \
        --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
        --output-dir training/bc_task_vlm/eval_manifests/exp1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from training.bc_task_vlm.dataset import (
    SFT_FORMAT_PLAIN,
    build_centralized_examples,
    list_task_trajectory_ids,
    select_held_out_trajectory_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="held_out_task_selection.json from data_analysis.select_held_out_tasks",
    )
    parser.add_argument("--validation-trajectory-fraction", type=float, default=0.1)
    parser.add_argument("--validation-split-seed", type=int, default=42)
    parser.add_argument(
        "--trajectories-per-split",
        type=int,
        default=75,
        help="Complete trajectories per headline manifest (~10 steps each).",
    )
    parser.add_argument(
        "--pilot-trajectories",
        type=int,
        default=12,
        help="Trajectories for the pilot manifest (disjoint from headline).",
    )
    parser.add_argument(
        "--train-get-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit get_image steps as eval targets (v3 active observation).",
    )
    parser.add_argument(
        "--partial-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Partial observability: per-agent private history + delivered messages.",
    )
    parser.add_argument(
        "--partial-step-index-mode",
        choices=("global", "local", "none"),
        default="global",
    )
    parser.add_argument(
        "--partial-observation-mode",
        choices=("cache", "consume-once"),
        default="consume-once",
    )
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stratified_trajectory_pick(
    pool_by_task: dict[str, list[str]],
    *,
    count: int,
    rng: random.Random,
    exclude: set[tuple[str, str]] | None = None,
) -> dict[str, list[str]]:
    """Round-robins tasks, picking one unused trajectory per pass, until count."""

    exclude = exclude or set()
    remaining = {
        task: [
            trajectory_id
            for trajectory_id in ids
            if (task, trajectory_id) not in exclude
        ]
        for task, ids in pool_by_task.items()
    }
    for ids in remaining.values():
        rng.shuffle(ids)
    task_order = sorted(remaining)
    rng.shuffle(task_order)

    picked: dict[str, list[str]] = {}
    total = 0
    while total < count:
        progressed = False
        for task in task_order:
            if total >= count:
                break
            if remaining[task]:
                picked.setdefault(task, []).append(remaining[task].pop())
                total += 1
                progressed = True
        if not progressed:
            break
    return picked


def build_manifest(
    *,
    split_name: str,
    dataset_root: Path,
    trajectory_ids_by_task: dict[str, list[str]],
    config: dict[str, Any],
    train_get_image: bool = False,
    partial_history: bool = False,
    partial_step_index_mode: str = "global",
    partial_observation_mode: str = "consume_once",
) -> dict[str, Any]:
    task_names = sorted(trajectory_ids_by_task)
    examples = build_centralized_examples(
        dataset_root=dataset_root,
        task_names=task_names,
        sft_format=SFT_FORMAT_PLAIN,
        train_get_image=train_get_image,
        partial_history=partial_history,
        partial_step_index_mode=partial_step_index_mode,
        partial_observation_mode=partial_observation_mode,
        trajectory_ids_by_task={
            task: set(ids) for task, ids in trajectory_ids_by_task.items()
        },
        example_build_workers=8,
        show_progress=True,
        progress_description=f"manifest:{split_name}",
    )
    examples.sort(key=lambda ex: (ex.task_name, ex.trajectory_id, ex.step_index))

    samples = [
        {
            "sample_id": example.sample_id,
            "task_name": example.task_name,
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "agent_id": example.agent_id,
            "target_tool": example.target_tool_call["name"],
        }
        for example in examples
    ]
    task_counts = Counter(sample["task_name"] for sample in samples)
    tool_counts = Counter(sample["target_tool"] for sample in samples)
    return {
        "split": split_name,
        "dataset_root": str(dataset_root),
        "config": config,
        "trajectory_ids_by_task": {
            task: sorted(ids) for task, ids in trajectory_ids_by_task.items()
        },
        "samples": samples,
        "summary": {
            "num_samples": len(samples),
            "num_trajectories": sum(
                len(ids) for ids in trajectory_ids_by_task.values()
            ),
            "num_tasks": len(task_names),
            "samples_per_task": dict(sorted(task_counts.items())),
            "samples_per_target_tool": dict(sorted(tool_counts.items())),
        },
    }


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    held_out_tasks: list[str] = selection["held_out_tasks"]
    train_tasks: list[str] = selection["train_tasks"]

    missing = [
        task
        for task in held_out_tasks + train_tasks
        if not (args.dataset_root / task).is_dir()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} tasks missing from {args.dataset_root}: {missing}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_config = {
        "selection_file": str(args.selection),
        "validation_trajectory_fraction": args.validation_trajectory_fraction,
        "validation_split_seed": args.validation_split_seed,
        "sample_seed": args.sample_seed,
        "held_out_tasks": held_out_tasks,
        "train_tasks": train_tasks,
        # Recorded so an eval run can be checked against the manifest it was
        # built from: get_image targets and the partial-observability prompt
        # shape must match between manifest, training, and eval.
        "train_get_image": args.train_get_image,
        "partial_history": args.partial_history,
        "partial_step_index_mode": args.partial_step_index_mode,
        "partial_observation_mode": args.partial_observation_mode.replace("-", "_"),
    }

    # Split A pool: the exact training-time validation holdout.
    holdout_ids_by_task = select_held_out_trajectory_ids(
        dataset_root=args.dataset_root,
        val_tasks=train_tasks,
        train_tasks=train_tasks,
        trajectories_per_task=0,
        trajectory_fraction=args.validation_trajectory_fraction,
        seed=args.validation_split_seed,
    )
    holdout_pool = {task: sorted(ids) for task, ids in holdout_ids_by_task.items()}

    rng = random.Random(args.sample_seed)
    headline_pick = stratified_trajectory_pick(
        holdout_pool,
        count=args.trajectories_per_split,
        rng=rng,
    )
    headline_keys = {
        (task, trajectory_id)
        for task, ids in headline_pick.items()
        for trajectory_id in ids
    }
    pilot_pick = stratified_trajectory_pick(
        holdout_pool,
        count=args.pilot_trajectories,
        rng=rng,
        exclude=headline_keys,
    )

    # Split B pool: every trajectory of the held-out tasks.
    heldout_task_pool = {
        task: list_task_trajectory_ids(
            dataset_root=args.dataset_root, task_name=task
        )
        for task in held_out_tasks
    }
    heldout_pick = stratified_trajectory_pick(
        heldout_task_pool,
        count=args.trajectories_per_split,
        rng=rng,
    )

    manifest_kwargs = {
        "train_get_image": args.train_get_image,
        "partial_history": args.partial_history,
        "partial_step_index_mode": args.partial_step_index_mode,
        "partial_observation_mode": args.partial_observation_mode.replace("-", "_"),
    }
    manifests = {
        "eval_manifest_heldout_trajectories.json": build_manifest(
            split_name="heldout_trajectories",
            dataset_root=args.dataset_root,
            trajectory_ids_by_task=headline_pick,
            config=shared_config,
            **manifest_kwargs,
        ),
        "eval_manifest_heldout_tasks.json": build_manifest(
            split_name="heldout_tasks",
            dataset_root=args.dataset_root,
            trajectory_ids_by_task=heldout_pick,
            config=shared_config,
            **manifest_kwargs,
        ),
        "eval_manifest_pilot.json": build_manifest(
            split_name="pilot",
            dataset_root=args.dataset_root,
            trajectory_ids_by_task=pilot_pick,
            config=shared_config,
            **manifest_kwargs,
        ),
    }
    for filename, manifest in manifests.items():
        path = args.output_dir / filename
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary = manifest["summary"]
        print(
            f"{filename}: {summary['num_samples']} steps / "
            f"{summary['num_trajectories']} trajectories / "
            f"{summary['num_tasks']} tasks"
        )

    # Leakage guards.
    heldout_overlap = set(held_out_tasks) & set(train_tasks)
    assert not heldout_overlap, f"held-out tasks overlap train: {heldout_overlap}"
    for task, ids in headline_pick.items():
        assert set(ids) <= holdout_ids_by_task[task]
    for task, ids in pilot_pick.items():
        assert set(ids) <= holdout_ids_by_task[task]
        assert not set(ids) & set(headline_pick.get(task, []))
    print("Leakage guards passed.")


if __name__ == "__main__":
    main()
