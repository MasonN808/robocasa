"""Build and freeze a reproducible live-simulator evaluation cohort.

The candidate manifest is deterministic but deliberately marked unfrozen.  A
candidate becomes a frozen benchmark only after an oracle run has produced one
successful, rejection-free result for every episode.  This avoids treating
stale validation fields embedded in rendered trajectories as current truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from training.bc_task_vlm.task_registry import get_task_metadata
from data_generation.task_level.tasks.specs import load_verified_task_specs
from data_generation.task_level.tasks.shared.workspace_semantics import (
    canonicalize_agent_locations,
)


SPLITS = ("train_task_types", "heldout_task_types")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_path(root: Path, task: str, trajectory_id: str) -> Path:
    dataset_name = get_task_metadata(task).dataset_name
    return root / dataset_name / trajectory_id / "original_trajectory.json"


def _candidate_ids(root: Path, task: str) -> list[str]:
    dataset_name = get_task_metadata(task).dataset_name
    return sorted(p.parent.name for p in (root / dataset_name).glob("traj_*/original_trajectory.json"))


def build_candidate_manifest(
    *,
    dataset_root: Path,
    selection: dict[str, Any],
    episodes_per_task: int,
    cohort_seed: int,
    layout: int,
    style: int,
    environment_seed: int,
) -> dict[str, Any]:
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    train_tasks = set(selection["train_tasks"])
    specs = {}
    for spec in load_verified_task_specs():
        specs[spec.composite_task] = spec
        specs[get_task_metadata(spec.composite_task).dataset_name] = spec
    declared_heldout = set(selection["held_out_tasks"])
    available_tasks = {
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and _candidate_ids(dataset_root, path.name)
    }
    overlap = train_tasks & declared_heldout
    if overlap:
        raise ValueError(f"tasks appear in both splits: {sorted(overlap)}")
    # Historical selection files can predate a newly verified task.  Training
    # membership is authoritative: every available task not used in SFT is a
    # held-out task type, rather than silently disappearing from evaluation.
    unassigned = available_tasks - train_tasks - declared_heldout
    task_splits = {
        "train_task_types": sorted(train_tasks),
        "heldout_task_types": sorted(declared_heldout | unassigned),
    }
    splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen_episode_ids: set[str] = set()
    for split, tasks in task_splits.items():
        splits[split] = {}
        for task in tasks:
            ids = _candidate_ids(dataset_root, task)
            if len(ids) < episodes_per_task:
                raise ValueError(
                    f"{task} has {len(ids)} candidates; need {episodes_per_task}"
                )
            rng = random.Random(f"{cohort_seed}:{split}:{task}")
            rng.shuffle(ids)
            by_coordinator: dict[str, list[tuple[str, dict[str, Any]]]] = {
                "agent_0": [], "agent_1": []
            }
            seen_configurations: set[str] = set()
            for trajectory_id in ids:
                trajectory = _read_json(_trajectory_path(dataset_root, task, trajectory_id))
                coordinator = trajectory.get("coordinator_id")
                if coordinator in by_coordinator:
                    canonical_state = canonicalize_agent_locations(
                        trajectory.get("initial_state") or specs[task].initial_state
                    )
                    canonical_signature = _sha256(
                        {
                            "initial_state": canonical_state,
                            "coordinator_id": coordinator,
                            "layout": layout,
                            "style": style,
                            "environment_seed": environment_seed,
                        }
                    )
                    if canonical_signature in seen_configurations:
                        continue
                    seen_configurations.add(canonical_signature)
                    by_coordinator[coordinator].append((trajectory_id, trajectory))
            required = {
                "agent_0": (episodes_per_task + 1) // 2,
                "agent_1": episodes_per_task // 2,
            }
            for coordinator, count in required.items():
                if len(by_coordinator[coordinator]) < count:
                    raise ValueError(
                        f"{task} has {len(by_coordinator[coordinator])} candidates for "
                        f"{coordinator}; need {count} for a balanced cohort"
                    )
            episodes = []
            offsets = {"agent_0": 0, "agent_1": 0}
            for rank in range(episodes_per_task):
                coordinator_id = f"agent_{rank % 2}"
                trajectory_id, trajectory = by_coordinator[coordinator_id][offsets[coordinator_id]]
                offsets[coordinator_id] += 1
                canonical_state = canonicalize_agent_locations(
                    trajectory.get("initial_state") or specs[task].initial_state
                )
                initial_state_signature = _sha256(canonical_state)
                configuration_signature = _sha256(
                    {
                        "initial_state": canonical_state,
                        "coordinator_id": coordinator_id,
                        "layout": layout,
                        "style": style,
                        "environment_seed": environment_seed,
                    }
                )
                episode_id = f"{split}:{task}:{rank:03d}:{configuration_signature[:12]}"
                if episode_id in seen_episode_ids:
                    raise AssertionError(f"duplicate episode_id: {episode_id}")
                seen_episode_ids.add(episode_id)
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "episode_rank": rank,
                        "task_name": task,
                        "trajectory_id": trajectory_id,
                        "coordinator_id": coordinator_id,
                        "initial_state_signature": initial_state_signature,
                        "configuration_signature": configuration_signature,
                        "scene": {
                            "layout": layout,
                            "style": style,
                            "seed": environment_seed,
                            "membership": "current_render_scene",
                        },
                        "provenance": {
                            "carrier_trajectory": str(
                                _trajectory_path(dataset_root, task, trajectory_id)
                            ),
                            "sampling_metadata": trajectory.get("sampling_metadata"),
                        },
                    }
                )
            splits[split][task] = episodes
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "fixed_live_sim",
        "frozen": False,
        "dataset_root": str(dataset_root.resolve()),
        "cohort_seed": cohort_seed,
        "selection_audit": {
            "declared_train_tasks": len(train_tasks),
            "declared_heldout_tasks": len(declared_heldout),
            "unassigned_tasks_added_to_heldout": sorted(unassigned),
        },
        "episodes_per_task": episodes_per_task,
        "contract": {
            "partial_history": True,
            "partial_step_index_mode": "none",
            "partial_observation_mode": "consume-once",
            "uniform_durations": True,
            "success_criterion": "fsm",
            "rejection_mode": "report_failed",
            "contention_policy": "concurrent_fsm",
        },
        "scene_policy": {
            "mode": "fixed_current_render_scene",
            "layout": layout,
            "style": style,
            "seed": environment_seed,
            "future_compatible_fields": ["scene.membership", "scene.signature"],
        },
        "splits": splits,
    }
    for tasks in manifest["splits"].values():
        for episodes in tasks.values():
            for episode in episodes:
                episode["scene"]["signature"] = _sha256(episode["scene"])
    manifest["content_hash"] = _sha256({k: v for k, v in manifest.items() if k != "content_hash"})
    return manifest


def _load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_id"):
                latest[row["episode_id"]] = row
    return latest


def freeze_manifest(manifest: dict[str, Any], result_paths: list[Path]) -> dict[str, Any]:
    results = _load_results(result_paths)
    failures = []
    count = 0
    for tasks in manifest["splits"].values():
        for episodes in tasks.values():
            for episode in episodes:
                count += 1
                result = results.get(episode["episode_id"])
                reasons = []
                if result is None:
                    reasons.append("missing_result")
                else:
                    if result.get("termination") == "harness_error":
                        reasons.append("harness_error")
                    if not result.get("fsm_goal_satisfied"):
                        reasons.append("fsm_goal_not_satisfied")
                    if result.get("had_rejection") or result.get("rejected_steps", 0):
                        reasons.append("rejection")
                if reasons:
                    failures.append({"episode_id": episode["episode_id"], "reasons": reasons})
    if failures:
        preview = ", ".join(f"{x['episode_id']}={x['reasons']}" for x in failures[:5])
        raise ValueError(f"oracle gate failed for {len(failures)}/{count} episodes: {preview}")
    frozen = json.loads(json.dumps(manifest))
    frozen["frozen"] = True
    frozen["oracle_validation"] = {
        "num_episodes": count,
        "result_files": [str(p.resolve()) for p in result_paths],
        "requirements": ["fsm_goal_satisfied", "no_rejections", "no_harness_error"],
    }
    frozen["content_hash"] = _sha256({k: v for k, v in frozen.items() if k != "content_hash"})
    return frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--cohort-seed", type=int, default=20260812)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--environment-seed", type=int, default=42)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--oracle-results", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_manifest:
        manifest = freeze_manifest(_read_json(args.candidate_manifest), args.oracle_results)
    else:
        if not args.dataset_root or not args.selection:
            raise SystemExit("building candidates requires --dataset-root and --selection")
        manifest = build_candidate_manifest(
            dataset_root=args.dataset_root,
            selection=_read_json(args.selection),
            episodes_per_task=args.episodes_per_task,
            cohort_seed=args.cohort_seed,
            layout=args.layout,
            style=args.style,
            environment_seed=args.environment_seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} (frozen={manifest['frozen']}, hash={manifest['content_hash']})")


if __name__ == "__main__":
    main()
