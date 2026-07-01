#!/usr/bin/env python3
"""Merge sharded HF staging outputs into one trainable dataset root.

Array staging jobs should write one independent dataset root and manifest per
shard. This helper creates a final dataset root by symlinking trajectory
directories from shard roots in repo-list order, optionally stopping at a global
episode cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        required=True,
        help="Directory containing shard_*/hf_stage_manifest.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Final dataset root to create for training.",
    )
    parser.add_argument(
        "--repo-list",
        type=Path,
        required=True,
        help="Newline-delimited repo list defining global selection order.",
    )
    parser.add_argument(
        "--max-total-episodes",
        type=int,
        default=None,
        help="Stop after linking this many episodes across repos.",
    )
    parser.add_argument(
        "--force-symlinks",
        action="store_true",
        help="Replace existing trajectory symlinks in the output root.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Write only hf_stage_manifest.json with shard references instead "
            "of creating one output symlink per trajectory."
        ),
    )
    return parser.parse_args()


def read_repo_list(path: Path) -> list[str]:
    repo_ids = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            repo_ids.append(line)
    return repo_ids


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_shard_manifests(shard_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for manifest_path in sorted(shard_root.glob("shard_*/hf_stage_manifest.json")):
        manifest = _load_json(manifest_path)
        manifest["_manifest_path"] = str(manifest_path)
        manifests.append(manifest)
    if not manifests:
        raise FileNotFoundError(
            f"No shard manifests found under {shard_root}/shard_*/hf_stage_manifest.json"
        )
    return manifests


def source_index_by_repo(
    manifests: Iterable[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    sources_by_repo: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for manifest in manifests:
        for source in manifest.get("sources", []):
            repo_id = source.get("repo_id")
            if not isinstance(repo_id, str) or not repo_id:
                continue
            if repo_id in sources_by_repo:
                raise ValueError(f"Duplicate staged source for repo {repo_id}")
            sources_by_repo[repo_id] = (manifest, source)
    return sources_by_repo


def _trajectory_dirs_for_source(
    *,
    manifest: dict[str, Any],
    source: dict[str, Any],
) -> list[tuple[str, Path]]:
    output_root = Path(str(manifest["output_root"]))
    trajectory_dirs: list[tuple[str, Path]] = []
    for task_name in source.get("tasks", []):
        task_root = output_root / str(task_name)
        if not task_root.is_dir():
            raise FileNotFoundError(f"Missing staged task directory: {task_root}")
        trajectory_dirs.extend(
            (str(task_name), path)
            for path in sorted(task_root.iterdir())
            if path.is_dir()
        )
    expected_count = int(source.get("num_episodes", len(trajectory_dirs)))
    if len(trajectory_dirs) < expected_count:
        raise ValueError(
            f"{source.get('repo_id')} manifest reports {expected_count} episodes "
            f"but only {len(trajectory_dirs)} trajectory dirs were found."
        )
    return trajectory_dirs[:expected_count]


def _link_trajectory_dir(
    *,
    source_dir: Path,
    target_dir: Path,
    force_symlinks: bool,
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() or target_dir.is_symlink():
        if not force_symlinks:
            raise FileExistsError(
                f"Output trajectory already exists: {target_dir}. "
                "Use --force-symlinks to replace existing symlinks."
            )
        if not target_dir.is_symlink():
            raise FileExistsError(
                f"Refusing to replace non-symlink output trajectory: {target_dir}"
            )
        target_dir.unlink()
    target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)


def merge_shards(
    *,
    shard_root: Path,
    output_root: Path,
    repo_ids: list[str],
    max_total_episodes: int | None,
    force_symlinks: bool = False,
    manifest_only: bool = False,
) -> dict[str, Any]:
    if max_total_episodes is not None and max_total_episodes < 1:
        raise ValueError("--max-total-episodes must be at least 1.")

    manifests = load_shard_manifests(shard_root)
    sources_by_repo = source_index_by_repo(manifests)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_sources: list[dict[str, Any]] = []
    task_episode_counts: dict[str, int] = {}
    total_episodes = 0
    missing_repos = [repo_id for repo_id in repo_ids if repo_id not in sources_by_repo]
    if missing_repos:
        missing_text = "\n".join(missing_repos)
        raise ValueError(f"Missing staged shard sources for repos:\n{missing_text}")

    for repo_id in repo_ids:
        if max_total_episodes is not None and total_episodes >= max_total_episodes:
            break
        manifest, source = sources_by_repo[repo_id]
        trajectory_dirs = _trajectory_dirs_for_source(
            manifest=manifest,
            source=source,
        )
        remaining = (
            len(trajectory_dirs)
            if max_total_episodes is None
            else max_total_episodes - total_episodes
        )
        selected_dirs = trajectory_dirs[:remaining]
        if not selected_dirs:
            continue

        for task_name, source_dir in selected_dirs:
            if not manifest_only:
                target_dir = output_root / task_name / source_dir.name
                _link_trajectory_dir(
                    source_dir=source_dir,
                    target_dir=target_dir,
                    force_symlinks=force_symlinks,
                )
            task_episode_counts[task_name] = task_episode_counts.get(task_name, 0) + 1

        selected_source = dict(source)
        selected_source["num_available_episodes"] = source.get("num_episodes")
        selected_source["num_episodes"] = len(selected_dirs)
        selected_source["shard_output_root"] = manifest.get("output_root")
        selected_source["shard_manifest_path"] = manifest.get("_manifest_path")
        selected_sources.append(selected_source)
        total_episodes += len(selected_dirs)

    merged_manifest = {
        "output_root": str(output_root.resolve()),
        "repo_order": repo_ids,
        "shard_root": str(shard_root.resolve()),
        "sources": selected_sources,
        "task_episode_counts": dict(sorted(task_episode_counts.items())),
        "total_episodes": total_episodes,
    }
    if max_total_episodes is not None:
        merged_manifest["max_total_episodes"] = max_total_episodes
        merged_manifest["max_total_episodes_reached"] = (
            total_episodes >= max_total_episodes
        )

    manifest_path = output_root / "hf_stage_manifest.json"
    manifest_path.write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return merged_manifest


def main() -> None:
    args = parse_args()
    manifest = merge_shards(
        shard_root=args.shard_root.resolve(),
        output_root=args.output_root.resolve(),
        repo_ids=read_repo_list(args.repo_list),
        max_total_episodes=args.max_total_episodes,
        force_symlinks=args.force_symlinks,
        manifest_only=args.manifest_only,
    )
    print(f"Merged staged HF shards at {manifest['output_root']}", flush=True)
    if args.manifest_only:
        print("Manifest-only merge: no trajectory symlinks were created.", flush=True)
    print(f"Total episodes: {manifest['total_episodes']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # pragma: no cover - shell pipeline behavior
        sys.exit(1)
