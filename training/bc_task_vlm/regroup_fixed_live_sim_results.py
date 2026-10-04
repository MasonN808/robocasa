#!/usr/bin/env python3
"""Regroup completed per-task live-sim episodes under a new task split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bc_task_vlm.live_sim_eval import _write_metrics


SPLITS = ("train_task_types", "heldout_task_types")


def _latest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["episode_id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    membership = {
        split: set(manifest["configurations"][split]) for split in SPLITS
    }
    if membership[SPLITS[0]] & membership[SPLITS[1]]:
        raise ValueError("target manifest task splits overlap")

    source_rows: dict[str, dict] = {}
    for split in SPLITS:
        path = args.source_run / split / "aggregate" / "live_sim_trajectories.jsonl"
        for episode_id, row in _latest(path).items():
            if episode_id in source_rows:
                raise ValueError(f"duplicate episode across source splits: {episode_id}")
            source_rows[episode_id] = row

    expected_tasks = membership[SPLITS[0]] | membership[SPLITS[1]]
    actual_tasks = {str(row["task_name"]) for row in source_rows.values()}
    if actual_tasks != expected_tasks:
        raise ValueError(
            f"task mismatch: missing={sorted(expected_tasks-actual_tasks)}, "
            f"extra={sorted(actual_tasks-expected_tasks)}"
        )

    receipt = {
        "schema_version": 1,
        "operation": "exact_task_membership_regroup",
        "source_run": str(args.source_run.resolve()),
        "target_manifest": str(args.target_manifest.resolve()),
        "target_manifest_content_hash": manifest.get("content_hash"),
        "source_episode_count": len(source_rows),
        "splits": {},
    }
    for split in SPLITS:
        selected = [
            dict(row, cohort_split=split)
            for row in source_rows.values()
            if str(row["task_name"]) in membership[split]
        ]
        selected.sort(key=lambda row: (str(row["task_name"]), str(row["episode_id"])))
        aggregate = args.output_run / split / "aggregate"
        aggregate.mkdir(parents=True, exist_ok=True)
        results = aggregate / "live_sim_trajectories.jsonl"
        results.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected),
            encoding="utf-8",
        )
        _write_metrics(results, aggregate)
        receipt["splits"][split] = {
            "num_tasks": len(membership[split]),
            "num_episodes": len(selected),
            "episode_ids": [str(row["episode_id"]) for row in selected],
        }
    if sum(item["num_episodes"] for item in receipt["splits"].values()) != len(source_rows):
        raise RuntimeError("regrouping did not preserve every source episode exactly once")
    (args.output_run / "regroup_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
