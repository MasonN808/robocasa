"""Verify that every frozen cohort configuration can use the certified scenes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_generation.task_level.scene_sampling import (
    compatible_scenes,
    load_compatibility_cache,
    sample_compatible_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-sampling-seed", type=int, default=20260819)
    args = parser.parse_args()

    cache = load_compatibility_cache(args.cache)
    targets = json.loads(args.targets.read_text())
    seen_configuration_signatures: set[tuple[str, str]] = set()
    compatibility_counts: Counter[int] = Counter()
    rows = 0
    sampled_scene_counts: Counter[str] = Counter()
    for split, tasks in targets["configurations"].items():
        for dataset_task, configurations in tasks.items():
            for row in configurations:
                key = (dataset_task, row["configuration_signature"])
                if key in seen_configuration_signatures:
                    raise ValueError(f"duplicate frozen configuration: {key}")
                seen_configuration_signatures.add(key)
                scenes = compatible_scenes(
                    cache,
                    task=row["composite_task"],
                    physical_configuration_signature=row[
                        "physical_configuration_signature"
                    ],
                )
                compatibility_counts[len(scenes)] += 1
                sampled = sample_compatible_scene(
                    cache,
                    task=row["composite_task"],
                    physical_configuration_signature=row[
                        "physical_configuration_signature"
                    ],
                    sampling_seed=args.scene_sampling_seed,
                    sample_index=rows,
                )
                repeated = sample_compatible_scene(
                    cache,
                    task=row["composite_task"],
                    physical_configuration_signature=row[
                        "physical_configuration_signature"
                    ],
                    sampling_seed=args.scene_sampling_seed,
                    sample_index=rows,
                )
                if sampled != repeated or sampled not in scenes:
                    raise ValueError(
                        "shared scene sampler is not deterministic or returned "
                        f"an incompatible scene for {key}"
                    )
                sampled_scene_counts[sampled["scene_signature"]] += 1
                rows += 1

    report = {
        "valid": True,
        "target_configuration_count": rows,
        "unique_configuration_count": len(seen_configuration_signatures),
        "compatible_scene_count_histogram": dict(sorted(compatibility_counts.items())),
        "scene_policy_version": cache["scene_policy_version"],
        "scene_sampling_seed": args.scene_sampling_seed,
        "sampled_unique_scene_count": len(sampled_scene_counts),
        "sampled_scene_count_range": [
            min(sampled_scene_counts.values()),
            max(sampled_scene_counts.values()),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
