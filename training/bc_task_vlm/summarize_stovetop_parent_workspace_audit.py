"""Strictly summarize the production stovetop parent-workspace regression."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-scene-count", type=int, default=75)
    parser.add_argument("--require-record-count", type=int, default=390)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    records = []
    errors = []
    for path in files:
        payload = json.loads(path.read_text())
        records.extend(payload.get("records") or [])
        errors.extend(payload.get("errors") or [])
    baseline_passes = sum(bool(row["baseline"]["success"]) for row in records)
    fixed_passes = sum(bool(row["fixed"]["success"]) for row in records)
    fixed_failures = [row for row in records if not row["fixed"]["success"]]
    summary = {
        "valid": not errors and not fixed_failures,
        "scene_count": len(files),
        "record_count": len(records),
        "baseline_passes": baseline_passes,
        "fixed_passes": fixed_passes,
        "fixed_failures": len(fixed_failures),
        "task_counts": dict(sorted(Counter(row["task"] for row in records).items())),
        "task_initialization_errors": errors,
    }
    if len(files) != args.require_scene_count:
        raise ValueError(f"found {len(files)} scenes, expected {args.require_scene_count}")
    if len(records) != args.require_record_count:
        raise ValueError(
            f"found {len(records)} applicable cases, expected {args.require_record_count}"
        )
    if errors:
        raise ValueError(f"audit contains {len(errors)} task initialization errors")
    if fixed_failures:
        raise ValueError(f"production fix failed {len(fixed_failures)} cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
