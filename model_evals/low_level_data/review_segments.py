"""Interactively accept or reject collected raw segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output", default=None, help="Optional JSONL of accepted segment metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_paths = sorted(Path(args.segments_dir).glob("**/metadata.json"))
    accepted_records = []
    for idx, path in enumerate(metadata_paths):
        metadata = json.loads(path.read_text())
        print("\n" + "=" * 80)
        print(f"[{idx + 1}/{len(metadata_paths)}] {path.parent}")
        for key in ("task", "tool", "agent", "prompt", "source", "backend", "predicate_success", "accepted"):
            print(f"{key}: {metadata.get(key)}")
        decision = input("[a] accept, [r] reject, [s] skip, [q] quit > ").strip().lower()
        if decision == "q":
            break
        if decision == "s":
            continue
        if decision not in {"a", "r"}:
            print("Unknown choice; leaving unchanged.")
            continue
        metadata["accepted"] = decision == "a"
        metadata["accepted_by_reviewer"] = decision == "a"
        path.write_text(json.dumps(metadata, indent=2, default=str))
        if decision == "a":
            accepted_records.append(metadata)
    if args.output:
        with Path(args.output).open("w") as f:
            for record in accepted_records:
                f.write(json.dumps(record, default=str) + "\n")
        print(f"Wrote accepted records: {args.output}")


if __name__ == "__main__":
    main()
