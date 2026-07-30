#!/usr/bin/env python
"""Finds the actual `<task>/traj_XXXXXX/...` root inside a downloaded dataset
snapshot, in case the task directories are nested under some prefix (e.g. a
generation-run timestamp directory) rather than sitting at the snapshot root.

Walks the tree breadth-first, bounded depth, looking for a directory whose
immediate children include at least half of the selection's task names.
Prints the resolved path on success; exits non-zero with a clear error if no
such directory is found within the depth bound.

Usage:
  python resolve_dataset_root.py --search-root <dir> \
    --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
    [--max-depth 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    all_tasks = set(selection["train_tasks"]) | set(selection["held_out_tasks"])

    frontier = [(args.search_root, 0)]
    best_match: tuple[Path, int] | None = None
    while frontier:
        current_dir, depth = frontier.pop(0)
        if not current_dir.is_dir():
            continue
        children = {p.name for p in current_dir.iterdir() if p.is_dir()}
        overlap = len(children & all_tasks)
        if overlap >= len(all_tasks) // 2:
            if best_match is None or overlap > best_match[1]:
                best_match = (current_dir, overlap)
            if overlap == len(all_tasks):
                break
        if depth < args.max_depth:
            for child in current_dir.iterdir():
                if child.is_dir():
                    frontier.append((child, depth + 1))

    if best_match is None:
        raise SystemExit(
            f"Could not find a directory under {args.search_root} whose children "
            f"match the {len(all_tasks)} task names (searched depth {args.max_depth})."
        )
    resolved_dir, overlap = best_match
    if overlap < len(all_tasks):
        print(
            f"warning: best match only covers {overlap}/{len(all_tasks)} tasks",
            file=sys.stderr,
        )
    print(resolved_dir)


if __name__ == "__main__":
    main()
