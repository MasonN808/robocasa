#!/usr/bin/env python
"""Find steps that collide when a trajectory is run in lock-step.

Under `--uniform-durations` every tool call costs one tick, so both agents
advance together and only a wait plus its release can order them. That makes
simultaneity a property of the trajectory rather than of a duration model: the
schedule is fully determined by the plan, so it can be computed offline.

The FSM cannot do this itself. It replays a flat list of steps one at a time
against one mutating state, so "these two happen at once" is not expressible in
its model -- answering it means evaluating two steps against the SAME pre-state
and intersecting what they touch. This runs as a separate pass over the pinned
lock-step schedule instead.

    python lockstep_check.py --root <dataset>          # count collisions
    python lockstep_check.py --root <dataset> --show 5 # print examples
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

# One source of truth: the schedule and footprint model live with the pass
# that acts on them, so the checker cannot drift from the inserter.
from insert_waits import footprint, schedule, tick_collisions  # noqa: F401


def collisions(steps, fixtures, objects):
    """Pairs sharing a tick whose footprints intersect, plus deadlock state."""

    _, deadlocked = schedule(steps)
    return tick_collisions(steps, fixtures, objects), deadlocked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--pattern", default="*/traj_*/original_trajectory.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    stats = Counter()
    resources = Counter()
    shown = 0
    for path in paths:
        payload = json.loads(open(path, encoding="utf-8").read())
        initial = payload.get("initial_state") or {}
        steps = payload.get("steps") or []
        found, deadlocked = collisions(
            steps,
            initial.get("fixtures") or {},
            set(initial.get("objects") or {}),
        )
        stats["trajectories"] += 1
        stats["deadlocked"] += bool(deadlocked)
        stats["colliding"] += bool(found)
        stats["collisions"] += len(found)
        for _, _, _, shared in found:
            resources.update(shared)
        if found and shown < args.show:
            shown += 1
            rel = os.path.relpath(path, args.root)
            print(f"\n{rel}")
            for tick, left, right, shared in found[:4]:
                a, b = steps[left], steps[right]
                print(f"  tick {tick}: {a['agent']} {a['tool']}{a.get('args')}")
                print(f"          {b['agent']} {b['tool']}{b.get('args')}")
                print(f"          both need: {', '.join(shared)}")

    print(f"\ntrajectories        : {stats['trajectories']}")
    print(f"with a collision    : {stats['colliding']}")
    print(f"total collisions    : {stats['collisions']}")
    print(f"deadlocked          : {stats['deadlocked']}")
    if resources:
        print(f"most contended      : {resources.most_common(8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
