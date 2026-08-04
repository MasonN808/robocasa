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

from insert_waits import (
    EXCLUSIVE_FIXTURE_TYPES,
    GIVE_SPACE_TOOL_NAMES,
    FIXTURE_ARG_NAMES,
    OBSERVATION_TOOL_NAMES,
    SOCIAL_TOOL_NAMES,
    _contested,
)

WAIT_TOOL = "wait_for_signal"


def _released_ids(step):
    """Symbolic ids this communicate hands over."""

    value = (step.get("args") or {}).get("releases") or []
    return {value} if isinstance(value, str) else {str(v) for v in value}


def schedule(steps):
    """Assign each step the tick it runs at under lock-step.

    Mirrors the live-sim scheduler: at each tick every agent that is not
    blocked on an undischarged wait executes its next step, all at once.

    Returns (ticks, deadlocked) where ticks[i] is the tick of steps[i], or
    None for steps never reached because the agents deadlocked.
    """

    order: dict[str, list[int]] = {}
    for index, step in enumerate(steps):
        order.setdefault(step.get("agent"), []).append(index)

    pointer = {agent: 0 for agent in order}
    ticks: list[int | None] = [None] * len(steps)
    released: set[tuple[str, str]] = set()  # (releasing agent, id)
    tick = 0

    while True:
        runnable = []
        for agent, indices in order.items():
            if pointer[agent] >= len(indices):
                continue
            index = indices[pointer[agent]]
            step = steps[index]
            if step.get("tool") == WAIT_TOOL:
                args = step.get("args") or {}
                key = (args.get("from"), str(args.get("about")))
                if key not in released:
                    continue  # still blocked
            runnable.append((agent, index))

        if not runnable:
            break

        # Every runnable agent acts at this same tick -- that is lock-step.
        for agent, index in runnable:
            ticks[index] = tick
            step = steps[index]
            if step.get("tool") == "communicate":
                for released_id in _released_ids(step):
                    released.add((agent, released_id))
            pointer[agent] += 1
        tick += 1

    deadlocked = any(pointer[a] < len(order[a]) for a in order)
    return ticks, deadlocked


def _footprint(step, fixtures, objects):
    """What this step occupies: contested ids plus the floor space it stands on."""

    if step.get("tool") in SOCIAL_TOOL_NAMES or step.get("tool") in OBSERVATION_TOOL_NAMES:
        return set()
    # give_space is a departure, not a claim. One agent stepping off a fixture
    # while the other steps on is the handoff working, not a collision --
    # insert_waits takes the same view and never guards it.
    if step.get("tool") in GIVE_SPACE_TOOL_NAMES:
        return set()
    out = set(_contested(step, fixtures, objects))
    # Two agents cannot approach one exclusive fixture at the same instant even
    # when they touch different objects there, and a fixture standing on
    # another shares its floor space.
    args = step.get("args") or {}
    for name in FIXTURE_ARG_NAMES:
        value = args.get(name)
        if not isinstance(value, str):
            continue
        spec = fixtures.get(value) or {}
        kind = str(spec.get("fixture_type", "")).lower()
        if kind in EXCLUSIVE_FIXTURE_TYPES:
            out.add(value)
            parent = spec.get("parent_fixture")
            if isinstance(parent, str):
                out.add(parent)
    return out


def collisions(steps, fixtures, objects):
    """Pairs of steps that run at the same tick and touch the same thing."""

    ticks, deadlocked = schedule(steps)
    by_tick: dict[int, list[int]] = {}
    for index, tick in enumerate(ticks):
        if tick is not None:
            by_tick.setdefault(tick, []).append(index)

    found = []
    for tick, indices in sorted(by_tick.items()):
        if len(indices) < 2:
            continue
        prints = {i: _footprint(steps[i], fixtures, objects) for i in indices}
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if steps[left].get("agent") == steps[right].get("agent"):
                    continue
                shared = prints[left] & prints[right]
                if shared:
                    found.append((tick, left, right, sorted(shared)))
    return found, deadlocked


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
