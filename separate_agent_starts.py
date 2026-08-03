#!/usr/bin/env python
"""Moves one agent off a shared start fixture where a separate one exists.

32 of 52 specs start both agents at the same fixture. That is only a defect when
the pair sits in one `parent_fixture` workspace: the agent standing there blocks
the other from reaching the paired fixture, and the trajectory dead-ends at
navigate_to_fixture. arrange_bread_bowl starts BOTH agents on the toaster_oven.

The prompt fix makes the generator yield correctly when this happens; this
removes the situation from the spec so it does not depend on the generator
noticing. Belt and braces, since the generator rewrites start positions freely.

Only agent_1 moves, and only to a fixture that is:
  - not agent_0's fixture,
  - not in the same parent_fixture workspace as agent_0's fixture,
  - actually declared in initial_state.fixtures.

A spec with no such fixture is left ALONE and reported: forcing a move there
would either recreate the conflict elsewhere or invent a fixture the scene has
no place for.

  python separate_agent_starts.py            # report
  python separate_agent_starts.py --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os

SPECS = "data_generation/task_level/tasks/specs/verified"


def workspace_of(fixture_id: str, fixtures: dict) -> set[str]:
    """The fixture plus anything sharing a footprint with it."""

    group = {fixture_id}
    state = fixtures.get(fixture_id) or {}
    parent = state.get("parent_fixture")
    if isinstance(parent, str):
        group.add(parent)
    for other_id, other in fixtures.items():
        if not isinstance(other, dict):
            continue
        if other.get("parent_fixture") in group:
            group.add(other_id)
    return group


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    moved = skipped = untouched = 0
    for path in sorted(glob.glob(f"{SPECS}/*.json")):
        name = os.path.basename(path)[:-5]
        payload = json.loads(open(path, encoding="utf-8").read())
        initial = payload.get("initial_state") or {}
        agents = initial.get("agents") or {}
        fixtures = initial.get("fixtures") or {}
        locations = {a: (s or {}).get("location") for a, s in agents.items()}
        here = locations.get("agent_1")
        there = locations.get("agent_0")
        if len(locations) < 2 or not here or not there:
            untouched += 1
            continue
        # The defect is a shared FOOTPRINT, not a shared id: cab/counter are two
        # fixture ids in one workspace, and an agent at either blocks the other.
        blocked = workspace_of(there, fixtures) | workspace_of(here, fixtures)
        if here not in workspace_of(there, fixtures):
            untouched += 1
            continue
        # Only fixtures the task can actually navigate to are legal stations. A
        # spec may declare scenery (stool, adjacent surface) that is not in the
        # navigate allowlist; parking an agent there strands it.
        navigable = (
            (payload.get("allowed_tool_specs") or {})
            .get("navigate_to_fixture", {})
            .get("allowed_fixture_ids")
        )
        navigable = set(navigable) if isinstance(navigable, list) else set(fixtures)
        candidates = [
            fixture_id
            for fixture_id in fixtures
            if fixture_id not in blocked and fixture_id in navigable
        ]
        if not candidates:
            print(f"  SKIP {name:28} agent_0={there} agent_1={here} - no separate navigable fixture")
            skipped += 1
            continue

        target = candidates[0]
        print(f"  move {name:28} agent_1: {here} -> {target}")
        moved += 1
        if args.apply:
            agents["agent_1"]["location"] = target
            open(path, "w", encoding="utf-8").write(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )

    print(f"\nmoved {moved}, skipped {skipped}, already separate {untouched}")
    if not args.apply:
        print("report only -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
