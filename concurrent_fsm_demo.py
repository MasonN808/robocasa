#!/usr/bin/env python
"""Worked examples of what the concurrent validator does and does not catch.

Each scenario isolates ONE behaviour. Run it and read the two columns: the row
label is the instant, and everything on a row happens at the same moment.

    python concurrent_fsm_demo.py            # all scenarios, lock-step
    python concurrent_fsm_demo.py --model executor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generation.task_level.tasks.shared.concurrent_fsm import (  # noqa: E402
    DURATION_MODELS,
    LOCK_STEP,
    ConcurrentTaskValidator,
)
from tests.test_concurrent_fsm import FakeValidator, OPEN, plan, step  # noqa: E402


def s(agent, tool, **args):
    return step(agent, tool, **args)


A0, A1 = "agent_0", "agent_1"


def talk(agent, text, **extra):
    other = A1 if agent == A0 else A0
    return s(agent, "communicate", to=other, message=text, **extra)


SCENARIOS: list[tuple[str, str, list[dict]]] = [
    (
        "1. Two streams, one clock",
        "Nothing is shared. Both agents just advance. Position in the file says\n"
        "nothing: agent_1's navigate is written LAST but runs at instant 1,\n"
        "because it is that agent's first move while agent_0 is two calls deep.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         s(A0, "give_space", fixture_id="cab"),
         s(A1, "navigate_to_fixture", fixture_id="sink")],
    ),
    (
        "2. Both grab the same object at once",
        "The plan reads fine top-to-bottom -- one pick_up after the other. On a\n"
        "clock they are the same instant. CAUGHT by invariant 1.",
        [*OPEN,
         s(A0, "pick_up_object", object_id="bowl", source_id="counter"),
         s(A1, "pick_up_object", object_id="bowl", source_id="counter")],
    ),
    (
        "3. The same object, genuinely one after the other",
        "Identical resource, but agent_1 spends three instants elsewhere first.\n"
        "Sequential use is not contention, so NO wait is demanded here. This is\n"
        "the case the old structural rule got wrong 220 times.",
        [*OPEN,
         s(A0, "pick_up_object", object_id="bowl", source_id="counter"),
         s(A1, "navigate_to_fixture", fixture_id="sink"),
         s(A1, "give_space", fixture_id="sink"),
         s(A1, "pick_up_object", object_id="bowl", source_id="counter")],
    ),
    (
        "4. A handover that works -- the four parts",
        "1 ASK, 2 BLOCK on the very next tick, then agent_0 works as long as it\n"
        "likes (agent_1's column is empty throughout -- that is what blocked looks\n"
        "like), 3a LEAVE, 3b REPORT on the very next tick, 4 MOVE IN.\n"
        "Two adjacencies (ask->block, leave->report) and one ordering (the waiter\n"
        "is blocked BEFORE the holder lets go). The gap in the middle is free.",
        [*OPEN,
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A0, "still working here"),
         talk(A0, "nearly done"),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is yours now", releases=["cab"]),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
    (
        "5. The release is forgotten",
        "Same plan, minus agent_0's releases field. A plain message does not wake\n"
        "a waiter. agent_1 blocks forever -- and this is 7 of the 12 trajectories\n"
        "the tick-format A/B produced.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is yours now")],
    ),
    (
        "6. The release arrives before anyone is waiting",
        "agent_0 releases at instant 1; agent_1 does not start waiting until 3.\n"
        "A release is an EVENT, not a standing fact -- the executor wakes a waiter\n"
        "by DELIVERING a message, and this one was delivered to nobody.",
        [*OPEN,
         talk(A0, "cab is free", releases=["cab"]),
         s(A1, "navigate_to_fixture", fixture_id="sink"),
         s(A1, "give_space", fixture_id="sink"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"})],
    ),
    (
        "7. Two arrivals, three instants apart",
        "Nobody acts at the same instant, so no same-instant rule fires. But\n"
        "agent_0 never left. CAUGHT by invariant 2, which reads OCCUPANCY rather\n"
        "than calls -- the blind spot every earlier pass had.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "on my way"),
         talk(A1, "still coming"),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
    (
        "8. Releasing a fixture you are still standing at",
        "agent_0 hands over the cabinet without leaving it: a promise, not a\n"
        "departure. This used to be invisible -- `releases` was a pure speech act,\n"
        "and whether it showed up depended on whether the arrival HAPPENED to\n"
        "overlap. Now it is rejected outright, under every duration model.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         talk(A0, "you can take it", releases=["cab"]),
         s(A1, "navigate_to_fixture", fixture_id="cab"),
         s(A0, "give_space", fixture_id="cab")],
    ),
    (
        "9. The same defect, one instant later",
        "Scenario 8 with agent_1 delayed one instant, so agent_0's give_space now\n"
        "lands before the arrival and the two never overlap. Under the old rules\n"
        "this was CLEAN -- same error, opposite verdict, decided by the clock.\n"
        "The structural rule catches it identically, which is the whole point of\n"
        "making it structural. date_night/traj_000000 is this shape.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         talk(A0, "you can take it", releases=["cab"]),
         talk(A1, "thanks, finishing up here"),
         s(A0, "give_space", fixture_id="cab"),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
    (
        "10. A gap between leaving and reporting",
        "agent_0 does leave, but chats for an instant before saying so. Every tick\n"
        "in that gap is a tick agent_1 sits blocked on a cabinet that is already\n"
        "free. The release must be the very next thing after the departure.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "just tidying up"),
         talk(A0, "all done now", releases=["cab"]),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
    (
        "11. Waiting for something already released",
        "agent_0 completes the handover before agent_1 ever blocks. There is\n"
        "nothing left to wait for, so the wait is dead weight -- and because a\n"
        "release only wakes an agent ALREADY waiting, it is worse than useless:\n"
        "agent_1 blocks on a message that has come and gone.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is free", releases=["cab"]),
         talk(A1, "one moment"),
         talk(A1, "nearly ready"),
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
    (
        "12. Working between the ask and the wait",
        "agent_1 asks, then goes off and does something else before blocking. If\n"
        "it had work available it was never stuck, and the ask told agent_0 a\n"
        "handover was owed one tick too early. Ask and block are adjacent.",
        [*OPEN,
         talk(A1, "tell me when the cabinet is free"),
         s(A1, "navigate_to_fixture", fixture_id="sink"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A0, "nearly done"),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is yours now", releases=["cab"]),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LOCK_STEP, choices=DURATION_MODELS)
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()

    for number, (title, blurb, steps) in enumerate(SCENARIOS, start=1):
        if args.only and args.only != number:
            continue
        validator = ConcurrentTaskValidator(FakeValidator())
        candidate = plan(*steps)
        print("=" * 78)
        print(title)
        print(blurb)
        print("-" * 78)
        print(validator.render(candidate, model=args.model, width=40))
        run = validator.replay(candidate, model=args.model, stop_on_step_error=False)
        parts = []
        if run.deadlocked:
            parts.append("DEADLOCK")
        if run.conflicts and not run.deadlocked:
            parts.append(f"{len(run.conflicts)} CONFLICT(S)")
        if run.protocol:
            parts.append(f"{len(run.protocol)} PROTOCOL ERROR(S)")
        verdict = " + ".join(parts) or "clean"
        print(f"\n  -> {verdict}   makespan={run.makespan:g}  idle={run.idle}")
        for line in run.conflicts + run.protocol:
            print(line)
        print()


if __name__ == "__main__":
    main()
