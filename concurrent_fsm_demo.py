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
        "4. A handover that works",
        "The four-part shape from TICK_FORMAT_RULES: ask, LEAVE, release, move in.\n"
        "The wait clears on the same instant the release fires.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is yours now", releases=["cab"]),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
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
         s(A0, "give_space", fixture_id="cab"),
         talk(A0, "cab is yours now"),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         s(A1, "navigate_to_fixture", fixture_id="cab")],
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
        "8. THE give_space LIMITATION -- caught, but for the wrong reason",
        "agent_0 releases the cabinet WITHOUT leaving it: a promise, not a\n"
        "departure. The protocol is formally perfect -- ask, wait, release,\n"
        "arrive -- and `releases` has no physical precondition, so nothing\n"
        "objects to it. The conflict below is reported because agent_1 HAPPENS\n"
        "to arrive while agent_0 is still standing there. The protocol error is\n"
        "never named; only its consequence is.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         talk(A0, "you can take it", releases=["cab"]),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         talk(A0, "just tidying up"),
         s(A1, "navigate_to_fixture", fixture_id="cab"),
         s(A0, "give_space", fixture_id="cab")],
    ),
    (
        "9. The same defect, one instant later -- not caught at all",
        "Byte-for-byte the same protocol error as scenario 8. agent_1 has one\n"
        "extra message to send, so agent_0's give_space lands first and the two\n"
        "never overlap. Clean. The plan is no more correct than scenario 8; the\n"
        "clock just absolved it. date_night/traj_000000 is exactly this -- dirty\n"
        "under lock_step, clean under executor.\n"
        "A STRUCTURAL rule -- 'do not release a fixture you are still standing\n"
        "at' -- needs no clock and would catch both 8 and 9.",
        [*OPEN,
         s(A0, "navigate_to_fixture", fixture_id="cab"),
         talk(A1, "tell me when the cabinet is free"),
         talk(A0, "you can take it", releases=["cab"]),
         s(A1, "wait_for_signal", **{"from": A0, "about": "cab"}),
         talk(A1, "thanks, finishing up here"),
         s(A0, "give_space", fixture_id="cab"),
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
        verdict = (
            "DEADLOCK" if run.deadlocked
            else f"{len(run.conflicts)} CONFLICT(S)" if run.conflicts
            else "clean"
        )
        print(f"\n  -> {verdict}   makespan={run.makespan:g}  idle={run.idle}")
        for line in run.conflicts:
            print(line)
        print()


if __name__ == "__main__":
    main()
