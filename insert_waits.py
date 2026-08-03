#!/usr/bin/env python
"""Phase 2: insert the sharing protocol into plans generated without it.

Phase 1 writes an ordinary sequential plan -- which the model does reliably --
and this computes where the two agents contend, with the whole plan in hand.
That is the part the model could not do while writing: four ordered obligations
spanning both agents and often ten steps. Four prompt-side approaches were tried
and all landed at 0-18%.

Contention is exactly two things:
  * the same OBJECT used by both agents;
  * the same EXCLUSIVE fixture (one opening, or too small for two robots).
A roomy counter shared by two agents on different objects is not contention.

For each one, three symbolically inert steps go in immediately before the
dependent action, so nothing downstream shifts in meaning:

    communicate(B -> A, "... <X> ...")     announcement: under partial
                                           observability the wait is invisible
                                           to A, so without this nobody knows
                                           to release it
    wait_for_signal(from=A, about=X)       B blocks
    communicate(A -> B, "... <X> ...")     release; A has already finished with
                                           X earlier in the recorded order

Nothing with world-state effects is inserted. `give_space` is never added: for a
fixture the releasing act is simply that A moved on, which the plan already has.

Deadlock is checked, not assumed: inserted waits can form a cycle (B waits on A
while A waits on B), which the FSM cannot see and which hangs live-sim. The
check replays the per-agent streams the way the scheduler does and fails loudly
if both agents end up blocked.

  python insert_waits.py --root <dir>           # report
  python insert_waits.py --root <dir> --out <dir> --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

from data_generation.task_level.tasks.shared.constants import (
    DEPENDENCY_ARG_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    FIXTURE_ARG_NAMES,
    OBSERVATION_TOOL_NAMES,
    SOCIAL_TOOL_NAMES,
)

# Varied phrasing on purpose. One fixed string would let a model resume on the
# template rather than on meaning, which is the degenerate policy this whole
# mechanism exists to measure.
ASKS = [
    "I need {x} next, so I am holding off until you are finished with it.",
    "Tell me when you are done with {x} and I will take it from there.",
    "I am waiting on {x} before I carry on with my part.",
    "Let me know once {x} is free; I cannot continue without it.",
]
FREES = [
    "I am finished with {x} now, it is all yours.",
    "Done with {x} -- go ahead whenever you are ready.",
    "{x} is free now, I have moved on to my next step.",
    "You can take {x}, I have finished what I needed there.",
]


def _contested(step, fixtures, objects):
    """Ids this step contends for: objects anywhere, exclusive fixtures only."""

    if step.get("tool") in SOCIAL_TOOL_NAMES or step.get("tool") in OBSERVATION_TOOL_NAMES:
        return []
    args = step.get("args") or {}
    out = []
    for name in DEPENDENCY_ARG_NAMES:
        value = args.get(name)
        if isinstance(value, str) and value in objects:
            out.append(value)
    for name in FIXTURE_ARG_NAMES:
        value = args.get(name)
        if not isinstance(value, str) or value not in fixtures:
            continue
        kind = str((fixtures.get(value) or {}).get("fixture_type", "")).lower()
        if kind in EXCLUSIVE_FIXTURE_TYPES:
            out.append(value)
    return out


def _guarded(steps, upto, actor, resource, holder, since):
    return any(
        s.get("tool") == "wait_for_signal"
        and s.get("agent") == actor
        and str((s.get("args") or {}).get("about")) == resource
        and (s.get("args") or {}).get("from") == holder
        for s in steps[since + 1 : upto]
    )


def insert(steps, fixtures, objects, rng):
    """Returns (new_steps, inserted_count)."""

    held: dict[str, tuple[str, int]] = {}
    out: list[dict] = []
    inserted = 0
    for index, step in enumerate(steps):
        actor = step.get("agent")
        for resource in _contested(step, fixtures, objects):
            holder, at = held.get(resource, (None, -1))
            if holder is None or holder == actor:
                continue
            if _guarded(steps, index, actor, resource, holder, at):
                continue
            out.append({"agent": actor, "tool": "communicate",
                        "args": {"to": holder,
                                 "message": rng.choice(ASKS).format(x=resource)},
                        "reasoning": f"I need {resource} and must wait."})
            out.append({"agent": actor, "tool": "wait_for_signal",
                        "args": {"from": holder, "about": resource},
                        "reasoning": f"Waiting for {resource}."})
            out.append({"agent": holder, "tool": "communicate",
                        "args": {"to": actor,
                                 "message": rng.choice(FREES).format(x=resource)},
                        "reasoning": f"I am done with {resource}."})
            inserted += 1
        out.append(step)
        for resource in _contested(step, fixtures, objects):
            held[resource] = (actor, index)

    # The model's own waits often carry a degenerate release -- a message whose
    # entire text is the id, which discharges nothing. Give those a real one.
    # It must go after the holder's LAST use of the thing, not straight after
    # the wait: dropping it next to the wait produces exactly the promise the
    # rule exists to reject, because the holder is still using it.
    def _uses(step, resource):
        if step.get("tool") in SOCIAL_TOOL_NAMES or step.get("tool") in OBSERVATION_TOOL_NAMES:
            return False
        if step.get("tool") in ("give_space",):
            return str((step.get("args") or {}).get("fixture_id")) == resource
        return any(str(v) == resource for v in (step.get("args") or {}).values())

    pending: dict[int, dict] = {}
    for index, step in enumerate(out):
        if step.get("tool") != "wait_for_signal":
            continue
        args = step.get("args") or {}
        about, holder = str(args.get("about")), args.get("from")
        needle = about.casefold()
        waiter_next = next(
            (j for j in range(index + 1, len(out))
             if out[j]["agent"] == step["agent"] and _uses(out[j], about)),
            len(out),
        )
        if any(
            s.get("tool") == "communicate" and s.get("agent") == holder
            and (s.get("args") or {}).get("to") == step["agent"]
            and needle in str((s.get("args") or {}).get("message", "")).casefold()
            and str((s.get("args") or {}).get("message", "")).casefold()
                 .replace(needle, "").strip(" .,;:!")
            for s in out[index + 1 : waiter_next]
        ):
            continue
        # after the holder finishes with it, and before the waiter needs it
        last_use = max(
            (j for j in range(index + 1, waiter_next)
             if out[j]["agent"] == holder and _uses(out[j], about)),
            default=index,
        )
        pending[last_use] = {
            "agent": holder, "tool": "communicate",
            "args": {"to": step["agent"],
                     "message": rng.choice(FREES).format(x=about)},
            "reasoning": f"I am done with {about}.",
        }
        inserted += 1

    repaired: list[dict] = []
    for index, step in enumerate(out):
        repaired.append(step)
        if index in pending:
            repaired.append(pending[index])

    for number, step in enumerate(repaired):
        step["step"] = number
    return repaired, inserted


def deadlocks(steps) -> bool:
    """Replays per-agent streams like the scheduler; True if both agents stick."""

    queues: dict[str, list[dict]] = {}
    for step in steps:
        queues.setdefault(step["agent"], []).append(step)
    waiting: dict[str, str | None] = {a: None for a in queues}
    delivered: dict[str, set[str]] = {a: set() for a in queues}
    while any(queues.values()):
        moved = False
        for agent, queue in queues.items():
            if not queue:
                continue
            if waiting[agent] is not None:
                if waiting[agent] in delivered[agent]:
                    waiting[agent] = None
                else:
                    continue
            step = queue[0]
            if step["tool"] == "wait_for_signal":
                about = str((step.get("args") or {}).get("about"))
                if about not in delivered[agent]:
                    waiting[agent] = about
                    continue
            queue.pop(0)
            moved = True
            if step["tool"] == "communicate":
                target = (step.get("args") or {}).get("to")
                message = str((step.get("args") or {}).get("message", ""))
                if target in delivered:
                    for token in message.replace(",", " ").split():
                        delivered[target].add(token.strip(".;:"))
        if not moved:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(os.path.join(args.root, "*", "trajectories", "*.json")))
    stats = Counter()
    for path in paths:
        payload = json.loads(open(path, encoding="utf-8").read())
        initial = payload.get("initial_state") or {}
        steps = payload.get("steps") or []
        new_steps, inserted = insert(
            steps, initial.get("fixtures") or {}, set(initial.get("objects") or {}), rng
        )
        stats["trajectories"] += 1
        stats["inserted"] += inserted
        stats["touched"] += bool(inserted)
        stats["grew"] += len(new_steps) - len(steps)
        if deadlocks(new_steps):
            stats["DEADLOCK"] += 1
            continue
        if args.apply and args.out:
            rel = os.path.relpath(path, args.root)
            dest = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            payload["steps"] = new_steps
            with open(dest, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

    n = max(stats["trajectories"], 1)
    print(f"trajectories        : {stats['trajectories']}")
    print(f"  needed a wait     : {stats['touched']}")
    print(f"  waits inserted    : {stats['inserted']}  ({stats['inserted']/n:.1f}/traj)")
    print(f"  steps added       : {stats['grew']}")
    print(f"  DEADLOCKED        : {stats['DEADLOCK']}")
    if not args.apply:
        print("\nreport only -- pass --out <dir> --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
