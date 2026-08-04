#!/usr/bin/env python
"""Tick-based trajectory format: the plan written the way it executes.

The flat step list is a *serialization* of two independent per-agent streams.
Reading it top to bottom gives one interleaving, but that is an artifact of how
the JSON was written, not a constraint on anything -- the agents advance at
their own rates, so position in the list says nothing about when a step runs.
Every coordination defect measured on this dataset traces to that one gap:

    2 of 2   lock-step collisions   (plan alternates correctly, ticks collide)
  332 of 357 inert waits            (release is later in the file, earlier in time)

In tick form there is no gap to fall into. Each row is one instant; both agents
act on it simultaneously; a blocked agent is visibly blocked. The model writes
in the frame the simulator executes in, so it can see that it is about to be
held up -- which is what it needs in order to say "tell me when the machine is
free" rather than "heading over now" and then wait.

    rows  -> steps : `to_steps`, for the FSM and everything downstream
    steps -> rows  : `to_rows`, for prompting and for inspection
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

BLOCKED = "(blocked)"
IDLE = "(idle)"


def to_steps(rows: Sequence[dict[str, Any]], agent_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Flatten tick rows into the numbered step list the FSM validates.

    Within one tick the agents are simultaneous, so their order in the flat
    list is arbitrary; `agent_ids` order is used for determinism. Anything the
    model leaves out, or marks blocked or idle, contributes no step.
    """

    steps: list[dict[str, Any]] = []
    for row in rows:
        for agent_id in agent_ids:
            action = row.get(agent_id)
            if not isinstance(action, dict) or not action.get("tool"):
                continue
            step = deepcopy(action)
            step["agent"] = agent_id
            step["step"] = len(steps)
            steps.append(step)
    return steps


def to_rows(
    steps: Sequence[dict[str, Any]],
    agent_ids: Sequence[str],
    ticks: Sequence[int | None] | None = None,
) -> list[dict[str, Any]]:
    """Group steps into tick rows, scheduling them if ticks are not supplied."""

    if ticks is None:
        from insert_waits import schedule

        ticks, _ = schedule(list(steps))

    rows: dict[int, dict[str, Any]] = {}
    for step, tick in zip(steps, ticks):
        if tick is None:
            continue
        row = rows.setdefault(tick, {"tick": tick})
        entry = {k: v for k, v in step.items() if k not in {"agent", "step"}}
        row[step["agent"]] = entry
    return [rows[tick] for tick in sorted(rows)]


def render(rows: Sequence[dict[str, Any]], agent_ids: Sequence[str], width: int = 46) -> str:
    """A readable two-column view, used in the prompt and for inspection."""

    lines = [f"{'tick':>4}  " + "  ".join(a.ljust(width) for a in agent_ids)]
    lines.append("-" * (6 + (width + 2) * len(agent_ids)))
    for row in rows:
        cells = []
        for agent_id in agent_ids:
            action = row.get(agent_id)
            if not isinstance(action, dict):
                cells.append("".ljust(width))
                continue
            args = action.get("args") or {}
            detail = ", ".join(f"{k}={v}" for k, v in args.items() if not isinstance(v, (dict, list)))
            cells.append(f"{action.get('tool')}({detail})"[:width].ljust(width))
        lines.append(f"{row.get('tick', 0):>4}  " + "  ".join(cells))
    return "\n".join(lines)


def build_tick_response_schema(
    *,
    agent_ids: Sequence[str],
    allowed_tool_specs: dict[str, Any],
) -> dict[str, Any]:
    """A response schema whose top level is ticks rather than steps."""

    from data_generation.task_level.tasks.shared.schema import (
        build_task_response_schema,
    )

    # Reuse the per-step schema the flat format already validates, then re-key
    # it by agent under each tick so one row is one instant.
    flat = build_task_response_schema(
        agent_ids=agent_ids,
        allowed_tool_specs=allowed_tool_specs,
    )
    step_schema = _find_step_schema(flat)
    action_schema = {
        "type": "object",
        "properties": {
            key: value
            for key, value in (step_schema.get("properties") or {}).items()
            if key not in {"agent", "step"}
        },
        "required": [
            name
            for name in (step_schema.get("required") or [])
            if name not in {"agent", "step"}
        ],
    }

    row_properties: dict[str, Any] = {
        "tick": {"type": "integer", "description": "0-based instant; both agents act on it."},
    }
    for agent_id in agent_ids:
        row_properties[agent_id] = {
            **deepcopy(action_schema),
            "description": (
                f"What {agent_id} does at this tick. Omit the field when "
                f"{agent_id} does nothing -- it is blocked on a wait, or has "
                f"no work available."
            ),
        }

    return {
        "type": "object",
        "properties": {
            "ticks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": row_properties,
                    "required": ["tick"],
                },
            }
        },
        "required": ["ticks"],
    }


def _find_step_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Locate the per-step object schema inside the flat response schema."""

    properties = schema.get("properties") or {}
    steps = properties.get("steps") or {}
    items = steps.get("items")
    if isinstance(items, dict):
        return items
    raise ValueError("flat response schema has no steps.items to re-key by tick")


TICK_FORMAT_RULES: tuple[str, ...] = (
    "Write the trajectory as a list of ticks, not as a list of steps. One tick "
    "is one instant of time.",
    "Both agents act SIMULTANEOUSLY on each tick. Whatever you put under "
    "agent_0 and under agent_1 in the same tick happens at the same moment.",
    "Because of that, the two agents must never act on the same object, or at "
    "the same exclusive fixture, on the same tick. Two robots cannot reach "
    "into one cabinet at once.",
    "Omit an agent from a tick when it does nothing. An agent that is waiting "
    "does nothing until it is released -- leave it out of those ticks.",
    "wait_for_signal(from, about) blocks the calling agent until the other "
    "agent sends a communicate whose releases names the same id. Nothing else "
    "wakes it, so put the wait where the agent genuinely cannot proceed.",
    "A WAIT IS HALF A PROTOCOL, AND YOU MUST WRITE BOTH HALVES. Every "
    "wait_for_signal(from=X, about=R) you write obliges agent X to send "
    "communicate(to=<waiter>, releases=[\"R\"]) on a LATER tick, naming R "
    "exactly. A plain message does not count -- only the releases field wakes "
    "the waiter. If you write the wait and forget X's release, that agent "
    "never moves again and every remaining tick of the plan is dead. Before "
    "you finish, go through your waits one by one and find the matching "
    "release for each.",
    "The wait must come BEFORE the release, never on the same tick and never "
    "after it. A release only wakes an agent that is ALREADY waiting -- one "
    "sent earlier was delivered to nobody and the waiter blocks forever. And "
    "if the resource was already released, there is nothing left to wait for: "
    "that wait is dead weight, so delete it rather than writing it.",
    "Release only what you have genuinely let go of, and release it "
    "IMMEDIATELY. Before X sends releases=[\"R\"]: if R is a fixture, X must "
    "already have called give_space on it or navigated somewhere else; if R "
    "is an object, X must have put it down. That departure must be the call "
    "RIGHT BEFORE the release -- no other messages or actions in between, "
    "because every tick in that gap is a tick the other agent sits blocked on "
    "something that is already free. Saying \"you can take it\" while still "
    "standing at the fixture is not a handover at all.",
    "Order what an agent says around what it is about to do. If you are about "
    "to be held up, ask BEFORE announcing that you are on your way: say "
    "\"tell me when the machine is free\", wait, and only then say \"heading "
    "over\". Announcing a move you cannot make yet is wrong.",
    "A handover has FOUR parts. They go in this order and no other:\n"
    "    tick 3   agent_1: communicate(to=agent_0, message=\"tell me when the "
    "cabinet is free\")       <- 1. ASK\n"
    "    tick 4   agent_1: wait_for_signal(from=agent_0, about=\"cab\")"
    "                       <- 2. BLOCK (agent_1 is now absent from every\n"
    "                                                                        "
    "                             tick until it is released)\n"
    "    tick 5   agent_0: give_space(fixture_id=\"cab\")"
    "                                  <- 3a. LEAVE, for real\n"
    "    tick 6   agent_0: communicate(to=agent_1, releases=[\"cab\"])"
    "                     <- 3b. REPORT it, immediately after\n"
    "    tick 7   agent_1: navigate_to_fixture(fixture_id=\"cab\")"
    "                        <- 4. MOVE IN\n"
    "Parts 3a and 3b are adjacent on purpose: the release is the report of the "
    "departure, so nothing goes between them. Part 2 comes before part 3b, "
    "because a release only reaches an agent that is already waiting.",
    "Being idle is a cost. If an agent has nothing to do for many ticks, the "
    "work is badly divided -- give it something, or shorten the wait.",
)
