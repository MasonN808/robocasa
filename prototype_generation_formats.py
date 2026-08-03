#!/usr/bin/env python
"""Prototypes two output formats against the sequential-list baseline.

Three prompt/output shapes generate the same task; each result is flattened to
the standard `steps` list and put through the real FSM. sweeten_coffee is the
subject because it failed 8 of 8 attempts under every configuration tried so
far, always for the same reason: `coffee` is handed between the agents and the
model never guards the handoff.

  timeline  - two columns, one row per tick. Simultaneity is structural: a row
              shows what BOTH agents do at the same moment, so a conflict is two
              cells in one row touching the same thing.
  ownership - every object has an owner; an agent may only act on what it owns;
              transfer needs an explicit handoff. The constraint becomes a thing
              the model cannot express illegally rather than a rule to recall.
  baseline  - the current numbered list, for reference.

  python prototype_generation_formats.py --format timeline --n 6
"""

from __future__ import annotations

import argparse
import json
import os
import re

TASK = "SweetenCoffee"
SPEC = "data_generation/task_level/tasks/specs/verified/sweetencoffee.json"

WORLD = """
Agents start at: agent_0 at `counter`, agent_1 at `fridge`.
Objects: `coffee` on counter, `milk` in fridge, `saucer_plate` on counter,
         `sugar_cube` on saucer_plate.
Fixtures: `counter` (roomy: both agents can work here at once),
          `fridge` (tight: only ONE agent at a time),
          `coffee_machine` (tight: only ONE agent at a time).
Tools: navigate_to_fixture(fixture_id), pick_up_object(object_id, source_id),
       place_in_receptacle(object_id, receptacle_id),
       place_next_to(object_id, reference_object_id),
       open_hinged_part(target_id, part_id), give_space(fixture_id),
       communicate(to, message), wait_for_signal(from, about)
Goal: the coffee ends up sweetened and served -- sugar_cube into coffee, and
      milk brought from the fridge to the counter.
"""

TIMELINE = """You are planning for TWO robots that act AT THE SAME TIME.

Write the plan as a TIMELINE: a list of ticks. At every tick BOTH robots do
something simultaneously. This is the whole point -- they are not taking turns.

Output JSON: {"timeline": [{"tick": 0, "agent_0": {...}, "agent_1": {...}}, ...]}
Each cell is {"tool": ..., "args": {...}, "reasoning": "one short sentence"}
or {"tool": "idle"} if that robot genuinely has nothing to do this tick.

Because both cells in a row happen at the same instant, look at each row and ask:
are these two robots reaching for the same object, or crowding the same tight
fixture? If so the row is illegal -- one of them must be waiting instead.

To hand something over, use this shape across ticks:
  - the receiver says what it is waiting for (naming the exact id),
  - the receiver's cells are wait_for_signal(from=..., about=<id>) while blocked,
  - the giver finishes with the thing,
  - THEN the giver sends a message naming the id saying it is done.
"""

OWNERSHIP = """You are planning for TWO robots that act AT THE SAME TIME.

Every object and every tight fixture has exactly ONE owner at any moment. A robot
may only act on things it currently owns. Ownership starts as:
  agent_0 owns: coffee, saucer_plate, sugar_cube, counter
  agent_1 owns: milk, fridge
  unowned: coffee_machine (whoever takes it first)

If a robot needs something it does not own, it cannot simply use it. It must
request a transfer, and the plan must show the transfer happening:
  1. the requester sends a message naming the exact id it needs,
  2. the requester calls wait_for_signal(from=<owner>, about=<id>),
  3. the owner finishes with it (last use, or give_space for a fixture),
  4. the owner sends a message naming the id -- ownership passes at this point,
  5. only now may the requester act on it.

Output JSON: {"steps": [{"step": 0, "agent": "agent_0", "tool": ..., "args": {...},
"reasoning": "..."}]}, and additionally a "transfers" list recording each
ownership change as {"id": ..., "from": ..., "to": ..., "at_step": N}.
"""

BASELINE = """Generate a multi-agent trajectory for this task.

Output JSON: {"steps": [{"step": 0, "agent": "agent_0", "tool": ..., "args": {...},
"reasoning": "..."}]}. Steps are numbered consecutively from 0. The two robots
execute concurrently, so whenever one uses an object or tight fixture the other
has used, the second must first call wait_for_signal(from=..., about=<id>),
having announced it, and the first must release it afterwards by name.
"""

PROMPTS = {"timeline": TIMELINE, "ownership": OWNERSHIP, "baseline": BASELINE}


def flatten(payload: dict) -> list[dict]:
    """Turns any of the three shapes into the standard ordered step list."""

    if "timeline" in payload:
        steps: list[dict] = []
        for row in payload["timeline"]:
            # ties inside a tick resolve agent_0 first, deterministically
            for agent in ("agent_0", "agent_1"):
                cell = row.get(agent)
                if not isinstance(cell, dict):
                    continue
                if cell.get("tool") in (None, "idle"):
                    continue
                steps.append({
                    "step": len(steps),
                    "agent": agent,
                    "tool": cell["tool"],
                    "args": cell.get("args") or {},
                    "reasoning": cell.get("reasoning") or "Acting.",
                })
        return steps
    steps = payload.get("steps") or []
    for index, step in enumerate(steps):
        step["step"] = index
        step.setdefault("reasoning", "Acting.")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=sorted(PROMPTS), required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    from data_generation.task_level.tasks.specs.runtime import (
        build_task_definition_from_spec,
    )
    from data_generation.task_level.tasks.specs import load_task_spec

    definition = build_task_definition_from_spec(load_task_spec(TASK))
    instance = definition.build_task_instance(0)
    validator = definition.validator_factory(instance)

    prompt = PROMPTS[args.format] + WORLD
    passed = 0
    reasons: list[str] = []
    for run in range(args.n):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt + f"\n\nVariation seed: {run}. Output JSON only.",
            )
            text = re.sub(r"^```(?:json)?|```$", "", response.text.strip(),
                          flags=re.M).strip()
            steps = flatten(json.loads(text))
            validator.validate({
                "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
                "steps": steps,
            })
            passed += 1
            print(f"  run {run}: PASS ({len(steps)} steps)")
        except Exception as exc:  # noqa: BLE001 - every failure mode is data here
            label = type(exc).__name__
            detail = str(exc).split("\n")[0][:110]
            reasons.append(f"{label}: {detail}")
            print(f"  run {run}: fail  {label}: {detail}")

    print(f"\n{args.format}: {passed}/{args.n} valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
