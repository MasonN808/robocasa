#!/usr/bin/env python
"""Work-partition coverage: |produced partitions| / |feasible partitions| per task.

A *work sequence* is the trajectory with all logistics stripped (observation,
communication, waiting, navigation, give_space) -- what is left is the real
labour. A *partition* is one labelling of that sequence with agent ids, e.g.
"001" = agent_0 does the first two steps, agent_1 the third.

Feasibility is decided by the real FSM, not by a reimplementation of it. We
synthesise a minimal candidate for a labelling and then repair it in a loop
driven by the validator's own errors: the FSM tells us which fixture an agent
must navigate to (`details["expected_location"]`) and where observation
brackets are missing, so we never have to re-derive fixture resolution. An
earlier version of this probe did re-derive it and got the precedence wrong --
for `place_next_to` it returned the carried object's origin instead of the
reference object's location, which reported 0 feasible partitions on 28 of 52
tasks.

INVARIANT: every partition actually observed in the data must come out
feasible. A task violating that is reported as `orphan` and its coverage is
withheld rather than averaged in.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path

REPO = Path("/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration")
sys.path.insert(0, str(REPO))

import insert_waits  # noqa: E402  (needs REPO on sys.path)

DATA = Path("/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset")

# Steps that carry no work: they exist to make the work legal or observable.
LOGISTICS = {
    "get_image",
    "communicate",
    "wait_for_signal",
    "navigate_to_fixture",
    "give_space",
}

MAX_REPAIRS = 120


def work_sequence(steps):
    """The trajectory stripped to real labour, plus its agent labelling."""

    work = [s for s in steps if s["tool"] not in LOGISTICS]
    signature = tuple((s["tool"], _arg_signature(s)) for s in work)
    labelling = "".join(str(s["agent"].rsplit("_", 1)[-1]) for s in work)
    return work, signature, labelling


def _arg_signature(step):
    """Args with agent-identity stripped, so two labellings of one sequence match."""

    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in (step.get("args") or {}).items()))


def observation_views(steps):
    """Reuse the view sets the generator actually emits for this task."""

    for step in steps:
        if step["tool"] == "get_image":
            return deepcopy(step["args"])
    return None


def build_candidate(work, labelling, agent_ids, views):
    """A minimal trajectory: opening coordination, then the labelled work."""

    steps = []

    def emit(agent, tool, args):
        steps.append({"step": len(steps), "agent": agent, "tool": tool,
                      "args": deepcopy(args), "reasoning": "probe"})

    # Both agents must communicate before the first task action.
    for i, agent in enumerate(agent_ids):
        other = agent_ids[1 - i]
        emit(agent, "communicate", {"to": other, "message": f"{agent} is starting the task."})

    for step, label in zip(work, labelling):
        agent = f"agent_{label}"
        emit(agent, step["tool"], step.get("args") or {})

    return steps


def renumber(steps):
    for i, step in enumerate(steps):
        step["step"] = i
    return steps


def repair_and_validate(steps, validator, agent_ids, views, initial_state):
    """Validate, repairing FSM-diagnosed logistics gaps until it passes or sticks."""

    from data_generation.task_level.tasks.shared.errors import (
        NavigationSemanticValidationError,
        ObservationSequenceSemanticValidationError,
        TrajectoryValidationError,
        WaitSignalSemanticValidationError,
    )

    steps = deepcopy(steps)
    last_error = None
    seen = set()
    wait_passes = 0

    for _ in range(MAX_REPAIRS):
        renumber(steps)
        candidate = {
            "agents": [{"agent": a} for a in agent_ids],
            "steps": deepcopy(steps),
        }
        try:
            validator.validate(candidate)
            return True, None, steps
        except NavigationSemanticValidationError as err:
            last_error = err
            idx = err.step
            agent = err.details.get("agent")
            target = err.details.get("expected_location")
            if idx is None or agent is None or not isinstance(target, str):
                break
            key = ("nav", idx, agent, target)
            if key in seen:
                break
            seen.add(key)
            steps.insert(idx, {"step": idx, "agent": agent, "tool": "navigate_to_fixture",
                               "args": {"fixture_id": target}, "reasoning": "probe"})
        except ObservationSequenceSemanticValidationError as err:
            last_error = err
            idx = err.details.get("step")
            position = err.details.get("position")
            if idx is None or views is None:
                break
            agent = steps[idx]["agent"]
            at = idx if position == "before" else idx + 1
            key = ("obs", idx, position, len(steps))
            if key in seen:
                break
            seen.add(key)
            steps.insert(at, {"step": at, "agent": agent, "tool": "get_image",
                              "args": deepcopy(views), "reasoning": "probe"})
        except WaitSignalSemanticValidationError as err:
            last_error = err
            # A cross-agent handoff needs the ask/wait/release protocol. Derive
            # it with the same pass production uses rather than hand-rolling
            # one here. insert() rebuilds every wait from scratch, so it is safe
            # to re-run: navigation repairs made after the first pass change who
            # occupies what, and the waits have to be recomputed against that.
            if wait_passes >= 4:
                break
            wait_passes += 1
            steps, _ = insert_waits.insert(
                steps,
                deepcopy(initial_state.get("fixtures") or {}),
                set(initial_state.get("objects") or {}),
                random.Random(0),
                {a: (s or {}).get("location")
                 for a, s in (initial_state.get("agents") or {}).items()},
            )
        except TrajectoryValidationError as err:
            last_error = err
            repaired = repair_occupancy(steps, err, agent_ids, seen)
            if not repaired:
                break

    return False, last_error, steps


def repair_occupancy(steps, err, agent_ids, seen):
    """If the blocker is another agent standing at the fixture, step them aside."""

    details = err.details or {}
    fixture = details.get("fixture_id") or details.get("expected_location")
    blocker = details.get("other_agent") or details.get("occupant")
    idx = err.step
    if idx is None or not isinstance(fixture, str):
        return False
    if blocker not in agent_ids:
        # Fall back to the other agent -- only two of them.
        acting = steps[idx]["agent"] if idx < len(steps) else None
        blocker = next((a for a in agent_ids if a != acting), None)
        if blocker is None:
            return False
    key = ("space", idx, blocker, fixture)
    if key in seen:
        return False
    seen.add(key)
    steps.insert(idx, {"step": idx, "agent": blocker, "tool": "give_space",
                       "args": {"fixture_id": fixture}, "reasoning": "probe"})
    return True


def make_validator(task_name, record):
    from data_generation.task_level.tasks.specs import load_task_spec
    from data_generation.task_level.tasks.specs.runtime import build_task_definition_from_spec
    from data_generation.task_level.tasks.specs.runtime import SpecDrivenTaskValidator

    spec = load_task_spec(task_name)
    definition = build_task_definition_from_spec(spec)
    validator = definition.validator_factory(None)
    # The per-trajectory instance randomises fixture ids, so replay against the
    # world this trajectory was actually generated in.
    validator.initial_state = deepcopy(record["initial_state"])
    return validator


def analyse_task(task_name, limit=None, data=None):
    task_dir = (data or DATA) / task_name
    trajectories = sorted(task_dir.glob("traj_*/original_trajectory.json"))
    if limit:
        trajectories = trajectories[:limit]
    if not trajectories:
        return None

    records = [json.loads(p.read_text()) for p in trajectories]
    agent_ids = [a["agent"] for a in records[0]["agents"]]

    by_signature = {}
    for record in records:
        work, signature, labelling = work_sequence(record["steps"])
        entry = by_signature.setdefault(signature, {"work": work, "record": record,
                                                    "produced": Counter()})
        entry["produced"][labelling] += 1

    # Coverage is measured against the dominant work sequence; alternates are
    # reported so we never silently average over a task with several.
    dominant = max(by_signature.values(), key=lambda e: sum(e["produced"].values()))
    work = dominant["work"]
    record = dominant["record"]
    produced = dominant["produced"]
    n = len(work)
    if n == 0 or n > 14:
        return {"task": task_name, "skipped": f"work length {n}"}

    views = observation_views(record["steps"])
    validator = make_validator(record["composite_task"], record)

    feasible = []
    detail = {}
    reasons = Counter()
    for labels in product("01", repeat=n):
        labelling = "".join(labels)
        steps = build_candidate(work, labelling, agent_ids, views)
        ok, err, repaired = repair_and_validate(steps, validator, agent_ids, views,
                                                record["initial_state"])
        if ok:
            feasible.append(labelling)
            load = [labelling.count(str(i)) for i in range(len(agent_ids))]
            detail[labelling] = {
                # One wait per cross-agent handoff -- insert_waits derives them,
                # so this is the dependency count rather than an estimate.
                "cross_agent_deps": sum(1 for x in repaired
                                        if x["tool"] == "wait_for_signal"),
                "balance": min(load) / max(load) if max(load) else 0.0,
                "load": load,
            }
        else:
            reasons[type(err).__name__ if err else "unknown"] += 1

    feasible_set = set(feasible)
    orphans = sorted(set(produced) - feasible_set)

    return {
        "task": task_name,
        "composite_task": record["composite_task"],
        "work_len": n,
        "work_sequences": len(by_signature),
        "trajectories": sum(produced.values()),
        "feasible": len(feasible_set),
        "produced": len(produced),
        "coverage": (len(set(produced) & feasible_set) / len(feasible_set)) if feasible_set else None,
        "orphans": orphans,
        "feasible_labellings": sorted(feasible_set),
        "partition_detail": detail,
        "work": [{"tool": w["tool"], "args": w.get("args") or {}} for w in work],
        "produced_counts": dict(produced.most_common()),
        "reasons": dict(reasons.most_common(4)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    # Both corpora use the same traj_*/original_trajectory.json layout, so the
    # before/after comparison runs through one code path rather than two probes.
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    data = Path(args.data) if args.data else DATA
    names = args.tasks or sorted(p.name for p in data.iterdir() if p.is_dir())
    results = []
    for name in names:
        try:
            result = analyse_task(name, args.limit, data=data)
        except Exception as exc:  # noqa: BLE001 - probe: report and continue
            result = {"task": name, "error": f"{type(exc).__name__}: {exc}"}
        if result is None:
            continue
        results.append(result)
        print(json.dumps(result), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
