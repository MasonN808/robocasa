#!/usr/bin/env python
"""Run the concurrent FSM over recorded trajectories and report what it sees.

Usage:
    python replay_report.py --limit 200            # summary over the subset
    python replay_report.py --show task/traj_000000  # two-column render
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from data_generation.task_level.tasks.shared.concurrent_fsm import (
    DURATION_MODELS,
    EXECUTOR,
    LOCK_STEP,
    ConcurrentTaskValidator,
)

ROOT = "/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset"


def build(record: dict, *, insert: bool):
    from data_generation.task_level.tasks.specs import load_task_spec
    from data_generation.task_level.tasks.specs.runtime import SpecDrivenTaskValidator

    spec = load_task_spec(record["composite_task"])
    inner = SpecDrivenTaskValidator(spec)
    initial = record["initial_state"]
    inner.initial_state = deepcopy(initial)
    steps = deepcopy(record["steps"])
    if insert:
        import insert_waits

        steps, _ = insert_waits.insert(
            steps,
            initial.get("fixtures") or {},
            set(initial.get("objects") or {}),
            random.Random(0),
            {a: (s or {}).get("location") for a, s in (initial.get("agents") or {}).items()},
        )
    return ConcurrentTaskValidator(inner), {"agents": record["agents"], "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--show", default=None)
    parser.add_argument("--model", default=LOCK_STEP, choices=DURATION_MODELS)
    parser.add_argument("--raw", action="store_true", help="skip insert_waits")
    args = parser.parse_args()

    if args.show:
        path = Path(ROOT) / args.show / "original_trajectory.json"
        record = json.loads(path.read_text())
        validator, candidate = build(record, insert=not args.raw)
        print(f"== {args.show}  [{record['composite_task']}]  model={args.model}")
        print(validator.render(candidate, model=args.model))
        run = validator.replay(candidate, model=args.model, stop_on_step_error=False)
        print(f"\nmakespan={run.makespan:g}  idle={run.idle}  goal_at={run.goal_at}")
        print(f"deadlocked={run.deadlocked}  conflicts={len(run.conflicts)}")
        for line in run.conflicts:
            print(line)
        for line in run.post_goal:
            print(line)
        if run.step_error:
            print(f"step_error: {type(run.step_error).__name__}: {run.step_error}")
        return

    paths = sorted(glob.glob(ROOT + "/*/traj_*/original_trajectory.json"))[: args.limit]
    tally: Counter[str] = Counter()
    for path in paths:
        record = json.loads(Path(path).read_text())
        try:
            validator, candidate = build(record, insert=not args.raw)
        except Exception as exc:
            tally[f"setup:{type(exc).__name__}"] += 1
            continue
        tally["total"] += 1
        for model in DURATION_MODELS:
            run = validator.replay(candidate, model=model, stop_on_step_error=False)
            if run.deadlocked:
                tally[f"{model}:deadlock"] += 1
            if run.conflicts and not run.deadlocked:
                tally[f"{model}:conflict"] += 1
            if run.step_error:
                tally[f"{model}:step_error"] += 1
            if run.goal_at is None:
                tally[f"{model}:no_goal"] += 1
            if not run.conflicts and not run.step_error and run.goal_at is not None:
                tally[f"{model}:clean"] += 1
    for key, count in sorted(tally.items()):
        print(f"{key:28s} {count}")


if __name__ == "__main__":
    main()
