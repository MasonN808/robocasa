#!/usr/bin/env python
"""Audit every recorded trajectory with the concurrent validator.

The repair-vs-regenerate decision needs the whole corpus, not the first 300:
how many trajectories are salvageable by the insertion pass, how many carry
defects only the model could have avoided, and how many are broken for reasons
that have nothing to do with coordination.
"""

from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import replay_report as rr
from data_generation.task_level.tasks.shared.concurrent_fsm import (
    EXECUTOR,
    LOCK_STEP,
)


def classify(run) -> str:
    if run.step_error:
        return "step_error"
    if run.deadlocked:
        return "deadlock"
    if run.protocol:
        return "protocol"
    if run.conflicts:
        return "conflict"
    if run.goal_at is None:
        return "no_goal"
    return "clean"


def main() -> None:
    paths = sorted(glob.glob(rr.ROOT + "/*/traj_*/original_trajectory.json"))
    tally: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    both_clean = 0

    for path in paths:
        task = path.split("/")[-3]
        record = json.loads(Path(path).read_text())
        tally["total"] += 1
        try:
            validator, candidate = rr.build(record, insert=True)
            runs = {
                model: validator.replay(
                    candidate, model=model, stop_on_step_error=False
                )
                for model in (LOCK_STEP, EXECUTOR)
            }
        except Exception as exc:
            tally[f"setup:{type(exc).__name__}"] += 1
            by_task[task]["setup_error"] += 1
            continue

        verdicts = {m: classify(r) for m, r in runs.items()}
        tally[f"lock_step:{verdicts[LOCK_STEP]}"] += 1
        tally[f"executor:{verdicts[EXECUTOR]}"] += 1
        by_task[task][verdicts[LOCK_STEP]] += 1
        if set(verdicts.values()) == {"clean"}:
            both_clean += 1
        elif "clean" in verdicts.values():
            tally["timing_dependent"] += 1

    print("=" * 62)
    for key, count in sorted(tally.items()):
        print(f"{key:34s} {count:5d}")
    print(f"{'CLEAN UNDER BOTH MODELS':34s} {both_clean:5d}")

    print("\nworst tasks by lock-step clean rate:")
    ranked = sorted(
        by_task.items(),
        key=lambda kv: kv[1]["clean"] / max(1, sum(kv[1].values())),
    )
    for task, counts in ranked[:12]:
        total = sum(counts.values())
        detail = " ".join(f"{k}={v}" for k, v in counts.most_common() if k != "clean")
        print(f"  {task:34s} {counts['clean']:3d}/{total:3d}  {detail}")


if __name__ == "__main__":
    main()
