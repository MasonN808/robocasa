#!/usr/bin/env python
"""Inject observations into tick trajectories and re-validate the result.

The point of the exercise: post-processing must not change the plan. Each
trajectory is validated as generated, put through `post_process_trajectory`,
and validated again. Anything that was clean before and is not clean after is
injection damage, and is printed in full.

Symbolic only -- nothing renders, so this is cheap enough to run on the whole
corpus before committing to an image pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_generation.task_level.generation.image.processor import (
    post_process_trajectory,
)
from data_generation.task_level.tasks.shared.concurrent_fsm import (
    ConcurrentTaskValidator,
)
import replay_report as rr


def _verdict(record: dict) -> tuple[bool, str]:
    """Validates one record exactly as generation does, with no wait insertion."""

    try:
        validator, candidate = rr.build(record, insert=False)
        validator.validate(candidate)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - every failure mode is reportable
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_enforced"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--backfill-rows",
        action="store_true",
        help=(
            "Reconstruct tick_rows for records generated before the rows were "
            "persisted. The reconstruction is a schedule, not necessarily THE "
            "schedule, so only the before/after comparison is meaningful."
        ),
    )
    args = parser.parse_args()

    totals = {"seen": 0, "clean_before": 0, "clean_after": 0, "broken": 0}
    for task_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        for path in sorted((task_dir / "trajectories").glob("*.json")):
            record = json.loads(path.read_text())
            if not isinstance(record.get("tick_rows"), list):
                if not args.backfill_rows:
                    continue
                from tick_format import to_rows

                agent_ids = [a["agent"] for a in record["agents"]]
                record["tick_rows"] = to_rows(record["steps"], agent_ids)
            totals["seen"] += 1
            before_ok, before_err = _verdict(record)
            totals["clean_before"] += before_ok

            processed = post_process_trajectory(record)
            after_ok, after_err = _verdict(processed)
            totals["clean_after"] += after_ok

            name = f"{task_dir.name}/{path.stem}"
            n_before = len(record.get("steps") or ())
            n_after = len(processed.get("steps") or ())
            n_img = sum(
                1 for s in processed["steps"] if s.get("tool") == "get_image"
            )
            flag = ""
            if before_ok and not after_ok:
                totals["broken"] += 1
                flag = "  <-- BROKEN BY INJECTION"
            print(
                f"{name:44s} {n_before:3d} -> {n_after:3d} steps "
                f"({n_img} obs)  before={'ok' if before_ok else 'BAD'} "
                f"after={'ok' if after_ok else 'BAD'}{flag}"
            )
            if before_ok and not after_ok:
                print(f"    {after_err}")
            elif not before_ok:
                print(f"    (was already bad: {before_err})")

            if args.out is not None:
                dst = args.out / task_dir.name / f"{path.stem}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(processed, indent=2))

    print(
        f"\n{totals['seen']} tick trajectories | clean before "
        f"{totals['clean_before']} | clean after {totals['clean_after']} | "
        f"broken by injection {totals['broken']}"
    )


if __name__ == "__main__":
    main()
