#!/usr/bin/env python
"""Summarise (and compare) work-partition coverage runs.

`coverage_work_partitions.py` emits one JSON record per task; this turns those
into the corpus-level numbers and, given two runs, the before/after table.

    python coverage_work_partitions.py --data <corpus> --out before.json
    python coverage_work_partitions.py --data <corpus> --out after.json
    python coverage_report.py before.json after.json

READ THE CAVEAT ON POOLED COVERAGE. `feasible` is a FLOOR, not the truth: the
enumerator labels work steps independently while pick/place pairs are atomic, so
it calls partitions infeasible that the corpus demonstrably contains. Tasks with
such orphans are excluded from pooled coverage, which means the two runs are
pooled over DIFFERENT task sets and the percentages are not comparable across
corpora. Compare `single_agent` and `distinct_partitions` -- both are measured
identically over every task and every trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(path: Path) -> dict:
    records = json.loads(path.read_text())
    clean = [
        r for r in records if r.get("coverage") is not None and not r.get("orphans")
    ]
    used = sum(
        len(set(r["produced_counts"]) & set(r["feasible_labellings"])) for r in clean
    )
    feasible = sum(r["feasible"] for r in clean)

    single = total = 0
    for record in records:
        for labelling, count in (record.get("produced_counts") or {}).items():
            total += count
            if len(set(labelling)) == 1:
                single += count

    return {
        "path": str(path),
        "tasks": len(records),
        "clean_tasks": len(clean),
        "orphan_tasks": sum(1 for r in records if r.get("orphans")),
        "skipped_tasks": sum(1 for r in records if r.get("error") or r.get("skipped")),
        "trajectories": total,
        "single_agent": single,
        "single_agent_rate": (single / total) if total else None,
        "distinct_partitions": sum(
            len(r.get("produced_counts") or {}) for r in records
        ),
        "pooled_used": used,
        "pooled_feasible": feasible,
        "pooled_coverage": (used / feasible) if feasible else None,
    }


def _row(label: str, value, width: int) -> str:
    return f"{value:>{width}}" if not isinstance(value, float) else f"{value:>{width}.1%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="one or more coverage --out JSON files")
    parser.add_argument("--labels", default=None, help="comma-separated column names")
    args = parser.parse_args()

    summaries = [summarise(Path(p)) for p in args.runs]
    labels = (
        args.labels.split(",")
        if args.labels
        else [Path(p).stem for p in args.runs]
    )

    rows = [
        ("trajectories", "trajectories"),
        ("single-agent", "single_agent"),
        ("single-agent rate", "single_agent_rate"),
        ("distinct partitions", "distinct_partitions"),
        ("pooled coverage*", "pooled_coverage"),
        ("clean tasks*", "clean_tasks"),
        ("orphan tasks", "orphan_tasks"),
    ]
    width = max(len(l) for l in labels) + 2
    print(f"{'metric':<22}" + "".join(f"{l:>{width}}" for l in labels))
    for title, key in rows:
        cells = "".join(_row(title, s[key], width) for s in summaries)
        print(f"{title:<22}{cells}")
    print(
        "\n* pooled coverage and clean-task counts are NOT comparable across "
        "corpora -- they are pooled over whichever tasks had no orphans, and "
        "that set differs per run. Compare single-agent rate and distinct "
        "partitions."
    )

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
