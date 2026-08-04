#!/usr/bin/env python
"""Measure how reliably the generator can produce each work partition.

The weights say how often each split SHOULD be sampled. They say nothing about
whether the model can actually produce it -- a split that is legal but awkward
will fail validation more often, and with a fixed retry budget it would quietly
end up under-represented no matter what its weight says.

This runs a fixed number of attempts per (task, partition) with retries off, so
each attempt is an independent Bernoulli trial, and reports:

    p            observed first-attempt success rate
    retries      ceil(log(1 - TARGET) / log(1 - p)) -- attempts needed to hit
                 TARGET success for that split
    verdict      ok / needs-retries / unreachable

Resumable: a pair whose result file already exists is skipped, so the sweep can
be run in slices and topped up.

    python calibrate_partitions.py --out RUN --tasks PrepareCoffee --dry-run
    python calibrate_partitions.py --out RUN --limit-pairs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TARGET = 0.95
PYTHON = "/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python"


def pairs_from_specs(task_filter, include_degenerate):
    """Every (composite_task, partition) the specs declare."""

    sys.path.insert(0, str(REPO))
    from data_generation.task_level.tasks.specs import load_all_task_specs
    from data_generation.task_level.tasks.shared.partitions import is_degenerate

    out = []
    for spec in load_all_task_specs():
        if task_filter and spec.composite_task not in task_filter:
            continue
        for partition in (spec.work_partitions or {}).get("partitions", []):
            if not include_degenerate and is_degenerate(partition):
                continue
            out.append((spec.composite_task, partition))
    return out


def run_pair(task, partition, out_root, attempts, timeout_sec, dry_run):
    """Generate `attempts` trajectories for one split, retries disabled."""

    labels = partition["labels"]
    workdir = out_root / task / labels
    result_path = workdir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text()), True

    command = [
        PYTHON, "-m", "data_generation.task_level.pipeline.cli",
        "--phase", "4", "--resume", str(workdir / "run"),
        "--num-runs", str(attempts),
        "--workers", "4",
        # One attempt per run: we are measuring the per-attempt success rate,
        # so a retry would contaminate the estimate it is meant to inform.
        "--max-retries", "1",
        "--generation-timeout-sec", str(timeout_sec),
        "--work-partition", labels,
        "--tasks", task,
    ]
    if dry_run:
        return {"task": task, "labels": labels, "command": " ".join(command)}, False

    workdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True,
        timeout=timeout_sec * attempts + 600,
    )
    produced = len(list((workdir / "run").glob("*/trajectories/*.json")))
    result = {
        "task": task,
        "labels": labels,
        "weight": partition.get("weight"),
        "balance": partition.get("balance"),
        "cross_agent_deps": partition.get("cross_agent_deps"),
        "attempts": attempts,
        "produced": produced,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:] if completed.returncode else "",
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result, False


def summarize(results):
    """Per-split success rate and the retry budget it implies."""

    rows = []
    for result in results:
        attempts = result.get("attempts") or 0
        produced = result.get("produced") or 0
        if not attempts:
            continue
        p = produced / attempts
        if p >= 1.0:
            retries = 1
            verdict = "ok"
        elif p <= 0.0:
            retries = None
            verdict = "unreachable"
        else:
            retries = math.ceil(math.log(1 - TARGET) / math.log(1 - p))
            verdict = "ok" if retries <= 3 else "needs-retries"
        rows.append({**result, "p": round(p, 3), "retries": retries,
                     "verdict": verdict})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--timeout-sec", type=int, default=600)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--include-degenerate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    pairs = pairs_from_specs(set(args.tasks) if args.tasks else None,
                             args.include_degenerate)
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]

    print(f"{len(pairs)} (task, partition) pairs x {args.attempts} attempts "
          f"= {len(pairs) * args.attempts} generations", flush=True)

    results = []
    for index, (task, partition) in enumerate(pairs, start=1):
        result, cached = run_pair(task, partition, out_root, args.attempts,
                                  args.timeout_sec, args.dry_run)
        results.append(result)
        if args.dry_run:
            print(f"  {result['task']} {result['labels']}")
            continue
        mark = "cached" if cached else "ran"
        print(f"[{index}/{len(pairs)}] {task} {partition['labels']}: "
              f"{result.get('produced')}/{result.get('attempts')} ({mark})",
              flush=True)

    if args.dry_run:
        return 0

    rows = summarize(results)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "calibration.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'task':28s} {'split':12s} {'p':>6s} {'retries':>8s}  verdict")
    for row in sorted(rows, key=lambda r: r["p"]):
        retries = row["retries"] if row["retries"] is not None else "-"
        print(f"{row['task'][:28]:28s} {row['labels'][:12]:12s} "
              f"{row['p']:6.2f} {str(retries):>8s}  {row['verdict']}")
    bad = [r for r in rows if r["verdict"] != "ok"]
    print(f"\n{len(rows)} splits measured, {len(bad)} needing attention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
