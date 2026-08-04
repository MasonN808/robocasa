#!/usr/bin/env python
"""Turn the feasible work partitions into per-task sampling weights.

Reads the coverage probe's output and writes a `work_partitions` block into
each task spec: the work sequence the partitions are labelled against, and one
weight per feasible labelling.

    weight  is-proportional-to  balance ** K  *  (1 + ALPHA * cross_agent_deps)

`balance` is min(load)/max(load) over the two agents, so an even split scores 1
and a 3/1 split scores 0.33; raising it to K>1 pulls sampling toward even
splits without collapsing onto them. `cross_agent_deps` is the number of
handoffs `insert_waits` derives for that labelling -- partitions that force
real coordination are what the model most needs to see.

Single-agent labellings are capped as a group at DEGENERATE_SHARE rather than
being scored, because balance**K would zero them out and they still need to
appear occasionally.

    python weight_partitions.py            # dry run, prints the table
    python weight_partitions.py --apply    # writes into the task specs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration")
SPECS = REPO / "data_generation/task_level/tasks/specs"
COVERAGE = Path(__file__).with_name("coverage_all.json")

K = 1.3
ALPHA = 0.5
DEGENERATE_SHARE = 0.05


def spec_path(composite_task: str) -> Path:
    name = "".join(c.lower() if c.isalnum() else "_" for c in composite_task)
    name = "_".join(p for p in name.split("_") if p) + ".json"
    for directory in (SPECS / "verified", SPECS):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def weights_for(detail: dict) -> dict[str, float]:
    """Split the probability mass across one task's feasible labellings."""

    degenerate = [k for k, v in detail.items() if v["balance"] == 0.0]
    shared = [k for k, v in detail.items() if v["balance"] > 0.0]

    if not shared:
        # Nothing is splittable (a two-step task on a single object): the cap
        # cannot apply, so spread evenly and let the caller flag the task.
        return {k: 1.0 / len(degenerate) for k in degenerate}

    scores = {
        k: detail[k]["balance"] ** K * (1.0 + ALPHA * detail[k]["cross_agent_deps"])
        for k in shared
    }
    total = sum(scores.values())
    budget = 1.0 - DEGENERATE_SHARE if degenerate else 1.0
    weights = {k: budget * v / total for k, v in scores.items()}
    if degenerate:
        weights.update({k: DEGENERATE_SHARE / len(degenerate) for k in degenerate})
    return weights


def assignment_for(labelling: str, work: list[dict]) -> dict[str, list[str]]:
    """Render one labelling as per-agent work, independent of step order."""

    assignment: dict[str, list[str]] = {}
    for label, step in zip(labelling, work):
        args = step.get("args") or {}
        subject = args.get("object_id") or args.get("target_id") or ""
        target = (args.get("support_object_id") or args.get("reference_object_id")
                  or args.get("reference_fixture_id") or args.get("receptacle_id")
                  or args.get("control_id") or args.get("part_id") or "")
        phrase = step["tool"] + (f" {subject}" if subject else "")
        phrase += f" -> {target}" if target else ""
        assignment.setdefault(f"agent_{label}", []).append(phrase)
    for agent in ("agent_0", "agent_1"):
        assignment.setdefault(agent, [])
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--task", default=None, help="show one task in full")
    args = ap.parse_args()

    rows = json.loads(COVERAGE.read_text())
    valid = [r for r in rows if r.get("coverage") is not None and not r.get("orphans")]

    written = 0
    flagged = []
    for row in valid:
        detail = row["partition_detail"]
        weights = weights_for(detail)
        if not any(v["balance"] > 0.0 for v in detail.values()):
            flagged.append(row["task"])

        partitions = [
            {
                # An assignment, not an ordering: which agent owns which piece
                # of work. A bit-string would only be meaningful against one
                # exact step order, and most tasks have several valid orders --
                # pinning one would trade sequence diversity for split
                # diversity. The generator stays free to sequence as it likes.
                "assignment": assignment_for(labelling, row["work"]),
                "labels": labelling,
                "weight": round(weights[labelling], 4),
                "balance": round(detail[labelling]["balance"], 3),
                "cross_agent_deps": detail[labelling]["cross_agent_deps"],
            }
            for labelling in sorted(weights, key=lambda k: -weights[k])
        ]

        if args.task in (None, row["task"]):
            degenerate_share = sum(w for k, w in weights.items()
                                   if detail[k]["balance"] == 0.0)
            print(f"\n{row['task']}  ({row['work_len']} work steps, "
                  f"{len(partitions)} feasible, degenerate {degenerate_share:.0%})")
            for part in partitions:
                marker = "*" if part["labels"] in row["produced_counts"] else " "
                print(f"  {marker} w={part['weight']:.3f} balance={part['balance']:.2f} "
                      f"deps={part['cross_agent_deps']}")
                for agent, items in sorted(part["assignment"].items()):
                    print(f"        {agent}: {'; '.join(items) or '(nothing)'}")

        if not args.apply:
            continue

        path = spec_path(row.get("composite_task") or row["task"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["work_partitions"] = {
            "work_sequence": row["work"],
            "partitions": partitions,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        written += 1

    print(f"\n{len(valid)} tasks weighted"
          + (f", {written} specs written" if args.apply else " (dry run)"))
    if flagged:
        print(f"structurally single-agent, cap cannot apply: {flagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
