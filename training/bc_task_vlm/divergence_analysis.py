"""Analyze where teacher-forced tool-call predictions first diverge.

Communication calls use ``comm_judge.jsonl`` equivalence when available;
all other calls use exact action-step match.  Positions are ranks within each
(task, trajectory), because stored ``step_index`` values include skipped image
steps and therefore are not contiguous action positions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze(predictions: Path, comm_judge: Path | None) -> list[dict[str, Any]]:
    judged = {}
    if comm_judge is not None and comm_judge.exists():
        judged = {row["sample_id"]: bool(row["equivalent"]) for row in _jsonl(comm_judge)}

    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(predictions):
        trajectories[(row["task_name"], row["trajectory_id"])].append(row)

    by_position: dict[int, list[bool]] = defaultdict(list)
    first_positions: Counter[int] = Counter()
    first_tools: Counter[str] = Counter()
    no_divergence = 0
    for rows in trajectories.values():
        rows.sort(key=lambda row: (int(row["step_index"]), row["sample_id"]))
        first: tuple[int, str] | None = None
        for position, row in enumerate(rows):
            target_tool = (row.get("target_tool_call") or {}).get("name", "unknown")
            correct = (
                judged.get(row["sample_id"], bool(row.get("exact_action_step_match")))
                if target_tool == "communicate"
                else bool(row.get("exact_action_step_match"))
            )
            by_position[position].append(correct)
            if not correct and first is None:
                first = (position, target_tool)
        if first is None:
            no_divergence += 1
        else:
            first_positions[first[0]] += 1
            first_tools[first[1]] += 1

    output: list[dict[str, Any]] = []
    for position, values in sorted(by_position.items()):
        output.append({
            "row_type": "position_accuracy", "key": position,
            "count": len(values), "value": sum(values) / len(values),
        })
    for position, count in sorted(first_positions.items()):
        output.append({
            "row_type": "first_divergence_position", "key": position,
            "count": count, "value": count / len(trajectories),
        })
    for tool, count in first_tools.most_common():
        total = sum(first_tools.values())
        output.append({
            "row_type": "first_divergence_tool", "key": tool,
            "count": count, "value": count / total,
        })
    output.append({
        "row_type": "no_divergence", "key": "trajectories",
        "count": no_divergence, "value": no_divergence / len(trajectories),
    })
    return output


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    positional = [r for r in rows if r["row_type"] == "position_accuracy"]
    first = [r for r in rows if r["row_type"] == "first_divergence_position"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([r["key"] for r in positional], [r["value"] for r in positional], marker="o")
    axes[0].set(xlabel="Action position", ylabel="Judged accuracy", ylim=(0, 1), title="Accuracy by position")
    axes[1].bar([str(r["key"]) for r in first], [r["count"] for r in first])
    axes[1].set(xlabel="First divergent position", ylabel="Trajectories", title="First divergence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--comm-judge", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = analyze(args.predictions, args.comm_judge)
    csv_path = args.output_dir / "divergence_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "key", "count", "value"))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output_dir / "divergence_analysis.png")
    print(csv_path)


if __name__ == "__main__":
    main()
