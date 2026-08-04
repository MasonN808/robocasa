"""Assign a work partition to each generation run, proportionally to its weight.

A task spec carries a `work_partitions` block: every division of the task's
work between the agents that the FSM accepts, each with a sampling weight (see
`weight_work_partitions.py`). This module picks which one a given run uses and
renders it into prompt rules.

Selection is deterministic in `run_index` and uses the Sainte-Lague divisor
method, so *every prefix* of the run sequence is already proportional. That is
what lets the dataset be generated in separate batches: a run of 50k and a
later run of 50k are each individually proportional, so their union is too, and
nothing has to be tracked across batches.

Retries reuse the run's TaskInstance, so a failed attempt is retried against the
same partition rather than silently drifting to an easier one.
"""

from __future__ import annotations

from typing import Any, Sequence

# Awarded partitions per (task signature) -- extended on demand, since a
# generation run asks for run_index 0, 1, 2, ... in order.
_AWARD_CACHE: dict[tuple, list[int]] = {}


def _signature(partitions: Sequence[dict[str, Any]]) -> tuple:
    return tuple((p.get("labels", ""), float(p.get("weight", 0.0))) for p in partitions)


def select_partition(
    partitions: Sequence[dict[str, Any]] | None,
    run_index: int,
) -> dict[str, Any] | None:
    """Returns the partition run `run_index` should generate, or None."""

    if not partitions or run_index < 0:
        return None
    weights = [float(p.get("weight", 0.0)) for p in partitions]
    if not any(weight > 0.0 for weight in weights):
        return None

    signature = _signature(partitions)
    awarded = _AWARD_CACHE.setdefault(signature, [])
    if run_index < len(awarded):
        return dict(partitions[awarded[run_index]])

    counts = [0] * len(partitions)
    for index in awarded:
        counts[index] += 1
    while len(awarded) <= run_index:
        # Sainte-Lague: award to whoever is furthest behind its share.
        best = max(
            range(len(partitions)),
            key=lambda i: (weights[i] / (2 * counts[i] + 1), -i),
        )
        awarded.append(best)
        counts[best] += 1
    return dict(partitions[awarded[run_index]])


def partition_rules(
    partition: dict[str, Any] | None,
    agent_ids: Sequence[str],
) -> tuple[str, ...]:
    """Renders one partition as prompt rules describing who does what."""

    if not partition:
        return ()
    assignment = partition.get("assignment") or {}
    lines: list[str] = [
        "Divide the work between the agents EXACTLY as follows. This division is "
        "required; do not move work from one agent to the other.",
    ]
    for agent_id in agent_ids:
        items = assignment.get(agent_id) or []
        if items:
            lines.append(f"  {agent_id} performs: " + "; ".join(items))
        else:
            lines.append(
                f"  {agent_id} performs none of the task actions, and instead "
                f"communicates and stays clear of the other agent's workspace."
            )
    lines.append(
        "The order of these actions is up to you, and either agent may act "
        "first, as long as each action is performed by the agent listed above."
    )
    if partition.get("cross_agent_deps"):
        lines.append(
            "This division requires the agents to hand work to each other: "
            "where one agent needs something the other is using or has not "
            "finished with, it must communicate and wait rather than proceed."
        )
    return tuple(lines)


def is_degenerate(partition: dict[str, Any] | None) -> bool:
    """True when one agent performs every task action."""

    if not partition:
        return False
    assignment = partition.get("assignment") or {}
    non_empty = [agent for agent, items in assignment.items() if items]
    return len(non_empty) == 1
