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

from .constants import EXCLUSIVE_FIXTURE_TYPES

# Awarded partitions per (task signature) -- extended on demand, since a
# generation run asks for run_index 0, 1, 2, ... in order.
_AWARD_CACHE: dict[tuple, list[int]] = {}


def _signature(partitions: Sequence[dict[str, Any]]) -> tuple:
    return tuple((p.get("labels", ""), float(p.get("weight", 0.0))) for p in partitions)


def select_partition(
    partitions: Sequence[dict[str, Any]] | None,
    run_index: int,
    *,
    policy: str = "weighted",
    initial_state: dict[str, Any] | None = None,
    work_sequence: Sequence[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    """Returns the partition run `run_index` should generate, or None."""

    if policy == "none" or not partitions or run_index < 0:
        return None
    if policy == "balanced_local":
        partitions = _balanced_local_candidates(
            partitions,
            initial_state=initial_state or {},
            work_sequence=work_sequence,
        )
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


def _entity_fixture(entity_id: Any, initial_state: dict[str, Any]) -> str | None:
    """Resolves a fixture or movable object's initial supporting fixture."""

    if not isinstance(entity_id, str):
        return None
    fixtures = initial_state.get("fixtures") or {}
    if entity_id in fixtures:
        return entity_id
    objects = initial_state.get("objects") or {}
    seen: set[str] = set()
    current = entity_id
    while current in objects and current not in seen:
        seen.add(current)
        location = (objects.get(current) or {}).get("location")
        if location in fixtures:
            return location
        if not isinstance(location, str):
            return None
        current = location
    return current if current in fixtures else None


def _action_fixture(action: dict[str, Any], initial_state: dict[str, Any]) -> str | None:
    """Finds the fixture at which one partitioned task action executes."""

    args = action.get("args") or {}
    for key in (
        "source_id", "reference_fixture_id", "fixture_id", "target_id",
        "support_id", "receptacle_id", "support_object_id", "reference_object_id",
    ):
        fixture = _entity_fixture(args.get(key), initial_state)
        if fixture is not None:
            return fixture
    return None


def partition_efficiency_cost(
    partition: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    work_sequence: Sequence[dict[str, Any]],
) -> tuple[int, int, int]:
    """Returns navigation, exclusive-sharing, and measured dependency costs."""

    labels = str(partition.get("labels") or "")
    agents = initial_state.get("agents") or {}
    first_fixture: dict[str, str] = {}
    fixture_owners: dict[str, set[str]] = {}
    fixtures = initial_state.get("fixtures") or {}
    for label, action in zip(labels, work_sequence):
        agent_id = f"agent_{label}"
        fixture_id = _action_fixture(action, initial_state)
        if fixture_id is None:
            continue
        first_fixture.setdefault(agent_id, fixture_id)
        fixture_type = str((fixtures.get(fixture_id) or {}).get("fixture_type", "")).lower()
        if fixture_type in EXCLUSIVE_FIXTURE_TYPES:
            fixture_owners.setdefault(fixture_id, set()).add(agent_id)
    initial_navigation_count = sum(
        (agents.get(agent_id) or {}).get("location") != fixture_id
        for agent_id, fixture_id in first_fixture.items()
    )
    exclusive_handover_count = sum(
        max(len(owners) - 1, 0) for owners in fixture_owners.values()
    )
    return (
        int(initial_navigation_count),
        int(exclusive_handover_count),
        int(partition.get("cross_agent_deps") or 0),
    )


def _balanced_local_candidates(
    partitions: Sequence[dict[str, Any]],
    *,
    initial_state: dict[str, Any],
    work_sequence: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keeps maximally balanced partitions with the lowest avoidable cost."""

    shared = [partition for partition in partitions if not is_degenerate(partition)]
    if not shared:
        return list(partitions)
    best_balance = max(float(partition.get("balance", 0.0)) for partition in shared)
    balanced = [
        partition for partition in shared
        if float(partition.get("balance", 0.0)) == best_balance
    ]
    costs = {
        str(partition.get("labels")): partition_efficiency_cost(
            partition,
            initial_state=initial_state,
            work_sequence=work_sequence,
        )
        for partition in balanced
    }
    totals = {label: sum(cost) for label, cost in costs.items()}
    best_total = min(totals.values())
    selected = []
    for partition in balanced:
        if totals[str(partition.get("labels"))] != best_total:
            continue
        selected.append({**partition, "weight": 1.0})
    return selected


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
        "Choose the execution order, but obey prerequisites and exclusive-resource "
        "handovers; these may require one assigned action before another. Each "
        "action must still be performed by the agent listed above."
    )
    if partition.get("cross_agent_deps"):
        lines.append(
            "This division requires the agents to hand work to each other: "
            "where one agent needs something the other is using or has not "
            "finished with, it must communicate and wait rather than proceed."
        )
    def owner_of(prefix: str) -> str | None:
        return next(
            (
                agent_id
                for agent_id in agent_ids
                for item in assignment.get(agent_id) or ()
                if item.startswith(prefix)
            ),
            None,
        )

    # These are physical precedence constraints, rendered with the sampled
    # owners so they remain correct when structured-random swaps the agents.
    for loaded_object in ("toaster_oven_bread", "meat2"):
        loader = owner_of(f"place_in_receptacle {loaded_object} -> bowl")
        mover = owner_of("pick_up_object bowl")
        if loader and mover:
            lines.append(
                f"Required order: {loader} places {loaded_object} into bowl while "
                f"bowl rests on counter; only afterward may {mover} pick up bowl."
            )
    return tuple(lines)


def is_degenerate(partition: dict[str, Any] | None) -> bool:
    """True when one agent performs every task action."""

    if not partition:
        return False
    assignment = partition.get("assignment") or {}
    non_empty = [agent for agent, items in assignment.items() if items]
    return len(non_empty) == 1
