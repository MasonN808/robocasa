"""Lock-step scheduling shared by the validator and the wait inserter.

Both need the same answer to "when does this step actually run", and when they
each had their own answer they drifted: the inserter reasoned about positions
in the file while the executor advanced two agents independently. Everything
downstream of that gap -- collisions, inert waits, deadlocks -- came from one
of them being wrong. There is one implementation here so that cannot recur.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def schedule(
    steps: Sequence[dict[str, Any]],
    *,
    permanent_releases: bool = False,
) -> tuple[list[int | None], bool]:
    """Assign each step the tick it runs at under lock-step.

    Every call costs one tick, so at each tick every agent not blocked on an
    undischarged wait acts -- all at once.

    A release is an EVENT, not a standing fact: it wakes only an agent already
    waiting when it fires. That matches the executor, which wakes a waiter by
    delivering a message -- one sent before the agent began waiting was never
    delivered to it and never will be. Treating releases as permanent made the
    schedule more permissive than the executor, so a plan could look clean here
    and hang there.

    Returns (ticks, deadlocked); a step never reached has tick None.
    """

    order: dict[str, list[int]] = {}
    for index, step in enumerate(steps):
        order.setdefault(step.get("agent"), []).append(index)

    pointer = {agent: 0 for agent in order}
    ticks: list[int | None] = [None] * len(steps)
    fired: dict[tuple[str, str], list[int]] = {}
    waiting_since: dict[str, int] = {}
    tick = 0

    while True:
        runnable: list[tuple[str, int]] = []
        for agent, indices in order.items():
            if pointer[agent] >= len(indices):
                continue
            index = indices[pointer[agent]]
            step = steps[index]
            if step.get("tool") == "wait_for_signal":
                args = step.get("args") or {}
                key = (args.get("from"), str(args.get("about")))
                since = waiting_since.setdefault(agent, tick)
                events = fired.get(key, ())
                if permanent_releases:
                    if not events:
                        continue
                elif not any(at >= since for at in events):
                    continue
                waiting_since.pop(agent, None)
            runnable.append((agent, index))

        if not runnable:
            break

        for agent, index in runnable:
            ticks[index] = tick
            step = steps[index]
            if step.get("tool") == "communicate":
                value = (step.get("args") or {}).get("releases") or []
                for released_id in ([value] if isinstance(value, str) else value):
                    fired.setdefault((agent, str(released_id)), []).append(tick)
            pointer[agent] += 1
        tick += 1

    return ticks, any(pointer[a] < len(order[a]) for a in order)


def usage_spans(
    steps: Sequence[dict[str, Any]],
    ticks: Sequence[int | None],
    resources_of,
) -> dict[str, dict[str, tuple[int, int]]]:
    """First and last tick at which each agent occupies each resource."""

    spans: dict[str, dict[str, list[int]]] = {}
    for index, step in enumerate(steps):
        if ticks[index] is None:
            continue
        agent = step.get("agent")
        for resource in resources_of(step):
            spans.setdefault(resource, {}).setdefault(agent, []).append(ticks[index])
    return {
        resource: {agent: (min(at), max(at)) for agent, at in by_agent.items()}
        for resource, by_agent in spans.items()
    }


def overlapping_users(
    spans: dict[str, dict[str, tuple[int, int]]],
) -> Iterable[tuple[str, str, str]]:
    """Yield (resource, first_user, later_user) where the tick spans overlap.

    Sequential use is not contention. The lock-step schedule is a function of
    the plan, so an agent that finishes with a bowl at tick 6 has finished
    before another that reaches it at tick 15, whether or not anything is
    said. Only an actual overlap needs ordering.
    """

    for resource, by_agent in spans.items():
        if len(by_agent) < 2:
            continue
        ordered = sorted(by_agent.items(), key=lambda kv: kv[1])
        for position, (agent, (start, end)) in enumerate(ordered):
            for other, (other_start, other_end) in ordered[position + 1 :]:
                if start <= other_end and other_start <= end:
                    yield resource, agent, other
