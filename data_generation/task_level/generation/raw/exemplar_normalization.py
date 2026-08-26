"""Conservative normalization of saved trajectory exemplars.

Only unambiguous message text is changed. Scheduling, tools, arguments, agent
assignments, and reasoning are deliberately preserved.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


class ExemplarNormalizationError(ValueError):
    """Raised when an old exemplar cannot be repaired without changing behavior."""


def _names_exact_id(message: str, resource_id: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(resource_id)}(?![A-Za-z0-9_])",
        message,
    ) is not None


def _requests_release(message: str) -> bool:
    return re.search(r"\breleas(?:e|es|ed|ing)\b", message, re.IGNORECASE) is not None


def normalize_exemplar_release_requests(candidate: dict[str, Any]) -> dict[str, Any]:
    """Add an exact release request before waits when that edit is unambiguous.

    A wait must already immediately follow a communication to its holder. If it
    does not, normalization refuses the exemplar because repairing it would alter
    the tick schedule rather than merely clarify existing language.
    """

    normalized = deepcopy(candidate)
    ticks = normalized.get("ticks")
    if not isinstance(ticks, list):
        raise ExemplarNormalizationError("exemplar must contain a ticks list")

    agent_ids = sorted(
        {
            key
            for row in ticks
            if isinstance(row, dict)
            for key in row
            if key.startswith("agent_")
        }
    )
    if not agent_ids:
        raise ExemplarNormalizationError("exemplar contains no agent fields")

    # Convert the legacy omitted-waiter representation into explicit markers.
    blocked: dict[str, tuple[str, str]] = {}
    for row_index, row in enumerate(ticks):
        if not isinstance(row, dict):
            raise ExemplarNormalizationError(f"tick row {row_index} is not an object")
        for agent_id in agent_ids:
            if agent_id not in row:
                if agent_id not in blocked:
                    raise ExemplarNormalizationError(
                        f"tick row {row_index} omits unblocked {agent_id}"
                    )
                row[agent_id] = {"state": "blocked"}
            elif agent_id in blocked and row[agent_id] != {"state": "blocked"}:
                raise ExemplarNormalizationError(
                    f"tick row {row_index} invokes blocked {agent_id}"
                )

        released: set[str] = set()
        new_waits: dict[str, tuple[str, str]] = {}
        for agent_id in agent_ids:
            call = row[agent_id]
            if not isinstance(call, dict):
                continue
            args = call.get("args") or {}
            if call.get("tool") == "communicate":
                waiter = args.get("to")
                release_id = args.get("releases")
                if waiter in blocked and release_id == blocked[waiter][1]:
                    released.add(waiter)
            elif call.get("tool") == "wait_for_signal":
                holder, about = args.get("from"), args.get("about")
                if isinstance(holder, str) and isinstance(about, str):
                    new_waits[agent_id] = (holder, about)
        for agent_id in released:
            blocked.pop(agent_id)
        blocked.update(new_waits)

    last_real_call: dict[str, tuple[dict[str, Any], int]] = {}
    for row_index, row in enumerate(ticks):
        if not isinstance(row, dict):
            raise ExemplarNormalizationError(f"tick row {row_index} is not an object")
        for agent_id, call in row.items():
            if agent_id == "tick" or not isinstance(call, dict):
                continue
            tool = call.get("tool")
            if tool == "get_image" or call == {"state": "blocked"}:
                continue
            if tool == "wait_for_signal":
                args = call.get("args") or {}
                about = args.get("about")
                holder = args.get("from")
                previous = last_real_call.get(agent_id)
                if not isinstance(about, str) or not isinstance(holder, str):
                    raise ExemplarNormalizationError(
                        f"{agent_id} wait at row {row_index} lacks string from/about"
                    )
                if previous is None or previous[0].get("tool") != "communicate":
                    raise ExemplarNormalizationError(
                        f"{agent_id} wait at row {row_index} does not immediately follow communicate"
                    )
                prior_call, prior_row = previous
                prior_args = prior_call.get("args") or {}
                if prior_args.get("to") != holder:
                    raise ExemplarNormalizationError(
                        f"{agent_id} wait at row {row_index} follows a message to the wrong agent"
                    )
                message = prior_args.get("message")
                if not isinstance(message, str):
                    raise ExemplarNormalizationError(
                        f"{agent_id} communicate at row {prior_row} lacks a string message"
                    )
                if not (_names_exact_id(message, about) and _requests_release(message)):
                    prior_args["message"] = (
                        f"{message.rstrip()} When done, release \"{about}\"."
                    )
                    prior_call["args"] = prior_args
            last_real_call[agent_id] = (call, row_index)
    return normalized
