#!/usr/bin/env python3
"""Make the model-facing communicate schema match the shared FSM contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data_generation/task_level/tasks/specs"
PHASES = [
    "propose",
    "await_plan",
    "await_confirmation",
    "confirm",
    "portion_complete",
]
ACTIVE_TOP_LEVEL = {"preparesandwichstation.json"}
PHASE_HELP = (
    " coordination_phase is optional and must be omitted from ordinary "
    "messages. Use it only for the opening propose/await_plan/"
    "await_confirmation/confirm handshake or for portion_complete."
)


def is_active_spec(path: Path) -> bool:
    relative = path.relative_to(SPECS)
    return relative.parent == Path("verified") or (
        relative.parent == Path(".") and relative.name in ACTIVE_TOP_LEVEL
    )


def main() -> int:
    changed = 0
    checked = 0
    for path in sorted(SPECS.rglob("*.json")):
        payload = json.loads(path.read_text())
        communicate = (payload.get("allowed_tool_specs") or {}).get("communicate")
        if communicate is None:
            continue
        checked += 1
        active = is_active_spec(path)
        optional = list(communicate.get("optional_tool_args", ()))
        if not active:
            optional = [arg for arg in optional if arg != "coordination_phase"]
            communicate["optional_tool_args"] = optional
            arg_types = dict(communicate.get("tool_arg_types", {}))
            arg_types.pop("coordination_phase", None)
            if arg_types:
                communicate["tool_arg_types"] = arg_types
            else:
                communicate.pop("tool_arg_types", None)
            allowed = dict(communicate.get("allowed_arg_values", {}))
            allowed.pop("coordination_phase", None)
            if allowed:
                communicate["allowed_arg_values"] = allowed
            else:
                communicate.pop("allowed_arg_values", None)
            communicate["description"] = str(
                communicate.get("description", "")
            ).replace(PHASE_HELP, "")
            rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            if rendered != path.read_text():
                path.write_text(rendered)
                changed += 1
            continue
        if "coordination_phase" not in optional:
            optional.append("coordination_phase")
        communicate["optional_tool_args"] = optional
        arg_types = dict(communicate.get("tool_arg_types", {}))
        arg_types["coordination_phase"] = "STRING"
        communicate["tool_arg_types"] = arg_types
        allowed = dict(communicate.get("allowed_arg_values", {}))
        allowed["coordination_phase"] = PHASES
        communicate["allowed_arg_values"] = allowed
        description = str(communicate.get("description", "")).rstrip()
        if "coordination_phase is optional" not in description:
            communicate["description"] = description + PHASE_HELP
        rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if rendered != path.read_text():
            path.write_text(rendered)
            changed += 1
    if checked == 0:
        raise RuntimeError(f"No communicate specs found under {SPECS}")
    print(f"checked={checked} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
