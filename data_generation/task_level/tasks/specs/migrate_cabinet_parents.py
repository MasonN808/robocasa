"""One-time migration for model-visible cabinet parent relationships.

This operates only on verified JSON specifications. It does not initialize a
simulator. Missing parents receive a scene-agnostic symbolic fixture whose
grounding is anchored to the cabinet; simulator-backed audit resolves that
symbol later when a scene is already being used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent / "verified"
FALLBACK_PARENT_ID = "cabinet_parent_counter"


def _is_model_relevant_cabinet(fixture_id: str, state: dict[str, Any]) -> bool:
    fixture_type = str(state.get("fixture_type") or "").lower()
    # ServeMealJuice's misleading ``sink`` is an internal native placement
    # anchor, not a model-relevant cabinet.
    return "cabinet" in fixture_type and fixture_id != "sink"


def migrate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    changes: list[str] = []
    initial_state = payload.get("initial_state") or {}
    fixtures = initial_state.get("fixtures") or {}
    symbols = ((payload.get("grounding") or {}).get("symbols") or {})

    missing = [
        fixture_id
        for fixture_id, fixture_state in fixtures.items()
        if isinstance(fixture_state, dict)
        and _is_model_relevant_cabinet(fixture_id, fixture_state)
        and not isinstance(fixture_state.get("parent_fixture"), str)
    ]
    if len(missing) > 1:
        raise ValueError(
            f"{payload.get('composite_task')} has multiple parentless cabinets; "
            "use an explicit per-parent migration rather than one fallback alias."
        )
    if missing:
        cabinet_id = missing[0]
        if FALLBACK_PARENT_ID in fixtures:
            raise ValueError(
                f"{payload.get('composite_task')} already defines {FALLBACK_PARENT_ID!r}."
            )
        fixtures[cabinet_id]["parent_fixture"] = FALLBACK_PARENT_ID
        fixtures[FALLBACK_PARENT_ID] = {
            "fixture_type": "counter",
            "parts": {},
            "controls": {},
        }
        symbols[FALLBACK_PARENT_ID] = {
            "entity_type": "fixture",
            "resolver": "nearest_placeable_surface_to_fixture",
            "anchor_fixture_symbol": cabinet_id,
            "preferred_fixture_types": ["counter"],
        }
        changes.append(f"added {cabinet_id}.parent_fixture={FALLBACK_PARENT_ID}")

    if payload.get("composite_task") == "ServeMealJuice":
        removed_state = fixtures.pop("sink", None)
        removed_symbol = symbols.pop("sink", None)
        if removed_state is not None or removed_symbol is not None:
            changes.append("removed unused model-visible sink fixture")

    # Hidden validation keeps both legacy cabinet navigation and the canonical
    # parent workspace legal. Model-facing generation later filters this union
    # down to only the canonical parent.
    for cabinet_id, fixture_state in list(fixtures.items()):
        if not isinstance(fixture_state, dict) or not _is_model_relevant_cabinet(
            cabinet_id, fixture_state
        ):
            continue
        parent_id = fixture_state.get("parent_fixture")
        if not isinstance(parent_id, str):
            continue
        for tool_name in ("navigate_to_fixture", "give_space"):
            tool_spec = (payload.get("allowed_tool_specs") or {}).get(tool_name)
            if not isinstance(tool_spec, dict):
                continue
            allowed = tool_spec.get("allowed_fixture_ids")
            if not isinstance(allowed, list) or cabinet_id not in allowed:
                continue
            if parent_id not in allowed:
                allowed.append(parent_id)
                changes.append(f"added {parent_id} to {tool_name} aliases")

    return payload, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write migrated JSON files.")
    args = parser.parse_args()
    changed_files = 0
    for path in sorted(ROOT.glob("*.json")):
        original = json.loads(path.read_text())
        migrated, changes = migrate_payload(original)
        if not changes:
            continue
        changed_files += 1
        print(f"{path.name}: {'; '.join(changes)}")
        if args.apply:
            path.write_text(json.dumps(migrated, indent=2) + "\n")
    print(f"changed_files={changed_files} mode={'apply' if args.apply else 'dry-run'}")


if __name__ == "__main__":
    main()
