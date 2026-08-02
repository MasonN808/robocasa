#!/usr/bin/env python
"""Writes each task's FULLY RESOLVED tool set back into its spec file.

Today the tool set is computed in one place and read from another:

    spec file declares      : communicate, give_space, navigate_to_fixture, ...
    initial_state implies   : hinged parts on {fridge, cabinet}
    GENERATOR validator sees: ... + open_hinged_part  (allowlists derived)
    EVALUATOR registry sees : ...                     (raw declared spec)

`task_registry.py` does `deepcopy(task_spec.allowed_tool_specs)` and never
calls `build_task_definition_from_spec`, so the evaluator misses every runtime
injection. That single divergence produces three observed symptoms:
"Tool open_hinged_part is not allowed here", `KeyError: 'hinged'`, and the
over-narrow `give_space` allowlist.

After this migration the spec IS the tool set: generator and evaluator read the
same field and cannot drift.

  python materialize_task_specs.py            # dry run, prints the diff
  python materialize_task_specs.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolved_tool_specs(task_spec):
    """The tool set the GENERATOR actually validates against."""

    from data_generation.task_level.tasks.specs.runtime import (
        build_task_definition_from_spec,
    )

    definition = build_task_definition_from_spec(task_spec)
    validator = definition.validator_factory(None)
    resolved = dict(getattr(validator, "allowed_tool_specs", {}) or {})

    # Fold in the give_space widening so it lives in the SPEC rather than as a
    # runtime patch -- two mechanisms is the defect we are removing. The
    # per-task allowlist was generated from fixtures the expert happened to
    # yield at, not a simulator constraint (SimToolExecutor.give_space only
    # requires the fixture to exist). Where a reachable fixture was missing,
    # two agents contending there had no legal way to resolve it; hot_dog_setup's
    # unyieldable `counter` scored 0/120.
    from training.bc_task_vlm.schema_utils import normalize_give_space_fixtures

    return normalize_give_space_fixtures(resolved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    from data_generation.task_level.tasks.specs import (
        _task_spec_paths,
        load_all_task_specs,
    )

    changed = 0
    for spec in load_all_task_specs():
        declared = set(spec.allowed_tool_specs)
        resolved = resolved_tool_specs(spec)
        added = sorted(set(resolved) - declared)
        # Constraint values can also change (derived allowlists, widened
        # give_space), so compare the whole structure, not just the key set.
        differs = any(
            spec.allowed_tool_specs.get(name) != resolved[name] for name in resolved
        )
        if not added and not differs:
            continue
        changed += 1
        print(f"\n{spec.composite_task}")
        if added:
            print(f"   + tools: {added}")
        for name in sorted(resolved):
            before = spec.allowed_tool_specs.get(name)
            if before == resolved[name]:
                continue
            b = {k: v for k, v in (before or {}).items() if k.startswith("allowed_")}
            a = {k: v for k, v in resolved[name].items() if k.startswith("allowed_")}
            if b != a:
                print(f"   ~ {name}: {b or '(absent)'} -> {a}")

        if not args.apply:
            continue

        path = next(p for p in _task_spec_paths(spec.composite_task) if p.exists())
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["allowed_tool_specs"] = resolved
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"\n{changed} spec(s) {'updated' if args.apply else 'would change'}")
    if not args.apply:
        print("dry run -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
