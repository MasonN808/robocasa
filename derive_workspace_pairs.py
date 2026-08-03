#!/usr/bin/env python
"""Derives which counter each storage/appliance fixture shares a footprint with.

`_fixtures_share_workspace` has two spatial branches and neither has data:
parent_fixture is declared on 0 of 128 fixtures, and the cluster gate compares
name-derived ids that never match (0 of 49 candidate pairs pass). The generator
prompt works around this by naming fixture TYPES ("cabinet/drawer fixtures and
their supporting counters"), which resolves only when a task has exactly one
counter -- true for 26 tasks, ambiguous for 8, and silent about appliances.

This measures the real geometry once so the relation can be declared instead of
guessed. Reports distances; --apply writes `parent_fixture` into the specs.

A fixture with no counter inside --max-distance gets NO field: a free-standing
fridge genuinely has no supporting counter, and forcing a nearest match is the
"fell back to nearest-center" behaviour the resolution logs already warn about.

  python derive_workspace_pairs.py                 # measure and report
  python derive_workspace_pairs.py --apply --max-distance 1.0
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

DATA = "/work/umass/shlomo_umass/dbenhamougol_umass/data"
ROOT = (
    f"{DATA}/structured_random_raw/data_generation/task_level"
    f"/data_structured_random/image/20260706T174208"
)

COUNTERLIKE = {
    "counter", "counter_non_dining", "counter_non_corner", "dining_counter",
    "island", "staging_surface", "dining_table", "counter_stove",
}
NEEDS_SUPPORT = {
    "cabinet", "cabinet_single_door", "cabinet_double_door", "cabinet_with_door",
    "drawer", "top_drawer", "fridge",
    "toaster_oven", "toaster", "coffee_machine", "blender", "stand_mixer",
    "electric_kettle",
}


def measure_task(task: str, composite: str):
    """Returns [(symbolic_fixture, nearest_counter, dist, runner_up_dist)]."""

    from training.bc_task_vlm.live_sim_eval import SimSession

    paths = sorted(glob.glob(f"{ROOT}/{task}/traj_*/original_trajectory.json"))
    if not paths:
        return []
    trajectory = json.loads(Path(paths[0]).read_text())
    scene = (trajectory.get("scene") or {})
    session = SimSession(
        composite_task=composite,
        sample_trajectory=trajectory,
        layout=int(scene.get("layout", 11)),
        style=int(scene.get("style", 34)),
        seed=int(scene.get("seed", 42)),
        gl_backend="egl",
        render_size=256,
    )
    adapter, _adapted = session.start_trajectory(trajectory)
    aliases = dict(getattr(adapter, "_fixture_aliases", {}) or {})
    concrete = session.executor.runner._fixtures
    position = {
        name: np.asarray(fixture.pos[:2], dtype=float)
        for name, fixture in concrete.items()
        if getattr(fixture, "pos", None) is not None
    }

    fixtures = (trajectory.get("initial_state") or {}).get("fixtures") or {}
    types = {
        fid: str((state or {}).get("fixture_type", "")).lower()
        for fid, state in fixtures.items()
    }
    counters = [f for f, t in types.items() if t in COUNTERLIKE]
    out = []
    for fid, ftype in types.items():
        if ftype not in NEEDS_SUPPORT:
            continue
        here = position.get(aliases.get(fid, fid))
        if here is None:
            continue
        distances = []
        for counter in counters:
            there = position.get(aliases.get(counter, counter))
            if there is None:
                continue
            distances.append((float(np.linalg.norm(here - there)), counter))
        if not distances:
            continue
        distances.sort()
        nearest_d, nearest = distances[0]
        runner_up = distances[1][0] if len(distances) > 1 else float("inf")
        out.append((fid, nearest, nearest_d, runner_up))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-distance", type=float, default=1.0)
    ap.add_argument("--tasks", type=str, default="")
    args = ap.parse_args()

    from data_generation.task_level.tasks.specs import (
        _task_spec_paths,
        load_all_task_specs,
    )
    from training.bc_task_vlm.task_registry import _camel_to_snake_case

    only = {t for t in args.tasks.split(",") if t}
    rows, written = [], 0
    for spec in load_all_task_specs():
        task = _camel_to_snake_case(spec.composite_task)
        if only and task not in only:
            continue
        try:
            measured = measure_task(task, spec.composite_task)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"  !! {task}: {type(exc).__name__}: {exc}")
            continue
        if not measured:
            continue
        assignments = {}
        for fid, counter, dist, runner_up in measured:
            keep = dist <= args.max_distance
            rows.append((task, fid, counter, dist, runner_up, keep))
            print(
                f"  {task:28} {fid:22} -> {counter:22} "
                f"{dist:5.2f} m (next {runner_up:5.2f}) {'KEEP' if keep else 'skip'}"
            )
            if keep:
                assignments[fid] = counter
        if args.apply and assignments:
            path = next(p for p in _task_spec_paths(spec.composite_task) if p.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            fixtures = payload.get("initial_state", {}).get("fixtures", {})
            for fid, counter in assignments.items():
                if fid in fixtures and isinstance(fixtures[fid], dict):
                    fixtures[fid]["parent_fixture"] = counter
                    written += 1
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    kept = sum(1 for r in rows if r[5])
    print(f"\nmeasured {len(rows)} fixtures; {kept} within {args.max_distance} m")
    if args.apply:
        print(f"wrote parent_fixture on {written} fixtures")
    else:
        print("dry run -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
