"""Does the flat step order the renderer walks match concurrent-sim order?

`sweep_trajectories.py` renders by walking `trajectory["steps"]` top to bottom.
The concurrent scheduler (`ConcurrentTaskValidator.replay`, and the live
executor it mirrors) runs ticks on a clock and, WITHIN a tick, applies calls in
agent-id order. Across ticks the two agree; within a tick nothing forces them
to. This reports how often they actually diverge, and how far.

    python order_ab.py [corpus]
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

REPO = Path("/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration")
sys.path.insert(0, str(REPO))
CORPUS = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image"
)


def one(path):
    from data_generation.task_level.tasks.shared import concurrent_fsm as cf
    from data_generation.task_level.tasks.specs import load_task_spec
    from data_generation.task_level.tasks.specs.runtime import SpecDrivenTaskValidator

    rec = json.loads(Path(path).read_text())
    inner = SpecDrivenTaskValidator(load_task_spec(rec["composite_task"]))
    inner.initial_state = deepcopy(rec["initial_state"])
    v = cf.ConcurrentTaskValidator(inner, models=(cf.LOCK_STEP,))
    try:
        replay = v.replay(
            {"agents": rec["agents"], "steps": rec["steps"]},
            model=cf.LOCK_STEP,
            stop_on_step_error=False,
        )
    except Exception as exc:
        return {"error": type(exc).__name__}

    order = [event.index for event in replay.events]
    steps = v.validator._normalize_steps(rec["steps"])
    if len(order) != len(steps):
        return {"error": f"partial replay {len(order)}/{len(steps)}"}

    # How many calls sit at a different position than the renderer gives them,
    # and -- the part that changes pixels -- how many observations have a
    # DIFFERENT set of calls applied before them in each order.
    #
    # Counted twice. `stale` takes every non-observation call, which overstates
    # the visual effect badly: `communicate` and `wait_for_signal` move the
    # symbolic state and nothing else, so reordering them around a camera
    # changes no pixel. `stale_physical` counts only calls that move the world,
    # and is the number to quote about images.
    moved = sum(1 for position, index in enumerate(order) if position != index)

    obs = {i for i, s in enumerate(steps) if s["tool"] in cf.OBSERVATION_TOOL_NAMES}
    invisible = cf.OBSERVATION_TOOL_NAMES | cf.SOCIAL_TOOL_NAMES | cf.WAIT_TOOL_NAMES
    physical = {i for i, s in enumerate(steps) if s["tool"] not in invisible}

    def prefix(sequence, counted):
        seen, out = set(), {}
        for index in sequence:
            if index in obs:
                out[index] = frozenset(seen)
            elif index in counted:
                seen.add(index)
        return out

    stale = sum(
        1
        for i in obs
        if prefix(range(len(steps)), set(range(len(steps))) - obs).get(i)
        != prefix(order, set(range(len(steps))) - obs).get(i)
    )
    flat_p = prefix(range(len(steps)), physical)
    conc_p = prefix(order, physical)
    stale_physical = sum(1 for i in obs if flat_p.get(i) != conc_p.get(i))

    return {
        "steps": len(steps),
        "moved": moved,
        "observations": len(obs),
        "stale_observations": stale,
        "stale_physical": stale_physical,
        "reordered": moved > 0,
        "any_stale": stale > 0,
        "any_stale_physical": stale_physical > 0,
    }


paths = sorted(str(p) for p in CORPUS.glob("*/trajectories/traj_*.json"))
print(f"corpus {CORPUS}\ntrajectories: {len(paths)}", flush=True)
with ProcessPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(one, paths, chunksize=8))

errors = Counter(r["error"] for r in results if "error" in r)
ok = [r for r in results if "error" not in r]
print(f"replayed {len(ok)}, errors {dict(errors)}")

n = len(ok) or 1
print(f"trajectories whose order differs:      {sum(r['reordered'] for r in ok)} "
      f"({sum(r['reordered'] for r in ok) / n:.1%})")
print(f"trajectories with a stale observation: {sum(r['any_stale'] for r in ok)} "
      f"({sum(r['any_stale'] for r in ok) / n:.1%})")
print(f"  ... counting only physical calls:    {sum(r['any_stale_physical'] for r in ok)} "
      f"({sum(r['any_stale_physical'] for r in ok) / n:.1%})")
total_steps = sum(r["steps"] for r in ok)
total_obs = sum(r["observations"] for r in ok)
moved = sum(r["moved"] for r in ok)
stale = sum(r["stale_observations"] for r in ok)
stale_p = sum(r["stale_physical"] for r in ok)
print(f"calls at a different position:         {moved}/{total_steps} "
      f"({moved / max(total_steps, 1):.1%})")
print(f"observations with a different prefix:  {stale}/{total_obs} "
      f"({stale / max(total_obs, 1):.1%})")
print(f"  ... where the prefix differs in a PHYSICAL call -- the ones whose")
print(f"      pixels can actually change:      {stale_p}/{total_obs} "
      f"({stale_p / max(total_obs, 1):.1%})")
