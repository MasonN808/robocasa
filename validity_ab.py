"""Corpus-wide validity before vs after the zero-duration change."""
import json, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

REPO = Path("/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration")
sys.path.insert(0, str(REPO))
CORPUS = Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image")
_OLD = {"communicate": 0.25, "get_image": 0.25, "navigate_to_fixture": 4.0}


def old_duration_of(tool_name, model):
    from data_generation.task_level.tasks.shared import concurrent_fsm as cf
    if model == cf.LOCK_STEP:
        return 1.0
    return _OLD.get(tool_name, 2.0)


def one(path):
    from data_generation.task_level.tasks.shared import concurrent_fsm as cf
    from data_generation.task_level.tasks.specs import load_task_spec
    from data_generation.task_level.tasks.specs.runtime import SpecDrivenTaskValidator
    rec = json.loads(Path(path).read_text())
    out = {}
    for label, charge in (("before", True), ("after", False)):
        inner = SpecDrivenTaskValidator(load_task_spec(rec["composite_task"]))
        inner.initial_state = deepcopy(rec["initial_state"])
        v = cf.ConcurrentTaskValidator(inner, models=(cf.LOCK_STEP,))
        saved = cf.duration_of
        if charge:
            cf.duration_of = old_duration_of
        try:
            r = v.validate({"agents": rec["agents"], "steps": rec["steps"]})
            out[label] = (bool(r.get("is_valid")), r.get("error_type"))
        except Exception as exc:
            out[label] = (False, type(exc).__name__)
        finally:
            cf.duration_of = saved
    return out


paths = sorted(str(p) for p in CORPUS.glob("*/trajectories/traj_*.json"))
print("trajectories:", len(paths), flush=True)
with ProcessPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(one, paths, chunksize=8))
for label in ("before", "after"):
    ok = sum(1 for r in results if r[label][0])
    errs = Counter(r[label][1] for r in results if not r[label][0])
    print(f"{label}: valid {ok}/{len(results)} = {ok / len(results):.2%}   failures {dict(errs.most_common())}")
