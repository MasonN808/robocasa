"""Compare current vs consistent stovetop-parent exclusion across one scene."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS

import robocasa.utils.occupancy_grid as occupancy_grid
from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.tasks import get_task_definition
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150_canonical_workspace.json"
STOVE_TOKENS = ("stove", "stovetop", "cooktop")
BASE_TOKENS = tuple(token for token in occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS if token not in STOVE_TOKENS)


def _trajectory(task: str, run_index: int) -> dict[str, Any]:
    cfg = _config(Namespace(task=task, num_runs=150, run_index=run_index, location="global", temperature=0.6), model=FLASH_MODEL, thinking="low")
    state = get_task_definition(task).build_task_instance(run_index, cfg).initial_state
    return {"trajectory_id": f"audit_{task}_{run_index}", "task": task, "composite_task": task, "initial_state": state, "grounding_map": build_grounding_map_for_task(task, state), "steps": []}


def _load(executor, adapted: dict[str, Any], *, fixed: bool) -> dict[str, Any]:
    occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS = BASE_TOKENS + (STOVE_TOKENS if fixed else ())
    executor.restore_baseline_state()
    try:
        executor.load_initial_state(adapted["initial_state"])
        return {"success": True, "positions": {f"agent_{i}": executor.runner._get_robot_position(i)[:2].tolist() for i in (0,1)}}
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}", "positions": {f"agent_{i}": executor.runner._get_robot_position(i)[:2].tolist() for i in (0,1)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--style", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(MANIFEST.read_text())
    records=[]; errors=[]; skipped=[]
    for task_row in source["tasks"]:
        task=task_row["task"]
        # Cheap symbolic prefilter; exact parent equality is checked after grounding.
        candidate_runs=[]; seen=set()
        for run in task_row["runs"]:
            locs=tuple((run["configuration"].get("agent_locations") or {}).values())
            key=tuple(locs)
            if key in seen: continue
            seen.add(key)
            if any("stove" in str(x).lower() for x in locs) and any("counter" in str(x).lower() for x in locs):
                candidate_runs.append(run["run_index"])
        if not candidate_runs: continue
        task_class=REGISTERED_ENVS[get_task_metadata(task).composite_task]
        if args.layout in task_class.EXCLUDE_LAYOUTS or args.style in task_class.EXCLUDE_STYLES:
            skipped.append(task); continue
        session=None
        try:
            sample=_trajectory(task,candidate_runs[0])
            session=SimSession(composite_task=task,sample_trajectory=sample,layout=args.layout,style=args.style,seed=args.seed,gl_backend="egl",render_size=256,map_dpi=60,map_renderer="raster")
            for run_index in candidate_runs:
                traj=_trajectory(task,run_index)
                adapter=TrajectoryAdapter(executor=session.executor,allow_approximate_ids=True)
                adapted=adapter.adapt(traj)
                agents=adapted["initial_state"]["agents"]
                scene=session.executor.get_scene_description().get("fixtures") or {}
                stove_agents=[a for a,s in agents.items() if "stove" in str((scene.get(s.get("location")) or {}).get("fixture_type") or "").lower()]
                if len(stove_agents)!=1: continue
                stove_agent=stove_agents[0]; stove_id=agents[stove_agent]["location"]
                parent=(scene.get(stove_id) or {}).get("parent_fixture")
                partner="agent_1" if stove_agent=="agent_0" else "agent_0"
                if agents[partner].get("location") != parent: continue
                records.append({"task":task,"run_index":run_index,"symbolic_locations":traj["initial_state"]["agents"],"resolved_locations":agents,"stove_id":stove_id,"parent_id":parent,"baseline":_load(session.executor,adapted,fixed=False),"fixed":_load(session.executor,adapted,fixed=True)})
        except Exception as exc:
            errors.append({"task":task,"error":f"{type(exc).__name__}: {exc}"})
        finally:
            if session is not None: session.close()
    occupancy_grid._COUNTERTOP_APPLIANCE_TOKENS = BASE_TOKENS
    payload={"scene":{"layout":args.layout,"style":args.style,"seed":args.seed},"records":records,"errors":errors,"skipped":skipped}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"scene":payload["scene"],"records":len(records),"errors":len(errors)},indent=2))


if __name__ == "__main__": main()
