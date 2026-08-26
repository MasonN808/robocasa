"""Audit every unique physical configuration against one candidate scene."""

from __future__ import annotations

import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
from typing import Any

import robocasa  # noqa: F401
from robosuite.environments.base import REGISTERED_ENVS

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.grounding_specs import build_grounding_map_for_task
from data_generation.task_level.scene_sampling import (
    SCENE_POLICY_VERSION,
    physical_configuration,
    physical_configuration_signature,
    scene_signature,
)
from data_generation.task_level.tasks import get_task_definition
from robocasa.utils.trajectory_adapter import TrajectoryAdapter
from training.bc_task_vlm.live_sim_eval import SimSession
from training.bc_task_vlm.task_registry import get_task_metadata


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150_canonical_workspace.json"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _trajectory(task: str, run_index: int) -> dict[str, Any]:
    cfg = _config(Namespace(task=task, num_runs=150, run_index=run_index, location="global", temperature=0.6), model=FLASH_MODEL, thinking="low")
    state = get_task_definition(task).build_task_instance(run_index, cfg).initial_state
    return {"trajectory_id": f"scene_audit_{task}_{run_index}", "task": task, "composite_task": task, "initial_state": state, "grounding_map": build_grounding_map_for_task(task, state), "steps": []}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout",type=int,required=True); parser.add_argument("--style",type=int,required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task", action="append", dest="tasks", help="Audit only this task; repeat for multiple tasks")
    args=parser.parse_args(); source=json.loads(MANIFEST.read_text())
    scene={"layout":args.layout,"style":args.style,"seed":args.seed}; records=[]; task_errors=[]; unsupported=[]
    for task_row in source["tasks"]:
        task=task_row["task"]; cls=REGISTERED_ENVS[get_task_metadata(task).composite_task]
        if args.tasks and task not in args.tasks:
            continue
        if args.layout in cls.EXCLUDE_LAYOUTS or args.style in cls.EXCLUDE_STYLES:
            unsupported.append(task); continue
        unique={}
        for run in task_row["runs"]:
            physical=physical_configuration(run["configuration"]); sig=physical_configuration_signature(run["configuration"])
            unique.setdefault(sig,(run["run_index"],physical))
        session=None
        try:
            sample=_trajectory(task,next(iter(unique.values()))[0])
            session=SimSession(composite_task=task,sample_trajectory=sample,**scene,gl_backend="egl",render_size=128,map_dpi=40,map_renderer="raster")
            for signature,(run_index,physical) in unique.items():
                trajectory=_trajectory(task,run_index)
                adapter=TrajectoryAdapter(executor=session.executor,allow_approximate_ids=True)
                try:
                    adapted=adapter.adapt(trajectory)
                    session.executor.restore_baseline_state()
                    session.executor.load_initial_state(adapted["initial_state"])
                    records.append({"task":task,"physical_configuration_signature":signature,"physical_configuration":physical,"representative_run_index":run_index,"compatible":True,"resolved_agent_locations":adapted["initial_state"].get("agents")})
                except Exception as exc:
                    records.append({"task":task,"physical_configuration_signature":signature,"physical_configuration":physical,"representative_run_index":run_index,"compatible":False,"error":f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            task_errors.append({"task":task,"error":f"{type(exc).__name__}: {exc}","configuration_count":len(unique)})
        finally:
            if session is not None: session.close()
    payload={"scene_policy_version":SCENE_POLICY_VERSION,"scene":{**scene,"scene_signature":scene_signature(scene)},"records":records,"unsupported_tasks":unsupported,"task_errors":task_errors}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"scene":scene,"records":len(records),"compatible":sum(r["compatible"] for r in records),"incompatible":sum(not r["compatible"] for r in records),"task_errors":len(task_errors)},indent=2))


if __name__=="__main__": main()
