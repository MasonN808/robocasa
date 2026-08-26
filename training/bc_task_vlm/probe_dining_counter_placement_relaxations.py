"""Probe narrow dining-counter placement fixes without changing production tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import robosuite.utils.transform_utils as T

from robocasa.environments.kitchen.composite.filling_serving_dishes.meat_skewer_assembly import MeatSkewerAssembly
from robocasa.environments.kitchen.composite.garnishing_dishes.garnish_cake import GarnishCake
from robocasa.utils import object_utils as OU
from robocasa.utils.placement import get_fixture_aabb
from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training/bc_task_vlm/eval_runs/dining_counter_placement_relaxation_probe"
TASKS = {
    "GarnishCake": (GarnishCake, "fruit_plate"),
    "MeatSkewerAssembly": (MeatSkewerAssembly, "oven_tray"),
}


def _patch(task_cls, incoming_name: str, mode: str):
    original = task_cls._get_obj_cfgs

    def modified(self):
        cfgs = original(self)
        for cfg in cfgs:
            if cfg["name"] != incoming_name:
                continue
            if mode == "widen_strip":
                old_size = cfg["placement"]["size"]
                cfg["placement"]["size"] = (old_size[0], 0.70)
            elif mode == "widen_along_edge":
                old_size = cfg["placement"]["size"]
                cfg["placement"]["size"] = (1.40, old_size[1])
            elif mode == "center_in_strip":
                cfg["placement"]["ensure_object_boundary_in_range"] = False
            else:
                raise ValueError(mode)
        return cfgs

    task_cls._get_obj_cfgs = modified
    return original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument(
        "--mode",
        choices=("widen_strip", "widen_along_edge", "center_in_strip"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    task_cls, incoming_name = TASKS[args.task]
    original = _patch(task_cls, incoming_name, args.mode)
    session = None
    result = {
        "task": args.task, "mode": args.mode, "layout": 15,
        "style": 14, "seed": args.seed, "initialized": False,
    }
    try:
        session = SimSession(
            composite_task=args.task,
            sample_trajectory=_trajectory(args.task, 0),
            layout=15, style=14, seed=args.seed, gl_backend="egl",
            render_size=900, map_dpi=130, map_renderer="raster",
        )
        env = session.executor.env
        fixture = env.dining_counter
        fmin, fmax = get_fixture_aabb(fixture)
        body_id = env.obj_body_id[incoming_name]
        position = np.asarray(env.sim.data.body_xpos[body_id])
        quaternion = T.convert_quat(env.sim.data.body_xquat[body_id], to="xyzw")
        bbox = np.asarray(env.objects[incoming_name].get_bbox_points(trans=position, rot=quaternion))
        # This dining counter is axis-aligned in the audited scene. Keep this
        # check independent from the preferred task sampling strip.
        footprint_on_counter_aabb = bool(
            np.all(bbox[:, 0] >= fmin[0] - 1e-6)
            and np.all(bbox[:, 0] <= fmax[0] + 1e-6)
            and np.all(bbox[:, 1] >= fmin[1] - 1e-6)
            and np.all(bbox[:, 1] <= fmax[1] + 1e-6)
        )
        contact = bool(OU.check_obj_fixture_contact(env, incoming_name, fixture))
        overlap_min = np.maximum(bbox[:, :2].min(axis=0), fmin[:2])
        overlap_max = np.minimum(bbox[:, :2].max(axis=0), fmax[:2])
        bbox_span = bbox[:, :2].max(axis=0) - bbox[:, :2].min(axis=0)
        overlap_span = np.maximum(overlap_max - overlap_min, 0.0)
        aabb_support_fraction = float(np.prod(overlap_span) / np.prod(bbox_span))
        # Check that the sampled support remains stable after physics advances.
        before_settle = position.copy()
        for _ in range(500):
            env.sim.step()
        after_settle = np.asarray(env.sim.data.body_xpos[body_id]).copy()
        contact_after_settle = bool(OU.check_obj_fixture_contact(env, incoming_name, fixture))
        image_name = f"{args.task.lower()}_{args.mode}_top.jpg"
        Image.fromarray(session.executor.runner._render_top_view()).save(OUT / image_name, quality=95)
        result.update({
            "initialized": True,
            "incoming_object": incoming_name,
            "incoming_position": position.tolist(),
            "incoming_bbox_xy": [bbox[:, :2].min(axis=0).tolist(), bbox[:, :2].max(axis=0).tolist()],
            "counter_aabb_xy": [fmin[:2].tolist(), fmax[:2].tolist()],
            "whole_footprint_inside_counter_aabb": footprint_on_counter_aabb,
            "incoming_bbox_aabb_support_fraction": aabb_support_fraction,
            "native_counter_contact": contact,
            "native_counter_contact_after_500_steps": contact_after_settle,
            "position_change_after_500_steps_m": float(np.linalg.norm(after_settle - before_settle)),
            "top_view": f"assets/{image_name}",
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        task_cls._get_obj_cfgs = original
        if session is not None:
            session.close()
    path = OUT / f"{args.task.lower()}_{args.mode}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
