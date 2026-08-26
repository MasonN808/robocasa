"""Run one RoboCasa task/scene initialization with verbose placement errors."""

from __future__ import annotations

import argparse
from copy import copy
import json

import numpy as np
import robocasa.macros as macros
from robocasa.utils.placement_samplers import SequentialCompositeSampler

from training.bc_task_vlm.audit_certified_scene_compatibility import _trajectory
from training.bc_task_vlm.live_sim_eval import SimSession


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--style", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    macros.VERBOSE = True
    original_sample = SequentialCompositeSampler.sample
    reported_failures: set[str] = set()

    def traced_sample(self, placed_objects=None, reference=None, on_top=True):
        placed_objects = {} if placed_objects is None else copy(placed_objects)
        for sampler, sample_args in zip(
            self.samplers.values(), self.sample_args.values()
        ):
            call_args = dict(sample_args or {})
            for name, value in (("reference", reference), ("on_top", on_top)):
                call_args.setdefault(name, value)
            try:
                new_placements = sampler.sample(
                    placed_objects=placed_objects, **call_args
                )
            except Exception as exc:
                object_geometry = []
                for obj in getattr(sampler, "mujoco_objects", []):
                    points = obj.get_bbox_points()
                    p0, px, py, pz = points[:4]
                    object_geometry.append(
                        {
                            "name": obj.name,
                            "bbox_size": [
                                float(px[0] - p0[0]),
                                float(py[1] - p0[1]),
                                float(pz[2] - p0[2]),
                            ],
                        }
                    )
                relevant_placed = {}
                for name, (position, _quat, obj) in placed_objects.items():
                    if not any(token in name for token in ("plate", "tray")):
                        continue
                    points = obj.get_bbox_points()
                    p0, px, py, pz = points[:4]
                    relevant_placed[name] = {
                        "position": np.asarray(position).tolist(),
                        "bbox_size": [
                            float(px[0] - p0[0]),
                            float(py[1] - p0[1]),
                            float(pz[2] - p0[2]),
                        ],
                    }
                print(
                    "OBJECT_SAMPLER_FAILURE "
                    f"composite={self.name!r} sampler={sampler.name!r} "
                    f"objects={[obj.name for obj in getattr(sampler, 'mujoco_objects', [])]!r} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if sampler.name not in reported_failures:
                    reported_failures.add(sampler.name)
                    print(
                        "OBJECT_SAMPLER_GEOMETRY "
                        + json.dumps(
                            {
                                "sampler": sampler.name,
                                "x_range": np.asarray(sampler.x_range).tolist(),
                                "y_range": np.asarray(sampler.y_range).tolist(),
                                "reference_pos": np.asarray(
                                    sampler.reference_pos
                                ).tolist(),
                                "reference_rot": float(sampler.reference_rot),
                                "objects": object_geometry,
                                "already_placed_supports": relevant_placed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                raise
            placed_objects.update(new_placements)
        sampled_names = [
            obj.name for sampler in self.samplers.values()
            for obj in sampler.mujoco_objects
        ]
        return {key: value for key, value in placed_objects.items() if key in sampled_names}

    SequentialCompositeSampler.sample = traced_sample
    trajectory = _trajectory(args.task, 0)
    session = SimSession(
        composite_task=args.task,
        sample_trajectory=trajectory,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        gl_backend="egl",
        render_size=128,
        map_dpi=40,
        map_renderer="raster",
    )
    session.close()
    SequentialCompositeSampler.sample = original_sample


if __name__ == "__main__":
    main()
