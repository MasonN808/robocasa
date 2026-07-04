"""Arbitrary state snapshot / restore for SimToolExecutor probes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExecutorStateSnapshot:
    sim_state: np.ndarray
    scene: dict[str, Any] | None
    object_locations: dict[str, str]
    held_objects: dict[int, str]
    held_object_offsets: dict[int, np.ndarray]
    support_parents: dict[str, str]
    recent_opened_sliding_fixture: dict[int, str]
    fixture_runtime_state: dict[str, dict[str, Any]]
    site_rgba: np.ndarray
    site_size: np.ndarray
    geom_rgba: np.ndarray


def capture_executor_state(executor) -> ExecutorStateSnapshot:
    executor.env.sim.forward()
    return ExecutorStateSnapshot(
        sim_state=np.array(executor.env.sim.get_state().flatten(), copy=True),
        scene=deepcopy(getattr(executor.runner, "_scene", None)),
        object_locations=dict(getattr(executor.runner, "_object_locations", {})),
        held_objects=deepcopy(getattr(executor, "_held_objects", {})),
        held_object_offsets=deepcopy(getattr(executor, "_held_object_offsets", {})),
        support_parents=deepcopy(getattr(executor, "_support_parents", {})),
        recent_opened_sliding_fixture=deepcopy(
            getattr(executor, "_recent_opened_sliding_fixture", {})
        ),
        fixture_runtime_state=executor._snapshot_fixture_runtime_state(),
        site_rgba=np.array(executor.env.sim.model.site_rgba, copy=True),
        site_size=np.array(executor.env.sim.model.site_size, copy=True),
        geom_rgba=np.array(executor.env.sim.model.geom_rgba, copy=True),
    )


def _restore_fixture_runtime_state(executor, fixture_runtime_state: dict[str, dict[str, Any]]) -> None:
    for fixture_id, fixture_state in fixture_runtime_state.items():
        fixture = executor.runner._fixtures.get(fixture_id)
        if fixture is None:
            continue
        for attr_name, attr_value in fixture_state.items():
            setattr(fixture, attr_name, deepcopy(attr_value))


def restore_executor_state(executor, snapshot: ExecutorStateSnapshot) -> None:
    executor.env.sim.set_state_from_flattened(snapshot.sim_state.copy())
    executor.env.sim.forward()
    executor.env.sim.model.site_rgba[:] = snapshot.site_rgba
    executor.env.sim.model.site_size[:] = snapshot.site_size
    executor.env.sim.model.geom_rgba[:] = snapshot.geom_rgba
    _restore_fixture_runtime_state(executor, snapshot.fixture_runtime_state)

    executor.runner._object_locations = dict(snapshot.object_locations)
    executor.runner._scene = deepcopy(snapshot.scene)
    executor._held_objects = deepcopy(snapshot.held_objects)
    executor._held_object_offsets = deepcopy(snapshot.held_object_offsets)
    executor._support_parents = deepcopy(snapshot.support_parents)
    executor._recent_opened_sliding_fixture = deepcopy(snapshot.recent_opened_sliding_fixture)

    if hasattr(executor.env, "update_sites"):
        executor.env.update_sites()
    if hasattr(executor.env, "update_state"):
        executor.env.update_state()
    executor.env.sim.forward()
    invalidate = getattr(executor, "_invalidate_visual_cache", None)
    if callable(invalidate):
        invalidate()
