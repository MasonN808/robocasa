"""Rendering must walk the steps the way the concurrent executor will."""

import json
from pathlib import Path

import pytest

from data_generation.task_level.tasks.shared.render_order import (
    concurrent_step_order,
    reorder_for_concurrent_render,
)

CORPUS = Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image")


def _sample(count: int = 12) -> list[Path]:
    paths = sorted(CORPUS.glob("*/trajectories/traj_*.json"))
    if not paths:
        pytest.skip(f"corpus not present at {CORPUS}")
    stride = max(len(paths) // count, 1)
    return paths[::stride][:count]


def test_a_trajectory_with_one_step_is_left_alone():
    trajectory = {"composite_task": "nope", "steps": [{"step": 0}]}
    assert concurrent_step_order(trajectory) == ([0], None)


def test_an_unloadable_task_falls_back_to_file_order_with_a_reason():
    trajectory = {
        "composite_task": "definitely_not_a_task",
        "agents": ["agent_0"],
        "steps": [{"step": 0}, {"step": 1}],
    }
    order, reason = concurrent_step_order(trajectory)
    assert order == [0, 1]
    assert reason


@pytest.mark.parametrize("path", _sample(), ids=lambda p: p.parent.parent.name)
def test_the_order_is_a_permutation_that_loses_no_step(path):
    trajectory = json.loads(path.read_text())
    order, reason = concurrent_step_order(trajectory)
    assert reason is None, reason
    assert sorted(order) == list(range(len(trajectory["steps"])))


@pytest.mark.parametrize("path", _sample(), ids=lambda p: p.parent.parent.name)
def test_reordering_preserves_every_step_object_and_its_step_field(path):
    trajectory = json.loads(path.read_text())
    reordered, order, _ = reorder_for_concurrent_render(trajectory)
    original = trajectory["steps"]
    assert len(reordered["steps"]) == len(original)
    # Same objects, same `step` labels -- only the sequence differs, so image
    # paths and any downstream join on step identity survive intact.
    assert {s["step"] for s in reordered["steps"]} == {s["step"] for s in original}
    for position, index in enumerate(order):
        assert reordered["steps"][position] is original[index]


@pytest.mark.parametrize("path", _sample(), ids=lambda p: p.parent.parent.name)
def test_each_agents_own_steps_keep_their_relative_order(path):
    """Reordering interleaves the agents; it never reshuffles one agent's plan."""

    trajectory = json.loads(path.read_text())
    reordered, _, _ = reorder_for_concurrent_render(trajectory)
    for agent in trajectory.get("agents", []):
        agent_id = agent if isinstance(agent, str) else agent.get("agent_id")
        before = [s["step"] for s in trajectory["steps"] if s.get("agent") == agent_id]
        after = [s["step"] for s in reordered["steps"] if s.get("agent") == agent_id]
        assert before == after


def test_the_corpus_really_does_reorder():
    """If nothing moved, the whole flag would be pointless -- guard that."""

    moved = 0
    for path in _sample(24):
        trajectory = json.loads(path.read_text())
        order, _ = concurrent_step_order(trajectory)
        moved += any(position != index for position, index in enumerate(order))
    assert moved > 0
