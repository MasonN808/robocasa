from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training.bc_task_vlm import dataset


def _step(index: int, agent: str, tool: str, args: dict) -> dict:
    return {"step": index, "agent": agent, "tool": tool, "args": args}


def _build_loaded_trajectory(raw_steps: list[dict]) -> dict:
    image_paths = {
        step["step"]: [f"/generated/{step['agent']}_{step['step']}.png"]
        for step in raw_steps
        if step["tool"] == "get_image"
    }
    plan_steps = []
    for raw_step in raw_steps:
        args = dict(raw_step["args"])
        if raw_step["tool"] == "get_image":
            args["image_paths"] = image_paths[raw_step["step"]]
        plan_steps.append(
            {
                "tool": raw_step["tool"],
                "args": args,
                "metadata": {
                    "step_index": raw_step["step"],
                    "source_agent": raw_step["agent"],
                },
            }
        )
    return {
        "original_trajectory.json": {
            "trajectory_id": "traj_partial",
            "steps": raw_steps,
        },
        "plan.json": plan_steps,
        "metadata.json": {
            "task": "test partial observability",
            "steps": [
                {"step_index": index, "success": True}
                for index in range(len(raw_steps))
            ],
        },
    }


def _build_examples(monkeypatch, tmp_path, raw_steps, **build_kwargs):
    loaded = _build_loaded_trajectory(raw_steps)
    monkeypatch.setenv("ROBOCASA_VALIDATE_IMAGE_PATHS", "false")
    with (
        patch.object(
            dataset,
            "get_task_metadata",
            return_value=SimpleNamespace(
                dataset_name="synthetic_task",
                composite_task="SyntheticTask",
                allowed_tool_specs={},
            ),
        ),
        patch.object(
            dataset,
            "_load_json",
            side_effect=lambda path: loaded[path.name],
        ),
    ):
        examples = dataset._build_centralized_examples_for_trajectory(
            task_name="synthetic_task",
            trajectory_dir=tmp_path,
            sft_format=dataset.SFT_FORMAT_PLAIN,
            response_schema={},
            tool_schemas=[],
            partial_history=True,
            **build_kwargs,
        )
    return {example.step_index: example for example in examples}


_RAW_STEPS = [
    _step(0, "agent_0", "get_image", {"views": ["top_view", "map"]}),
    _step(1, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    _step(2, "agent_0", "communicate", {"to": "agent_1", "message": "doing X"}),
    _step(3, "agent_1", "get_image", {"views": ["robot0_agentview_center"]}),
    _step(4, "agent_1", "pick_up_object", {"object_id": "cheese"}),
    _step(5, "agent_1", "communicate", {"to": "agent_0", "message": "done Y"}),
    _step(6, "agent_0", "navigate_to_fixture", {"fixture_id": "sink"}),
]


def test_partial_history_is_agent_private(monkeypatch, tmp_path):
    # Pinned to "global" numbering because this test is about WHICH steps an
    # agent sees, not how they are numbered. The builder's default moved from
    # "global" to "local" (commit 8881b4b) and the renumbering silently broke
    # the assertion below, which looks like a privacy regression and is not.
    by_step = _build_examples(
        monkeypatch, tmp_path, _RAW_STEPS, partial_step_index_mode="global"
    )

    # agent_1's example never contains agent_0's private navigate_to_fixture.
    agent1_history_tools = [s["tool"] for s in by_step[4].history_steps]
    assert "navigate_to_fixture" not in agent1_history_tools
    assert by_step[4].history_steps == [
        {"step": 2, "agent": "agent_0", "tool": "communicate", "args": {"to": "agent_1", "message": "doing X"}}
    ]


def test_partial_history_is_agent_private_under_the_default_numbering(
    monkeypatch, tmp_path
):
    """The privacy property must not depend on partial_step_index_mode."""

    by_step = _build_examples(monkeypatch, tmp_path, _RAW_STEPS)

    assert "navigate_to_fixture" not in [s["tool"] for s in by_step[4].history_steps]
    history = by_step[4].history_steps
    assert [(s["agent"], s["tool"]) for s in history] == [("agent_0", "communicate")]
    # "local" renumbers by position in this agent's own history, so the joint
    # index 2 is gone -- that is the point of the mode.
    assert [s["step"] for s in history] == [0]

    # agent_0's later example sees its own prior actions plus the delivered
    # message from agent_1, but never agent_1's private pick_up_object.
    agent0_history_tools = [s["tool"] for s in by_step[6].history_steps]
    assert agent0_history_tools == ["navigate_to_fixture", "communicate", "communicate"]
    assert "pick_up_object" not in agent0_history_tools


def test_partial_history_message_delivery_timing(monkeypatch, tmp_path):
    # "global" keeps the joint index, which is what identifies the step being
    # asserted about here. See the note in test_partial_history_is_agent_private.
    by_step = _build_examples(
        monkeypatch, tmp_path, _RAW_STEPS, partial_step_index_mode="global"
    )

    # The step-2 message must not appear in agent_1's history before it was
    # sent, only from the first agent_1 example emitted after step 2.
    assert by_step[4].history_steps[-1]["step"] == 2


def test_partial_history_prefix_invariance(monkeypatch, tmp_path):
    mutated_steps = list(_RAW_STEPS[:5]) + [
        _step(5, "agent_1", "communicate", {"to": "agent_0", "message": "COMPLETELY DIFFERENT"}),
        _step(6, "agent_0", "place_in_receptacle", {"object_id": "cheese", "receptacle_id": "plate"}),
    ]

    original = _build_examples(monkeypatch, tmp_path, _RAW_STEPS)
    mutated = _build_examples(monkeypatch, tmp_path, mutated_steps)

    for step_index in (1, 2, 4):
        assert original[step_index].history_steps == mutated[step_index].history_steps
        assert original[step_index].image_paths == mutated[step_index].image_paths


@pytest.mark.parametrize(
    ("predict_agent", "train_get_image"),
    # (False, True) is NOT here: that combination is partial-observability v3
    # (get_image supervised, caller identity still fixed), which is legal.
    [(True, False), (True, True)],
)
def test_partial_history_requires_no_agent_prediction(
    tmp_path, predict_agent, train_get_image
):
    with pytest.raises(ValueError, match="requires predict_agent=False"):
        dataset.build_centralized_examples(
            dataset_root=tmp_path,
            task_names=[],
            predict_agent=predict_agent,
            train_get_image=train_get_image,
            partial_history=True,
        )


def test_partial_history_changes_example_cache_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dataset,
        "get_task_metadata",
        lambda _name: SimpleNamespace(
            dataset_name="synthetic_task",
            composite_task="SyntheticTask",
            allowed_tool_specs={},
        ),
    )
    monkeypatch.setattr(dataset, "_trajectory_dirs_for_task", lambda **_kwargs: [])

    legacy = dataset.build_example_cache_fingerprint(
        dataset_root=tmp_path,
        task_name="synthetic_task",
    )
    partial = dataset.build_example_cache_fingerprint(
        dataset_root=tmp_path,
        task_name="synthetic_task",
        partial_history=True,
    )

    assert "partial_history" not in legacy
    assert partial["partial_history"] is True
    assert partial != legacy


_V3_RAW_STEPS = [
    _step(0, "agent_0", "get_image", {"views": ["top_view", "room_view", "map"]}),
    _step(1, "agent_1", "get_image", {"views": ["top_view", "room_view", "map"]}),
    _step(2, "agent_0", "get_image", {"views": ["agentview_center"]}),
    _step(3, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    _step(4, "agent_1", "get_image", {"views": ["wrist"]}),
    _step(5, "agent_1", "pick_up_object", {"object_id": "cheese"}),
]


def _build_v3(monkeypatch, tmp_path):
    return _build_examples(
        monkeypatch, tmp_path, _V3_RAW_STEPS, train_get_image=True
    )


def test_partial_v3_keeps_true_requester_on_global_views(monkeypatch, tmp_path):
    """Global-view requests must NOT be relabelled to agent_0: requester
    identity is part of the partial-observability state."""
    by_step = _build_v3(monkeypatch, tmp_path)

    assert by_step[0].agent_id == "agent_0"
    assert by_step[1].agent_id == "agent_1"


def test_partial_v3_get_image_target_keeps_own_prior_cache(monkeypatch, tmp_path):
    by_step = _build_v3(monkeypatch, tmp_path)

    # agent_0's first request has no prior cache; its second does, and that
    # cache is the one agent_0 itself acquired at step 0.
    assert by_step[0].image_paths == []
    assert by_step[2].image_paths == ["/generated/agent_0_0.png"]
    # agent_1's later request sees only its OWN step-1 cache, never agent_0's.
    assert by_step[4].image_paths == ["/generated/agent_1_1.png"]


def test_partial_v3_history_stays_agent_private(monkeypatch, tmp_path):
    by_step = _build_v3(monkeypatch, tmp_path)

    agents_in_history = {h["agent"] for h in by_step[5].history_steps}
    assert agents_in_history <= {"agent_1"}


def test_partial_history_still_rejects_agent_prediction(tmp_path):
    with pytest.raises(ValueError, match="requires predict_agent=False"):
        dataset.build_centralized_examples(
            dataset_root=tmp_path,
            task_names=[],
            predict_agent=True,
            partial_history=True,
        )


@pytest.mark.parametrize(
    ("mode", "expect_physical", "expect_communicate", "expect_get_image"),
    [
        # cache: persistent per-agent observation, everyone sees pixels
        ("cache", True, True, True),
        # consume_once: a get_image result feeds only that agent's NEXT call
        ("consume_once", True, True, False),
    ],
)
def test_partial_observation_modes_control_image_attachment(
    monkeypatch, tmp_path, mode, expect_physical, expect_communicate, expect_get_image
):
    raw_steps = [
        _step(0, "agent_0", "get_image", {"views": ["agentview_center"]}),
        _step(1, "agent_0", "communicate", {"to": "agent_1", "message": "hi"}),
        _step(2, "agent_0", "get_image", {"views": ["wrist"]}),
        _step(3, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
    ]
    by_step = _build_examples(
        monkeypatch,
        tmp_path,
        raw_steps,
        train_get_image=True,
        partial_observation_mode=mode,
    )

    # step 3 is the physical action; step 1 communicate; step 2 a get_image
    # that itself follows a get_image (so "consume_once" would still feed it).
    assert bool(by_step[3].image_paths) is expect_physical
    assert bool(by_step[1].image_paths) is expect_communicate
    # step 2 follows a *communicate*, which already consumed the pending image
    # under consume_once, so this get_image target sees nothing.
    assert bool(by_step[2].image_paths) is expect_get_image


def test_partial_observation_mode_rejects_unknown_value(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="partial_observation_mode must be one of"):
        _build_examples(
            monkeypatch, tmp_path, _RAW_STEPS, partial_observation_mode="bogus"
        )
