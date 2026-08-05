from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training.bc_task_vlm import dataset


def _step(index: int, agent: str, tool: str, args: dict) -> dict:
    return {"step": index, "agent": agent, "tool": tool, "args": args}


def test_causal_single_cache_is_prefix_derived_and_agent_private(
    monkeypatch, tmp_path
):
    raw_steps = [
        _step(0, "agent_1", "get_image", {"views": ["top_view", "map"]}),
        _step(1, "agent_1", "communicate", {"message": "seen"}),
        _step(2, "agent_1", "get_image", {"views": ["robot0_agentview_center"]}),
        _step(3, "agent_0", "get_image", {"views": ["top_view", "map"]}),
        _step(4, "agent_0", "communicate", {"message": "seen"}),
        _step(5, "agent_0", "navigate_to_fixture", {"fixture_id": "counter"}),
        _step(6, "agent_0", "communicate", {"message": "after action"}),
    ]
    image_paths = {
        0: ["/generated/agent1_global.png"],
        2: ["/generated/agent1_wrist.png"],
        3: ["/generated/agent0_global.png"],
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
    loaded = {
        "original_trajectory.json": {
            "trajectory_id": "traj_causal",
            "steps": raw_steps,
        },
        "plan.json": plan_steps,
        "metadata.json": {
            "task": "test causal observations",
            "steps": [
                {"step_index": index, "success": True}
                for index in range(len(raw_steps))
            ],
        },
    }
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
            predict_agent=True,
            train_get_image=True,
            # by_step[7] below is the synthetic terminal task_complete example,
            # which only exists when task_complete is supervised. It used to
            # ride along with predict_agent; commit 31b7b71 split it out and the
            # default is now off, so this test has to ask for it explicitly.
            predict_task_complete=True,
            causal_single_cache=True,
        )

    by_step = {example.step_index: example for example in examples}
    assert by_step[0].agent_id == "agent_1"
    assert by_step[0].image_paths == []
    assert by_step[1].image_paths == image_paths[0]
    assert by_step[2].image_paths == image_paths[0]
    assert by_step[3].agent_id == "agent_0"
    assert by_step[3].image_paths == image_paths[2]
    assert by_step[4].image_paths == image_paths[3]
    assert by_step[5].image_paths == image_paths[3]
    assert by_step[6].image_paths == []
    assert by_step[7].image_paths == []
    prompt_text = by_step[3].messages[1]["content"][-1]["text"]
    assert "Active observation owner: agent_1" in prompt_text


def test_causal_single_cache_changes_example_cache_fingerprint(monkeypatch, tmp_path):
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
        predict_agent=True,
        train_get_image=True,
    )
    causal = dataset.build_example_cache_fingerprint(
        dataset_root=tmp_path,
        task_name="synthetic_task",
        predict_agent=True,
        train_get_image=True,
        causal_single_cache=True,
    )

    assert "causal_single_cache" not in legacy
    assert causal["causal_single_cache"] is True
    assert causal != legacy


@pytest.mark.parametrize(
    ("predict_agent", "train_get_image"),
    [(False, False), (True, False), (False, True)],
)
def test_causal_single_cache_requires_v3_flags(
    tmp_path, predict_agent, train_get_image
):
    with pytest.raises(ValueError, match="requires predict_agent=True"):
        dataset.build_centralized_examples(
            dataset_root=tmp_path,
            task_names=[],
            predict_agent=predict_agent,
            train_get_image=train_get_image,
            causal_single_cache=True,
        )


def test_legacy_task_complete_keeps_last_agent_observation():
    example = dataset._build_task_complete_example(
        task_metadata=SimpleNamespace(
            dataset_name="synthetic_task",
            composite_task="SyntheticTask",
        ),
        metadata={"task": "finish"},
        trajectory_id="traj_legacy",
        raw_steps=[
            _step(0, "agent_1", "communicate", {"message": "done"}),
        ],
        history_steps=[
            _step(0, "agent_1", "communicate", {"message": "done"}),
        ],
        latest_observations_by_agent={
            "agent_1": (["/generated/agent1.png"], ["top_view"]),
        },
        allowed_tool_specs={},
        sft_format=dataset.SFT_FORMAT_PLAIN,
        response_schema={},
        tool_schemas=[],
        causal_single_cache=False,
    )

    assert example.image_paths == ["/generated/agent1.png"]
