from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from training.bc_task_vlm import dataset


def _step(index: int, agent: str, tool: str, args: dict) -> dict:
    return {"step": index, "agent": agent, "tool": tool, "args": args}



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
