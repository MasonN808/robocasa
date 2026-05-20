from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training.scripts import stage_hf_sweep_datasets as stage


class FakeDataset:
    def __init__(
        self,
        rows: list[dict],
        *,
        column_names: list[str] | None = None,
        loaded_indices: list[int] | None = None,
    ) -> None:
        self.rows = rows
        self.column_names = column_names or list(rows[0])
        self.loaded_indices = loaded_indices if loaded_indices is not None else []

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index: int) -> dict:
        self.loaded_indices.append(index)
        return self.rows[index]

    def remove_columns(self, columns: list[str]):
        removed = set(columns)
        return FakeDataset(
            [
                {key: value for key, value in row.items() if key not in removed}
                for row in self.rows
            ],
            column_names=[
                column_name
                for column_name in self.column_names
                if column_name not in removed
            ],
            loaded_indices=self.loaded_indices,
        )


def _trajectory_row(episode_id: str) -> dict:
    original_trajectory = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "move",
                "args": {},
                "reasoning": "",
            }
        ]
    }
    return {
        "episode_id": episode_id,
        "task": "make a hot dog",
        "task_dir": "hot_dog_setup",
        "original_trajectory": json.dumps(original_trajectory),
        "execution_metadata": json.dumps({"task": "make a hot dog"}),
        "step_index": [0],
        "tool_name": ["move"],
        "tool_args": ["{}"],
        "robot_idx": [0],
        "success": [True],
        "room_view": ["image column should not be loaded for skipped rows"],
    }


def _write_stage_file(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_trajectory(
    output_root: Path,
    *,
    task_dir: str = "hot_dog_setup",
    trajectory_id: str = "traj_000000",
    plan_steps: list[dict] | None = None,
) -> Path:
    trajectory_dir = output_root / task_dir / trajectory_id
    trajectory_dir.mkdir(parents=True)
    _write_stage_file(trajectory_dir / "original_trajectory.json", {"steps": []})
    _write_stage_file(trajectory_dir / "adapted_trajectory.json", {})
    _write_stage_file(trajectory_dir / "plan.json", plan_steps or [])
    _write_stage_file(trajectory_dir / "metadata.json", {})
    _write_stage_file(trajectory_dir / "trajectory_execution_metadata.json", {})
    (trajectory_dir / "images" / trajectory_id).mkdir(parents=True)
    return trajectory_dir


class StageHfSweepResumeTests(unittest.TestCase):
    def test_existing_trajectory_validation_checks_image_paths(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = _write_complete_trajectory(
            temp_root,
            plan_steps=[
                {
                    "tool": "get_image",
                    "metadata": {
                        "image_paths": [
                            "images/traj_000000/0_room_view_agent_0.jpg",
                        ],
                    },
                }
            ],
        )
        image_path = (
            trajectory_dir / "images" / "traj_000000" / "0_room_view_agent_0.jpg"
        )
        image_path.write_bytes(b"not a real image but enough for path validation")

        self.assertTrue(
            stage._existing_trajectory_is_complete(trajectory_dir, validate=True)
        )
        image_path.unlink()
        self.assertFalse(
            stage._existing_trajectory_is_complete(trajectory_dir, validate=True)
        )
        self.assertTrue(
            stage._existing_trajectory_is_complete(trajectory_dir, validate=False)
        )

    def test_resume_skips_complete_existing_episode_and_stages_missing_tail(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        _write_complete_trajectory(temp_root)
        dataset = FakeDataset(
            [
                _trajectory_row("episode_000000"),
                _trajectory_row("episode_000001"),
            ]
        )
        repo_state = stage.RepoStageState(
            repo_id="DorianAtSchool/robocasa_20260430T030150Z_full_run_hot_dog_setup",
            split="train",
            revision=None,
            sidecar_root=None,
            dataset=dataset,
            row_granularity=stage._row_granularity_for_dataset(dataset),
            episodes=iter(stage._iter_episode_sources(dataset)),
        )

        with mock.patch.object(stage, "_load_repo_states", return_value=[repo_state]):
            staged_sources = stage.stage_repos(
                repo_ids=[repo_state.repo_id],
                output_root=temp_root,
                split="train",
                revision=None,
                trust_remote_code=False,
                counters_by_task=defaultdict(int),
                load_workers=1,
                workers=1,
                max_in_flight=1,
                progress_interval=0,
                resume_existing=True,
            )

        self.assertEqual(dataset.loaded_indices, [1])
        self.assertTrue((temp_root / "hot_dog_setup" / "traj_000001").is_dir())
        self.assertEqual(staged_sources[0]["num_episodes"], 2)
        self.assertEqual(staged_sources[0]["num_skipped_existing"], 1)
        self.assertEqual(staged_sources[0]["num_staged_new_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
