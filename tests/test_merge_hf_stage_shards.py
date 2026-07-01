from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from training.scripts import merge_hf_stage_shards as merge


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trajectory(root: Path, *, task: str, trajectory_id: str) -> None:
    trajectory_dir = root / task / trajectory_id
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "plan.json").write_text("[]", encoding="utf-8")


class MergeHfStageShardsTests(unittest.TestCase):
    def test_merge_shards_links_repos_in_repo_list_order_with_cap(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shard_root = temp_root / "shards"
        output_root = temp_root / "merged"
        first_shard = shard_root / "shard_000"
        second_shard = shard_root / "shard_001"

        for index in range(3):
            _write_trajectory(
                first_shard,
                task="hot_dog_setup",
                trajectory_id=f"shard_000_traj_{index:06d}",
            )
        for index in range(2):
            _write_trajectory(
                second_shard,
                task="spicy_marinade",
                trajectory_id=f"shard_001_traj_{index:06d}",
            )

        first_repo = "DorianAtSchool/robocasa_20260430T030150Z_full_hot_dog_setup"
        second_repo = "DorianAtSchool/robocasa_20260430T030150Z_full_spicy_marinade"
        _write_manifest(
            first_shard / "hf_stage_manifest.json",
            {
                "output_root": str(first_shard),
                "sources": [
                    {
                        "repo_id": first_repo,
                        "num_episodes": 3,
                        "tasks": ["hot_dog_setup"],
                    }
                ],
            },
        )
        _write_manifest(
            second_shard / "hf_stage_manifest.json",
            {
                "output_root": str(second_shard),
                "sources": [
                    {
                        "repo_id": second_repo,
                        "num_episodes": 2,
                        "tasks": ["spicy_marinade"],
                    }
                ],
            },
        )

        manifest = merge.merge_shards(
            shard_root=shard_root,
            output_root=output_root,
            repo_ids=[first_repo, second_repo],
            max_total_episodes=4,
            force_symlinks=False,
        )

        self.assertEqual(manifest["total_episodes"], 4)
        self.assertEqual(
            manifest["task_episode_counts"],
            {"hot_dog_setup": 3, "spicy_marinade": 1},
        )
        self.assertTrue(
            (output_root / "hot_dog_setup" / "shard_000_traj_000000").is_symlink()
        )
        self.assertTrue(
            (output_root / "spicy_marinade" / "shard_001_traj_000000").is_symlink()
        )
        self.assertFalse(
            (output_root / "spicy_marinade" / "shard_001_traj_000001").exists()
        )
        self.assertTrue((output_root / "hf_stage_manifest.json").is_file())

    def test_read_repo_list_ignores_comments_and_blank_lines(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        repo_list = temp_root / "repos.txt"
        repo_list.write_text(
            "\n# comment\nDorianAtSchool/repo_a\nDorianAtSchool/repo_b # trailing\n",
            encoding="utf-8",
        )

        self.assertEqual(
            merge.read_repo_list(repo_list),
            ["DorianAtSchool/repo_a", "DorianAtSchool/repo_b"],
        )


if __name__ == "__main__":
    unittest.main()
