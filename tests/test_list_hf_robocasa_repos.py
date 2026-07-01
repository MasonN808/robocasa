from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from training.scripts import list_hf_robocasa_repos as list_repos


@dataclass(frozen=True)
class FakeDatasetInfo:
    id: str


class FakeHfApi:
    def __init__(self, repo_ids: list[str]) -> None:
        self.repo_ids = repo_ids

    def list_datasets(self, *, author: str):
        self.author = author
        return [FakeDatasetInfo(repo_id) for repo_id in self.repo_ids]


class ListHfRoboCasaReposTests(unittest.TestCase):
    def test_discover_repo_ids_filters_and_sorts_matching_datasets(self):
        api = FakeHfApi(
            [
                "DorianAtSchool/unrelated",
                "Other/robocasa_20260430T030150Z_full_run_other",
                "DorianAtSchool/robocasa_20260430T030150Z_full_run_z_task",
                "DorianAtSchool/robocasa_20260430T030150Z_full_run_a_task",
            ]
        )

        repo_ids = list_repos.discover_repo_ids(
            author="DorianAtSchool",
            prefix="robocasa_20260430T030150Z_full_",
            api=api,
        )

        self.assertEqual(api.author, "DorianAtSchool")
        self.assertEqual(
            repo_ids,
            [
                "DorianAtSchool/robocasa_20260430T030150Z_full_run_a_task",
                "DorianAtSchool/robocasa_20260430T030150Z_full_run_z_task",
            ],
        )

    def test_write_repo_outputs_writes_lines_and_env_file(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        output_path = temp_root / "repos.txt"
        env_output_path = temp_root / "repos.env"

        list_repos.write_repo_outputs(
            ["DorianAtSchool/repo_a", "DorianAtSchool/repo_b"],
            output_path=output_path,
            env_output_path=env_output_path,
        )

        self.assertEqual(
            output_path.read_text(encoding="utf-8"),
            "DorianAtSchool/repo_a\nDorianAtSchool/repo_b\n",
        )
        self.assertEqual(
            env_output_path.read_text(encoding="utf-8"),
            "HF_DATASET_REPOS=DorianAtSchool/repo_a,DorianAtSchool/repo_b\n",
        )


if __name__ == "__main__":
    unittest.main()
