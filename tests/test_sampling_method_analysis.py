from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_analysis_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "data_analysis"
        / "analyze_sampling_methods.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_sampling_methods", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analysis = load_analysis_module()


class SamplingMethodAnalysisTests(unittest.TestCase):
    def write_trajectory(
        self,
        root: Path,
        *,
        method: str = "base",
        task: str = "prepare_coffee",
        traj_id: str = "traj_000000",
        valid: bool = True,
        first_message: str = "agent_1, grab the mug.",
    ) -> Path:
        path = root / method / task / "trajectories" / f"{traj_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trajectory_id": traj_id,
            "composite_task": "PrepareCoffee",
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "communicate",
                    "args": {"to": "agent_1", "message": first_message},
                    "reasoning": "I need to coordinate.",
                },
                {
                    "step": 1,
                    "agent": "agent_1",
                    "tool": "pick_up_object",
                    "args": {"object_id": "mug"},
                    "reasoning": "I am picking up the mug.",
                },
                {
                    "step": 2,
                    "agent": "agent_1",
                    "tool": "communicate",
                    "args": {"to": "agent_0", "message": "I have the mug."},
                    "reasoning": "I should update my teammate.",
                },
            ],
            "validation": {"is_valid": valid},
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_load_records_and_summarize_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root)
            self.write_trajectory(root, traj_id="traj_000001", valid=False)

            records, warnings = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
            )
            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sampling_method, "base")
            self.assertEqual(records[0].task, "prepare_coffee")
            self.assertEqual(records[0].environment, "prepare_coffee")
            self.assertIn(
                "agent_0 -> agent_1: agent_1, grab the mug.",
                records[0].conversation_text,
            )

            summary = analysis.summarize_records(
                records,
                sampling_method="base",
                task="prepare_coffee",
            )
            self.assertEqual(summary["num_trajectories"], 1)
            self.assertEqual(summary["num_steps"], 3)
            self.assertEqual(
                summary["tool_call_distribution"]["communicate"]["count"], 2
            )
            self.assertEqual(summary["conversation_messages"]["total"], 2.0)
            self.assertEqual(summary["conversation_length"]["texts"], 1)

    def test_build_trajectory_embedding_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root)
            records, _ = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
            )

            items = analysis.build_trajectory_embedding_items(
                records,
                instruction="Represent this robot trajectory conversation.",
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].sampling_method, "base")
            self.assertEqual(items[0].task, "prepare_coffee")
            self.assertTrue(items[0].text.startswith("Instruct:"))
            self.assertIn("I have the mug.", items[0].text)

    def test_max_trajectories_applies_after_invalid_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root, traj_id="traj_000000", valid=False)
            self.write_trajectory(root, traj_id="traj_000001", valid=True)
            self.write_trajectory(root, traj_id="traj_000002", valid=True)

            records, warnings = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=1,
            )

            self.assertEqual(warnings, [])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].trajectory_id, "traj_000001")

    def test_parallel_loading_matches_sequential_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root, task="prepare_coffee", traj_id="traj_000000")
            self.write_trajectory(root, task="arrange_tea", traj_id="traj_000001")

            sequential, _ = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
                load_workers=1,
            )
            parallel, _ = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
                load_workers=2,
            )

            self.assertEqual(
                [(record.task, record.trajectory_id) for record in parallel],
                [(record.task, record.trajectory_id) for record in sequential],
            )

    def test_exact_pairwise_similarity_summary(self):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        summary = analysis.exact_pairwise_similarity_summary(
            vectors,
            chunk_size=2,
        )
        self.assertTrue(summary["exact"])
        self.assertEqual(summary["num_items"], 3)
        self.assertEqual(summary["num_pairs"], 3)
        self.assertEqual(summary["computed_pairs"], 3)
        self.assertAlmostEqual(summary["min"], 0.0)
        self.assertAlmostEqual(summary["max"], 1.0)
        self.assertAlmostEqual(summary["avg"], 1.0 / 3.0)

    def test_compute_similarity_maps_by_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root, traj_id="traj_000000")
            self.write_trajectory(
                root,
                traj_id="traj_000001",
                first_message="agent_1, wait by the counter.",
            )
            records, _ = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
            )
            items = analysis.build_trajectory_embedding_items(records, instruction="")
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

            maps = analysis.compute_similarity_maps(
                items,
                embeddings,
                chunk_size=2,
            )
            summary = maps["by_task"][("base", "prepare_coffee")]
            self.assertEqual(summary["num_items"], 2)
            self.assertEqual(summary["num_pairs"], 1)
            self.assertAlmostEqual(summary["avg"], 0.0)

    def test_embedding_cache_round_trip_and_metadata_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_trajectory(root)
            records, _ = analysis.load_records(
                root,
                include_invalid=False,
                max_trajectories_per_task=None,
            )
            items = analysis.build_trajectory_embedding_items(records, instruction="")
            metadata = analysis.build_embedding_cache_metadata(
                items,
                provider="vllm",
                model_name="Qwen/Qwen3-Embedding-4B",
                instruction="",
                dtype="float16",
                max_model_len=None,
            )
            embeddings = np.asarray([[0.5, 0.5]], dtype=np.float32)

            analysis.write_cached_embeddings(root, metadata, embeddings)
            cached = analysis.load_cached_embeddings(root, metadata)
            self.assertIsNotNone(cached)
            np.testing.assert_allclose(cached, embeddings)

            mismatched_metadata = dict(metadata)
            mismatched_metadata["model"] = "other-model"
            self.assertIsNone(
                analysis.load_cached_embeddings(root, mismatched_metadata)
            )

    def test_main_reuses_matching_embedding_cache_without_vllm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_root = temp_path / "input"
            output_dir = temp_path / "output"
            self.write_trajectory(input_root, traj_id="traj_000000")
            self.write_trajectory(
                input_root,
                traj_id="traj_000001",
                first_message="agent_1, wait by the counter.",
            )

            records, _ = analysis.load_records(
                input_root,
                include_invalid=False,
                max_trajectories_per_task=30,
                load_workers=1,
            )
            items = analysis.build_trajectory_embedding_items(records, instruction="")
            metadata = analysis.build_embedding_cache_metadata(
                items,
                provider="vllm",
                model_name="Qwen/Qwen3-Embedding-4B",
                instruction="",
                dtype=None,
                max_model_len=None,
            )
            analysis.write_cached_embeddings(
                output_dir,
                metadata,
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )

            with contextlib.redirect_stderr(io.StringIO()):
                exit_code = analysis.main(
                    [
                        "--input-root",
                        str(input_root),
                        "--output-dir",
                        str(output_dir),
                        "--embedding-provider",
                        "vllm",
                        "--embedding-model",
                        "Qwen/Qwen3-Embedding-4B",
                        "--load-workers",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            with (output_dir / "sampling_method_analysis.json").open(
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            similarity = payload["by_task"][0]["trajectory_embedding_similarity"]
            self.assertEqual(similarity["num_pairs"], 1)
            self.assertAlmostEqual(similarity["avg"], 0.0)


if __name__ == "__main__":
    unittest.main()
