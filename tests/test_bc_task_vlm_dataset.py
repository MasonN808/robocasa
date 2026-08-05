from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

from training.bc_task_vlm import dataset as dataset_module
from training.bc_task_vlm.dataset import (
    build_example_cache_fingerprint,
    build_example_cache_path,
    build_centralized_examples,
    estimate_centralized_example_length,
    list_available_task_names,
    list_task_trajectory_ids,
    load_examples_from_cache,
    save_examples_to_cache,
)


# A real rendered trajectory, vendored into the repo. These tests used to copy
# from a generated dataset directory (data/image/<timestamp>/), which was deleted
# when that vintage was superseded -- every test here then failed on a missing
# file rather than on anything about the code. The fixture is checked in so the
# suite does not depend on which corpus happens to be on disk.
#
# It is a --step-order concurrent render, so plan.json is a genuine permutation
# of the trajectory's steps and these tests exercise the reordered path.
FIXTURE_ROOT = Path(__file__).resolve().parent / "data" / "render_fixture"
FIXTURE_TASK = "add_lemon_to_fish"
FIXTURE_TRAJECTORY_ID = "traj_000000"
SOURCE_TRAJECTORY_DIR = FIXTURE_ROOT / FIXTURE_TASK / FIXTURE_TRAJECTORY_ID


class CentralizedExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        # The fixture's image paths point at the render that produced it, which
        # is not part of the repo. These tests are about example construction,
        # not about pixels being on disk.
        patcher = unittest.mock.patch.dict(
            os.environ, {"ROBOCASA_VALIDATE_IMAGE_PATHS": "false"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build_single_trajectory_dataset_root(self) -> Path:
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = temp_root / FIXTURE_TASK / FIXTURE_TRAJECTORY_ID
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("original_trajectory.json", "plan.json", "metadata.json"):
            shutil.copy2(SOURCE_TRAJECTORY_DIR / file_name, trajectory_dir / file_name)
        return temp_root

    def test_joint_demo_builds_one_centralized_example_per_action(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=[FIXTURE_TASK],
        )

        self.assertTrue(examples)
        self.assertTrue(
            all(example.trajectory_id == FIXTURE_TRAJECTORY_ID for example in examples)
        )
        self.assertEqual(
            {example.agent_id for example in examples},
            {"agent_0", "agent_1"},
        )

        for example in examples:
            self.assertNotIn("reasoning", example.target_payload["steps"][0])
            self.assertNotIn(
                "reasoning",
                example.to_feature_dict()["target_payload"]["steps"][0],
            )
            self.assertNotIn("reasoning", example.to_manifest_entry())
            self.assertNotIn("reasoning", json.dumps(example.history_steps))
            self.assertNotIn("reasoning", json.dumps(example.response_schema))
            roles = [message["role"] for message in example.messages]
            self.assertEqual(roles, ["system", "user", "assistant"])
            num_image_placeholders = sum(
                1
                for message in example.messages
                for item in message.get("content", [])
                if isinstance(item, dict) and item.get("type") == "image"
            )
            self.assertEqual(num_image_placeholders, len(example.image_paths))

    def test_length_estimate_accounts_for_images_and_respects_image_cap(self):
        example = SimpleNamespace(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "move"}]}
            ],
            tool_schemas=[{"type": "function", "function": {"name": "move"}}],
            image_paths=["one.png", "two.png", "three.png"],
        )

        uncapped = estimate_centralized_example_length(example)
        capped = estimate_centralized_example_length(
            example,
            max_images_per_sample=1,
        )

        self.assertGreater(uncapped, capped)
        self.assertEqual(
            uncapped - capped,
            256 * (len(example.image_paths) - 1),
        )

    def test_plain_sft_centralized_examples_use_text_targets(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=[FIXTURE_TASK],
            sft_format="plain",
        )

        self.assertTrue(examples)
        example = examples[0]
        assistant_message = example.messages[-1]
        target = json.loads(example.target_text)

        self.assertEqual(example.response_schema, {})
        self.assertEqual(set(target), {"tool", "args"})
        self.assertNotIn("reasoning", target)
        self.assertNotIn("reasoning", example.target_payload["steps"][0])
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["content"], example.target_text)
        self.assertNotIn("tool_calls", assistant_message)
        self.assertNotIn("reasoning_content", assistant_message)

    def test_manifest_only_stage_root_resolves_shard_trajectories(self):
        shard_root = self._build_single_trajectory_dataset_root()
        manifest_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (manifest_root / "hf_stage_manifest.json").write_text(
            json.dumps(
                {
                    "output_root": str(manifest_root),
                    "sources": [
                        {
                            "repo_id": "example/fixture",
                            "tasks": [FIXTURE_TASK],
                            "num_episodes": 1,
                            "shard_output_root": str(shard_root),
                        }
                    ],
                    "total_episodes": 1,
                }
            ),
            encoding="utf-8",
        )

        examples = build_centralized_examples(
            dataset_root=manifest_root,
            task_names=[FIXTURE_TASK],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=manifest_root,
            task_name=FIXTURE_TASK,
        )

        self.assertEqual(list_available_task_names(manifest_root), [FIXTURE_TASK])
        self.assertEqual(
            list_task_trajectory_ids(
                dataset_root=manifest_root,
                task_name=FIXTURE_TASK,
            ),
            [FIXTURE_TRAJECTORY_ID],
        )
        self.assertTrue(examples)
        self.assertTrue(
            all(example.trajectory_id == FIXTURE_TRAJECTORY_ID for example in examples)
        )
        self.assertEqual(
            [entry["trajectory_id"] for entry in fingerprint["trajectories"]],
            [FIXTURE_TRAJECTORY_ID],
        )

    def test_centralized_examples_round_trip_through_cache(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        cache_dir = dataset_root / ".cache"
        centralized_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=[FIXTURE_TASK],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )
        cache_path = build_example_cache_path(
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )

        save_examples_to_cache(
            cache_path=cache_path,
            fingerprint=fingerprint,
            examples=centralized_examples,
        )
        compressed_cache_path = dataset_module._preferred_compressed_example_cache_path(
            cache_path
        )

        self.assertTrue(compressed_cache_path.is_file())
        self.assertFalse(cache_path.exists())

        cached_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
        )

        self.assertEqual(cached_examples, centralized_examples)

    def test_legacy_json_cache_load_migrates_to_compressed_cache(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        cache_dir = dataset_root / ".cache"
        centralized_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=[FIXTURE_TASK],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )
        legacy_fingerprint = json.loads(json.dumps(fingerprint))
        legacy_fingerprint["builder_dependencies"]["dataset.py"] = {
            "mtime_ns": 1,
            "size": 1,
        }
        cache_path = build_example_cache_path(
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_format_version": dataset_module._EXAMPLE_CACHE_FORMAT_VERSION,
                    "fingerprint": legacy_fingerprint,
                    "examples": [example.__dict__ for example in centralized_examples],
                }
            ),
            encoding="utf-8",
        )
        compressed_cache_path = dataset_module._preferred_compressed_example_cache_path(
            cache_path
        )

        self.assertFalse(compressed_cache_path.exists())

        cached_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
        )

        self.assertEqual(cached_examples, centralized_examples)
        self.assertTrue(compressed_cache_path.is_file())

        cache_path.unlink()
        cached_compressed_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
        )

        self.assertEqual(cached_compressed_examples, centralized_examples)

    def test_cache_fingerprint_includes_sft_format(self):
        dataset_root = self._build_single_trajectory_dataset_root()

        tool_call_fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
            sft_format="tool_call",
        )
        plain_fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
            sft_format="plain",
        )

        self.assertNotEqual(tool_call_fingerprint, plain_fingerprint)
        self.assertEqual(tool_call_fingerprint["sft_format"], "tool_call")
        self.assertEqual(plain_fingerprint["sft_format"], "plain")

    def test_cache_invalidates_when_trajectory_inputs_change(self):
        dataset_root = self._build_single_trajectory_dataset_root()
        cache_dir = dataset_root / ".cache"
        centralized_examples = build_centralized_examples(
            dataset_root=dataset_root,
            task_names=[FIXTURE_TASK],
        )
        fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )
        cache_path = build_example_cache_path(
            cache_dir=cache_dir,
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )
        save_examples_to_cache(
            cache_path=cache_path,
            fingerprint=fingerprint,
            examples=centralized_examples,
        )

        metadata_path = dataset_root / FIXTURE_TASK / FIXTURE_TRAJECTORY_ID / "metadata.json"
        metadata_path.write_text(
            metadata_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        updated_fingerprint = build_example_cache_fingerprint(
            dataset_root=dataset_root,
            task_name=FIXTURE_TASK,
        )

        cached_examples = load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=updated_fingerprint,
        )

        self.assertIsNone(cached_examples)


if __name__ == "__main__":
    unittest.main()
