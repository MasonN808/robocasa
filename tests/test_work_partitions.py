"""Tests for weighted work-partition assignment."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.task_level.tasks.shared.partitions import (  # noqa: E402
    is_degenerate,
    partition_rules,
    select_partition,
)

PARTITIONS = [
    {"labels": "001", "weight": 0.475, "cross_agent_deps": 1,
     "assignment": {"agent_0": ["pick_up_object mug"], "agent_1": ["press_button"]}},
    {"labels": "110", "weight": 0.475, "cross_agent_deps": 1,
     "assignment": {"agent_0": ["press_button"], "agent_1": ["pick_up_object mug"]}},
    {"labels": "000", "weight": 0.025, "cross_agent_deps": 0,
     "assignment": {"agent_0": ["pick_up_object mug", "press_button"], "agent_1": []}},
    {"labels": "111", "weight": 0.025, "cross_agent_deps": 0,
     "assignment": {"agent_0": [], "agent_1": ["pick_up_object mug", "press_button"]}},
]


class SelectPartitionTests(unittest.TestCase):
    def test_selection_is_deterministic_in_run_index(self):
        first = [select_partition(PARTITIONS, i)["labels"] for i in range(50)]
        second = [select_partition(PARTITIONS, i)["labels"] for i in range(50)]
        self.assertEqual(first, second)

    def test_empirical_frequencies_track_the_weights(self):
        counts = Counter(select_partition(PARTITIONS, i)["labels"] for i in range(2000))
        for partition in PARTITIONS:
            share = counts[partition["labels"]] / 2000
            self.assertAlmostEqual(share, partition["weight"], delta=0.005)

    def test_separate_batches_are_each_proportional(self):
        # The dataset may be generated 50k at a time; each batch has to be
        # proportional on its own, with no cross-batch bookkeeping.
        head = Counter(select_partition(PARTITIONS, i)["labels"] for i in range(1000))
        tail = Counter(select_partition(PARTITIONS, i)["labels"] for i in range(1000, 2000))
        for partition in PARTITIONS:
            label = partition["labels"]
            self.assertAlmostEqual(head[label] / 1000, tail[label] / 1000, delta=0.01)

    def test_single_agent_partitions_stay_within_their_cap(self):
        counts = Counter(select_partition(PARTITIONS, i)["labels"] for i in range(2000))
        degenerate = counts["000"] + counts["111"]
        self.assertLessEqual(degenerate / 2000, 0.055)

    def test_absent_or_unweighted_partitions_select_nothing(self):
        self.assertIsNone(select_partition(None, 0))
        self.assertIsNone(select_partition([], 0))
        self.assertIsNone(select_partition([{"labels": "0", "weight": 0.0}], 0))

    def test_none_policy_disables_ownership_partition(self):
        self.assertIsNone(select_partition(PARTITIONS, 0, policy="none"))

    def test_balanced_local_rejects_degenerate_and_prefers_aligned_starts(self):
        initial_state = {
            "agents": {
                "agent_0": {"location": "coffee_machine"},
                "agent_1": {"location": "cab"},
            },
            "fixtures": {
                "cab": {"fixture_type": "cabinet"},
                "coffee_machine": {"fixture_type": "coffee_machine"},
            },
            "objects": {"mug": {"location": "cab"}},
        }
        work = [
            {"tool": "pick_up_object", "args": {"object_id": "mug", "source_id": "cab"}},
            {"tool": "place_under", "args": {"object_id": "mug", "reference_fixture_id": "coffee_machine"}},
            {"tool": "press_button", "args": {"target_id": "coffee_machine", "control_id": "start_button"}},
        ]
        selected = select_partition(
            PARTITIONS,
            0,
            policy="balanced_local",
            initial_state=initial_state,
            work_sequence=work,
        )
        self.assertEqual(selected["labels"], "110")


class PartitionRuleTests(unittest.TestCase):
    def test_rules_name_each_agent_and_its_work(self):
        rules = partition_rules(PARTITIONS[0], ("agent_0", "agent_1"))
        text = "\n".join(rules)
        self.assertIn("agent_0 performs: pick_up_object mug", text)
        self.assertIn("agent_1 performs: press_button", text)

    def test_handoff_is_called_out_only_when_the_split_needs_one(self):
        with_dep = "\n".join(partition_rules(PARTITIONS[0], ("agent_0", "agent_1")))
        without = "\n".join(partition_rules(PARTITIONS[2], ("agent_0", "agent_1")))
        self.assertIn("hand work to each other", with_dep)
        self.assertNotIn("hand work to each other", without)

    def test_idle_agent_is_given_something_to_do(self):
        rules = "\n".join(partition_rules(PARTITIONS[2], ("agent_0", "agent_1")))
        self.assertIn("agent_1 performs none of the task actions", rules)

    def test_no_partition_produces_no_rules(self):
        self.assertEqual(partition_rules(None, ("agent_0", "agent_1")), ())

    def test_bowl_loading_precedence_uses_sampled_owners(self):
        partition = {
            "assignment": {
                "agent_0": ["pick_up_object bowl"],
                "agent_1": [
                    "place_in_receptacle toaster_oven_bread -> bowl"
                ],
            }
        }
        text = "\n".join(partition_rules(partition, ("agent_0", "agent_1")))
        self.assertIn(
            "agent_1 places toaster_oven_bread into bowl while bowl rests on "
            "counter; only afterward may agent_0 pick up bowl",
            text,
        )


class DegenerateTests(unittest.TestCase):
    def test_split_partitions_are_not_degenerate(self):
        self.assertFalse(is_degenerate(PARTITIONS[0]))

    def test_single_agent_partitions_are_degenerate(self):
        self.assertTrue(is_degenerate(PARTITIONS[2]))
        self.assertTrue(is_degenerate(PARTITIONS[3]))

    def test_absent_partition_is_not_degenerate(self):
        self.assertFalse(is_degenerate(None))


class SpecIntegrationTests(unittest.TestCase):
    def test_every_weighted_spec_builds_instances_and_prompts(self):
        from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY

        weighted = 0
        for definition in SPEC_TASK_REGISTRY.values():
            instance = definition.build_task_instance(0)
            prompt = definition.build_prompt("v0", instance)
            if instance.work_partition is None:
                continue
            weighted += 1
            self.assertIn("Divide the work between the agents EXACTLY", prompt)
            # A deliberately single-agent run must not also be told that a
            # single agent should not do all the subtasks.
            if is_degenerate(instance.work_partition):
                self.assertNotIn("a single agent should not do all subtasks", prompt)
            else:
                self.assertIn("a single agent should not do all subtasks", prompt)
        self.assertGreater(weighted, 40)


if __name__ == "__main__":
    unittest.main()
