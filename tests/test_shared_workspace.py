"""Countertop appliances share a workspace with the counter they stand on.

Without this, no give_space is injected when one agent occupies the counter and
another must reach the appliance, and the simulator silently teleports the
blocker instead -- a move no policy chose, which hides the coordination the
task exists to require.
"""
import unittest

from data_generation.task_level.pipeline.phase1 import _fixtures_share_workspace
from data_generation.task_level.tasks.specs import load_all_task_specs, load_task_spec

APPLIANCES = {"toaster_oven", "toaster", "coffee_machine",
              "blender", "stand_mixer", "electric_kettle"}


class SharedWorkspace(unittest.TestCase):
    def test_counter_shares_with_appliance_on_it(self):
        fx = load_task_spec("ArrangeBreadBowl").initial_state["fixtures"]
        self.assertTrue(_fixtures_share_workspace("counter", "toaster_oven",
                                                  fixture_states=fx))

    def test_two_separate_counters_do_not_share(self):
        fx = load_task_spec("ArrangeBreadBowl").initial_state["fixtures"]
        self.assertFalse(_fixtures_share_workspace("counter", "dining_counter",
                                                   fixture_states=fx))

    def test_blast_radius_is_only_appliance_tasks(self):
        """The clause must not make unrelated fixture pairs share a workspace."""
        affected = set()
        for spec in load_all_task_specs():
            fx = spec.initial_state.get("fixtures") or {}
            for a in fx:
                for b in fx:
                    if a >= b:
                        continue
                    if not _fixtures_share_workspace(a, b, fixture_states=fx):
                        continue
                    types = {
                        str((fx[a] or {}).get("fixture_type", "")).lower(),
                        str((fx[b] or {}).get("fixture_type", "")).lower(),
                    }
                    if types & APPLIANCES:
                        affected.add(spec.composite_task)
        self.assertEqual(
            affected,
            {"ArrangeBreadBowl", "SweetenCoffee", "PrepareSandwichStation"},
            "appliance clause changed which tasks are affected",
        )


if __name__ == "__main__":
    unittest.main()
