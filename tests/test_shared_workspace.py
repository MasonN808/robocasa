"""Workspace sharing is declared per fixture, derived from scene geometry.

Both spatial branches of _fixtures_share_workspace were inert: parent_fixture
was declared on 0 of 128 fixtures, and the cluster gate compares name-derived
ids that never match (0 of 49 candidate pairs). The generator prompt papered
over this by naming fixture TYPES ("cabinet/drawer and their supporting
counters"), which resolves only when a task has exactly one counter -- true for
26 tasks, ambiguous for 8, and silent about appliances entirely.

parent_fixture is now derived from measured geometry (derive_workspace_pairs.py)
and the prompt rule reads the field instead of enumerating types.
"""
import unittest

from data_generation.task_level.pipeline.phase1 import _fixtures_share_workspace
from data_generation.task_level.tasks.specs import load_all_task_specs, load_task_spec


class SharedWorkspace(unittest.TestCase):
    def test_appliance_shares_with_the_counter_it_stands_on(self):
        fx = load_task_spec("ArrangeBreadBowl").initial_state["fixtures"]
        # measured 0.30 m to `counter`; next counter is 4.20 m away
        self.assertTrue(
            _fixtures_share_workspace("counter", "toaster_oven", fixture_states=fx)
        )

    def test_appliance_does_not_share_with_a_distant_counter(self):
        fx = load_task_spec("ArrangeBreadBowl").initial_state["fixtures"]
        self.assertFalse(
            _fixtures_share_workspace(
                "dining_counter", "toaster_oven", fixture_states=fx
            )
        )

    def test_cabinet_above_counter_also_shares(self):
        fx = load_task_spec("SetupWineGlasses").initial_state["fixtures"]
        self.assertTrue(
            _fixtures_share_workspace("cabinet", "counter", fixture_states=fx)
        )

    def test_far_apart_fixtures_never_share(self):
        """A fridge metres from the counter must not be treated as blocking it."""
        fx = load_task_spec("PrepareCheeseStation").initial_state["fixtures"]
        # measured: fridge -> counter 3.63 m, well beyond the 1.0 m cutoff
        self.assertFalse(
            _fixtures_share_workspace("fridge", "counter", fixture_states=fx)
        )

    def test_the_check_is_no_longer_inert(self):
        pairs = 0
        for spec in load_all_task_specs():
            fx = spec.initial_state.get("fixtures") or {}
            for a in fx:
                for b in fx:
                    if a < b and _fixtures_share_workspace(a, b, fixture_states=fx):
                        pairs += 1
        self.assertGreaterEqual(pairs, 11, "workspace sharing regressed to inert")


if __name__ == "__main__":
    unittest.main()
