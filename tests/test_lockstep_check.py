"""Tests for the lock-step simultaneity check."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lockstep_check import collisions, schedule  # noqa: E402

FIXTURES = {
    "cab": {"fixture_type": "cabinet"},
    "counter": {"fixture_type": "counter"},
    "toaster": {"fixture_type": "toaster_oven", "parent_fixture": "counter"},
}
OBJECTS = {"bowl", "mug"}


def step(agent, tool, **args):
    return {"agent": agent, "tool": tool, "args": args, "reasoning": "r"}


class ScheduleTests(unittest.TestCase):
    def test_both_agents_advance_together(self):
        steps = [
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="counter"),
        ]
        ticks, deadlocked = schedule(steps)
        self.assertEqual(ticks, [0, 0])
        self.assertFalse(deadlocked)

    def test_each_agent_spends_one_tick_per_call(self):
        steps = [
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "pick_up_object", object_id="bowl", source_id="cab"),
        ]
        ticks, _ = schedule(steps)
        self.assertEqual(ticks, [0, 1])

    def test_a_wait_stalls_until_its_release(self):
        steps = [
            step("agent_0", "wait_for_signal", **{"from": "agent_1", "about": "bowl"}),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "communicate", to="agent_0", message="yours", releases="bowl"),
        ]
        ticks, deadlocked = schedule(steps)
        self.assertFalse(deadlocked)
        # agent_1 acts at ticks 0 and 1; the release lands at 1, so the waiter
        # cannot clear before tick 2.
        self.assertEqual(ticks[1], 0)
        self.assertEqual(ticks[2], 1)
        self.assertEqual(ticks[0], 2)

    def test_a_wait_that_is_never_released_deadlocks(self):
        steps = [
            step("agent_0", "wait_for_signal", **{"from": "agent_1", "about": "bowl"}),
            step("agent_1", "communicate", to="agent_0", message="unrelated"),
        ]
        ticks, deadlocked = schedule(steps)
        self.assertTrue(deadlocked)
        self.assertIsNone(ticks[0])


class CollisionTests(unittest.TestCase):
    def _collide(self, steps):
        return collisions(steps, FIXTURES, OBJECTS)[0]

    def test_two_agents_entering_one_cabinet_together_collide(self):
        found = self._collide([
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][3], ["cab"])

    def test_different_fixtures_do_not_collide(self):
        self.assertEqual(self._collide([
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="counter"),
        ]), [])

    def test_a_shared_object_collides(self):
        found = self._collide([
            step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            step("agent_1", "place_on_object", object_id="mug", support_object_id="bowl"),
        ])
        self.assertTrue(found)
        self.assertIn("bowl", found[0][3])

    def test_a_child_fixture_collides_with_its_parents_floor_space(self):
        # A toaster oven resting on a counter shares that counter's floor.
        found = self._collide([
            step("agent_0", "navigate_to_fixture", fixture_id="toaster"),
            step("agent_1", "pick_up_object", object_id="mug", source_id="toaster"),
        ])
        self.assertTrue(found)

    def test_give_space_is_a_departure_not_a_claim(self):
        # One agent stepping off as the other steps on is the handoff working.
        self.assertEqual(self._collide([
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ]), [])

    def test_communication_never_collides(self):
        self.assertEqual(self._collide([
            step("agent_0", "communicate", to="agent_1", message="hello"),
            step("agent_1", "communicate", to="agent_0", message="hi"),
        ]), [])

    def test_a_wait_separates_two_arrivals(self):
        # The same two arrivals, ordered by a wait, land on different ticks.
        self.assertEqual(self._collide([
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "wait_for_signal", **{"from": "agent_0", "about": "cab"}),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="clear", releases="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ]), [])

    def test_positional_ordering_alone_does_not_separate_arrivals(self):
        # candle_cleanup/traj_000025 in miniature: the plan alternates
        # correctly in written order, but nothing forces the second arrival to
        # follow the first departure, so lock-step collapses them onto one tick.
        found = self._collide([
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "pick_up_object", object_id="bowl", source_id="cab"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ])
        self.assertTrue(found, "positional alternation is not an ordering constraint")


if __name__ == "__main__":
    unittest.main()
