"""Tests for structural wait discharge via communicate.releases."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.bc_task_vlm import live_sim_eval  # noqa: E402


class ReleaseDischargeTests(unittest.TestCase):
    """The live-sim rule that decides whether a message wakes a waiter."""

    WAITING = {"from": "agent_1", "about": "bowl"}

    def setUp(self):
        live_sim_eval._LENIENT_WAIT_DISCHARGE = False

    tearDown = setUp

    @staticmethod
    def _message(**args):
        return {"agent": "agent_1", "tool": "communicate",
                "args": {"to": "agent_0", "message": "...", **args}}

    def test_release_naming_the_awaited_id_discharges(self):
        step = self._message(releases="bowl")
        self.assertTrue(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_release_list_containing_the_awaited_id_discharges(self):
        step = self._message(releases=["tray", "bowl"])
        self.assertTrue(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_release_of_something_else_does_not_discharge(self):
        step = self._message(releases="tray")
        self.assertFalse(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_unrelated_chatter_does_not_discharge(self):
        # The arrange_bread_bowl failure: an agent waiting on the bowl was woken
        # by a message about something else and then collided over it.
        step = self._message(message="I am heading to the counter now.")
        self.assertFalse(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_message_mentioning_the_id_in_text_alone_does_not_discharge(self):
        step = self._message(message="I still need the bowl for a moment.")
        self.assertFalse(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_lenient_mode_restores_the_old_wake_on_anything_rule(self):
        live_sim_eval._LENIENT_WAIT_DISCHARGE = True
        step = self._message(message="unrelated")
        self.assertTrue(live_sim_eval._releases_awaited(step, self.WAITING))

    def test_wait_naming_nothing_falls_back_to_waking(self):
        # Nothing specific was named, so nothing specific can hand it over;
        # waiting forever would be worse than the old behaviour.
        step = self._message(message="unrelated")
        self.assertTrue(live_sim_eval._releases_awaited(step, {"about": ""}))


class CommunicateArgumentTests(unittest.TestCase):
    """The FSM's handling of the releases argument."""

    def _validator(self):
        from data_generation.task_level.tasks.specs import load_task_spec
        from data_generation.task_level.tasks.specs.runtime import (
            build_task_definition_from_spec,
        )

        spec = load_task_spec("PrepareCoffee")
        return build_task_definition_from_spec(spec).validator_factory(None)

    def _step(self, **args):
        return {"step": 0, "agent": "agent_0", "tool": "communicate",
                "reasoning": "r", "args": {"to": "agent_1", "message": "done", **args}}

    def test_known_id_is_accepted_and_normalized_to_a_list(self):
        step = self._step(releases="mug")
        self._validator()._validate_communicate_step(step)
        self.assertEqual(step["args"]["releases"], ["mug"])

    def test_unknown_id_is_rejected(self):
        from data_generation.task_level.tasks.shared.errors import (
            CommunicationStepSemanticValidationError,
        )

        with self.assertRaises(CommunicationStepSemanticValidationError):
            self._validator()._validate_communicate_step(self._step(releases="teapot"))

    def test_message_without_releases_is_unchanged(self):
        step = self._step()
        self._validator()._validate_communicate_step(step)
        self.assertNotIn("releases", step["args"])

    def test_unsupported_argument_is_still_rejected(self):
        from data_generation.task_level.tasks.shared.errors import (
            CommunicationStepSemanticValidationError,
        )

        with self.assertRaises(CommunicationStepSemanticValidationError):
            self._validator()._validate_communicate_step(self._step(urgency="high"))


class InsertWaitsTests(unittest.TestCase):
    def test_every_derived_wait_gets_a_matching_structural_release(self):
        import glob
        import json
        import random
        from copy import deepcopy

        import insert_waits

        root = "/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset"
        paths = sorted(glob.glob(root + "/*/traj_000000/original_trajectory.json"))
        if not paths:
            self.skipTest("trajectory subset not present")

        waits_seen = 0
        for path in paths[:20]:
            record = json.loads(Path(path).read_text())
            initial = record["initial_state"]
            steps, _ = insert_waits.insert(
                deepcopy(record["steps"]),
                initial.get("fixtures") or {},
                set(initial.get("objects") or {}),
                random.Random(0),
                {a: (s or {}).get("location")
                 for a, s in (initial.get("agents") or {}).items()},
            )
            released = set()
            for step in steps:
                value = (step.get("args") or {}).get("releases")
                if isinstance(value, str):
                    released.add(value)
                elif value:
                    released.update(value)
            for step in steps:
                if step["tool"] != "wait_for_signal":
                    continue
                waits_seen += 1
                self.assertIn(step["args"]["about"], released)
        self.assertGreater(waits_seen, 0)


if __name__ == "__main__":
    unittest.main()
