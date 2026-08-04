"""Coordination is derived from the plan, not asked of the model."""

from __future__ import annotations

import random
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import insert_waits  # noqa: E402
from data_generation.task_level.generation.raw import runtime_support  # noqa: E402

FIXTURES = {
    "counter": {"fixture_type": "counter", "parts": {}, "controls": {}},
    "cab": {"fixture_type": "cabinet", "parts": {}, "controls": {}},
}
INITIAL = {
    "agents": {"agent_0": {"location": "counter", "held_object": None},
               "agent_1": {"location": "cab", "held_object": None}},
    "objects": {"bowl": {"object_type": "bowl", "location": "counter"}},
    "fixtures": FIXTURES,
}


def step(agent, tool, **args):
    return {"agent": agent, "tool": tool, "args": args, "reasoning": "r"}


class ModelToolRegistryTests(unittest.TestCase):
    def test_the_model_is_never_offered_wait_for_signal(self):
        from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY

        definition = SPEC_TASK_REGISTRY["PrepareCoffee"]
        self.assertNotIn("wait_for_signal", str(definition.response_schema))
        self.assertNotIn("wait_for_signal", definition.build_prompt("v0"))

    def test_the_validator_still_accepts_derived_waits(self):
        from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY

        validator = SPEC_TASK_REGISTRY["PrepareCoffee"].validator_factory(None)
        self.assertIn("wait_for_signal", validator.allowed_tool_specs)


class DeriveCoordinationTests(unittest.TestCase):
    def _candidate(self):
        return {"agents": [{"agent": "agent_0"}, {"agent": "agent_1"}], "steps": [
            step("agent_0", "communicate", to="agent_1", message="start"),
            step("agent_1", "communicate", to="agent_0", message="ok"),
            step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            step("agent_1", "pick_up_object", object_id="bowl", source_id="counter"),
        ]}

    def test_a_shared_object_gains_a_wait_and_a_release(self):
        out = runtime_support._derive_coordination(self._candidate(), INITIAL)
        waits = [s for s in out["steps"] if s["tool"] == "wait_for_signal"]
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]["args"]["about"], "bowl")
        released = [s for s in out["steps"]
                    if (s.get("args") or {}).get("releases") == "bowl"]
        self.assertTrue(released, "a derived wait must come with its release")

    def test_the_input_candidate_is_not_mutated(self):
        candidate = self._candidate()
        runtime_support._derive_coordination(candidate, INITIAL)
        self.assertEqual(len(candidate["steps"]), 4)

    def test_derivation_is_deterministic_for_one_plan(self):
        first = runtime_support._derive_coordination(self._candidate(), INITIAL)
        second = runtime_support._derive_coordination(self._candidate(), INITIAL)
        self.assertEqual(first["steps"], second["steps"])

    def test_an_empty_plan_is_left_alone(self):
        candidate = {"agents": [], "steps": []}
        self.assertEqual(runtime_support._derive_coordination(candidate, INITIAL),
                         candidate)


class LockstepInsertionTests(unittest.TestCase):
    def _insert(self, steps):
        return insert_waits.insert(
            deepcopy(steps), FIXTURES, {"bowl"}, random.Random(0),
            {"agent_0": "counter", "agent_1": "cab"},
        )[0]

    def test_positionally_ordered_arrivals_are_given_a_real_ordering(self):
        # The candle_cleanup/traj_000025 shape: correct on the page, collapsed
        # onto one tick in execution because nothing forces the order.
        steps = [
            step("agent_0", "communicate", to="agent_1", message="start"),
            step("agent_1", "communicate", to="agent_0", message="ok"),
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ]
        self.assertTrue(insert_waits.tick_collisions(steps, FIXTURES, {"bowl"}))
        out = self._insert(steps)
        self.assertEqual(insert_waits.tick_collisions(out, FIXTURES, {"bowl"}), [])

    def test_insertion_does_not_deadlock(self):
        steps = [
            step("agent_0", "communicate", to="agent_1", message="start"),
            step("agent_1", "communicate", to="agent_0", message="ok"),
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        ]
        self.assertFalse(insert_waits.deadlocks(self._insert(steps)))

    def test_rerunning_the_pass_is_idempotent(self):
        steps = [
            step("agent_0", "communicate", to="agent_1", message="start"),
            step("agent_1", "communicate", to="agent_0", message="ok"),
            step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            step("agent_1", "pick_up_object", object_id="bowl", source_id="counter"),
        ]
        once = self._insert(steps)
        twice = self._insert(once)
        self.assertEqual([s["tool"] for s in once], [s["tool"] for s in twice])


class SharedSubstrateTests(unittest.TestCase):
    def test_the_checker_and_the_inserter_use_one_schedule(self):
        import lockstep_check

        self.assertIs(lockstep_check.schedule, insert_waits.schedule)
        self.assertIs(lockstep_check.tick_collisions, insert_waits.tick_collisions)


if __name__ == "__main__":
    unittest.main()
