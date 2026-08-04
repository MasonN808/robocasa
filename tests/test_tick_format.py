"""Tests for the tick-based trajectory format."""

from __future__ import annotations

import glob
import json
import random
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import insert_waits  # noqa: E402
import tick_format  # noqa: E402
from data_generation.task_level.generation.raw import runtime_support  # noqa: E402

AGENTS = ("agent_0", "agent_1")


def action(tool, **args):
    return {"tool": tool, "args": args, "reasoning": "r"}


class FlattenTests(unittest.TestCase):
    def test_both_agents_in_one_tick_become_adjacent_steps(self):
        rows = [{"tick": 0, "agent_0": action("get_image"), "agent_1": action("get_image")}]
        steps = tick_format.to_steps(rows, AGENTS)
        self.assertEqual([s["agent"] for s in steps], ["agent_0", "agent_1"])
        self.assertEqual([s["step"] for s in steps], [0, 1])

    def test_an_omitted_agent_contributes_no_step(self):
        rows = [{"tick": 0, "agent_0": action("get_image")}]
        self.assertEqual(len(tick_format.to_steps(rows, AGENTS)), 1)

    def test_steps_are_numbered_consecutively_across_ticks(self):
        rows = [
            {"tick": 0, "agent_0": action("get_image"), "agent_1": action("get_image")},
            {"tick": 1, "agent_0": action("give_space", fixture_id="cab")},
        ]
        self.assertEqual([s["step"] for s in tick_format.to_steps(rows, AGENTS)], [0, 1, 2])

    def test_a_malformed_action_is_dropped_rather_than_crashing(self):
        rows = [{"tick": 0, "agent_0": "not a dict", "agent_1": {"no_tool": 1}}]
        self.assertEqual(tick_format.to_steps(rows, AGENTS), [])


class RoundTripTests(unittest.TestCase):
    def test_real_trajectories_survive_steps_to_rows_and_back(self):
        root = "/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset"
        paths = sorted(glob.glob(root + "/*/traj_000000/original_trajectory.json"))
        if not paths:
            self.skipTest("trajectory subset not present")

        for path in paths[:12]:
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
            rows = tick_format.to_rows(steps, AGENTS)
            restored = tick_format.to_steps(rows, AGENTS)

            with self.subTest(path=path):
                self.assertEqual(
                    sorted((s["agent"], s["tool"]) for s in steps),
                    sorted((s["agent"], s["tool"]) for s in restored),
                )
                original_ticks, _ = insert_waits.schedule(steps)
                restored_ticks, _ = insert_waits.schedule(restored)
                self.assertEqual(
                    max(t for t in original_ticks if t is not None),
                    max(t for t in restored_ticks if t is not None),
                )


class SchemaTests(unittest.TestCase):
    def _definition(self):
        from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY

        return SPEC_TASK_REGISTRY["PrepareCoffee"]

    def test_the_top_level_is_ticks_not_steps(self):
        schema = self._definition().tick_response_schema
        self.assertEqual(list(schema["properties"]), ["ticks"])

    def test_a_row_has_a_tick_and_one_slot_per_agent(self):
        row = self._definition().tick_response_schema["properties"]["ticks"]["items"]
        self.assertEqual(set(row["properties"]), {"tick", "agent_0", "agent_1"})
        self.assertEqual(row["required"], ["tick"], "an agent may sit a tick out")

    def test_waits_are_offered_in_tick_form_and_withheld_in_flat_form(self):
        # In tick form the model can see that an agent is blocked, so it is
        # asked to place its own coordination.
        definition = self._definition()
        self.assertIn("wait_for_signal", json.dumps(definition.tick_response_schema))
        self.assertNotIn("wait_for_signal", json.dumps(definition.response_schema))

    def test_an_action_slot_carries_no_agent_or_step_field(self):
        row = self._definition().tick_response_schema["properties"]["ticks"]["items"]
        self.assertNotIn("agent", row["properties"]["agent_0"]["properties"])
        self.assertNotIn("step", row["properties"]["agent_0"]["properties"])


class ValidationPathTests(unittest.TestCase):
    def _candidate(self):
        return {"agents": [{"agent": "agent_0"}, {"agent": "agent_1"}], "ticks": [
            {"tick": 0,
             "agent_0": action("communicate", to="agent_1", message="start"),
             "agent_1": action("communicate", to="agent_0", message="ok")},
            {"tick": 1, "agent_0": action("pick_up_object", object_id="mug", source_id="cab")},
        ]}

    def test_tick_output_is_recognized_by_shape(self):
        self.assertTrue(runtime_support._is_tick_format(self._candidate()))
        self.assertFalse(runtime_support._is_tick_format({"steps": []}))

    def test_flattening_produces_steps_and_keeps_the_rows(self):
        out = runtime_support._flatten_tick_candidate(self._candidate())
        self.assertEqual(len(out["steps"]), 3)
        self.assertEqual(len(out["tick_rows"]), 2)
        self.assertNotIn("ticks", out)

    def test_agent_order_falls_back_when_agents_are_absent(self):
        candidate = self._candidate()
        candidate.pop("agents")
        out = runtime_support._flatten_tick_candidate(candidate)
        self.assertEqual([s["agent"] for s in out["steps"]][:2], ["agent_0", "agent_1"])


class RenderTests(unittest.TestCase):
    def test_render_shows_one_row_per_tick(self):
        rows = [{"tick": 0, "agent_0": action("get_image"), "agent_1": action("get_image")},
                {"tick": 1, "agent_0": action("give_space", fixture_id="cab")}]
        text = tick_format.render(rows, AGENTS)
        self.assertIn("agent_0", text)
        self.assertIn("give_space(fixture_id=cab)", text)
        self.assertEqual(len(text.splitlines()), 4)  # header, rule, two rows


if __name__ == "__main__":
    unittest.main()
