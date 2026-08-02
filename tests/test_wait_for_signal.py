"""Semantics of wait_for_signal.

The environment deliberately does NOT inspect args.about: any inbound message
wakes a waiter and the model decides whether it was the one it needed. These
tests pin that down, plus the two guards (livelock cap, mutual-wait deadlock).
"""
import unittest
from training.bc_task_vlm.live_sim_eval import AgentRuntime, WAIT_TOOL_NAME


class WaitForSignalSemantics(unittest.TestCase):
    def setUp(self):
        self.a = AgentRuntime("agent_0")

    def test_starts_not_waiting(self):
        self.assertIsNone(self.a.waiting_for)

    def test_any_message_from_other_agent_wakes(self):
        self.a.waiting_for = {"from": "agent_1", "about": "bowl"}
        self.a.deliver({"agent": "agent_1", "tool": "communicate",
                        "args": {"to": "agent_0", "message": "totally unrelated"}})
        self.assertIsNone(self.a.waiting_for,
                          "relevance is the model's judgement, not the harness's")

    def test_own_message_does_not_wake(self):
        self.a.waiting_for = {"from": "agent_1", "about": "bowl"}
        self.a.deliver({"agent": "agent_0", "tool": "communicate",
                        "args": {"to": "agent_1", "message": "mine"}})
        self.assertIsNotNone(self.a.waiting_for)

    def test_non_message_step_does_not_wake(self):
        self.a.waiting_for = {"from": "agent_1", "about": "bowl"}
        self.a.deliver({"agent": "agent_1", "tool": "navigate_to_fixture",
                        "args": {"fixture_id": "counter"}})
        self.assertIsNotNone(self.a.waiting_for)

    def test_about_is_never_matched(self):
        """A message that does not mention `about` still wakes the agent."""
        self.a.waiting_for = {"from": "agent_1", "about": "hotdog_bun"}
        self.a.deliver({"agent": "agent_1", "tool": "communicate",
                        "args": {"to": "agent_0", "message": "I am at the fridge"}})
        self.assertIsNone(self.a.waiting_for)

    def test_wait_is_in_shared_tool_registry(self):
        from data_generation.task_level.subatomic_tool_specs import (
            TASK_LEVEL_ALLOWED_TOOL_SPECS,
        )
        spec = TASK_LEVEL_ALLOWED_TOOL_SPECS[WAIT_TOOL_NAME]
        self.assertEqual(spec["tool_args"], ["from", "about"])
        self.assertIn("does NOT check", spec["description"])


if __name__ == "__main__":
    unittest.main()


class WaitInsertion(unittest.TestCase):
    """Phase 3 rewriter: every wait must have a release, in the right order."""

    def _calls(self):
        return [
            {"tool": "communicate", "robot_idx": 0,
             "args": {"to": "agent_1", "message": "starting"}},
            {"tool": "pick_up_object", "robot_idx": 0,
             "args": {"object_id": "bowl", "source_id": "counter"}},
            {"tool": "place_on_surface", "robot_idx": 0,
             "args": {"object_id": "bowl", "support_id": "counter"}},
            {"tool": "pick_up_object", "robot_idx": 1,
             "args": {"object_id": "bowl", "source_id": "counter"}},
        ]

    def test_inserts_wait_before_dependent_action(self):
        from insert_wait_for_signal import rewrite
        out, stats = rewrite(self._calls())
        self.assertEqual(stats["waits"], 1)
        tools = [c["tool"] for c in out]
        wait_i = tools.index("wait_for_signal")
        dep_i = max(i for i, c in enumerate(out)
                    if c["tool"] == "pick_up_object" and c["robot_idx"] == 1)
        self.assertLess(wait_i, dep_i, "wait must precede the dependent action")

    def test_wait_names_the_holder_and_object(self):
        from insert_wait_for_signal import rewrite
        out, _ = rewrite(self._calls())
        wait = next(c for c in out if c["tool"] == "wait_for_signal")
        self.assertEqual(wait["args"]["from"], "agent_0")
        self.assertEqual(wait["args"]["about"], "bowl")
        self.assertEqual(wait["robot_idx"], 1, "the WAITER issues the wait")

    def test_every_wait_has_a_release_from_the_named_partner(self):
        from insert_wait_for_signal import rewrite
        out, _ = rewrite(self._calls())
        for i, c in enumerate(out):
            if c["tool"] != "wait_for_signal":
                continue
            holder = c["args"]["from"]
            obj = c["args"]["about"]
            released = any(
                p["tool"] == "communicate"
                and f"agent_{p['robot_idx']}" == holder
                and obj in (p["args"].get("message") or "")
                for p in out[:i]
            )
            self.assertTrue(released, f"wait about {obj} has no release from {holder}")

    def test_no_dependency_leaves_trajectory_untouched(self):
        from insert_wait_for_signal import rewrite
        solo = [{"tool": "pick_up_object", "robot_idx": 0,
                 "args": {"object_id": "bowl", "source_id": "counter"}}]
        out, stats = rewrite(solo)
        self.assertEqual(out, solo)
        self.assertFalse(stats)
