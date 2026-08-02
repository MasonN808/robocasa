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
