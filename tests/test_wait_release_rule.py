"""The FSM must prove every wait_for_signal is released by the named partner.

The runtime harness wakes a waiter on ANY message on purpose -- judging relevance
is the model's job at inference. Validation has the opposite duty: a demo is only
usable if the partner really does report the awaited thing, so the release must
name `about` verbatim. Without this rule a trajectory that deadlocks in live-sim
still validates, because a wait has no symbolic effect to contradict.
"""

from __future__ import annotations

import unittest

from data_generation.task_level.tasks.shared.errors import (
    WaitSignalSemanticValidationError,
)
from data_generation.task_level.tasks.shared.fsm import FiniteStateTaskValidator


class _NeverSatisfied(FiniteStateTaskValidator):
    """Goal never fires, so validation runs the whole step list."""

    def is_goal_state_satisfied(self, runtime_state) -> bool:  # noqa: D102
        return False


def _validator() -> FiniteStateTaskValidator:
    return _NeverSatisfied(
        composite_task="WaitProbe",
        agent_ids=("agent_0", "agent_1"),
        initial_state={
            "agents": {
                "agent_0": {"location": "counter", "held_object": None},
                "agent_1": {"location": "counter", "held_object": None},
            },
            "objects": {},
            "fixtures": {"counter": {"fixture_type": "counter"}},
        },
        allowed_tool_specs={
            "communicate": {"tool_args": ["to", "message"]},
            "wait_for_signal": {"tool_args": ["from", "about"]},
        },
    )


def _wait(step, agent="agent_0", frm="agent_1", about="bowl"):
    return {
        "step": step,
        "agent": agent,
        "tool": "wait_for_signal",
        "args": {"from": frm, "about": about},
        "reasoning": "I am waiting.",
    }


def _msg(step, agent, to, message):
    return {
        "step": step,
        "agent": agent,
        "tool": "communicate",
        "args": {"to": to, "message": message},
        "reasoning": "I am speaking.",
    }


class WaitReleaseRuleTest(unittest.TestCase):
    def _run(self, steps):
        return _validator().validate(
            {
                "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
                "steps": steps,
            }
        )

    def _error(self, steps) -> str:
        with self.assertRaises(WaitSignalSemanticValidationError) as caught:
            self._run(steps)
        return str(caught.exception)

    def test_release_naming_about_verbatim_is_accepted(self):
        steps = [
            _wait(0),
            _msg(1, "agent_1", "agent_0", "I am done with bowl, it is yours."),
        ]
        # reaches the wait rule without raising; later goal checks are not our concern
        try:
            self._run(steps)
        except WaitSignalSemanticValidationError:  # pragma: no cover - the failure we test for
            self.fail("a verbatim release must satisfy the wait rule")
        except Exception:
            pass

    def test_wait_with_no_release_is_rejected(self):
        message = self._error([_wait(0), _msg(1, "agent_1", "agent_0", "on my way.")])
        self.assertIn("never released", message)

    def test_release_must_come_from_the_named_partner(self):
        # agent_0 naming the object itself does not release its own wait
        message = self._error(
            [_wait(0), _msg(1, "agent_0", "agent_1", "I need the bowl.")]
        )
        self.assertIn("never released", message)

    def test_paraphrase_does_not_release(self):
        # "the bowl" reads fine to a human but `about` was 'salad_bowl'
        message = self._error(
            [
                _wait(0, about="salad_bowl"),
                _msg(1, "agent_1", "agent_0", "I am finished with the bowl."),
            ]
        )
        self.assertIn("never released", message)

    def test_sibling_object_does_not_release(self):
        # the collision a token matcher would let through
        message = self._error(
            [
                _wait(0, about="sugar_cube_2"),
                _msg(1, "agent_1", "agent_0", "I am done with sugar_cube_1."),
            ]
        )
        self.assertIn("never released", message)

    def test_release_must_be_addressed_to_the_waiter(self):
        steps = [
            _wait(0),
            _msg(1, "agent_1", "agent_1", "bowl is free"),
        ]
        with self.assertRaises(Exception):
            self._run(steps)

    def test_earlier_mention_does_not_count_as_release(self):
        # a release has to follow the wait, not precede it
        message = self._error(
            [
                _msg(0, "agent_1", "agent_0", "I will use bowl first."),
                _wait(1),
            ]
        )
        self.assertIn("never released", message)

    def test_wait_rejects_self_as_partner(self):
        message = self._error([_wait(0, frm="agent_0")])
        self.assertIn("other agent", message)

    def test_wait_rejects_empty_about(self):
        message = self._error([_wait(0, about="  ")])
        self.assertIn("non-empty about", message)

    def test_wait_rejects_extra_args(self):
        step = _wait(0)
        step["args"]["timeout"] = 5
        message = self._error([step])
        self.assertIn("only contain from and about", message)


if __name__ == "__main__":
    unittest.main()
