"""The validator runs the plan on a clock instead of reading it down the page."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_generation.task_level.tasks.shared import concurrent_fsm  # noqa: E402
from data_generation.task_level.tasks.shared.concurrent_fsm import (  # noqa: E402
    EXECUTOR,
    LOCK_STEP,
    ConcurrentTaskValidator,
)
from data_generation.task_level.tasks.shared.state import (  # noqa: E402
    AgentRuntimeState,
    TaskRuntimeState,
)

AGENTS = ("agent_0", "agent_1")
FIXTURES = {
    "counter": {"fixture_type": "counter", "parts": {}, "controls": {}},
    "cab": {"fixture_type": "cabinet", "parts": {}, "controls": {}},
    "sink": {"fixture_type": "sink", "parts": {}, "controls": {}},
}
OBJECTS = {"bowl": {"object_type": "bowl", "location": "counter"}}


def step(agent, tool, **args):
    return {"agent": agent, "tool": tool, "args": args, "reasoning": "r", "step": 0}


def plan(*steps):
    numbered = [dict(s, step=i) for i, s in enumerate(steps)]
    return {"agents": [{"agent": a} for a in AGENTS], "steps": numbered}


class FakeValidator:
    """A minimal stand-in exposing the surface ConcurrentTaskValidator uses.

    The real SpecDrivenTaskValidator needs a task spec and a sampled world; the
    scheduling behaviour under test is independent of both, so the seams are
    stubbed and only location/held effects are modelled.
    """

    composite_task = "Fake"
    agent_ids = AGENTS
    allowed_tool_specs = {
        "navigate_to_fixture": {}, "pick_up_object": {}, "give_space": {},
        "place_on_surface": {}, "communicate": {}, "wait_for_signal": {},
    }
    _all_checks = ["fake"]

    def __init__(self, *, goal_after: int | None = None) -> None:
        self.initial_state = {"fixtures": FIXTURES, "objects": OBJECTS,
                              "agents": {a: {"location": None} for a in AGENTS}}
        self._goal_after = goal_after
        self._applied = 0

    # seams the concurrent validator calls
    def _normalize_agents(self, value):
        return [{"agent": a} for a in AGENTS]

    def _normalize_steps(self, value):
        return list(value or [])

    def _build_runtime_state(self, agents):
        self._applied = 0
        return TaskRuntimeState(
            agents={a: AgentRuntimeState(location=None, held_object=None)
                    for a in AGENTS},
            objects={}, fixtures={}, machine_state={},
        )

    def _validation_error_with_step(self, exc, step_index):
        return exc

    def _is_allowed_observation_tool(self, tool_name):
        return True

    def _validate_communicate_step(self, step):
        return None

    def _validate_task_local_symbolic_constraints(self, step):
        return None

    def _validate_generic_transition(self, step, state):
        return None

    def validate_task_preconditions(self, step, state):
        return None

    def apply_task_effects(self, step, state):
        return None

    def _apply_generic_effects(self, step, state):
        agent = state.agents[step["agent"]]
        args = step.get("args") or {}
        if step["tool"] == "navigate_to_fixture":
            agent.location = args.get("fixture_id")
        elif step["tool"] == "give_space":
            agent.location = None
        elif step["tool"] == "pick_up_object":
            agent.held_object = args.get("object_id")
        elif step["tool"] == "place_on_surface":
            agent.held_object = None
        self._applied += 1

    def is_goal_state_satisfied(self, state):
        if self._goal_after is None:
            return False
        return self._applied >= self._goal_after

    def _build_final_state(self, state):
        return {}

    def trajectory_signature(self, candidate):
        return "sig"


def build(**kwargs):
    return ConcurrentTaskValidator(FakeValidator(**kwargs))


OPEN = (step("agent_0", "communicate", to="agent_1", message="a b c d"),
        step("agent_1", "communicate", to="agent_0", message="a b c d"))


class ScheduleTests(unittest.TestCase):
    def test_both_agents_advance_on_the_same_instant(self):
        run = build().replay(plan(*OPEN), model=LOCK_STEP)
        self.assertEqual([e.start for e in run.events], [0.0, 0.0])

    def test_lock_step_instants_are_integer_ticks(self):
        run = build().replay(
            plan(*OPEN, step("agent_0", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual([e.tick for e in run.events], [0, 0, 1])

    def test_executor_durations_desynchronise_the_two_streams(self):
        # navigate costs 4.0 and communicate 0.25, so the streams stop being
        # aligned -- which is the whole reason a tick model is an assumption.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "communicate", to="agent_0", message="e f g h")),
            model=EXECUTOR,
        )
        starts = {(e.agent, e.index): e.start for e in run.events}
        self.assertEqual(starts[("agent_0", 2)], 0.25)
        self.assertEqual(starts[("agent_1", 4)], 0.5)

    def test_position_in_the_file_does_not_order_the_agents(self):
        # agent_1's step is written last but runs first: it is that agent's
        # opening move, while agent_0 is three calls deep.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink")),
            model=LOCK_STEP,
        )
        by_index = {e.index: e.tick for e in run.events}
        self.assertEqual(by_index[4], 1)
        self.assertEqual(by_index[3], 2)


class WaitTests(unittest.TestCase):
    def _waiting_plan(self, *, release: bool):
        release_args = {"releases": ["cab"]} if release else {}
        return plan(
            *OPEN,
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "wait_for_signal", **{"from": "agent_0", "about": "cab"}),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="cab is free now",
                 **release_args),
        )

    def test_a_wait_blocks_until_its_release_fires(self):
        run = build().replay(self._waiting_plan(release=True), model=LOCK_STEP)
        by_index = {e.index: e.tick for e in run.events}
        self.assertEqual(by_index[6], 3, "the release is sent at t=3")
        self.assertEqual(by_index[3], 3, "the wait clears on the same instant")
        self.assertEqual(by_index[4], 4, "and the arrival follows it")

    def test_a_wait_nothing_releases_is_a_deadlock_not_a_pass(self):
        run = build().replay(self._waiting_plan(release=False), model=LOCK_STEP)
        self.assertTrue(run.deadlocked)
        self.assertEqual(run.blocked, ["agent_1"])

    def test_a_release_sent_before_the_wait_never_discharges_it(self):
        # The executor wakes a waiter by DELIVERING a message. One sent while
        # the agent was still busy was never delivered to it.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "communicate", to="agent_1", message="cab is free",
                      releases=["cab"]),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(run.deadlocked)


class ContentionTests(unittest.TestCase):
    def test_two_agents_reaching_for_one_object_at_one_instant_collide(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
                 step("agent_1", "pick_up_object", object_id="bowl", source_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(len(run.conflicts), 1)
        self.assertIn("bowl", run.conflicts[0])

    def test_sequential_use_of_one_object_is_not_contention(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink"),
                 step("agent_1", "pick_up_object", object_id="bowl", source_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_co_occupancy_is_caught_even_when_the_arrivals_are_apart(self):
        # The blind spot every earlier pass had: nobody ACTS at the same
        # instant, but agent_0 never left, so both are standing there.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "communicate", to="agent_0", message="e f g h"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertTrue(any("cab" in c for c in run.conflicts))

    def test_a_handoff_through_give_space_is_not_a_collision(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_a_shared_non_exclusive_fixture_is_fine(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="counter"),
                 step("agent_1", "navigate_to_fixture", fixture_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_agents_starting_co_located_are_not_blamed_on_the_plan(self):
        validator = build()
        state = validator.validator._build_runtime_state(None)
        original = validator.validator._build_runtime_state

        def seeded(agents):
            fresh = original(agents)
            for agent in fresh.agents.values():
                agent.location = "cab"
            return fresh

        validator.validator._build_runtime_state = seeded
        run = validator.replay(
            plan(*OPEN, step("agent_0", "give_space", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])
        del state

    def test_the_starting_overlap_excuse_expires_once_they_separate(self):
        # alcohol_serving_prep starts both agents at the cabinet. Excusing that
        # is right; excusing a RETURN to it after leaving is not, and a blanket
        # suppression hid exactly that case.
        validator = build()
        original = validator.validator._build_runtime_state

        def seeded(agents):
            fresh = original(agents)
            for agent in fresh.agents.values():
                agent.location = "cab"
            return fresh

        validator.validator._build_runtime_state = seeded
        run = validator.replay(
            plan(*OPEN,
                 step("agent_1", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.initial_overlap, ["cab"])
        self.assertTrue(any("cab" in c for c in run.conflicts))


class PromptedProtocolTests(unittest.TestCase):
    """The handover written into TICK_FORMAT_RULES must actually work.

    The rules show the model a four-part shape. If the implementation and the
    example ever disagree, the model is being taught to produce deadlocks --
    which is exactly what the first tick A/B measured, so pin it here.
    """

    def _handover(self):
        # ASK, then BLOCK on the very next tick; the holder keeps working for
        # as long as it likes, and only then LEAVES and REPORTS back to back.
        return plan(
            *OPEN,
            step("agent_1", "communicate", to="agent_0",
                 message="tell me when the cabinet is free"),
            step("agent_1", "wait_for_signal", **{"from": "agent_0", "about": "cab"}),
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="still working here"),
            step("agent_0", "communicate", to="agent_1", message="nearly done"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="cab is yours now",
                 releases=["cab"]),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        )

    def test_the_prompted_handover_runs_clean(self):
        for model in (LOCK_STEP, EXECUTOR):
            with self.subTest(model=model):
                run = build().replay(self._handover(), model=model)
                self.assertFalse(run.deadlocked)
                self.assertEqual(run.conflicts, [])
                self.assertEqual(run.protocol, [])

    def test_the_wait_actually_blocks(self):
        # agent_1 reaches its wait at instant 2 and is not woken until the
        # release at 3, so it genuinely sits still for an instant. The wait
        # EVENT is recorded at the instant it discharges, which is why the
        # blocking shows up as idle time rather than as a tick difference.
        run = build().replay(self._handover(), model=LOCK_STEP)
        self.assertEqual(run.protocol, [], "not flagged inert")
        self.assertGreater(run.idle["agent_1"], 0.0)

    def test_the_rules_state_the_release_obligation(self):
        import tick_format

        text = "\n".join(tick_format.TICK_FORMAT_RULES)
        self.assertIn("releases", text)
        self.assertIn("give_space", text, "release is coupled to departure")
        self.assertIn("on a LATER tick", text)
        self.assertIn("IMMEDIATELY", text, "no gap between departure and report")
        self.assertIn("very next tick", text, "no gap between ask and wait")

    def test_the_wait_must_immediately_follow_the_ask(self):
        steps = list(self._handover()["steps"])
        steps.insert(3, step("agent_1", "get_image"))  # inert, must be allowed
        self.assertEqual(build().replay(plan(*steps), model=LOCK_STEP).protocol, [])

        steps[3] = step("agent_1", "communicate", to="agent_0", message="one more thing")
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertEqual(run.protocol, [], "an extra ask is still an ask")

        steps[3] = step("agent_1", "navigate_to_fixture", fixture_id="sink")
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(any("not the request" in p for p in run.protocol))

    def test_the_holder_must_still_hold_it_when_the_waiter_blocks(self):
        # give_space moved before the wait: nothing was ever contested.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="one moment"),
                 step("agent_1", "communicate", to="agent_0",
                      message="tell me when the cabinet is free"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"}),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertTrue(any("had already let it go" in p for p in run.protocol))

    def test_dropping_the_release_from_the_prompted_shape_deadlocks(self):
        # The failure mode the rules exist to prevent, held fixed.
        steps = [s for s in self._handover()["steps"]
                 if not (s.get("args") or {}).get("releases")]
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(run.deadlocked)


class ReleaseProtocolTests(unittest.TestCase):
    """Three rules that need no clock, so no duration model can excuse them."""

    def test_releasing_a_fixture_you_are_standing_at_is_rejected(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1",
                      message="you can take it", releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("still standing at it" in p for p in run.protocol))

    def test_releasing_an_object_you_are_still_holding_is_rejected(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl",
                      source_id="counter"),
                 step("agent_0", "communicate", to="agent_1",
                      message="bowl is yours", releases=["bowl"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("still holding it" in p for p in run.protocol))

    def test_a_gap_between_leaving_and_reporting_is_rejected(self):
        # Every tick in the gap is a tick the waiter sits blocked on something
        # that is already free.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="just tidying"),
                 step("agent_0", "communicate", to="agent_1", message="all done",
                      releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("not the handover" in p for p in run.protocol))

    def test_observations_between_the_departure_and_the_report_are_fine(self):
        # get_image is inert and post-processing interleaves it everywhere.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "get_image"),
                 step("agent_0", "communicate", to="agent_1", message="all done",
                      releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertEqual(run.protocol, [])

    def test_a_wait_released_on_its_own_instant_is_inert(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "communicate", to="agent_0", message="ok then"),
                 step("agent_1", "communicate", to="agent_0", message="still here"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(any("inert" in p for p in run.protocol))

    def test_a_wait_that_arrives_after_the_release_says_so(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "communicate", to="agent_0", message="one"),
                 step("agent_1", "communicate", to="agent_0", message="two"),
                 step("agent_1", "communicate", to="agent_0", message="three"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(run.deadlocked)
        self.assertTrue(any("too late" in c for c in run.conflicts))


class GoalTests(unittest.TestCase):
    def test_work_begun_after_the_goal_is_reported(self):
        # The goal lands on the two navigates at t=1; the give_space at t=2 is
        # started afterwards and has nothing left to contribute.
        run = build(goal_after=2).replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink")),
            model=LOCK_STEP,
        )
        self.assertIsNotNone(run.goal_at)
        self.assertTrue(run.post_goal)

    def test_work_already_in_flight_when_the_goal_lands_is_not_post_goal(self):
        # Both agents commit to their call on the same instant; one of them
        # happening to complete the goal does not make the other's call late.
        run = build(goal_after=3).replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.post_goal, [])


class ReportingTests(unittest.TestCase):
    def test_idle_time_is_reported_per_agent(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.makespan, 3.0)
        self.assertEqual(run.idle["agent_1"], 2.0)

    def test_render_puts_one_agent_in_each_column(self):
        text = build().render(plan(*OPEN), model=LOCK_STEP)
        self.assertIn("agent_0", text.splitlines()[0])
        self.assertIn("agent_1", text.splitlines()[0])
        self.assertEqual(len(text.splitlines()), 3)

    def test_durations_match_the_live_evaluation(self):
        self.assertEqual(concurrent_fsm.duration_of("navigate_to_fixture", EXECUTOR), 4.0)
        self.assertEqual(concurrent_fsm.duration_of("communicate", EXECUTOR), 0.25)
        self.assertEqual(concurrent_fsm.duration_of("pick_up_object", EXECUTOR), 2.0)
        self.assertEqual(concurrent_fsm.duration_of("navigate_to_fixture", LOCK_STEP), 1.0)


if __name__ == "__main__":
    unittest.main()
