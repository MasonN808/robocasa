"""Scheduler-contract tests for partial-observability live sim.

These exercise the pieces that encode the distributed contract — resource
claims, consume-once observations, private history/delivery, and rejection
handling — without a model, a GPU, or a simulator.
"""

from __future__ import annotations

import pytest

from training.bc_task_vlm.live_sim_eval import (
    REJECTION_MODE_REPORT_FAILED,
    REJECTION_MODE_SILENT_RETRY,
    RESOURCE_LOCKS_FIXTURES,
    RESOURCE_LOCKS_NONE,
    RESOURCE_LOCKS_OBJECTS,
    AgentRuntime,
    _handle_rejection,
    _proposal_key,
    _tool_duration,
    resource_claims,
)


def _nav(fixture: str) -> dict:
    return {"tool": "navigate_to_fixture", "args": {"fixture_id": fixture}}


def _pick(obj: str, source: str) -> dict:
    return {"tool": "pick_up_object", "args": {"object_id": obj, "source_id": source}}


def _comm(to: str, message: str = "hi") -> dict:
    return {"tool": "communicate", "args": {"to": to, "message": message}}


def _conflicts(a: dict, b: dict, mode: str) -> bool:
    return bool(resource_claims(a, mode=mode) & resource_claims(b, mode=mode))


# --- resource claims -------------------------------------------------------


def test_disjoint_fixtures_do_not_conflict():
    assert not _conflicts(_nav("counter"), _nav("fridge"), RESOURCE_LOCKS_FIXTURES)


def test_same_fixture_conflicts():
    assert _conflicts(_nav("counter"), _nav("counter"), RESOURCE_LOCKS_FIXTURES)


def test_source_id_is_a_location_not_an_object():
    """pick_up_object(source_id=X) reaches into fixture X, so it must collide
    with a navigate to X — the bug this test pins is source_id being treated as
    an object name, which silently prevented that conflict."""

    assert _conflicts(
        _nav("counter"), _pick("cheese", "counter"), RESOURCE_LOCKS_FIXTURES
    )


def test_read_only_tools_never_conflict():
    gi = {"tool": "get_image", "args": {"views": ["wrist"]}}
    assert resource_claims(gi, mode=RESOURCE_LOCKS_FIXTURES) == frozenset()
    assert resource_claims(_comm("agent_1"), mode=RESOURCE_LOCKS_FIXTURES) == frozenset()
    assert not _conflicts(_comm("agent_1"), _comm("agent_0"), RESOURCE_LOCKS_FIXTURES)


def test_lock_modes_narrow_what_is_claimed():
    nav = _nav("counter")
    assert resource_claims(nav, mode=RESOURCE_LOCKS_FIXTURES) == {"counter"}
    # objects mode leaves fixtures shareable, keeping give_space meaningful
    assert resource_claims(nav, mode=RESOURCE_LOCKS_OBJECTS) == frozenset()
    assert resource_claims(nav, mode=RESOURCE_LOCKS_NONE) == frozenset()
    # both agents reaching for the same item always conflicts
    assert _conflicts(
        _pick("cheese", "fridge"), _pick("cheese", "counter"), RESOURCE_LOCKS_OBJECTS
    )


# --- consume-once observations ---------------------------------------------


def test_observation_is_consumed_exactly_once():
    agent = AgentRuntime("agent_0")
    agent.pending_obs = (["/img/a.png"], ["wrist"])

    paths, views = agent.take_observation()
    assert paths == ["/img/a.png"] and views == ["wrist"]

    # second decision gets nothing: the image was spent
    assert agent.take_observation() == ([], [])


# --- private history and delivery ------------------------------------------


def test_private_history_is_agent_scoped():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    a0.deliver({"agent": "agent_0", "tool": "navigate_to_fixture",
                "args": {"fixture_id": "counter"}})

    assert len(a0.private_history) == 1
    assert a1.private_history == []  # agent_1 never sees agent_0's physical act


def test_communicate_reaches_recipient_but_images_do_not():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    a0.pending_obs = (["/img/secret.png"], ["wrist"])
    msg = {"agent": "agent_0", "tool": "communicate",
           "args": {"to": "agent_1", "message": "counter is clear"}}
    a0.deliver(msg)
    a1.deliver(msg)  # delivery at execution time

    assert a1.private_history[-1]["args"]["message"] == "counter is clear"
    assert a1.pending_obs is None  # images are never transferred


# --- rejection handling ----------------------------------------------------


def _reject(agent, agents, *, mode, max_silent=3, clock=0.0):
    record: dict = {}
    _handle_rejection(
        agent=agent, agents=agents, step=_nav("counter"), reason="resource conflict",
        clock=clock, mode=mode, max_silent=max_silent, multiplier=1.0, record=record,
    )
    return record


def test_silent_retry_leaves_state_untouched_and_waits_for_a_world_event():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    a1.ready_at = 7.5  # the other agent is mid-tool
    agents = {"agent_0": a0, "agent_1": a1}

    record = _reject(a0, agents, mode=REJECTION_MODE_SILENT_RETRY)

    assert record["escalated"] is False
    assert a0.private_history == []  # no FAILED line: no distribution shift
    # sleeps until the world can actually change, not a fixed delay — a constant
    # sleep would reproduce the identical call under greedy decoding forever
    assert a0.ready_at == 7.5


def test_silent_retry_escalates_after_the_cap():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    agents = {"agent_0": a0, "agent_1": a1}

    for _ in range(3):
        record = _reject(a0, agents, mode=REJECTION_MODE_SILENT_RETRY, max_silent=3)
        assert record["escalated"] is False
    assert a0.private_history == []

    record = _reject(a0, agents, mode=REJECTION_MODE_SILENT_RETRY, max_silent=3)
    assert record["escalated"] is True
    assert a0.private_history[-1]["error"].startswith("resource conflict")


def test_report_failed_mode_escalates_immediately():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    record = _reject(a0, {"agent_0": a0, "agent_1": a1},
                     mode=REJECTION_MODE_REPORT_FAILED)

    assert record["escalated"] is True
    assert "error" in a0.private_history[-1]


def test_rejection_never_touches_the_other_agent():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    agents = {"agent_0": a0, "agent_1": a1}
    for mode in (REJECTION_MODE_SILENT_RETRY, REJECTION_MODE_REPORT_FAILED):
        _reject(a0, agents, mode=mode, max_silent=0)
        assert a1.private_history == []


def test_silent_retry_still_advances_the_clock_when_no_event_is_pending():
    """With nothing else in flight there is no world event to wait for, so the
    agent must still be pushed forward rather than retrying at zero cost."""

    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    _reject(a0, {"agent_0": a0, "agent_1": a1},
            mode=REJECTION_MODE_SILENT_RETRY, clock=2.0)
    assert a0.ready_at > 2.0


# --- durations -------------------------------------------------------------


def test_durations_separate_tool_classes_even_when_measurement_is_flat():
    """Teleporting execution can make measured sim-steps near-uniform; the
    per-class floors are what stop both agents staying in lockstep."""

    comm = _tool_duration("communicate", sim_steps=None, multiplier=1.0)
    nav = _tool_duration("navigate_to_fixture", sim_steps=None, multiplier=1.0)
    pick = _tool_duration("pick_up_object", sim_steps=None, multiplier=1.0)
    assert comm < pick < nav


def test_measured_steps_override_the_floor_when_larger():
    assert _tool_duration("communicate", sim_steps=100, multiplier=1.0) == 100.0


# --- repeat detection ------------------------------------------------------


def test_proposal_key_detects_identical_retries():
    assert _proposal_key(_nav("counter")) == _proposal_key(_nav("counter"))
    assert _proposal_key(_nav("counter")) != _proposal_key(_nav("fridge"))
