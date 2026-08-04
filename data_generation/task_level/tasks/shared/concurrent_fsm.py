"""A validator that replays a trajectory the way the executor actually runs it.

The old validator walks the flat step list top to bottom against one state. That
list is a *serialization* of two independent per-agent streams, so reading it in
order describes an interleaving that never happens: a step written at index 5 may
run long after one written at index 6. Every coordination defect measured on this
dataset traces back to that gap, and each was patched separately -- a lock-step
collision checker here, a tick-span contention rule there, a wait-necessity
heuristic in a third place. They were all approximations of one thing the
validator was not doing: running the plan.

This one runs it. Two agents, a clock, per-tool durations, and waits that block
until a release is *delivered*. Contention stops being something to infer -- if
two agents end up holding the same object or standing at the same exclusive
fixture at the same instant, the replay is simply in that state, and the check is
a state invariant rather than a guess about plan positions.

Two duration models, both faithful to a real executor regime:

    LOCK_STEP  every call costs 1.0  -- `live_sim_eval._UNIFORM_DURATIONS`
    EXECUTOR   the per-class floors  -- `live_sim_eval._tool_duration` default

LOCK_STEP is the "tick" regime: instants are integers and one instant is one
tick. EXECUTOR is what the live evaluation runs by default. A plan that is
correct only under one of them is correct only by luck, so `validate` checks
both by default; see `DURATION_MODELS`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from .constants import (
    DEPENDENCY_ARG_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    FIXTURE_ARG_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
    SOCIAL_TOOL_NAMES,
    WAIT_TOOL_NAMES,
)
from .errors import (
    DeadlockSemanticValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    PostGoalActionSemanticValidationError,
    ResourceConflictSemanticValidationError,
    TrajectoryValidationError,
    UnsatisfiedGoalSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    WaitSignalSemanticValidationError,
)

LOCK_STEP = "lock_step"
EXECUTOR = "executor"
DURATION_MODELS = (LOCK_STEP, EXECUTOR)

# Mirrors live_sim_eval._tool_duration. The executor teleports, so measured
# sim-steps come out ~0 and these floors ARE the duration model there too.
_EXECUTOR_FLOORS = {"communicate": 0.25, "get_image": 0.25, "navigate_to_fixture": 4.0}
_EXECUTOR_DEFAULT_FLOOR = 2.0

# Float clocks accumulate error; instants are compared, not stored, so round.
_EPS = 1e-6

_WAIT_TOOL = "wait_for_signal"


def duration_of(tool_name: str, model: str) -> float:
    """Virtual-clock cost of one completed call under `model`."""

    if model == LOCK_STEP:
        return 1.0
    return _EXECUTOR_FLOORS.get(tool_name, _EXECUTOR_DEFAULT_FLOOR)


@dataclass
class Event:
    """One call, placed on the clock."""

    index: int  # position in the flat step list
    agent: str
    tool: str
    start: float
    end: float

    @property
    def tick(self) -> int:
        """Instant ordinal; under LOCK_STEP this is literally the tick."""

        return int(round(self.start))


@dataclass
class Replay:
    """What happened when the plan was run."""

    model: str
    events: list[Event] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    idle: dict[str, float] = field(default_factory=dict)
    makespan: float = 0.0
    goal_at: float | None = None
    post_goal: list[str] = field(default_factory=list)
    # Handover protocol violations. Unlike `conflicts` these need no clock: a
    # release that reports a departure that did not happen is wrong under every
    # duration model, whether or not the timing exposes it.
    protocol: list[str] = field(default_factory=list)
    first_action: bool = False
    # Exclusive fixtures the sampled initial state already put both agents at.
    initial_overlap: list[str] = field(default_factory=list)
    runtime_state: Any = None
    step_error: TrajectoryValidationError | None = None

    @property
    def deadlocked(self) -> bool:
        return bool(self.blocked)

    def ticks(self, total_steps: int) -> list[int | None]:
        """Per-step instant ordinals, indexed like the flat step list."""

        out: list[int | None] = [None] * total_steps
        for event in self.events:
            out[event.index] = event.tick
        return out


def _released_ids(step: dict[str, Any]) -> list[str]:
    value = (step.get("args") or {}).get("releases") or []
    values = [value] if isinstance(value, str) else list(value)
    return [str(item).strip() for item in values if str(item).strip()]


class ConcurrentTaskValidator:
    """Replays a candidate as concurrent per-agent streams on a virtual clock.

    Composes an existing `FiniteStateTaskValidator` (or any subclass, so the
    spec-driven task semantics come along unchanged) and reuses its per-step
    precondition and effect machinery. What is new here is only *when* each step
    is applied -- and that turns out to be where all the bugs were.
    """

    def __init__(self, validator: Any) -> None:
        self.validator = validator
        self.composite_task = validator.composite_task
        self.agent_ids = tuple(validator.agent_ids)

    # -- replay -----------------------------------------------------------

    def replay(
        self,
        candidate: dict[str, Any],
        *,
        model: str = LOCK_STEP,
        stop_on_step_error: bool = True,
    ) -> Replay:
        """Runs the plan on a clock and reports what the run looked like."""

        validator = self.validator
        agents = validator._normalize_agents(candidate.get("agents"))
        steps = validator._normalize_steps(candidate.get("steps"))
        state = validator._build_runtime_state(agents)

        result = Replay(model=model, runtime_state=state)
        if not steps:
            return result

        streams: dict[str, list[int]] = {agent_id: [] for agent_id in self.agent_ids}
        for index, step in enumerate(steps):
            streams.setdefault(step["agent"], []).append(index)

        cursor = {agent_id: 0 for agent_id in streams}
        ready_at = {agent_id: 0.0 for agent_id in streams}
        waiting_since: dict[str, float] = {}
        blocked_on: dict[str, tuple[str, str]] = {}
        # A release is an EVENT: it wakes only an agent already waiting when it
        # fires, exactly as AgentRuntime.deliver does -- a message sent before
        # the waiter blocked was never delivered to it and never will be.
        fired: dict[tuple[str, str], list[float]] = {}
        seen_conflicts: set[tuple[str, ...]] = set()
        # When each wait first blocked, and when each resource was actually
        # let go of, so the two orderings can be checked after the run.
        wait_started: dict[int, float] = {}
        departures: dict[tuple[str, str], list[float]] = {}
        # Last non-observation step each agent took, so a release can be checked
        # against what its sender did immediately before speaking.
        previous: dict[str, dict[str, Any] | None] = {a: None for a in streams}
        # Agents that START co-located are a property of the sampled initial
        # state, not of the plan, so the opening overlap is excused. The grace
        # expires the moment they separate: coming BACK to a fixture the other
        # never left is the plan's doing, and several trajectories do exactly
        # that under cover of having begun there.
        grace: set[tuple[str, ...]] = {
            ("at", location, *sorted(here))
            for location, here in self._occupancy(state).items()
            if len(here) > 1
        }
        result.initial_overlap = sorted(key[1] for key in grace)
        clock = 0.0

        while True:
            # An agent blocked on its LAST step still counts: it has nothing
            # left to do, but it is stuck rather than finished, and the
            # executor would sit there too.
            pending = [
                a for a in streams
                if cursor[a] < len(streams[a]) or a in blocked_on
            ]
            if not pending:
                break

            # A wait is CALLED once and then the agent sits blocked, exactly as
            # AgentRuntime does: `waiting_for` is set when the call is made and
            # cleared when a message is delivered. Deferring the call until it
            # could succeed put it on the clock at the instant it discharged,
            # which drew the agent as idle for the whole block and then calling
            # wait at the very moment it was already free.
            runnable: list[str] = []
            for agent_id in pending:
                key = blocked_on.get(agent_id)
                if key is not None:
                    woken = [
                        at for at in fired.get(key, ())
                        if at >= waiting_since[agent_id] - _EPS
                    ]
                    if not woken:
                        continue
                    blocked_on.pop(agent_id)
                    # A woken agent resumes on the instant AFTER the release,
                    # never on the release itself. Same-instant resumption is a
                    # race the executor would have to arbitrate -- the message
                    # has to be delivered before the waiter's next proposal is
                    # built -- and it is one more thing for the model to get
                    # subtly wrong. Costing the wake a tick removes both.
                    ready_at[agent_id] = max(
                        ready_at[agent_id],
                        min(woken) + duration_of(_WAIT_TOOL, model),
                    )
                if cursor[agent_id] < len(streams[agent_id]):
                    runnable.append(agent_id)

            if not runnable and not blocked_on:
                break

            if not runnable:
                result.blocked = sorted(blocked_on)
                for agent_id in result.blocked:
                    index = streams[agent_id][cursor[agent_id] - 1]
                    step = steps[index]
                    args = step.get("args") or {}
                    earlier = fired.get(blocked_on[agent_id], ())
                    since = waiting_since[agent_id]
                    detail = (
                        f"the release fired at t={max(earlier):g}, before this "
                        f"agent began waiting at t={since:g}, so it was "
                        f"delivered to nobody -- the wait is too late to be "
                        f"woken by it"
                        if earlier
                        else "it is never released"
                    )
                    result.conflicts.append(
                        f"  {agent_id} is blocked forever from t={since:g}, at step "
                        f"{step['step']}: wait_for_signal(from={args.get('from')!r}, "
                        f"about={str(args.get('about'))!r}) -- {detail}."
                    )
                break

            clock = max(clock, min(ready_at[a] for a in runnable))
            acting = sorted(a for a in runnable if ready_at[a] <= clock + _EPS)

            instant = [(a, streams[a][cursor[a]]) for a in acting]
            self._check_simultaneous_use(steps, instant, clock, seen_conflicts, result)

            for agent_id, index in instant:
                step = steps[index]
                tool = step["tool"]

                if tool in WAIT_TOOL_NAMES:
                    self._check_ask(step, agent_id, previous[agent_id], result)
                    args = step.get("args") or {}
                    key = (args.get("from"), str(args.get("about")))
                    blocked_on[agent_id] = key
                    waiting_since[agent_id] = clock
                    wait_started[index] = clock
                    # A release fired at the very instant the wait is called is
                    # a race in the executor -- whether the message lands before
                    # or after `waiting_for` is set depends on the order the
                    # cycle happens to process proposals. Discharge stays
                    # permissive; the plan is flagged instead of gambled on.
                    if any(at <= clock + _EPS for at in fired.get(key, ())):
                        result.protocol.append(
                            f"  step {step['step']}: {agent_id} calls "
                            f"wait_for_signal on {str(args.get('about'))!r} at "
                            f"t={clock:g}, but {args.get('from')} released it at "
                            f"t={max(fired[key]):g}. The wait is inert -- block "
                            f"before the release, or drop it."
                        )
                else:
                    try:
                        self._apply(step, state)
                    except TrajectoryValidationError as exc:
                        result.step_error = validator._validation_error_with_step(
                            exc, step["step"]
                        )
                        if stop_on_step_error:
                            result.makespan = clock
                            return result
                    if tool not in OBSERVATION_TOOL_NAMES | SOCIAL_TOOL_NAMES:
                        result.first_action = True

                if tool == "communicate":
                    for released in _released_ids(step):
                        self._check_release(
                            step, agent_id, released, state,
                            previous[agent_id], clock, result,
                        )
                        fired.setdefault((agent_id, released), []).append(clock)

                for resource in self._let_go_of(step, previous[agent_id]):
                    departures.setdefault((agent_id, resource), []).append(clock)

                if tool not in OBSERVATION_TOOL_NAMES:
                    previous[agent_id] = step

                end = clock + duration_of(tool, model)
                result.events.append(
                    Event(index=index, agent=agent_id, tool=tool, start=clock, end=end)
                )
                ready_at[agent_id] = end
                cursor[agent_id] += 1

            self._check_co_location(state, acting, clock, seen_conflicts, grace, result)

            if result.goal_at is None and validator.is_goal_state_satisfied(state):
                result.goal_at = clock
            elif result.goal_at is not None:
                for agent_id, index in instant:
                    step = steps[index]
                    if step["tool"] in OBSERVATION_TOOL_NAMES:
                        continue
                    result.post_goal.append(
                        f"  step {step['step']} ({step['tool']} by {agent_id}) starts "
                        f"at t={clock:g}, after the goal was reached at "
                        f"t={result.goal_at:g}."
                    )

        result.makespan = max((event.end for event in result.events), default=0.0)
        busy = {agent_id: 0.0 for agent_id in streams}
        for event in result.events:
            busy[event.agent] += event.end - event.start
        result.idle = {a: round(result.makespan - b, 3) for a, b in busy.items()}
        return result

    # -- per-step application ---------------------------------------------

    def _apply(self, step: dict[str, Any], state: Any) -> None:
        """Applies one step, mirroring live_sim_eval.FsmMirror.step."""

        validator = self.validator
        if step["tool"] in OBSERVATION_TOOL_NAMES:
            # Observation never reaches FsmMirror in the live evaluation either:
            # get_image is served from the renderer and has no symbolic effect.
            # Some specs never offer it, yet post-processing writes it into the
            # recorded trajectories, so gating on allowed_tool_specs here would
            # reject data the executor runs happily.
            return

        if step["tool"] == "communicate":
            validator._validate_communicate_step(step)
            state.communicated_agents.add(step["agent"])
            validator.apply_task_effects(step, state)
            return

        if step["tool"] not in OBSERVATION_TOOL_NAMES:
            if state.communicated_agents != set(self.agent_ids):
                raise MissingInitialCommunicationSemanticValidationError(
                    "Both agents must coordinate via communication before the "
                    "first task action.",
                    details={"agent": step["agent"], "tool": step["tool"]},
                )

        if step["tool"] not in validator.allowed_tool_specs and not (
            validator._is_allowed_observation_tool(step["tool"])
        ):
            raise UnsupportedToolSemanticValidationError(
                f"Tool {step['tool']} is not allowed for {self.composite_task}.",
                details={"tool": step["tool"], "composite_task": self.composite_task},
            )

        validator._validate_task_local_symbolic_constraints(step)
        validator._validate_generic_transition(step, state)
        validator.validate_task_preconditions(step, state)
        validator._apply_generic_effects(step, state)
        validator.apply_task_effects(step, state)

    # -- the two contention invariants ------------------------------------

    def _exclusive(self, fixture_id: str | None) -> bool:
        fixtures = self.validator.initial_state.get("fixtures") or {}
        state = fixtures.get(fixture_id)
        if not isinstance(state, dict):
            return False
        return str(state.get("fixture_type", "")).lower() in EXCLUSIVE_FIXTURE_TYPES

    def _named_resources(self, step: dict[str, Any]) -> list[str]:
        """Objects and exclusive fixtures this call reaches for."""

        tool = step["tool"]
        if tool in SOCIAL_TOOL_NAMES or tool in OBSERVATION_TOOL_NAMES:
            return []
        # give_space is the actor LEAVING; two agents swapping places at one
        # fixture is the handoff working, not a collision.
        if tool in GIVE_SPACE_TOOL_NAMES:
            return []
        objects = set(self.validator.initial_state.get("objects") or {})
        args = step.get("args") or {}
        found: list[str] = []
        for name in DEPENDENCY_ARG_NAMES:
            value = args.get(name)
            if isinstance(value, str) and value in objects:
                found.append(value)
        for name in FIXTURE_ARG_NAMES:
            value = args.get(name)
            if isinstance(value, str) and self._exclusive(value):
                found.append(value)
        return found

    def _check_simultaneous_use(
        self,
        steps: Sequence[dict[str, Any]],
        instant: list[tuple[str, int]],
        clock: float,
        seen: set[tuple[str, ...]],
        result: Replay,
    ) -> None:
        """Invariant 1: two agents must not reach for one thing at one instant."""

        if len(instant) < 2:
            return
        by_resource: dict[str, list[tuple[str, int]]] = {}
        for agent_id, index in instant:
            for resource in self._named_resources(steps[index]):
                by_resource.setdefault(resource, []).append((agent_id, index))
        for resource, users in by_resource.items():
            if len(users) < 2:
                continue
            key = ("use", resource, *sorted(a for a, _ in users))
            if key in seen:
                continue
            seen.add(key)
            detail = ", ".join(
                f"{a} {steps[i]['tool']} (step {steps[i]['step']})" for a, i in users
            )
            result.conflicts.append(
                f"  t={clock:g}: {detail} -- both reach for {resource!r} at the "
                f"same instant. One of them must wait_for_signal on it and the "
                f"other must release it."
            )

    def _let_go_of(
        self, step: dict[str, Any], previous: dict[str, Any] | None
    ) -> list[str]:
        """Resources this call actually hands back: a fixture left, a thing put down."""

        args = step.get("args") or {}
        if step["tool"] in GIVE_SPACE_TOOL_NAMES:
            return [str(args.get("fixture_id"))]
        if step["tool"] in RELEASE_TOOL_NAMES:
            objects = set(self.validator.initial_state.get("objects") or {})
            return [
                str(value)
                for value in args.values()
                if isinstance(value, str) and value in objects
            ]
        return []

    def _check_ask(
        self,
        step: dict[str, Any],
        agent_id: str,
        previous: dict[str, Any] | None,
        result: Replay,
    ) -> None:
        """A wait is announced before it is taken.

        The partner cannot see that an agent has stopped -- a blocked agent is
        simply absent from the plan. The ask is what tells it a handover is
        owed, so the wait goes IMMEDIATELY after it: any call in between is the
        agent doing other work while claiming it cannot proceed.
        """

        holder = (step.get("args") or {}).get("from")
        prefix = f"  step {step['step']}: {agent_id} waits on"
        about = str((step.get("args") or {}).get("about"))
        if not self._known_id(about):
            # Coordinating on a fiction. The model names the MOMENT it is
            # waiting for -- "mug_placed", "counter_free" -- instead of the
            # thing, and since a consistent fiction matches itself, a replay
            # that only pairs waits with releases would let it through.
            result.protocol.append(
                f"{prefix} {about!r}, which is not an object or fixture in "
                f"initial_state. `about` names a THING that exists, never an "
                f"event or a state."
            )
            return
        if previous is None or previous["tool"] != "communicate":
            was = "nothing" if previous is None else previous["tool"]
            result.protocol.append(
                f"{prefix} {about!r} but its previous call was {was}, not the "
                f"request. Ask {holder} for it on the tick before you block, "
                f"so the other agent knows a handover is owed."
            )
            return
        if (previous.get("args") or {}).get("to") != holder:
            result.protocol.append(
                f"{prefix} {about!r} on {holder}, but the message right before "
                f"it was addressed to "
                f"{(previous.get('args') or {}).get('to')!r}. Ask the agent you "
                f"are about to wait on."
            )

    # NOTE: an earlier rule here rejected a wait whose holder had already let
    # the resource go before the wait began. Measured against real tick output
    # it was wrong 6 times in 7: the holder vacates on the SAME instant the
    # waiter blocks, which is not a mistake but the tightest correct handover
    # there is -- the release still lands afterwards and still wakes the
    # waiter. The one genuinely-early case was equally harmless. What the rule
    # was reaching for is a wait that cannot be discharged, and that is already
    # caught exactly: by the inert check when the release lands on the wait's
    # own instant, and by the deadlock check when it lands before.

    def _check_release(
        self,
        step: dict[str, Any],
        agent_id: str,
        released: str,
        state: Any,
        previous: dict[str, Any] | None,
        clock: float,
        result: Replay,
    ) -> None:
        """A release is a report of a departure, so the departure must be real.

        `releases` is otherwise a pure speech act: nothing stops an agent
        handing over a fixture it is still standing at. The waiter then wakes
        and walks into an occupied space, and whether that shows up as a
        collision depends entirely on which duration model is running -- see
        `concurrent_fsm_demo.py` scenarios 8 and 9, the same defect with
        opposite verdicts. These three rules need no clock.
        """

        agent_state = state.agents.get(agent_id)
        if agent_state is None:
            return
        prefix = f"  step {step['step']}: {agent_id} releases {released!r}"

        if not self._known_id(released):
            result.protocol.append(
                f"{prefix}, which is not an object or fixture in initial_state. "
                f"`releases` names a THING that exists, never an event or a state."
            )
            return

        # 1. Still holding it.
        if agent_state.held_object == released:
            result.protocol.append(
                f"{prefix} while still holding it. Put it down first."
            )
            return

        # 2. Still standing at it.
        if self._exclusive(released) and agent_state.location == released:
            result.protocol.append(
                f"{prefix} while still standing at it. Call "
                f"give_space(fixture_id={released!r}) first -- announcing a "
                f"handover is not performing one."
            )
            return

        # 3. Released, but not as the report of the departure.
        #
        # Only things the agent took POSSESSION of need a departure to report:
        # a fixture it stood at, an object it held. A resource merely used --
        # picking a drumstick out of a pan, say -- is handed back the moment
        # that use ends, because there is nothing to put down or step off.
        # Demanding a give_space there flagged 31 correct handovers.
        if previous is None or released in self._named_resources(previous):
            return
        args = previous.get("args") or {}
        departed = (
            # Stepping clear of ANY fixture counts. Placing an object down and
            # then giving space before announcing is not a gap in the handover,
            # it IS the handover -- the agent finishes withdrawing and only then
            # says so. Requiring the released id to match rejected all 5 of the
            # real cases, every one of them correct.
            previous["tool"] in GIVE_SPACE_TOOL_NAMES
        ) or (
            previous["tool"] in NAVIGATION_TOOL_NAMES
            and str(args.get("fixture_id")) != released
        ) or (
            previous["tool"] in RELEASE_TOOL_NAMES
            and released in {str(value) for value in args.values()}
        )
        if not departed:
            result.protocol.append(
                f"{prefix}, but its previous call was "
                f"{previous['tool']}, not the handover. The release must be "
                f"the very next thing the agent does after letting go: "
                f"anything in between is time the waiter is blocked on a "
                f"resource that is already free."
            )

    def _known_id(self, symbol: str) -> bool:
        """True when this names something that actually exists in the scene."""

        initial = self.validator.initial_state
        return symbol in (initial.get("objects") or {}) or symbol in (
            initial.get("fixtures") or {}
        )

    def _occupancy(self, state: Any) -> dict[str, list[str]]:
        """Which agents are standing at each exclusive fixture right now."""

        occupants: dict[str, list[str]] = {}
        for agent_id, agent_state in state.agents.items():
            location = agent_state.location
            if location and self._exclusive(location):
                occupants.setdefault(location, []).append(agent_id)
        return occupants

    def _check_co_location(
        self,
        state: Any,
        acting: Sequence[str],
        clock: float,
        seen: set[tuple[str, ...]],
        grace: set[tuple[str, ...]],
        result: Replay,
    ) -> None:
        """Invariant 2: an exclusive fixture holds at most one agent.

        Occupancy persists between calls -- an agent that navigated to the
        cabinet is still standing there until it leaves. Checking only the
        calls, as every earlier pass did, cannot see two agents that arrived at
        different instants and never left.
        """

        occupancy = self._occupancy(state)
        live = {
            ("at", fixture_id, *sorted(here))
            for fixture_id, here in occupancy.items()
            if len(here) > 1
        }
        grace.intersection_update(live)  # separated at least once -> no excuse left

        for fixture_id, agents_here in occupancy.items():
            if len(agents_here) < 2 or not any(a in acting for a in agents_here):
                continue
            key = ("at", fixture_id, *sorted(agents_here))
            if key in seen or key in grace:
                continue
            seen.add(key)
            result.conflicts.append(
                f"  t={clock:g}: {' and '.join(sorted(agents_here))} are both at "
                f"{fixture_id!r}, which only one agent fits at. The one arriving "
                f"second must wait until the first calls give_space and releases it."
            )

    # -- drop-in validation ------------------------------------------------

    def validate(
        self,
        candidate: dict[str, Any],
        *,
        models: Sequence[str] = DURATION_MODELS,
    ) -> dict[str, Any]:
        """Validates under every duration model; returns the FSM result shape.

        A plan that survives only one timing regime survived by luck, so all of
        them have to agree. The reported final state comes from LOCK_STEP, the
        regime with exactly one schedule.
        """

        replays = {model: self.replay(candidate, model=model) for model in models}
        primary = replays.get(LOCK_STEP) or next(iter(replays.values()))

        for model in models:
            run = replays[model]
            if run.step_error is not None:
                raise run.step_error
            if run.blocked:
                raise DeadlockSemanticValidationError(
                    f"Under the {model} schedule the trajectory deadlocks:\n"
                    + "\n".join(run.conflicts),
                    details={"model": model, "blocked": run.blocked},
                )
            if run.conflicts:
                raise ResourceConflictSemanticValidationError(
                    f"Under the {model} schedule the agents collide:\n"
                    + "\n".join(run.conflicts),
                    details={"model": model, "conflicts": run.conflicts},
                )
            if run.protocol:
                raise WaitSignalSemanticValidationError(
                    "The handover protocol is not followed:\n"
                    + "\n".join(run.protocol),
                    details={"model": model, "protocol": run.protocol},
                )
            if run.post_goal:
                raise PostGoalActionSemanticValidationError(
                    f"Work is started after the {self.composite_task} goal is "
                    f"already satisfied ({model} schedule):\n"
                    + "\n".join(run.post_goal),
                    details={"model": model, "composite_task": self.composite_task},
                )
            if not run.first_action:
                raise MissingTaskActionSemanticValidationError(
                    "Trajectory did not contain any task action steps."
                )
            if run.goal_at is None:
                raise UnsatisfiedGoalSemanticValidationError(
                    f"Trajectory never satisfied the {self.composite_task} goal "
                    f"state under the {model} schedule."
                )

        validator = self.validator
        agents = validator._normalize_agents(candidate.get("agents"))
        steps = validator._normalize_steps(candidate.get("steps"))
        return {
            "is_valid": True,
            "checks": list(validator._all_checks) + ["concurrent_replay"],
            "final_state": validator._build_final_state(primary.runtime_state),
            "normalized_candidate": {"agents": agents, "steps": steps},
            "signature": validator.trajectory_signature(
                {"agents": agents, "steps": steps}
            ),
            "schedule": {
                model: {
                    "makespan": round(run.makespan, 3),
                    "idle": run.idle,
                    "goal_at": run.goal_at,
                }
                for model, run in replays.items()
            },
        }

    # -- inspection --------------------------------------------------------

    def render(
        self,
        candidate: dict[str, Any],
        *,
        model: str = LOCK_STEP,
        width: int = 44,
    ) -> str:
        """A two-column view of the run: one row per instant, one column per agent."""

        run = self.replay(candidate, model=model, stop_on_step_error=False)
        steps = self.validator._normalize_steps(candidate.get("steps"))
        rows: dict[float, dict[str, str]] = {}
        for event in run.events:
            args = steps[event.index].get("args") or {}
            detail = ", ".join(
                f"{k}={v}" for k, v in args.items() if not isinstance(v, (dict, list))
            )
            rows.setdefault(event.start, {})[event.agent] = f"{event.tool}({detail})"

        header = f"{'t':>6}  " + "  ".join(a.ljust(width) for a in self.agent_ids)
        lines = [header, "-" * len(header)]
        for at in sorted(rows):
            cells = [rows[at].get(a, "")[:width].ljust(width) for a in self.agent_ids]
            lines.append(f"{at:>6g}  " + "  ".join(cells))
        return "\n".join(lines)
