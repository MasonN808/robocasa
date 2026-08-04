"""Closed-loop (live-sim) evaluation of tool-calling VLMs.

Implements training/bc_task_vlm/plans/live_sim_eval_plan.md: the model controls
both agents live in the MuJoCo sim under the v2 self-scheduled contract (the
model emits the acting agent and may call task_complete), success is judged by
the task env's own ``_check_success`` plus the symbolic FSM goal checker, and
visuals are rendered from current sim state according to the selected
observation contract.

Backends:
  oracle      replay the expert trajectory's own steps (verification gate:
              expect ~100% success; no model involved)
  degenerate  emit garbage every turn (verification gate: expect 0% success,
              termination by rejections/budget, no crashes)
  hf          a local HF VLM (base weights or base+LoRA adapter)
  vllm        a localhost vLLM OpenAI-compatible server
  gemini      Vertex Gemini with native function calling

Example (oracle smoke):
  python -m training.bc_task_vlm.live_sim_eval --backend oracle \
    --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_tasks.json \
    --dataset-root training/bc_task_vlm/eval_data_subset \
    --output-dir training/bc_task_vlm/eval_runs/live_sim_oracle__heldout_tasks \
    --max-trajectories 2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import re
from typing import Any
from urllib import error as urllib_error, request as urllib_request

from training.bc_task_vlm.prompting import (
    build_messages,
    build_user_prompt,
)
from training.bc_task_vlm.schema_utils import (
    augment_tool_specs_with_get_image,
    TASK_COMPLETE_TOOL_NAME,
    augment_tool_specs_for_agent_prediction,
    compact_json_dumps,
    apply_eval_fixture_aliases,
)
from training.bc_task_vlm.task_registry import AGENT_IDS, get_task_metadata
from training.bc_task_vlm.tool_calling import (
    build_tool_schemas,
    parse_first_qwen_tool_call,
    pop_agent_argument,
    tool_call_to_single_step_payload,
)

# Expert view-selection rule, measured over 162 trajectories (~3,300 calls):
# views deterministically match the NEXT action (scout set before navigate,
# wrist set before manipulation, overhead at the episode opening). The next
# action is unknown before generation, so the first pass renders the overhead
# set and the second pass re-renders the canonical set for the predicted
# action (see --two-pass-views).
OVERHEAD_VIEWS = ("top_view", "room_view", "map")
SCOUT_VIEWS = ("agentview_center", "agentview_left", "agentview_right")
WRIST_VIEWS = ("wrist", "agentview_center")
# After the two required opening communications, the routing pass must not use
# a view family that was paired exclusively with one action family during SFT.
# Every overhead-only exp52 example targets ``communicate``. A fixed mixed
# bundle gives the router global and agent-centric evidence without conditioning
# its prediction on the eventual tool choice. Canonical refinement remains
# in-distribution for the selected tool.
ROUTER_VIEWS = ("top_view", "wrist", "agentview_center")
OPENING_COMMUNICATION_STEP_INDICES = frozenset({2, 3})
NAVIGATE_TOOLS = {"navigate_to_fixture"}
GET_IMAGE_OBSERVATION_MODE_NEXT_TURN = "next_turn"
GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE = "causal_cache"
GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE_CENTRALIZED = (
    "causal_cache_centralized"
)
GET_IMAGE_OBSERVATION_MODES = (
    GET_IMAGE_OBSERVATION_MODE_NEXT_TURN,
    GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE,
    GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE_CENTRALIZED,
)
_NUMBERED_OBJECT_RE = re.compile(r"^obj_(\d+)$")


def canonical_views_for_tool(tool_name: str) -> tuple[str, ...]:
    if tool_name in NAVIGATE_TOOLS:
        return SCOUT_VIEWS
    if tool_name in {"communicate", TASK_COMPLETE_TOOL_NAME}:
        return OVERHEAD_VIEWS
    return WRIST_VIEWS


def _global_step_index_for_turn(turn_index: int) -> int:
    """Map a live action turn to the global indices used during SFT.

    Demonstrations begin with one observation for each agent (global steps 0
    and 1), followed by the two opening communication actions (steps 2 and 3).
    Thereafter the live loop acquires a fresh observation before every action,
    so each additional decision advances the global index by two.
    """

    if turn_index < 0:
        raise ValueError("turn_index must be non-negative")
    if turn_index < 2:
        return turn_index + 2
    return 2 * turn_index + 1


def _step_index_for_turn(
    turn_index: int, *, active_observation: bool
) -> int:
    """Return the SFT-compatible index for the selected observation mode."""

    if active_observation:
        if turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        return turn_index
    return _global_step_index_for_turn(turn_index)


def _cached_observation_for_agent(
    cached_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    agent_id: str | None,
) -> tuple[list[str] | None, tuple[str, ...]]:
    """Return only the selected agent's prefix-derived visual cache."""

    if agent_id is None:
        return None, ()
    cached = cached_observations_by_agent.get(agent_id)
    if cached is None:
        return None, ()
    image_paths, view_names = cached
    return list(image_paths), tuple(view_names)


def _record_agent_observation(
    cached_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    *,
    agent_id: str,
    image_paths: list[str],
    view_names: list[str],
) -> None:
    """Update one logical agent cache without granting access to the other."""

    cached_observations_by_agent[agent_id] = (
        list(image_paths),
        list(view_names),
    )


def _uses_causal_cached_observations(observation_mode: str) -> bool:
    """Return whether a mode uses prefix-derived per-agent image caches."""

    return observation_mode in {
        GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE,
        GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE_CENTRALIZED,
    }


def _causal_cache_rejection_reason(
    *,
    tool_name: str,
    proposed_agent: str,
    active_observation_agent: str | None,
    allow_cross_owner_communication: bool = False,
) -> str | None:
    """Enforce the selected causal-cache action-ownership contract."""

    if tool_name in {"get_image", TASK_COMPLETE_TOOL_NAME}:
        return None
    if tool_name == "communicate" and allow_cross_owner_communication:
        return None
    if active_observation_agent is None:
        if tool_name == "communicate":
            return None
        return "physical action requires a preceding agent-owned get_image"
    if proposed_agent != active_observation_agent:
        return (
            f"active observation belongs to {active_observation_agent}, "
            f"not {proposed_agent}"
        )
    return None


def _check_pruned_organize_condiments_success(
    env,
    *,
    object_utils=None,
) -> bool:
    """Evaluate OrganizeCondiments when trajectory pruning omitted distractors.

    The upstream task's native checker unconditionally indexes ``distractor``.
    Generated trajectories intentionally omit that non-goal object, so the
    trajectory-compatible native condition retains the three condiment and
    gripper-clearance requirements while omitting only the impossible
    distractor-on-counter and distractor-clearance clauses.
    """

    required_objects = ("condiment1", "condiment2", "condiment3")
    objects = getattr(env, "objects", {})
    missing = [name for name in required_objects if name not in objects]
    if missing:
        raise KeyError(
            "missing goal-relevant OrganizeCondiments object(s): "
            + ", ".join(missing)
        )
    if object_utils is None:
        import robocasa.utils.object_utils as object_utils

    return all(
        object_utils.obj_inside_of(env, name, env.cab)
        for name in required_objects
    ) and all(
        object_utils.gripper_obj_far(env, name)
        for name in required_objects
    )


# ---------------------------------------------------------------------------
# Partial-observability runtime: per-agent private state + concurrent scheduler
# ---------------------------------------------------------------------------

WAIT_TOOL_NAME = "wait_for_signal"
REJECTION_MODE_SILENT_RETRY = "silent-retry"
REJECTION_MODE_REPORT_FAILED = "report-failed"
RESOURCE_LOCKS_NONE = "none"
RESOURCE_LOCKS_OBJECTS = "objects"
RESOURCE_LOCKS_FIXTURES = "fixtures"

# Tools that mutate nothing and therefore claim no shared resource.
_NON_CLAIMING_TOOLS = frozenset({"communicate", "get_image", TASK_COMPLETE_TOOL_NAME})
# Arg names that name a fixture vs a movable object, used to derive claims
# without inventing a new schema.
# Movable objects an agent would be holding.
_OBJECT_ARG_NAMES = ("object_id", "reference_object_id")
# Places an agent must occupy or reach into. `source_id` belongs here, not with
# objects: pick_up_object(source_id=...) names the fixture being reached into
# (fridge/cabinet/counter), so it must collide with a navigate to that fixture.
_LOCATION_ARG_NAMES = (
    "fixture_id",
    "source_id",
    "target_id",
    "receptacle_id",
    "reference_fixture_id",
)


def resource_claims(step: dict[str, Any], *, mode: str) -> frozenset[str]:
    """Resources a proposed tool call would hold, derived from its arguments.

    `communicate`/`get_image` are read-only and claim nothing, so two agents may
    always look and talk simultaneously.
    """

    if mode == RESOURCE_LOCKS_NONE:
        return frozenset()
    tool = step.get("tool")
    if tool in _NON_CLAIMING_TOOLS:
        return frozenset()
    args = step.get("args") or {}
    # Flat namespace: the same entity named through different argument slots
    # must collide, so claims are bare IDs rather than typed keys.
    claims: set[str] = {
        str(args[name]) for name in _OBJECT_ARG_NAMES if args.get(name)
    }
    if mode == RESOURCE_LOCKS_FIXTURES:
        claims.update(
            str(args[name]) for name in _LOCATION_ARG_NAMES if args.get(name)
        )
    return frozenset(claims)


def _sim_clock(session: "SimSession") -> float | None:
    """MuJoCo simulation time in seconds, or None if unavailable.

    ToolResult carries no timing, so the only real measure of how long a tool
    took is the simulator's own clock across the call.
    """

    try:
        return float(session.executor.env.sim.data.time)
    except Exception:
        return None


# Schedule-robustness probe. Because the executor teleports, measured sim-steps
# are ~0 and the floors below ARE the duration model -- so --duration-multiplier
# cannot perturb a schedule, and a null result from varying it means nothing.
# Jitter scales each completed call instead, which is the only knob that changes
# who wins a race. Off by default; a run that sets it is asking whether the
# plans survive a different interleaving, not measuring their quality.
_DURATION_JITTER = 0.0
_DURATION_RNG: random.Random | None = None
# Lock-step. Every call costs one tick, so both agents advance together and the
# ONLY thing that can order them is a wait and its release -- no timing is left
# to rescue a plan that forgot one. This REMOVES the timing dimension rather
# than testing it: there is exactly one schedule, so a plan that works is
# correct rather than lucky, and wait necessity becomes decidable by ablation.
# Two costs, both deliberate: a communicate costs the same tick as a navigate,
# so makespan degenerates to step count; and intra-tick order still exists (two
# sequential executor calls), so the contention model stays load-bearing.
_UNIFORM_DURATIONS = False


def _tool_duration(
    tool_name: str, *, sim_steps: float | None, multiplier: float
) -> float:
    """Virtual-clock duration for a completed call.

    Measured executor sim-steps scaled by `multiplier`, with per-class floors.
    The floors matter because the executor teleports: if measured costs come out
    uniform, equal durations from a common start would keep both agents in
    lockstep forever, which is precisely the artificial regime the virtual clock
    exists to avoid -- unless lock-step is what you asked for.
    """

    if _UNIFORM_DURATIONS:
        return 1.0
    if tool_name in ("communicate", "get_image"):
        floor = 0.25
    elif tool_name == "navigate_to_fixture":
        floor = 4.0
    else:
        floor = 2.0
    measured = 0.0 if not sim_steps else float(sim_steps) * multiplier
    duration = max(floor, measured)
    if _DURATION_JITTER and _DURATION_RNG is not None:
        duration *= _DURATION_RNG.uniform(
            max(0.05, 1.0 - _DURATION_JITTER), 1.0 + _DURATION_JITTER
        )
    return duration


class AgentRuntime:
    """Private state for one logical agent under partial observability."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.private_history: list[dict[str, Any]] = []
        self.pending_obs: tuple[list[str], list[str]] | None = None
        self.ready_at: float = 0.0
        self.local_turn = 0
        self.silent_retries = 0
        self.active_s = 0.0
        self.last_proposal_key: str | None = None
        self.repeated_proposals = 0
        # wait_for_signal state. `waiting_for` records the DECLARED intent
        # ({"from","about"}); the harness never checks `about` against message
        # content -- any delivered message wakes the agent and the model decides
        # whether it was the one it needed. That is deliberate: a harness that
        # guaranteed relevance would leave nothing to learn and nothing to
        # measure, whereas this makes comprehension an observable decision.
        self.waiting_for: dict[str, Any] | None = None
        self.consecutive_waits = 0

    def deliver(self, step: dict[str, Any], clock: float | None = None) -> None:
        """Appends to this agent's private history (own act or delivered msg)."""

        self.private_history.append(deepcopy(step))
        # ANY inbound message from another agent wakes a waiter. Relevance is
        # the model's judgement, not the environment's.
        if (
            self.waiting_for is not None
            and step.get("tool") == "communicate"
            and step.get("agent") != self.agent_id
        ):
            self.waiting_for = None
            # Resume at the moment of release. `ready_at` still holds the stale
            # time the wait WOULD have expired, and the scheduler's catch-up
            # cannot repair it: a freshly woken agent is exactly the one that
            # sets the next clock minimum, so `ready_at < clock` is never true
            # for it. It then acts in the past -- traj_000004 of
            # arrange_bread_bowl released at t=10.5 and resumed at t=9.25.
            if clock is not None and self.ready_at < clock:
                self.ready_at = clock

    def take_observation(self) -> tuple[list[str], list[str]]:
        """Consume-once: hand over the pending observation and clear it."""

        if self.pending_obs is None:
            return [], []
        image_paths, view_names = self.pending_obs
        self.pending_obs = None
        return list(image_paths), list(view_names)


def _proposal_key(step: dict[str, Any]) -> str:
    return json.dumps(
        {"tool": step.get("tool"), "args": step.get("args") or {}}, sort_keys=True
    )


def _tool_invalidates_active_observation(tool_name: str) -> bool:
    """Physical tools mutate the scene or robot pose; communication does not."""

    return tool_name not in {
        "communicate",
        "get_image",
        TASK_COMPLETE_TOOL_NAME,
    }


def _routing_views_for_step(step_index: int) -> tuple[str, ...]:
    """Use trained opening views, then a target-independent routing bundle."""

    if step_index in OPENING_COMMUNICATION_STEP_INDICES:
        return OVERHEAD_VIEWS
    return ROUTER_VIEWS


# ---------------------------------------------------------------------------
# FSM mirror: incremental legality + goal checking over symbolic state
# ---------------------------------------------------------------------------


def _canonical_symbolic_initial_state(*, task_spec, trajectory: dict[str, Any]):
    """Normalize legacy trajectory symbols without introducing simulator IDs."""

    initial_state = deepcopy(trajectory.get("initial_state") or task_spec.initial_state)
    source_grounding = (
        (trajectory.get("grounding_map") or {}).get("symbols") or {}
    )
    target_grounding = (task_spec.grounding or {}).get("symbols") or {}
    legacy_aliases = (task_spec.grounding or {}).get("legacy_symbol_aliases") or {}
    symbol_map: dict[str, str] = {}

    def type_values(
        *,
        state_entry: dict[str, Any],
        grounding_entry: dict[str, Any],
        entity_type: str,
    ) -> set[str]:
        type_key = "object_type" if entity_type == "object" else "fixture_type"
        values = {
            str(value)
            for value in (state_entry.get(type_key), grounding_entry.get(type_key))
            if value
        }
        if entity_type == "fixture":
            values.update(
                str(value)
                for value in grounding_entry.get("preferred_fixture_types", ())
                if value
            )
        return values

    for state_key, entity_type in (("objects", "object"), ("fixtures", "fixture")):
        source_entities = initial_state.get(state_key) or {}
        target_entities = task_spec.initial_state.get(state_key) or {}
        used_targets: set[str] = set()

        for source_symbol in source_entities:
            target_symbol = source_symbol
            if target_symbol not in target_entities:
                target_symbol = legacy_aliases.get(source_symbol)
            if target_symbol in target_entities:
                symbol_map[source_symbol] = target_symbol
                used_targets.add(target_symbol)

        for source_symbol, source_entry in source_entities.items():
            if source_symbol in symbol_map:
                continue
            source_descriptor = source_grounding.get(source_symbol) or {}
            source_types = type_values(
                state_entry=source_entry,
                grounding_entry=source_descriptor,
                entity_type=entity_type,
            )
            candidates = []
            for target_symbol, target_entry in target_entities.items():
                if target_symbol in used_targets:
                    continue
                target_descriptor = target_grounding.get(target_symbol) or {}
                target_types = type_values(
                    state_entry=target_entry,
                    grounding_entry=target_descriptor,
                    entity_type=entity_type,
                )
                if source_types & target_types:
                    candidates.append(target_symbol)

            source_index = source_descriptor.get("index")
            if len(candidates) > 1 and source_index is not None:
                indexed = [
                    candidate
                    for candidate in candidates
                    if (target_grounding.get(candidate) or {}).get("index")
                    == source_index
                ]
                if len(indexed) == 1:
                    candidates = indexed
            if len(candidates) == 1:
                symbol_map[source_symbol] = candidates[0]
                used_targets.add(candidates[0])

    def rewrite(value):
        if isinstance(value, dict):
            rewritten = {
                symbol_map.get(key, key): rewrite(item)
                for key, item in value.items()
            }
            if len(rewritten) != len(value):
                raise ValueError("symbol normalization produced duplicate IDs")
            return rewritten
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return symbol_map.get(value, value)
        return value

    # The SAME symbol map must be applied to the trajectory's steps, not only
    # to initial_state. original_trajectory.json (what both training and
    # live-sim read) may name fixtures by role -- hot_dog_setup uses
    # `sausage_source_fixture`, `serving_surface`, `bun_source_fixture` -- and
    # leaving those unrewritten made every step referencing them illegal, since
    # the task's allowlist is expressed in canonical names.
    # Apply the SAME symbol map to the trajectory's steps, not only to
    # initial_state. original_trajectory.json (read by BOTH training and
    # live-sim) may name entities in a vocabulary the task allowlist does not
    # use -- hot_dog_setup says `bun` and `bun_source_fixture` where the spec
    # says `hotdog_bun` and `counter` -- and leaving steps unrewritten made
    # every such step illegal.
    #
    # Caveat worth knowing: this assumes the spec name is also resolvable in
    # the simulator. Where it is not, the call becomes legal-but-unexecutable
    # rather than rejected. prepare_cheese_station is the one known case
    # (spec `lettuce_bowl` vs scene `salad_bowl`); that is a spec/data naming
    # defect to reconcile, not a reason to skip the rewrite.
    trajectory["steps"] = rewrite(trajectory.get("steps") or [])
    # The TrajectoryAdapter is built from the ORIGINAL trajectory, so its alias
    # map keys on the pre-canonical names (bun_source_fixture). Rewriting steps
    # in place afterwards leaves the adapter unable to resolve the new names,
    # which is why a pickup worked only when a navigate happened to bind the
    # fixture first (6/6 with a prior navigate passed, 4/4 without failed).
    # Stash the map so callers can register the canonical aliases too.
    trajectory["_canonical_symbol_map"] = dict(symbol_map)
    return rewrite(initial_state)


def _register_canonical_fixture_aliases(adapter, trajectory: dict[str, Any]) -> None:
    """Teaches the adapter the canonical names the steps were rewritten to.

    Without this, `counter` (canonical) has no alias while `bun_source_fixture`
    (original) does, and the executor raises
    `Unknown support fixture/site 'counter'` unless some earlier navigate
    happened to bind it.
    """

    symbol_map = trajectory.get("_canonical_symbol_map") or {}
    aliases = getattr(adapter, "_fixture_aliases", None)
    if not symbol_map or aliases is None:
        return
    for original, canonical in symbol_map.items():
        if canonical in aliases:
            continue
        concrete = aliases.get(original)
        if concrete:
            aliases[canonical] = concrete


class FsmMirror:
    """Replays the FiniteStateTaskValidator loop body one step at a time.

    Mirrors fsm.py::FiniteStateTaskValidator.validate's per-step behavior
    (communicate gate, allowed tools, preconditions, effects, goal check)
    against the TRAJECTORY's sampled initial state instead of the spec's
    template. Legality failures are returned, never raised.
    """

    def __init__(self, *, composite_task: str, trajectory: dict[str, Any]) -> None:
        from data_generation.task_level.tasks.specs import load_task_spec
        from data_generation.task_level.tasks.specs.runtime import (
            SpecDrivenTaskValidator,
        )

        spec = load_task_spec(composite_task)
        self.validator = SpecDrivenTaskValidator(spec)
        initial_state = _canonical_symbolic_initial_state(
            task_spec=spec,
            trajectory=trajectory,
        )
        if initial_state:
            self.validator.initial_state = initial_state
        agents = self.validator._normalize_agents(trajectory.get("agents"))
        self.runtime_state = self.validator._build_runtime_state(agents)
        self.goal_satisfied = self.validator.is_goal_state_satisfied(
            self.runtime_state
        )

    def partial_goal_fraction(self) -> float:
        """Fraction of top-level spec goal conditions currently satisfied."""

        spec = self.validator._task_spec
        conditions = spec.goal_conditions
        if not conditions:
            return float(self.goal_satisfied)
        satisfied = 0
        for condition in conditions:
            original = self.validator._task_spec
            try:
                self.validator._task_spec = replace(
                    original, goal_conditions=(condition,)
                )
                satisfied += int(
                    self.validator.is_goal_state_satisfied(self.runtime_state)
                )
            finally:
                self.validator._task_spec = original
        return satisfied / len(conditions)

    def step(self, step: dict[str, Any]) -> tuple[bool, str | None]:
        """Applies one symbolic step; returns (legal, reason_if_not)."""

        validator = self.validator
        state = self.runtime_state
        try:
            if step["tool"] == "communicate":
                validator._validate_communicate_step(step)
                state.communicated_agents.add(step["agent"])
                validator.apply_task_effects(step, state)
            else:
                if state.communicated_agents != set(validator.agent_ids):
                    return False, (
                        "Both agents must communicate before the first task "
                        "action."
                    )
                if step["tool"] not in validator.allowed_tool_specs:
                    return False, f"Tool {step['tool']} is not allowed here."
                validator._validate_task_local_symbolic_constraints(step)
                validator._validate_generic_transition(step, state)
                validator.validate_task_preconditions(step, state)
                validator._apply_generic_effects(step, state)
                validator.apply_task_effects(step, state)
            self.goal_satisfied = validator.is_goal_state_satisfied(state)
            return True, None
        except Exception as exc:  # validator raises typed validation errors
            return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Sim session: one env per task, rewound per trajectory
# ---------------------------------------------------------------------------


class SimSession:
    def __init__(
        self,
        *,
        composite_task: str,
        sample_trajectory: dict[str, Any],
        layout: int,
        style: int,
        seed: int,
        gl_backend: str,
        render_size: int,
        map_dpi: int = 300,
        map_renderer: str = "legacy",
    ) -> None:
        from robocasa.utils.sim_tool_executor import SimToolExecutor
        from robocasa.utils.trajectory_pruning import (
            build_trajectory_pruning_config,
        )

        self.composite_task = composite_task
        pruning = build_trajectory_pruning_config(sample_trajectory, layout=layout)
        kwargs: dict[str, Any] = {}
        if any(pruning.get(key) for key in (
            "update_fxtr_cfg_dict",
            "trajectory_object_names",
            "trajectory_object_types",
            "trajectory_object_specs",
        )):
            kwargs.update(
                update_fxtr_cfg_dict=pruning.get("update_fxtr_cfg_dict"),
                trajectory_object_names=pruning.get("trajectory_object_names"),
                trajectory_object_types=pruning.get("trajectory_object_types"),
                trajectory_object_specs=pruning.get("trajectory_object_specs"),
            )
        self.executor = SimToolExecutor(
            task_name=composite_task,
            robots=2,
            layout=layout,
            style=style,
            seed=seed,
            gl_backend=gl_backend,
            render_width=render_size,
            render_height=render_size,
            map_dpi=map_dpi,
            map_renderer=map_renderer,
            robot_spawn="trajectory",
            **kwargs,
        )
        self._synchronize_pruned_task_cardinality(composite_task)

    def _synchronize_pruned_task_cardinality(self, composite_task: str) -> None:
        """Keep native task counters consistent with trajectory pruning.

        Some RoboCasa tasks sample a variable object count while fixtures are
        being initialized.  Trajectory pruning changes RNG consumption and can
        subsequently remove objects that are absent from the recorded initial
        state, leaving the task's cached count larger than ``env.objects``.
        BeverageOrganization's native checker indexes every object up to that
        cached count, so the mismatch otherwise produces ``KeyError: obj_N``.

        Keep this compatibility fix deliberately task-specific: similarly named
        counters in other tasks need not describe contiguous ``obj_N`` objects.
        """

        if composite_task != "BeverageOrganization":
            return
        indices = sorted(
            int(match.group(1))
            for name in self.executor.env.objects
            if (match := _NUMBERED_OBJECT_RE.fullmatch(str(name)))
        )
        if indices == list(range(len(indices))) and indices:
            self.executor.env.num_bev = len(indices)

    def start_trajectory(self, trajectory: dict[str, Any]):
        from robocasa.utils.trajectory_adapter import TrajectoryAdapter

        self.executor.restore_baseline_state()
        adapter = TrajectoryAdapter(
            executor=self.executor,
            allow_approximate_ids=True,
        )
        adapted = adapter.adapt(trajectory)
        self.executor.load_initial_state(adapted["initial_state"])
        # load_initial_state may prune a sampled variable-cardinality object
        # set, so counters derived during env construction must be repaired
        # after loading each trajectory rather than only once in __init__.
        self._synchronize_pruned_task_cardinality(self.composite_task)
        return adapter, adapted

    def native_success(self) -> tuple[bool | None, str | None]:
        """(success, error). Unwraps common env wrappers to find _check_success."""

        env = self.executor.env
        for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
            if candidate is None or not hasattr(candidate, "_check_success"):
                continue
            try:
                if (
                    self.composite_task == "OrganizeCondiments"
                    and "distractor" not in getattr(candidate, "objects", {})
                ):
                    return (
                        bool(
                            _check_pruned_organize_condiments_success(
                                candidate
                            )
                        ),
                        None,
                    )
                return bool(candidate._check_success()), None
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, f"no _check_success on {type(env).__name__}"

    def render_views(
        self,
        views: tuple[str, ...],
        *,
        agent_id: str,
        out_dir: Path,
        tag: str,
    ) -> tuple[list[str], list[str]]:
        """Renders the requested views to files; returns (paths, view_names)."""

        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            str(out_dir / f"{tag}_{view}.jpg") for view in views
        ]
        result = self.executor.get_image(
            views=list(views),
            image_paths=image_paths,
            agent_id=agent_id,
        )
        produced = result.details.get("image_paths") or image_paths
        return [str(p) for p in produced], list(views)

    def close(self) -> None:
        try:
            self.executor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class OraclePolicy:
    """Replays the expert trajectory's own action steps."""

    def __init__(self, raw_steps: list[dict[str, Any]]) -> None:
        self._steps = [s for s in raw_steps if s.get("tool") != "get_image"]
        self._cursor = 0

    def next_step(self, **_ignored) -> dict[str, Any] | None:
        if self._cursor >= len(self._steps):
            return None
        step = self._steps[self._cursor]
        self._cursor += 1
        return {
            "agent": step["agent"],
            "tool": step["tool"],
            "args": deepcopy(step.get("args", {})),
        }


class DegeneratePolicy:
    """Emits an unparseable/illegal call every turn."""

    def next_step(self, **_ignored) -> dict[str, Any] | None:
        return {
            "agent": "agent_0",
            "tool": "pick_up_object",
            "args": {"object_id": "nonexistent_object_xyz", "source_id": "counter"},
        }


class HfPolicy:
    """One local HF VLM shared across trajectories; generates one step/turn."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from training.bc_task_vlm.evaluation import VisionGenerationCollator

        self._torch = torch
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
        if args.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, str(args.adapter_path)
            )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.collator = VisionGenerationCollator(
            processor_name_or_path=args.model_name_or_path,
            max_length=args.max_length,
            trust_remote_code=False,
            sft_format="tool_call",
            image_resolution=args.image_resolution,
        )
        from transformers import AutoProcessor as _AP

        self.processor = _AP.from_pretrained(args.model_name_or_path)
        self.max_new_tokens = args.max_new_tokens

    def generate(self, feature: dict[str, Any]) -> str:
        torch = self._torch
        batch = self.collator([feature])
        batch.pop("sample_metadata", None)
        device = next(self.model.parameters()).device
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[0][batch["input_ids"].shape[1] :]
        return self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class VllmPolicy:
    """OpenAI-compatible client for a localhost vLLM multimodal server."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.endpoint = (
            args.vllm_base_url.rstrip("/") + "/chat/completions"
        )
        self.model = args.vllm_model
        self.api_key = args.vllm_api_key
        self.request_timeout = args.vllm_request_timeout
        self.max_new_tokens = args.max_new_tokens
        self.seed = args.seed
        self.last_usage: dict[str, Any] = {}

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _request_messages(cls, feature: dict[str, Any]) -> list[dict[str, Any]]:
        image_paths = iter(feature["image_paths"])
        used_images = 0
        messages: list[dict[str, Any]] = []
        # Match VisionGenerationCollator: the final labeled assistant turn is
        # a training-shape placeholder and is removed for generation.
        for message in feature["messages"][:-1]:
            content = message.get("content", "")
            if not isinstance(content, list):
                messages.append(dict(message))
                continue
            converted: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") != "image":
                    converted.append(dict(item))
                    continue
                try:
                    image_path = next(image_paths)
                except StopIteration as exc:
                    raise ValueError(
                        "Message contains more image slots than image_paths."
                    ) from exc
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": cls._image_data_url(image_path)},
                    }
                )
                used_images += 1
            messages.append({**message, "content": converted})
        if used_images != len(feature["image_paths"]):
            raise ValueError(
                "image_paths contains images without matching message slots."
            )
        return messages

    def generate(self, feature: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._request_messages(feature),
            "tools": feature["tool_schemas"],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self.request_timeout
            ) as response:
                response_payload = json.loads(response.read())
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP {exc.code}: {body[:1000]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc.reason}") from exc

        self.last_usage = dict(response_payload.get("usage") or {})
        try:
            message = response_payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Malformed vLLM response: {response_payload!r}"
            ) from exc
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise RuntimeError("vLLM tool-call arguments are not an object.")
            return (
                "<tool_call>\n"
                + json.dumps(
                    {
                        "name": function.get("name"),
                        "arguments": arguments,
                    },
                    separators=(",", ":"),
                )
                + "\n</tool_call>"
            )
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        raise RuntimeError(
            f"vLLM response contained no text or tool call: {message!r}"
        )


class GeminiPolicy:
    """Vertex Gemini native-function-calling policy shared across turns."""

    def __init__(self, args: argparse.Namespace) -> None:
        import os

        from data_generation.task_level.runtime.client import (
            build_raw_google_genai_client,
            load_dotenv_file,
        )

        load_dotenv_file()
        location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        # Kept so the client can be rebuilt after a wedged connection: httpx
        # pools connections, so once one hangs every retry on that pool hangs
        # too, and a timeout alone just re-wedges at the next call.
        self._client_args = (args.project, location, args.request_timeout)
        self.client = self._build_client()
        self.model = args.model
        self.temperature = args.temperature
        self.max_output_tokens = args.max_output_tokens
        self.thinking_budget = args.thinking_budget
        self.max_retries = args.max_retries
        # Hard ceiling per SDK call. Healthy calls return in seconds -- whole
        # trajectories finish in 23-72s -- so 90s is ~20x headroom while
        # capping a wedged connection at 90s instead of the 360s a 2x-of-
        # request-timeout default cost. Wedges here are intermittent and hit
        # arbitrary tasks, so the cost per occurrence is what matters.
        self.hard_timeout = float(
            getattr(args, "generate_hard_timeout", 0) or 90.0
        )

    def _build_client(self):
        from data_generation.task_level.runtime.client import (
            build_raw_google_genai_client,
        )

        project, location, timeout_sec = self._client_args
        return build_raw_google_genai_client(
            project, location, timeout_sec=timeout_sec
        )

    def _call_with_deadline(self, fn):
        """Runs one SDK call under a hard wall-clock bound.

        The SDK's own timeout is advisory here: it retries internally, so a
        stuck connection never surfaces as an error. This turns that into a
        TimeoutError the retry loop above can act on.
        """

        deadline = float(getattr(self, "hard_timeout", 0) or 0)
        if deadline <= 0:
            return fn()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(fn).result(timeout=deadline)
        finally:
            # Do not join a thread that may still be blocked in ssl.read.
            pool.shutdown(wait=False)

    def generate(self, feature: dict[str, Any]) -> str:
        from google.genai import types

        from training.bc_task_vlm.eval_standalone import (
            _GEMINI_MIME_BY_SUFFIX,
            _message_text,
            build_vertex_function_declarations,
        )

        messages = feature["messages"]
        parts: list[Any] = []
        for image_path in feature["image_paths"]:
            path = Path(image_path)
            parts.append(types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=_GEMINI_MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg"),
            ))
        # build_messages() appends a trailing assistant turn to preserve the
        # training-example shape, and live-sim builds it with target_text="",
        # so messages[-1] is an EMPTY assistant turn -- not the prompt. Taking
        # it sent Gemini nothing but tool declarations, and because the tool
        # config forces a call (mode=ANY), the model still emitted plausible
        # calls with zero task context instead of failing loudly. Always read
        # the last user turn.
        user_message = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if user_message is None:
            raise ValueError("No user message to send to Gemini.")
        prompt_text = _message_text(user_message)
        if not prompt_text.strip():
            raise ValueError("Refusing to query Gemini with an empty prompt.")
        parts.append(types.Part.from_text(text=prompt_text))
        declarations = [
            types.FunctionDeclaration(**item)
            for item in build_vertex_function_declarations(
                feature["allowed_tool_specs"],
                # Under the partial contract the scheduler fixes the caller, so
                # the tools take no "agent" argument and the parsed call is not
                # stripped of one. Advertising the parameter anyway made Gemini
                # emit it on every call, which then failed validation -- the
                # agent relaunched the same rejected proposal until its
                # rejection budget drained, scoring 0 for harness reasons.
                include_agent_param=bool(feature.get("predict_agent", True)),
            )
        ]
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": _message_text(messages[0]),
            "tools": [types.Tool(function_declarations=declarations)],
            "tool_config": types.ToolConfig(function_calling_config=
                types.FunctionCallingConfig(mode="ANY")),
        }
        if self.thinking_budget >= 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        import os as _os

        if _os.environ.get("LIVESIM_DEBUG_PAYLOAD"):
            img_bytes = sum(
                len(getattr(getattr(p, "inline_data", None), "data", b"") or b"")
                for p in parts
            )
            print(
                f"[payload] images={len(feature['image_paths'])} "
                f"image_bytes={img_bytes/1e6:.2f}MB "
                f"prompt_chars={len(prompt_text)} "
                f"tools={len(declarations)}",
                flush=True,
            )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # --request-timeout reaches httpx, but google-genai wraps each
                # request in its own tenacity retry, so a wedged connection
                # retries instead of failing: one request was observed blocked
                # in ssl.read for 78 minutes under a 180s timeout, silently
                # eating the job's walltime and truncating the cell. Bound it
                # ourselves. The orphaned thread is leaked deliberately -- it
                # is bounded by max_retries and cannot be cancelled.
                response = self._call_with_deadline(
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=parts,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                )
                for candidate in getattr(response, "candidates", None) or []:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        call = getattr(part, "function_call", None)
                        if call is not None and getattr(call, "name", None):
                            return (
                                "<tool_call>\n"
                                + json.dumps({"name": call.name, "arguments": dict(call.args or {})})
                                + "\n</tool_call>"
                            )
                raise ValueError("Gemini response contained no function call")
            except Exception as exc:
                last_error = exc
                if isinstance(exc, FuturesTimeoutError):
                    # The abandoned request still owns its pooled connection,
                    # so retrying on the same client wedges again immediately.
                    # Observed: a job sat 37min at 0 trajectories, every retry
                    # timing out. A fresh client gets a fresh pool.
                    try:
                        self.client = self._build_client()
                    except Exception:  # noqa: BLE001 -- keep the original error
                        pass
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error


def _uses_model_generation(policy: Any) -> bool:
    """Return whether a policy implements the shared model generate contract."""

    return callable(getattr(policy, "generate", None))


# ---------------------------------------------------------------------------
# The live loop for one trajectory
# ---------------------------------------------------------------------------


def run_trajectory(
    *,
    session: SimSession,
    policy,
    args: argparse.Namespace,
    task_name: str,
    composite_task: str,
    trajectory: dict[str, Any],
    frames_dir: Path | None,
) -> dict[str, Any]:
    task_metadata = get_task_metadata(task_name)
    train_get_image = bool(getattr(args, "train_get_image", False))
    get_image_observation_mode = getattr(
        args,
        "get_image_observation_mode",
        GET_IMAGE_OBSERVATION_MODE_NEXT_TURN,
    )
    causal_cached_observations = (
        train_get_image
        and _uses_causal_cached_observations(get_image_observation_mode)
    )
    centralized_causal_communication = (
        get_image_observation_mode
        == GET_IMAGE_OBSERVATION_MODE_CAUSAL_CACHE_CENTRALIZED
    )
    tool_specs = augment_tool_specs_for_agent_prediction(
        task_metadata.allowed_tool_specs,
        include_get_image=train_get_image,
        include_task_complete=bool(
            getattr(args, "predict_task_complete", False)
        ),
    )
    tool_schemas = build_tool_schemas(
        agent_ids=AGENT_IDS,
        allowed_tool_specs=tool_specs,
        include_agent_param=True,
    )
    adapter, adapted = session.start_trajectory(trajectory)
    # Simulator-adapted IDs are concrete scene names. The FSM keeps the
    # trajectory's symbolic state and normalizes legacy symbols to the current
    # verified task spec inside FsmMirror.
    mirror = FsmMirror(composite_task=composite_task, trajectory=trajectory)
    _register_canonical_fixture_aliases(adapter, trajectory)
    if frames_dir is not None:
        session.executor.save_scene_frames(str(frames_dir), prefix="step_-001")

    expert_action_steps = [
        s for s in trajectory["steps"] if s.get("tool") != "get_image"
    ]
    budget_multiplier = 2 if train_get_image else 1
    step_budget = max(
        4, int(len(expert_action_steps) * args.step_budget_factor * budget_multiplier)
    )
    is_model_policy = _uses_model_generation(policy)

    history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consecutive_rejections = 0
    declared_complete = False
    termination = "budget_exhausted"
    last_executed_tool: str | None = None
    turn_index = 0
    requested_views: tuple[str, ...] | None = None
    requested_agent: str | None = None
    cached_observations_by_agent: dict[
        str, tuple[list[str], list[str]]
    ] = {}
    active_observation_agent: str | None = None

    while turn_index < step_budget:
        step_index = _step_index_for_turn(
            turn_index, active_observation=train_get_image
        )
        proposal_observation_agent = (
            active_observation_agent
            if causal_cached_observations
            else requested_agent
        )
        views_used: tuple[str, ...] | None = None
        proposal_started = time.perf_counter()
        if is_model_policy:
            proposal, views_used = _model_propose_step(
                session=session,
                policy=policy,
                args=args,
                task_metadata=task_metadata,
                tool_specs=tool_specs,
                tool_schemas=tool_schemas,
                trajectory=trajectory,
                history=history,
                step_index=step_index,
                last_executed_tool=last_executed_tool,
                frames_dir=frames_dir,
                requested_views=requested_views,
                requested_agent=requested_agent,
                cached_observations_by_agent=cached_observations_by_agent,
                active_observation_agent=active_observation_agent,
            )
            requested_views = None
            requested_agent = None
        else:
            proposal = policy.next_step()
        if proposal is None:
            termination = "policy_exhausted"
            break

        record: dict[str, Any] = {
            "step_index": step_index,
            "proposal": deepcopy(proposal),
            "views": list(views_used) if views_used else None,
            "proposal_elapsed_s": round(time.perf_counter() - proposal_started, 3),
            "observation_agent": (
                proposal_observation_agent if train_get_image else None
            ),
        }
        if "error" in proposal:
            # Unparseable model output: no-op with feedback.
            record.update(legal=False, executed=False, reason=proposal["error"])
            history.append(
                {
                    "step": step_index,
                    "agent": proposal.get("agent") or "agent_0",
                    "tool": proposal.get("tool") or "invalid",
                    "args": proposal.get("args") or {},
                    "error": proposal["error"][:120],
                }
            )
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            turn_index += 1
            continue

        if proposal["tool"] == TASK_COMPLETE_TOOL_NAME:
            declared_complete = True
            record.update(legal=True, executed=False, reason=None)
            records.append(record)
            termination = "task_complete_declared"
            break

        if proposal["tool"] == "get_image":
            views = tuple(proposal.get("args", {}).get("views", ()))
            known_views = set(OVERHEAD_VIEWS + SCOUT_VIEWS + WRIST_VIEWS)
            if not views or any(view not in known_views for view in views):
                record.update(
                    legal=False,
                    executed=False,
                    reason="invalid observation views",
                )
                consecutive_rejections += 1
            else:
                record.update(legal=True, executed=True, reason=None)
                history.append(
                    {
                        "step": step_index,
                        "agent": proposal["agent"],
                        "tool": "get_image",
                        "args": {"views": list(views)},
                    }
                )
                if causal_cached_observations:
                    render_dir = frames_dir or Path(args.output_dir) / "_tmp_views"
                    image_paths, view_names = session.render_views(
                        views,
                        agent_id=proposal["agent"],
                        out_dir=render_dir,
                        tag=f"turn_{step_index:03d}_request_{len(views)}v",
                    )
                    _record_agent_observation(
                        cached_observations_by_agent,
                        agent_id=proposal["agent"],
                        image_paths=image_paths,
                        view_names=view_names,
                    )
                    active_observation_agent = proposal["agent"]
                else:
                    requested_views = views
                    requested_agent = proposal["agent"]
                consecutive_rejections = 0
            records.append(record)
            turn_index += 1
            continue

        if causal_cached_observations:
            cache_reason = _causal_cache_rejection_reason(
                tool_name=proposal["tool"],
                proposed_agent=proposal["agent"],
                active_observation_agent=active_observation_agent,
                allow_cross_owner_communication=(
                    centralized_causal_communication
                ),
            )
            if cache_reason is not None:
                record.update(legal=False, executed=False, reason=cache_reason)
                history.append(
                    {
                        "step": step_index,
                        "agent": proposal["agent"],
                        "tool": proposal["tool"],
                        "args": deepcopy(proposal["args"]),
                        "error": cache_reason[:120],
                    }
                )
                consecutive_rejections += 1
                records.append(record)
                if consecutive_rejections >= args.max_consecutive_rejections:
                    termination = "max_consecutive_rejections"
                    break
                turn_index += 1
                continue

        symbolic_step = {
            "step": step_index,
            "agent": proposal["agent"],
            "tool": proposal["tool"],
            "args": deepcopy(proposal["args"]),
        }
        symbolic_state_before = deepcopy(mirror.runtime_state)
        symbolic_goal_before = mirror.goal_satisfied
        legal, reason = mirror.step(symbolic_step)
        record.update(legal=legal, reason=reason)

        # Only the oracle (expert replay) bypasses the legality gate; model
        # and degenerate policies are subject to it.
        should_execute = legal or isinstance(policy, OraclePolicy)
        if not should_execute:
            record["executed"] = False
            history.append({**symbolic_step, "error": (reason or "illegal")[:120]})
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            turn_index += 1
            continue

        try:
            tool_call = adapter._adapt_step(
                symbolic_step,
                resolved_initial_state=adapted["initial_state"],
                output_dir=None,
            )
            result = session.executor.execute(
                tool_call["tool"],
                robot_idx=tool_call.get("robot_idx", 0),
                **tool_call.get("args", {}),
            )
            record.update(
                executed=True,
                sim_success=bool(result.success),
            )
            if not result.success:
                mirror.runtime_state = symbolic_state_before
                mirror.goal_satisfied = symbolic_goal_before
                details = getattr(result, "details", None) or {}
                sim_reason = str(
                    details.get("error")
                    or details.get("reason")
                    or "simulator reported an unsuccessful tool call"
                )
                record["sim_error"] = sim_reason
                history.append({**symbolic_step, "error": sim_reason[:120]})
                consecutive_rejections += 1
                records.append(record)
                if consecutive_rejections >= args.max_consecutive_rejections:
                    termination = "max_consecutive_rejections"
                    break
                turn_index += 1
                continue
        except Exception as exc:
            mirror.runtime_state = symbolic_state_before
            mirror.goal_satisfied = symbolic_goal_before
            record.update(
                executed=False,
                sim_success=False,
                sim_error=f"{type(exc).__name__}: {exc}",
            )
            history.append(
                {**symbolic_step, "error": f"{type(exc).__name__}: {exc}"[:120]}
            )
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            turn_index += 1
            continue

        consecutive_rejections = 0
        last_executed_tool = symbolic_step["tool"]
        history.append(symbolic_step)
        if causal_cached_observations and _tool_invalidates_active_observation(
            symbolic_step["tool"]
        ):
            active_observation_agent = None
        if frames_dir is not None:
            try:
                session.executor.save_scene_frames(
                    str(frames_dir), prefix=f"step_{step_index:03d}"
                )
            except Exception:
                pass

        record["fsm_goal"] = mirror.goal_satisfied
        native_now, native_err = session.native_success()
        record["native_success"] = native_now
        if native_err:
            record["native_error"] = native_err
        records.append(record)

        # Honor --success-criterion here exactly as run_trajectory_partial
        # does. This path used to hardcode "native and fsm", which scored
        # centralized rollouts on a strictly harder bar than partial ones and
        # made the centralized-vs-partial comparison invalid: a rollout that
        # satisfies the task spec but fails the teleporting executor's
        # geometric native check terminated as goal_satisfied under partial
        # and ran to budget_exhausted under centralized.
        criterion = getattr(args, "success_criterion", "fsm")
        reached = (
            mirror.goal_satisfied
            if criterion == "fsm"
            else (mirror.goal_satisfied and native_now)
        )
        if reached:
            termination = "goal_satisfied"
            turn_index += 1
            break
        turn_index += 1

    native, native_error = session.native_success()
    fsm_goal = mirror.goal_satisfied
    return {
        "task_name": task_name,
        "composite_task": composite_task,
        "trajectory_id": trajectory.get("trajectory_id"),
        "get_image_observation_mode": (
            get_image_observation_mode if train_get_image else None
        ),
        "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
        "expert_steps": len(expert_action_steps),
        "steps_used": len(records),
        "executed_steps": sum(1 for r in records if r.get("executed")),
        "rejected_steps": sum(
            1
            for r in records
            if r.get("legal") is False or r.get("sim_success") is False
        ),
        "termination": termination,
        "declared_complete": declared_complete,
        "native_success": native,
        "native_error": native_error,
        "fsm_goal_satisfied": fsm_goal,
        "partial_goal_fraction": mirror.partial_goal_fraction(),
        "step_efficiency_ratio": (
            len(records) / len(expert_action_steps) if expert_action_steps else None
        ),
        "agent_turns": [
            r["proposal"].get("agent") for r in records if r.get("executed")
        ],
        "steps": records,
    }


def run_trajectory_partial(
    *,
    session: SimSession,
    policy,
    args: argparse.Namespace,
    task_name: str,
    composite_task: str,
    trajectory: dict[str, Any],
    frames_dir: Path | None,
) -> dict[str, Any]:
    """Partial-observability episode: per-agent private state, virtual clock.

    Each agent becomes ready when its own tool completes, so the two agents
    desynchronize naturally and simultaneous proposals are the exception rather
    than every cycle. Neither proposal sees the other's result before it is
    generated. Execution stays serial (the executor runs a tool to completion);
    only readiness ordering is event-timed.
    """

    task_metadata = get_task_metadata(task_name)
    # Partial v3 supervises get_image with a FIXED caller, so the tool set gains
    # get_image but never task_complete or the agent argument.
    tool_specs = augment_tool_specs_with_get_image(task_metadata.allowed_tool_specs)
    tool_schemas = build_tool_schemas(
        agent_ids=AGENT_IDS,
        allowed_tool_specs=tool_specs,
        include_agent_param=False,
    )
    adapter, adapted = session.start_trajectory(trajectory)
    mirror = FsmMirror(composite_task=composite_task, trajectory=trajectory)
    _register_canonical_fixture_aliases(adapter, trajectory)
    if frames_dir is not None:
        session.executor.save_scene_frames(str(frames_dir), prefix="step_-001")

    expert_action_steps = [
        s for s in trajectory["steps"] if s.get("tool") != "get_image"
    ]
    step_budget = max(4, int(len(expert_action_steps) * args.step_budget_factor * 2))
    lock_mode = getattr(args, "resource_locks", RESOURCE_LOCKS_FIXTURES)
    lock_priority = getattr(args, "conflict_priority", "agent-order")
    rejection_mode = getattr(args, "rejection_mode", REJECTION_MODE_SILENT_RETRY)
    max_silent = int(getattr(args, "max_silent_retries", 3))
    multiplier = float(getattr(args, "duration_multiplier", 1.0))
    global _DURATION_JITTER, _DURATION_RNG, _UNIFORM_DURATIONS
    _UNIFORM_DURATIONS = bool(getattr(args, "uniform_durations", False))
    _DURATION_JITTER = float(getattr(args, "duration_jitter", 0.0))
    # Per trajectory, so the same trajectory gets the same perturbed schedule
    # regardless of which shard or order it ran in.
    # sha256, not hash(): str hashing is salted per process, so hash() would
    # make a "reproducible" seed differ between runs.
    _DURATION_RNG = (
        random.Random(
            hashlib.sha256(
                f"{int(getattr(args, 'duration_jitter_seed', 0))}|"
                f"{composite_task}|{trajectory.get('trajectory_id', '')}".encode()
            ).hexdigest()
        )
        if _DURATION_JITTER else None
    )

    agents = {aid: AgentRuntime(aid) for aid in AGENT_IDS}
    is_model_policy = _uses_model_generation(policy)
    # Non-model policies (oracle/degenerate) replay a JOINT expert sequence, but
    # a per-agent scheduler asks a specific agent what it wants to do. Split the
    # expert steps into per-agent queues so "agent A's next expert action" is
    # well defined regardless of the order the scheduler picks agents in.
    # ORDER-PRESERVING REPLAY. Splitting the joint expert plan into per-agent
    # queues let the scheduler interleave it differently from how it was
    # written, which invalidates plans that depend on their own ordering. In
    # arrange_bread_bowl/traj_000009 agent_1 picked up the bowl and carried it
    # to dining_counter before agent_0 could place bread into it, so a correct
    # plan was rejected for "missing navigation". Non-model policies now follow
    # the recorded joint order exactly; only model policies are scheduled.
    expert_sequence: list[dict[str, Any]] = []
    if not is_model_policy:
        for raw in trajectory["steps"]:
            if raw.get("tool") == "get_image":
                continue
            if raw.get("agent") in agents:
                expert_sequence.append(
                    {"agent": raw["agent"], "tool": raw["tool"],
                     "args": deepcopy(raw.get("args", {}))}
                )
    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    clock = 0.0
    turn_index = 0          # productive (executed) steps -- what step_budget caps
    proposal_index = 0      # every model call, for logging/ids
    rejected_total = 0
    consecutive_rejections = 0
    max_consec = int(getattr(args, "max_consecutive_rejections", 8))
    # A rejected call is not productive work, so it must not consume the budget
    # meant for it -- but it cannot be free either, or a livelock runs forever.
    # Hence a separate allowance, generous relative to the step budget.
    max_rejected = int(step_budget * float(getattr(args, "rejection_budget_ratio", 1.0)))
    termination = "budget_exhausted"
    simultaneous_cycles = 0
    mutual_wait_deadlocks = 0
    waits_issued = 0
    wait_caps = 0
    max_consecutive_waits = int(getattr(args, 'max_consecutive_waits', 3))
    escalations = 0
    silent_resolutions = 0

    # Concurrent replay lets each agent advance its own stream under the timing
    # scheduler, the regime a model actually meets at eval. It is only sound
    # when cross-agent ordering is explicit (wait_for_signal); with implicit
    # ordering the scheduler re-interleaves and breaks correct plans, which is
    # why ordered replay is the default.
    concurrent_replay = bool(getattr(args, "concurrent_expert_replay", False))
    expert_queues: dict[str, list[dict[str, Any]]] = {}
    if concurrent_replay:
        for entry in expert_sequence:
            expert_queues.setdefault(entry["agent"], []).append(entry)

    def _propose(agent: AgentRuntime) -> tuple[dict[str, Any], tuple[str, ...] | None]:
        """One decision for one agent, using only that agent's private state."""

        if not is_model_policy:
            if concurrent_replay:
                queue = expert_queues.get(agent.agent_id) or []
                if not queue:
                    return {"exhausted": True}, None
                return queue.pop(0), None
            if not expert_sequence or expert_sequence[0]["agent"] != agent.agent_id:
                return {"exhausted": True}, None
            return expert_sequence.pop(0), None
        image_paths, view_names = agent.take_observation()
        render_dir = frames_dir or Path(args.output_dir) / "_tmp_views"
        proposal = _generate_once(
            session=session,
            policy=policy,
            args=args,
            task_metadata=task_metadata,
            tool_specs=tool_specs,
            tool_schemas=tool_schemas,
            trajectory=trajectory,
            history=agent.private_history,
            step_index=turn_index,
            views=tuple(view_names),
            render_dir=render_dir,
            agent_hint=agent.agent_id,
            pre_rendered_image_paths=image_paths if image_paths else None,
            partial_caller=agent.agent_id,
        )
        return proposal, tuple(view_names) if view_names else None

    while turn_index < step_budget:
        if not is_model_policy and not concurrent_replay:
            # Replay follows the recorded joint order; the timing scheduler is
            # bypassed entirely so the plan cannot be re-interleaved.
            if not expert_sequence:
                termination = "policy_exhausted"
                break
            actor = agents[expert_sequence[0]["agent"]]
            if actor.ready_at == float("inf"):
                actor.ready_at = clock
            clock = max(clock, actor.ready_at)
            ready = [actor]
        else:
            if concurrent_replay and not any(expert_queues.values()):
                termination = "policy_exhausted"
                break
            if all(a.ready_at == float("inf") for a in agents.values()):
                termination = "policy_exhausted"
                break
            # A waiting agent is not polled. If EVERY agent is waiting nobody
            # can ever send the message that would wake them, so break the
            # deadlock rather than hanging: wake all, and record it -- a hang
            # is operationally far worse than budget exhaustion.
            if agents and all(a.waiting_for is not None for a in agents.values()):
                mutual_wait_deadlocks += 1
                for a in agents.values():
                    a.waiting_for = None
                    a.consecutive_waits = 0
            # A blocked agent must not drive the clock. Its ready_at is stale
            # -- the moment its wait WOULD have expired -- so including it here
            # pulled the clock back every iteration and the loop never
            # advanced past a partner that was merely busy.
            runnable_times = [
                a.ready_at for a in agents.values() if a.waiting_for is None
            ] or [a.ready_at for a in agents.values()]
            # Monotonic, but only over agents that still have work. `inf` is the
            # sentinel for an EXHAUSTED agent, and clamping upward against it
            # pins the clock at infinity forever -- after which the catch-up
            # below hands every other agent an infinite ready_at and nothing is
            # ever schedulable again. Without work to order there is no time to
            # advance to, so leave the clock where it is and let the
            # nobody-is-ready branch decide what happens next.
            finite = [t for t in runnable_times if t != float("inf")]
            if finite:
                clock = max(clock, min(finite))
            # A wait is released by a message, so the agent resumes at the
            # moment of release. Leaving its stale ready_at -- the time the
            # wait would have expired -- let it act "in the past", inverting
            # sim_time against the real execution order.
            for a in agents.values():
                if a.waiting_for is None and a.ready_at < clock:
                    a.ready_at = clock
            ready = sorted(
                (a for a in agents.values()
                 if a.ready_at <= clock and a.ready_at != float("inf")
                 and a.waiting_for is None),
                key=lambda a: a.agent_id,
            )
            if not ready:
                # A partner that is merely BUSY must not cancel a wait. This
                # fired whenever nobody was runnable at the current instant,
                # so every wait expired as soon as the other agent was
                # mid-action -- waits behaved as a 2s pause rather than a
                # block, which is why inserting them changed nothing.
                busy = [
                    a for a in agents.values()
                    if a.waiting_for is None and a.ready_at != float("inf")
                ]
                if busy:
                    clock = max(clock, min(a.ready_at for a in busy))
                    continue
                waiting = [a for a in agents.values() if a.waiting_for is not None]
                if not waiting:
                    break
                clock = max(clock, min(a.ready_at for a in waiting))
                for a in waiting:
                    a.waiting_for = None
                continue
        if len(ready) > 1:
            simultaneous_cycles += 1

        # --- concurrent generation: neither sees the other's result ---
        started = time.perf_counter()
        if len(ready) == 1 or not _uses_model_generation(policy):
            proposals = [(a, *_propose(a)) for a in ready]
        else:
            with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                futures = {pool.submit(_propose, a): a for a in ready}
                proposals = [(futures[f], *f.result()) for f in futures]
            proposals.sort(key=lambda item: item[0].agent_id)
        elapsed = round(time.perf_counter() - started, 3)

        # --- resource conflict resolution against the same pre-batch state ---
        claims = {
            agent.agent_id: resource_claims(prop, mode=lock_mode)
            for agent, prop, _ in proposals
            if "error" not in prop
        }
        # Deterministic priority breaks the tie. Rejecting BOTH proposals is
        # symmetric, so two agents wanting the same fixture would each retry the
        # same call forever (observed: 6 conflicts, zero progress, livelock).
        # One agent proceeds; the loser gets an ordinary tool error and replans.
        conflicted: set[str] = set()
        ids = sorted(claims)
        if lock_priority == "seeded":
            rng = random.Random(f"{args.seed}:{turn_index}")
            rng.shuffle(ids)
        held: set[str] = set()
        for a_id in ids:
            if claims[a_id] & held:
                conflicted.add(a_id)
            else:
                held |= claims[a_id]

        for agent, proposal, views_used in proposals:
            record: dict[str, Any] = {
                "step_index": turn_index,
                "sim_time": round(clock, 3),
                "agent": agent.agent_id,
                "local_turn": agent.local_turn,
                "proposal": deepcopy(proposal),
                "views": list(views_used) if views_used else None,
                "proposal_elapsed_s": elapsed,
                "cycle_agents": [a.agent_id for a in ready],
            }
            agent.local_turn += 1
            proposal_index += 1

            key = _proposal_key(proposal)
            if key == agent.last_proposal_key:
                agent.repeated_proposals += 1
            agent.last_proposal_key = key

            # wait_for_signal is a scheduling primitive, not a sim action: it
            # blocks the agent instead of touching the world. `about` is
            # recorded but never checked -- any inbound message wakes the agent
            # (see AgentRuntime.deliver) and judging relevance is the model's
            # job. The first wait is free; later consecutive waits consume
            # budget so a wait->irrelevant-message->wait livelock is bounded
            # and visible rather than an unbounded stall.
            if proposal.get("tool") == WAIT_TOOL_NAME and "error" not in proposal:
                wait_args = proposal.get("args") or {}
                agent.consecutive_waits += 1
                agent.waiting_for = {
                    "from": wait_args.get("from"),
                    "about": wait_args.get("about"),
                    "declared_at": turn_index,
                }
                record["legal"] = True
                record["executed"] = True
                record["waited_for"] = deepcopy(agent.waiting_for)
                record["consecutive_waits"] = agent.consecutive_waits
                waits_issued += 1
                if agent.consecutive_waits > max_consecutive_waits:
                    # Treat as unproductive: stop blocking and make it cost a turn.
                    agent.waiting_for = None
                    agent.consecutive_waits = 0
                    record["wait_capped"] = True
                    wait_caps += 1
                    turn_index += 1
                agent.deliver({"agent": agent.agent_id, "tool": WAIT_TOOL_NAME,
                               "args": deepcopy(wait_args)})
                agent.ready_at = clock + _tool_duration(
                    WAIT_TOOL_NAME, sim_steps=None, multiplier=multiplier
                )
                records.append(record)
                continue
            if proposal.get("tool") != WAIT_TOOL_NAME:
                agent.consecutive_waits = 0

            if proposal.get("error") or proposal.get("_rejected"):
                pass  # counted below
            if proposal.get("exhausted"):
                # This agent has no expert steps left; retire it from scheduling.
                agent.ready_at = float("inf")
                agent.local_turn -= 1
                turn_index -= 1
                continue

            if "error" in proposal:
                record.update(legal=False, executed=False, reason=proposal["error"])
                agent.ready_at = clock + _tool_duration(
                    "communicate", sim_steps=None, multiplier=multiplier
                )
                records.append(record)
                continue

            # Resource conflict -> ordinary tool error handed back to the agent.
            if agent.agent_id in conflicted:
                reason = "resource conflict: another agent holds a required resource"
                record.update(
                    legal=False, executed=False, reason=reason, conflict=True
                )
                conflicts.append(
                    {"agent": agent.agent_id, "sim_time": round(clock, 3),
                     "tool": proposal.get("tool"), "claims": sorted(claims[agent.agent_id])}
                )
                _handle_rejection(
                    agent=agent,
                    agents=agents,
                    step={"agent": agent.agent_id, "tool": proposal["tool"],
                          "args": deepcopy(proposal["args"])},
                    reason=reason,
                    clock=clock,
                    mode=rejection_mode,
                    max_silent=max_silent,
                    multiplier=multiplier,
                    record=record,
                )
                if not is_model_policy and not record.get("escalated"):
                    # Silent retry leaves state untouched, so an expert step
                    # consumed by a rejected proposal must go back on the queue.
                    if concurrent_replay:
                        expert_queues.setdefault(
                            proposal["agent"], []
                        ).insert(0, proposal)
                    else:
                        expert_sequence.insert(0, proposal)
                if record.get("escalated"):
                    escalations += 1
                else:
                    silent_resolutions += 1
                records.append(record)
                continue

            # --- get_image: render, deliver to the requester only ---
            if proposal["tool"] == "get_image":
                views = tuple(proposal.get("args", {}).get("views", ()))
                known = set(OVERHEAD_VIEWS + SCOUT_VIEWS + WRIST_VIEWS)
                if not views or any(v not in known for v in views):
                    record.update(
                        legal=False, executed=False, reason="invalid observation views"
                    )
                    agent.ready_at = clock + _tool_duration(
                        "get_image", sim_steps=None, multiplier=multiplier
                    )
                    records.append(record)
                    continue
                render_dir = frames_dir or Path(args.output_dir) / "_tmp_views"
                image_paths, view_names = session.render_views(
                    views,
                    agent_id=agent.agent_id,
                    out_dir=render_dir,
                    tag=f"t{turn_index:03d}_{agent.agent_id}",
                )
                # Consume-once: only the requester ever sees this.
                turn_index += 1
                consecutive_rejections = 0
                agent.pending_obs = (image_paths, view_names)
                agent.deliver(
                    {"agent": agent.agent_id, "tool": "get_image",
                     "args": {"views": list(views)}}
                )
                record.update(legal=True, executed=True, reason=None)
                agent.ready_at = clock + _tool_duration(
                    "get_image", sim_steps=None, multiplier=multiplier
                )
                agent.silent_retries = 0
                records.append(record)
                continue

            symbolic_step = {
                "step": turn_index,
                "agent": agent.agent_id,
                "tool": proposal["tool"],
                "args": deepcopy(proposal["args"]),
            }
            state_before = deepcopy(mirror.runtime_state)
            goal_before = mirror.goal_satisfied
            legal, reason = mirror.step(symbolic_step)
            record.update(legal=legal, reason=reason)

            if not legal:
                record["executed"] = False
                _handle_rejection(
                    agent=agent, agents=agents, step=symbolic_step,
                    reason=reason or "illegal", clock=clock, mode=rejection_mode,
                    max_silent=max_silent, multiplier=multiplier, record=record,
                )
                if record.get("escalated"):
                    escalations += 1
                else:
                    silent_resolutions += 1
                records.append(record)
                continue

            try:
                tool_call = adapter._adapt_step(
                    symbolic_step,
                    resolved_initial_state=adapted["initial_state"],
                    output_dir=None,
                )
                sim_t0 = _sim_clock(session)
                result = session.executor.execute(
                    tool_call["tool"],
                    robot_idx=tool_call.get("robot_idx", 0),
                    **tool_call.get("args", {}),
                )
                record.update(executed=True, sim_success=bool(result.success))
                if not result.success:
                    mirror.runtime_state = state_before
                    mirror.goal_satisfied = goal_before
                    details = getattr(result, "details", None) or {}
                    sim_reason = str(
                        details.get("error")
                        or details.get("reason")
                        or "simulator reported an unsuccessful tool call"
                    )
                    record["sim_error"] = sim_reason
                    _handle_rejection(
                        agent=agent, agents=agents, step=symbolic_step,
                        reason=sim_reason, clock=clock, mode=rejection_mode,
                        max_silent=max_silent, multiplier=multiplier, record=record,
                    )
                    if record.get("escalated"):
                        escalations += 1
                    else:
                        silent_resolutions += 1
                    records.append(record)
                    continue
            except Exception as exc:
                mirror.runtime_state = state_before
                mirror.goal_satisfied = goal_before
                record.update(
                    executed=False, sim_success=False,
                    sim_error=f"{type(exc).__name__}: {exc}",
                )
                _handle_rejection(
                    agent=agent, agents=agents, step=symbolic_step,
                    reason=f"{type(exc).__name__}: {exc}", clock=clock,
                    mode=rejection_mode, max_silent=max_silent,
                    multiplier=multiplier, record=record,
                )
                records.append(record)
                continue

            # --- committed: private history, then delivery to the recipient ---
            agent.silent_retries = 0
            turn_index += 1
            consecutive_rejections = 0
            agent.deliver(symbolic_step)
            if symbolic_step["tool"] == "communicate":
                recipient = (symbolic_step.get("args") or {}).get("to")
                if recipient in agents and recipient != agent.agent_id:
                    agents[recipient].deliver(symbolic_step, clock=clock)
                    record["delivered_to"] = recipient
            sim_t1 = _sim_clock(session)
            measured = (
                (sim_t1 - sim_t0)
                if (sim_t0 is not None and sim_t1 is not None and sim_t1 > sim_t0)
                else None
            )
            duration = _tool_duration(
                symbolic_step["tool"], sim_steps=measured, multiplier=multiplier
            )
            record["measured_sim_seconds"] = (
                round(measured, 3) if measured is not None else None
            )
            agent.ready_at = clock + duration
            agent.active_s += duration
            record["duration"] = round(duration, 3)

            if frames_dir is not None:
                try:
                    session.executor.save_scene_frames(
                        str(frames_dir), prefix=f"step_{turn_index:03d}"
                    )
                except Exception:
                    pass

            record["fsm_goal"] = mirror.goal_satisfied
            native_now, native_err = session.native_success()
            record["native_success"] = native_now
            if native_err:
                record["native_error"] = native_err
            records.append(record)

            criterion = getattr(args, "success_criterion", "fsm")
            reached = (
                mirror.goal_satisfied
                if criterion == "fsm"
                else (mirror.goal_satisfied and native_now)
            )
            if reached:
                termination = "goal_satisfied"
                break
        if termination in ("goal_satisfied", "max_consecutive_rejections",
                           "rejection_budget_exhausted"):
            break

    native, native_error = session.native_success()
    finite = [a.ready_at for a in agents.values() if a.ready_at != float("inf")]
    makespan = max(finite, default=0.0)
    return {
        "task_name": task_name,
        "composite_task": composite_task,
        "trajectory_id": trajectory.get("trajectory_id"),
        "partial_history": True,
        "resource_locks": lock_mode,
        "rejection_mode": rejection_mode,
        "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
        "expert_steps": len(expert_action_steps),
        "steps_used": len(records),
        "productive_steps": turn_index,
        "rejected_total": rejected_total,
        "executed_steps": sum(1 for r in records if r.get("executed")),
        "rejected_steps": sum(
            1 for r in records
            if r.get("legal") is False or r.get("sim_success") is False
        ),
        "termination": termination,
        "declared_complete": False,
        "native_success": native,
        "native_error": native_error,
        "fsm_goal_satisfied": mirror.goal_satisfied,
        "partial_goal_fraction": mirror.partial_goal_fraction(),
        "makespan": round(makespan, 3),
        "simultaneous_cycles": simultaneous_cycles,
        "waits_issued": waits_issued,
        "wait_caps": wait_caps,
        "mutual_wait_deadlocks": mutual_wait_deadlocks,
        "resource_conflicts": len(conflicts),
        "conflict_details": conflicts,
        "silent_resolutions": silent_resolutions,
        "escalations": escalations,
        "per_agent": {
            aid: {
                "turns": a.local_turn,
                "active_sim_time": round(a.active_s, 3),
                "idle_sim_time": round(max(0.0, makespan - a.active_s), 3),
                "repeated_proposals": a.repeated_proposals,
                "private_history_len": len(a.private_history),
            }
            for aid, a in agents.items()
        },
        "agent_turns": [r["agent"] for r in records if r.get("executed")],
        "steps": records,
    }


def _handle_rejection(
    *,
    agent: AgentRuntime,
    agents: dict[str, AgentRuntime],
    step: dict[str, Any],
    reason: str,
    clock: float,
    mode: str,
    max_silent: int,
    multiplier: float,
    record: dict[str, Any],
) -> None:
    """Silent-retry by default; escalate to a FAILED: entry only as a last resort.

    Silent retry sleeps the agent until the NEXT WORLD EVENT rather than a fixed
    delay. That matters: under greedy decoding an unchanged prompt reproduces the
    identical call, so a constant sleep would loop forever. Waiting for the other
    agent to finish means the retry sees a released lock, a satisfied
    precondition, or a delivered message.
    """

    agent.silent_retries += 1
    escalate = (
        mode == REJECTION_MODE_REPORT_FAILED or agent.silent_retries > max_silent
    )
    record["silent_retries"] = agent.silent_retries
    record["escalated"] = escalate
    if escalate:
        # Accepts the distribution shift: the model never saw FAILED: in training.
        agent.deliver({**step, "error": reason[:120]})
        agent.silent_retries = 0
        agent.ready_at = clock + _tool_duration(
            step.get("tool", "communicate"), sim_steps=None, multiplier=multiplier
        )
        return
    # State untouched: no history entry, observation not consumed.
    others = [a.ready_at for a in agents.values() if a.agent_id != agent.agent_id]
    next_event = max((t for t in others if t > clock), default=None)
    if next_event is None:
        next_event = clock + _tool_duration(
            "navigate_to_fixture", sim_steps=None, multiplier=multiplier
        )
    agent.ready_at = next_event


def _model_propose_step(
    *,
    session: SimSession,
    policy: "HfPolicy | VllmPolicy | GeminiPolicy",
    args: argparse.Namespace,
    task_metadata,
    tool_specs: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    trajectory: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    last_executed_tool: str | None,
    frames_dir: Path | None,
    requested_views: tuple[str, ...] | None = None,
    requested_agent: str | None = None,
    cached_observations_by_agent: dict[
        str, tuple[list[str], list[str]]
    ] | None = None,
    active_observation_agent: str | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Propose under the selected active-observation or legacy view contract."""

    render_dir = (frames_dir or Path(args.output_dir) / "_tmp_views")
    if getattr(args, "train_get_image", False):
        observation_mode = getattr(
            args,
            "get_image_observation_mode",
            GET_IMAGE_OBSERVATION_MODE_NEXT_TURN,
        )
        if _uses_causal_cached_observations(observation_mode):
            cached_image_paths, views = _cached_observation_for_agent(
                cached_observations_by_agent or {},
                active_observation_agent,
            )
            proposal = _generate_once(
                session=session,
                policy=policy,
                args=args,
                task_metadata=task_metadata,
                tool_specs=tool_specs,
                tool_schemas=tool_schemas,
                trajectory=trajectory,
                history=history,
                step_index=step_index,
                views=views,
                render_dir=render_dir,
                agent_hint=active_observation_agent,
                pre_rendered_image_paths=cached_image_paths,
            )
            return proposal, views
        views = requested_views or ()
        proposal = _generate_once(
            session=session,
            policy=policy,
            args=args,
            task_metadata=task_metadata,
            tool_specs=tool_specs,
            tool_schemas=tool_schemas,
            trajectory=trajectory,
            history=history,
            step_index=step_index,
            views=views,
            render_dir=render_dir,
            agent_hint=requested_agent,
        )
        return proposal, views
    views = _routing_views_for_step(step_index)
    proposal = _generate_once(
        session=session,
        policy=policy,
        args=args,
        task_metadata=task_metadata,
        tool_specs=tool_specs,
        tool_schemas=tool_schemas,
        trajectory=trajectory,
        history=history,
        step_index=step_index,
        views=views,
        render_dir=render_dir,
    )
    if not args.two_pass_views or "error" in proposal:
        return proposal, views
    canonical = canonical_views_for_tool(proposal["tool"])
    if canonical == views:
        return proposal, views
    second = _generate_once(
        session=session,
        policy=policy,
        args=args,
        task_metadata=task_metadata,
        tool_specs=tool_specs,
        tool_schemas=tool_schemas,
        trajectory=trajectory,
        history=history,
        step_index=step_index,
        views=canonical,
        render_dir=render_dir,
        agent_hint=proposal.get("agent"),
    )
    if "error" in second:
        return proposal, views
    return second, canonical


def _generate_once(
    *,
    session: SimSession,
    policy: "HfPolicy | VllmPolicy | GeminiPolicy",
    args: argparse.Namespace,
    task_metadata,
    tool_specs: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    trajectory: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    views: tuple[str, ...],
    render_dir: Path,
    agent_hint: str | None = None,
    pre_rendered_image_paths: list[str] | None = None,
    partial_caller: str | None = None,
) -> dict[str, Any]:
    if pre_rendered_image_paths is not None:
        image_paths = list(pre_rendered_image_paths)
        view_names = list(views)
    elif views:
        image_paths, view_names = session.render_views(
            views,
            agent_id=agent_hint or "agent_0",
            out_dir=render_dir,
            tag=f"turn_{step_index:03d}_{len(views)}v",
        )
    else:
        image_paths, view_names = [], []
    partial = partial_caller is not None
    if partial:
        # The caller is the actor: no agent prediction, no joint step index,
        # and observation ownership is implicit in the private state.
        index_mode = getattr(args, "partial_step_index_mode", "none")
        if index_mode == "local":
            prompt_step_index: int | None = len(history)
            step_index_label = "Next local agent turn index"
        elif index_mode == "none":
            prompt_step_index = None
            step_index_label = "Next global step index"
        else:
            prompt_step_index = step_index
            step_index_label = "Next global step index"
        user_prompt = build_user_prompt(
            composite_task=task_metadata.composite_task,
            task_instruction=trajectory.get("task", ""),
            agent_id=partial_caller,
            next_step_index=prompt_step_index,
            step_index_label=step_index_label,
            observation_views=view_names,
            history_steps=history,
            allowed_tool_specs=tool_specs,
            sft_format="tool_call",
            predict_agent=False,
        )
    else:
        user_prompt = build_user_prompt(
            composite_task=task_metadata.composite_task,
            task_instruction=trajectory.get("task", ""),
            agent_id="",
            next_step_index=step_index,
            observation_views=view_names,
            history_steps=history,
            allowed_tool_specs=tool_specs,
            sft_format="tool_call",
            observation_owner=agent_hint,
            include_observation_owner=(
                _uses_causal_cached_observations(
                    getattr(
                        args,
                        "get_image_observation_mode",
                        GET_IMAGE_OBSERVATION_MODE_NEXT_TURN,
                    )
                )
            ),
            predict_agent=True,
        )
    feature = {
        "sample_id": f"live/{trajectory.get('trajectory_id')}/turn_{step_index}",
        "task_name": task_metadata.dataset_name,
        "trajectory_id": str(trajectory.get("trajectory_id") or ""),
        "step_index": step_index,
        "agent_id": partial_caller or agent_hint or "",
        # Policies that rebuild tool declarations from allowed_tool_specs need
        # to know whether the "agent" argument is part of this contract.
        "predict_agent": not partial,
        "target_payload": None,
        "target_tool_call": None,
        "target_text": "",
        "messages": build_messages(
            user_prompt=user_prompt,
            num_images=len(image_paths),
            predict_agent=not partial,
            train_get_image=bool(getattr(args, "train_get_image", False)),
            # The generation collator strips the labeled assistant turn before
            # tokenization, so retain its expected training-example shape.
            target_text="",
        ),
        "image_paths": image_paths,
        "tool_schemas": tool_schemas,
        "allowed_tool_specs": tool_specs,
    }
    from training.bc_task_vlm.prompting import append_few_shot_block

    if args.task_spec_detail:
        feature["messages"] = append_few_shot_block(
            feature["messages"], args.task_spec_blocks[task_metadata.dataset_name]
        )
    if args.few_shot:
        feature["messages"] = append_few_shot_block(
            feature["messages"], args.few_shot_blocks[task_metadata.dataset_name]
        )
    try:
        decoded = policy.generate(feature)
        parsed = parse_first_qwen_tool_call(decoded)
        # eval-only cab<->cabinet leniency; never makes a legal call illegal
        parsed = apply_eval_fixture_aliases(parsed, tool_specs)
        if partial:
            # Caller identity is fixed by the scheduler; the model never names
            # an acting agent under the distributed contract.
            agent, stripped = partial_caller, parsed
        else:
            agent, stripped = pop_agent_argument(parsed)
        if agent not in AGENT_IDS:
            return {"error": f'missing/invalid "agent" argument: {agent!r}'}
        payload = tool_call_to_single_step_payload(
            stripped,
            step_index=step_index,
            agent_id=agent,
            agent_ids=AGENT_IDS,
            allowed_tool_specs=tool_specs,
        )
        step = payload["steps"][0]
        return {"agent": step["agent"], "tool": step["tool"], "args": step["args"]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _latest_trajectory_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the latest append-only record for each trajectory key."""

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record.get("task_name"), record.get("trajectory_id"))
        latest[key] = record
    return list(latest.values())


def _completed_trajectory_keys(
    records: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return resumable completions, leaving latest harness errors pending."""

    return {
        (record["task_name"], record["trajectory_id"])
        for record in _latest_trajectory_records(records)
        if record.get("termination") != "harness_error"
    }


def _finalize_first_recording(
    *,
    frames_dir: Path | None,
    output_dir: Path,
    task_name: str,
    success: bool,
    firsts: dict[tuple[str, bool], bool],
    record_firsts: bool,
    save_frames: bool,
    record_fps: int,
) -> None:
    """Classify a recorded first outcome when frames were actually created."""

    if frames_dir is None or not frames_dir.exists():
        return
    if not firsts.get((task_name, success)):
        firsts[(task_name, success)] = True
        if record_firsts and not save_frames:
            classified = (
                output_dir
                / "recordings"
                / task_name
                / ("success" if success else "failure")
            )
            if classified.exists():
                shutil.rmtree(classified)
            frames_dir.rename(classified)
            _assemble_recording_videos(classified, fps=record_fps)
    elif record_firsts and not save_frames:
        shutil.rmtree(frames_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("oracle", "degenerate", "hf", "vllm", "gemini"), required=True
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save per-camera stills for every evaluated trajectory.",
    )
    parser.add_argument("--record-fps", type=int, default=2)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gl-backend", default="osmesa")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument(
        "--map-dpi",
        type=int,
        default=300,
        help=(
            "Placement-map export DPI. The compatibility default is 300 "
            "(6000x4800); lower values trade source-map fidelity for speed."
        ),
    )
    parser.add_argument(
        "--map-renderer",
        choices=("legacy", "raster"),
        default="legacy",
        help="Placement-grid renderer. Legacy is compatible; raster is faster.",
    )
    parser.add_argument("--step-budget-factor", type=float, default=2.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=3)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-firsts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record stills for the first success and first failure per task.",
    )
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Local LoRA directory or Hugging Face adapter repository ID.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM API base URL.",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Served vLLM model or LoRA name used in requests.",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="Optional API key for the vLLM server.",
    )
    parser.add_argument(
        "--vllm-request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout for the vLLM server.",
    )
    parser.add_argument(
        "--predict-task-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Match the adapter's training flag. --no-predict-task-complete "
            "removes task_complete from the tool set, so a centralized rollout "
            "can only end on goal satisfaction, budget exhaustion or "
            "rejections -- the same terminations available to partial-obs."
        ),
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sft-format", choices=("tool_call",), default="tool_call")
    parser.add_argument("--no-forced-json", action="store_true")
    parser.add_argument(
        "--train-get-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the model request observations with get_image.",
    )
    parser.add_argument(
        "--get-image-observation-mode",
        choices=GET_IMAGE_OBSERVATION_MODES,
        default=GET_IMAGE_OBSERVATION_MODE_NEXT_TURN,
        help=(
            "How model-emitted get_image observations condition later calls. "
            "next_turn sends them to the immediately following proposal. "
            "Both causal-cache modes render each successful request "
            "immediately, keep agent caches separate, and expose only the "
            "previously active agent's cache. causal_cache enforces that "
            "communication uses that owner; causal_cache_centralized allows "
            "the centralized policy to select either communication speaker. "
            "Physical actions require the active owner and clear the active "
            "visual context. No step parity or current-target information is "
            "used."
        ),
    )
    parser.add_argument("--task-spec-detail", action="store_true")
    parser.add_argument("--few-shot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model", default="gemini-3.5-flash-preview")
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-consecutive-waits", type=int, default=3,
        help="Consecutive wait_for_signal calls before the wait is capped "
             "and charged a turn. Bounds a wait->irrelevant-message->wait livelock.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--generate-hard-timeout", type=float, default=0.0,
        help="Hard per-call wall-clock bound in seconds (0 = 90s). "
             "Guards against google-genai retrying a wedged connection forever.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--thinking-budget", type=int, default=0)
    partial = parser.add_argument_group("partial observability")
    partial.add_argument(
        "--partial-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the distributed contract: per-agent private history (own "
        "actions + delivered messages), consume-once observations, and a "
        "concurrent per-agent scheduler. The caller IS the actor, so the model "
        "no longer predicts the acting agent.",
    )
    partial.add_argument(
        "--partial-step-index-mode",
        choices=("global", "local", "none"),
        default="none",
        help="How step indices are rendered; 'none' (default) omits them, since "
        "a joint index leaks the other agent's hidden activity.",
    )
    partial.add_argument(
        "--partial-observation-mode",
        choices=("cache", "consume-once"),
        default="consume-once",
        help="'consume-once' feeds a get_image result to that agent's next call "
        "then discards it, matching the trained contract.",
    )
    partial.add_argument(
        "--resource-locks",
        choices=(RESOURCE_LOCKS_NONE, RESOURCE_LOCKS_OBJECTS, RESOURCE_LOCKS_FIXTURES),
        default=RESOURCE_LOCKS_FIXTURES,
        help="Which shared resources a tool call claims. Conflicts REJECT and "
        "hand the problem back to the agent; they never queue.",
    )
    partial.add_argument(
        "--conflict-priority",
        choices=("agent-order", "seeded"),
        default="agent-order",
        help="Tie-break when two agents claim the same resource. 'agent-order' "
        "lets the lower agent id proceed; 'seeded' shuffles per turn. Rejecting "
        "both is symmetric and livelocks under deterministic retry.",
    )
    partial.add_argument(
        "--duration-multiplier",
        type=float,
        default=1.0,
        help="Scales measured executor sim-steps into virtual-clock duration.",
    )
    partial.add_argument(
        "--uniform-durations",
        action="store_true",
        help="Lock-step: every call costs one tick, so both agents advance "
             "together and only a wait+release can order them. Removes the "
             "timing dimension instead of testing it.",
    )
    partial.add_argument(
        "--duration-jitter",
        type=float,
        default=0.0,
        help="Randomly scale each call's duration by 1+-J. The only knob that "
             "actually changes the interleaving, since the executor teleports "
             "and measured sim-steps never reach the floors.",
    )
    partial.add_argument(
        "--duration-jitter-seed",
        type=int,
        default=0,
        help="Seed for --duration-jitter, so a perturbed schedule is reproducible.",
    )
    partial.add_argument(
        "--rejection-mode",
        choices=(REJECTION_MODE_SILENT_RETRY, REJECTION_MODE_REPORT_FAILED),
        # report-failed by default: with silent-retry the agent's history is
        # unchanged after a rejection, so at temperature 0 it re-emits the
        # identical call. Measured adaptation after a rejection: 8.0% when
        # silent vs 100% once FAILED: is shown (untrained 9.4% -> 94.1%), and
        # 28-32% of every trajectory's budget went to re-emitting known-failed
        # calls. Trained models never saw FAILED: in training, so they are
        # mildly OOD on it; that is the accepted cost of not wasting a third
        # of the budget.
        default=REJECTION_MODE_REPORT_FAILED,
        help="silent-retry: a rejected call leaves state untouched and the agent "
        "sleeps until the next world event, then retries (no distribution "
        "shift). report-failed: append a FAILED: line the model never saw in "
        "training.",
    )
    partial.add_argument(
        "--rejection-budget-ratio",
        type=float,
        default=1.0,
        help="Rejected proposals get their own allowance, sized as this "
        "multiple of the step budget, instead of consuming the budget meant "
        "for productive work.",
    )
    partial.add_argument(
        "--concurrent-expert-replay",
        action="store_true",
        help=(
            "Replay expert trajectories through the timing scheduler instead of "
            "in recorded joint order, so each agent advances its own stream. "
            "Only sound when cross-agent ordering is explicit (wait_for_signal); "
            "with implicit ordering the scheduler re-interleaves correct plans."
        ),
    )
    partial.add_argument(
        "--success-criterion",
        choices=("fsm", "native-and-fsm"),
        default="fsm",
        help="What ends an episode successfully. 'fsm' (default) stops when the "
        "verified task spec's symbolic goal is satisfied; native success is "
        "still recorded as a diagnostic. Native disagreement is a property of "
        "the teleporting executor's geometric predicates, not of the policy, "
        "and requiring both wastes budget after the task is already done.",
    )
    partial.add_argument(
        "--max-silent-retries",
        type=int,
        default=3,
        help="Silent retries before a rejection escalates to a FAILED: entry.",
    )
    parser.add_argument(
        "--two-pass-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-render canonical views for the predicted tool and re-predict.",
    )
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.map_dpi < 1:
        raise SystemExit("--map-dpi must be at least 1")
    if (
        args.get_image_observation_mode != GET_IMAGE_OBSERVATION_MODE_NEXT_TURN
        and not args.train_get_image
    ):
        raise SystemExit(
            "non-default --get-image-observation-mode requires "
            "--train-get-image"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    trajectory_ids_by_task: dict[str, list[str]] = {
        task: sorted(ids)
        for task, ids in manifest["trajectory_ids_by_task"].items()
    }
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",")}
        trajectory_ids_by_task = {
            t: ids for t, ids in trajectory_ids_by_task.items() if t in keep
        }
        if not trajectory_ids_by_task:
            available = ", ".join(sorted(manifest["trajectory_ids_by_task"]))
            raise SystemExit(
                f"--tasks matched no manifest tasks. Available tasks: {available}"
            )

    from training.bc_task_vlm.eval_standalone import (
        _load_few_shot_blocks,
        _load_task_spec_blocks,
    )

    task_names = sorted(trajectory_ids_by_task)
    args.task_spec_blocks = (
        _load_task_spec_blocks(task_names) if args.task_spec_detail else {}
    )
    args.few_shot_blocks = _load_few_shot_blocks(task_names) if args.few_shot else {}

    results_path = args.output_dir / "live_sim_trajectories.jsonl"
    done: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        existing_records = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
        done = _completed_trajectory_keys(existing_records)

    policy = None
    policy_started = time.perf_counter()
    if args.backend == "hf":
        if not args.model_name_or_path:
            raise SystemExit("--backend hf requires --model-name-or-path")
        policy = HfPolicy(args)
    elif args.backend == "vllm":
        if not args.vllm_model:
            raise SystemExit("--backend vllm requires --vllm-model")
        if args.vllm_request_timeout <= 0:
            raise SystemExit("--vllm-request-timeout must be positive")
        policy = VllmPolicy(args)
    elif args.backend == "gemini":
        policy = GeminiPolicy(args)
    elif args.backend == "degenerate":
        policy = DegeneratePolicy()
    policy_load_s = round(time.perf_counter() - policy_started, 3)
    print(f"[timing] policy_load_s={policy_load_s}", flush=True)

    total_done = len(done)
    firsts: dict[tuple[str, bool], bool] = {}
    if args.record_firsts:
        for task_name in trajectory_ids_by_task:
            for success, label in ((True, "success"), (False, "failure")):
                firsts[(task_name, success)] = (
                    args.output_dir / "recordings" / task_name / label
                ).exists()
    with results_path.open("a", encoding="utf-8") as out:
        for task_name, trajectory_ids in trajectory_ids_by_task.items():
            task_metadata = get_task_metadata(task_name)
            composite = task_metadata.composite_task
            session = None
            for trajectory_id in trajectory_ids:
                if (task_name, trajectory_id) in done:
                    continue
                if (
                    args.max_trajectories is not None
                    and total_done >= args.max_trajectories
                ):
                    break
                traj_path = (
                    args.dataset_root
                    / task_metadata.dataset_name
                    / trajectory_id
                    / "original_trajectory.json"
                )
                trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
                if session is None:
                    print(f"[{task_name}] building sim session...", flush=True)
                    session_started = time.perf_counter()
                    try:
                        session = SimSession(
                            composite_task=composite,
                            sample_trajectory=trajectory,
                            layout=args.layout,
                            style=args.style,
                            seed=args.seed,
                            gl_backend=args.gl_backend,
                            render_size=args.render_size,
                            map_dpi=args.map_dpi,
                            map_renderer=args.map_renderer,
                        )
                        session_build_s = round(
                            time.perf_counter() - session_started, 3
                        )
                        print(
                            f"[{task_name}] session_build_s={session_build_s}",
                            flush=True,
                        )
                    except Exception as exc:
                        result = {
                            "task_name": task_name,
                            "composite_task": composite,
                            "trajectory_id": trajectory_id,
                            "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
                            "termination": "harness_error",
                            "native_success": None,
                            "fsm_goal_satisfied": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-2000:],
                        }
                        out.write(json.dumps(result, ensure_ascii=True) + "\n")
                        out.flush()
                        total_done += 1
                        print(
                            f"[{task_name}/{trajectory_id}] session build failed: "
                            f"{result['error']}", flush=True
                        )
                        break
                frames_dir = None
                if args.save_frames:
                    frames_dir = args.output_dir / "frames" / task_name / trajectory_id
                elif args.record_firsts:
                    # Record until both a success and a failure exist for the
                    # task; the final classification renames the directory.
                    if not (
                        firsts.get((task_name, True))
                        and firsts.get((task_name, False))
                    ):
                        frames_dir = (
                            args.output_dir
                            / "recordings"
                            / task_name
                            / trajectory_id
                        )
                started = time.time()
                try:
                    if args.backend == "oracle":
                        trajectory_policy = OraclePolicy(trajectory["steps"])
                    else:
                        trajectory_policy = policy
                    runner = (
                        run_trajectory_partial
                        if getattr(args, "partial_history", False)
                        else run_trajectory
                    )
                    result = runner(
                        session=session,
                        policy=trajectory_policy,
                        args=args,
                        task_name=task_name,
                        composite_task=composite,
                        trajectory=trajectory,
                        frames_dir=frames_dir,
                    )
                except Exception as exc:
                    result = {
                        "task_name": task_name,
                        "composite_task": composite,
                        "trajectory_id": trajectory_id,
                        "termination": "harness_error",
                        "native_success": None,
                        "fsm_goal_satisfied": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                result["elapsed_s"] = round(time.time() - started, 1)
                if "session_build_s" in locals() and session_build_s is not None:
                    result["session_build_s"] = session_build_s
                    session_build_s = None
                out.write(json.dumps(result, ensure_ascii=True) + "\n")
                out.flush()
                total_done += 1
                success = bool(result.get("native_success"))
                _finalize_first_recording(
                    frames_dir=frames_dir,
                    output_dir=args.output_dir,
                    task_name=task_name,
                    success=success,
                    firsts=firsts,
                    record_firsts=args.record_firsts,
                    save_frames=args.save_frames,
                    record_fps=args.record_fps,
                )
                native_error_text = (
                    f" native_error={result['native_error']}"
                    if result.get("native_error")
                    else ""
                )
                print(
                    f"[{task_name}/{trajectory_id}] "
                    f"native={result.get('native_success')}"
                    f"{native_error_text} "
                    f"fsm={result.get('fsm_goal_satisfied')} "
                    f"term={result.get('termination')} "
                    f"({result['elapsed_s']}s)",
                    flush=True,
                )
            if session is not None:
                session.close()

    _write_metrics(
        results_path,
        args.output_dir,
        policy_load_s=policy_load_s,
        total_elapsed_s=round(time.perf_counter() - run_started, 3),
    )


def _write_metrics(
    results_path: Path,
    output_dir: Path,
    *,
    policy_load_s: float | None = None,
    total_elapsed_s: float | None = None,
) -> None:
    raw_records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = _latest_trajectory_records(raw_records)
    n = len(records)
    if not n:
        return
    native = sum(1 for r in records if r.get("native_success"))
    fsm = sum(1 for r in records if r.get("fsm_goal_satisfied"))
    agree = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and bool(r.get("native_success")) == bool(r.get("fsm_goal_satisfied"))
    )
    comparable = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and r.get("fsm_goal_satisfied") is not None
    )
    metrics = {
        "num_trajectories": n,
        "num_jsonl_records": len(raw_records),
        "num_superseded_records": len(raw_records) - n,
        "policy_load_s": policy_load_s,
        "total_elapsed_s": total_elapsed_s,
        "native_success_rate": native / n,
        "fsm_goal_rate": fsm / n,
        "judge_agreement_rate": agree / comparable if comparable else None,
        "declared_complete_rate": sum(
            1 for r in records if r.get("declared_complete")
        ) / n,
        "harness_error_rate": sum(
            1 for r in records if r.get("termination") == "harness_error"
        ) / n,
        "terminations": {},
        "mean_steps_used": sum(r.get("steps_used", 0) for r in records) / n,
        "mean_trajectory_elapsed_s": _mean_present(
            r.get("elapsed_s") for r in records
        ),
        "total_session_build_s": sum(
            r.get("session_build_s", 0.0) for r in records
        ),
        "mean_model_proposal_elapsed_s": _mean_present(
            step.get("proposal_elapsed_s")
            for r in records
            for step in r.get("steps", [])
        ),
        "mean_expert_steps": sum(r.get("expert_steps", 0) for r in records) / n,
        "mean_partial_goal_fraction": sum(
            r.get("partial_goal_fraction", 0.0) for r in records
        ) / n,
        "mean_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio") for r in records
        ),
        "mean_success_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio")
            for r in records
            if r.get("native_success")
        ),
        "rejection_rate": (
            sum(r.get("rejected_steps", 0) for r in records)
            / max(1, sum(r.get("steps_used", 0) for r in records))
        ),
        "agent_turn_counts": {},
        "same_agent_transition_rate": None,
        "same_agent_run_lengths": {},
    }
    turns = [turn for r in records for turn in r.get("agent_turns", [])]
    metrics["agent_turn_counts"] = {
        agent: turns.count(agent) for agent in AGENT_IDS
    }
    same = total_transitions = 0
    run_lengths: dict[str, int] = {}
    for record in records:
        trajectory_turns = record.get("agent_turns", [])
        total_transitions += max(0, len(trajectory_turns) - 1)
        same += sum(a == b for a, b in zip(trajectory_turns, trajectory_turns[1:]))
        for length in _run_lengths(trajectory_turns):
            key = str(length)
            run_lengths[key] = run_lengths.get(key, 0) + 1
    metrics["same_agent_transition_rate"] = (
        same / total_transitions if total_transitions else None
    )
    metrics["same_agent_run_lengths"] = run_lengths

    # Partial-observability scheduler metrics. Only emitted for partial runs so
    # centralized outputs keep their existing shape.
    partial_records = [r for r in records if r.get("partial_history")]
    if partial_records:
        conflicts = sum(r.get("resource_conflicts", 0) for r in partial_records)
        escalations = sum(r.get("escalations", 0) for r in partial_records)
        silent = sum(r.get("silent_resolutions", 0) for r in partial_records)
        rejections = escalations + silent
        cycles = sum(r.get("steps_used", 0) for r in partial_records)
        per_agent: dict[str, dict[str, float]] = {}
        for agent_id in AGENT_IDS:
            active = [
                r["per_agent"][agent_id]["active_sim_time"]
                for r in partial_records
                if r.get("per_agent", {}).get(agent_id)
            ]
            idle = [
                r["per_agent"][agent_id]["idle_sim_time"]
                for r in partial_records
                if r.get("per_agent", {}).get(agent_id)
            ]
            repeats = sum(
                r.get("per_agent", {}).get(agent_id, {}).get("repeated_proposals", 0)
                for r in partial_records
            )
            per_agent[agent_id] = {
                "mean_active_sim_time": _mean_present(active),
                "mean_idle_sim_time": _mean_present(idle),
                "repeated_proposals": repeats,
            }
        metrics["partial"] = {
            "trajectories": len(partial_records),
            "mean_makespan": _mean_present(
                r.get("makespan") for r in partial_records
            ),
            "simultaneous_cycle_rate": (
                sum(r.get("simultaneous_cycles", 0) for r in partial_records)
                / max(1, cycles)
            ),
            "resource_conflicts": conflicts,
            "rejections_observed": rejections,
            # Rates over rejections are only interpretable with enough of them;
            # report the raw count alongside so a ratio from five events is
            # visible as such.
            "silent_resolve_rate": (silent / rejections) if rejections else None,
            "escalation_rate": (escalations / rejections) if rejections else None,
            "conflict_metrics_interpretable": rejections >= 20,
            "per_agent": per_agent,
        }
    for r in records:
        term = r.get("termination", "unknown")
        metrics["terminations"][term] = metrics["terminations"].get(term, 0) + 1
    (output_dir / "live_sim_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _mean_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _run_lengths(turns: list[str]) -> list[int]:
    if not turns:
        return []
    lengths: list[int] = []
    current = turns[0]
    length = 1
    for turn in turns[1:]:
        if turn == current:
            length += 1
        else:
            lengths.append(length)
            current, length = turn, 1
    lengths.append(length)
    return lengths


def _assemble_recording_videos(recording_dir: Path, *, fps: int) -> None:
    """Create one MP4 per camera from the saved step stills."""

    import imageio.v2 as imageio

    for camera_dir in sorted(path for path in recording_dir.iterdir() if path.is_dir()):
        frames = sorted(camera_dir.glob("*.jpg"))
        if not frames:
            continue
        writer = imageio.get_writer(recording_dir / f"{camera_dir.name}.mp4", fps=fps)
        try:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
