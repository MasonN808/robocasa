"""Render shared task prompts from task metadata and task state."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    WAIT_TOOL_NAMES,
    INTERACTION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
)
from .types import TaskInstance, TaskPromptBuilder


def _canonical_arg_groups_for_prompt(
    tool_name: str,
    tool_spec: dict[str, Any],
) -> list[tuple[str, ...]]:
    """Render generation-facing arg alternatives without legacy aliases.

    We keep runtime alias support for backward compatibility, but generation
    prompts should teach one canonical form per semantic role.
    """

    canonical_overrides: dict[str, list[tuple[str, ...]]] = {
        "place_on_surface": [("support_id",)],
        "place_in_receptacle": [("receptacle_id",)],
        "place_on_object": [("support_object_id",)],
        "place_under": [("reference_fixture_id",)],
        "place_next_to": [("reference_object_id", "reference_fixture_id")],
    }
    if tool_name in canonical_overrides:
        return canonical_overrides[tool_name]

    groups: list[tuple[str, ...]] = []
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        normalized_group = tuple(
            arg_name for arg_name in arg_group if isinstance(arg_name, str)
        )
        if normalized_group:
            groups.append(normalized_group)
    return groups


def _format_agent_id_list(agent_ids: Sequence[str]) -> str:
    """Formats agent IDs into a short prompt-facing list."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if len(agent_ids) == 1:
        return agent_ids[0]
    if len(agent_ids) == 2:
        return f"{agent_ids[0]} and {agent_ids[1]}"
    return f"{', '.join(agent_ids[:-1])}, and {agent_ids[-1]}"


def _format_initial_agent_positions(
    initial_state: dict[str, Any],
    agent_ids: Sequence[str],
) -> str:
    """Formats the prompt-facing list of starting fixture positions."""

    position_lines: list[str] = []
    for agent_id in agent_ids:
        agent_state = initial_state.get("agents", {}).get(agent_id, {})
        location = agent_state.get("location")
        location_label = (
            location if isinstance(location, str) and location else "unknown"
        )
        position_lines.append(f"- {agent_id}: {location_label}")
    return "\n".join(position_lines)


def _build_fsm_prompt_rules(
    allowed_tool_specs: dict[str, dict[str, Any]],
    *,
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    task_effects: Sequence[dict[str, Any]] | None = None,
    extra_rules: Sequence[str] | None = None,
) -> list[str]:
    """Builds concise prompt rules that mirror the FSM validator."""

    # Keep the prompt rules derived from the same tool set the validator uses so
    # the model sees the key legality constraints up front.
    allowed_tool_names = set(allowed_tool_specs)
    prompt_rules: list[str] = []

    if "communicate" in allowed_tool_names:
        prompt_rules.append(
            "Before the first task action, both agents must communicate at least once."
        )
    if "navigate_to_fixture" in allowed_tool_names:
        prompt_rules.append(
            "Before interacting with a fixture, surface, receptacle, or dispenser, first navigate to that place."
        )
    if allowed_tool_names & (OPEN_PART_TOOL_NAMES | CLOSE_PART_TOOL_NAMES):
        prompt_rules.append(
            "Only open or close a fixture part after navigating to that same fixture."
        )
    if allowed_tool_names & ACQUIRE_TOOL_NAMES:
        prompt_rules.append(
            "Only use pick_up_object when the object is still at the listed source_id, and never pick up a second object while already holding one."
        )
        prompt_rules.append(
            "After picking up an object, that agent should only navigate or place that same object until it is no longer holding anything."
        )
    if allowed_tool_names & RELEASE_TOOL_NAMES:
        prompt_rules.append(
            "Only use a placement tool for the exact object the acting agent is currently holding."
        )
        prompt_rules.append(
            "Do not use a movable object that any agent is currently holding as a source, support, receptacle, or reference target. If a bowl, plate, tray, or other movable support needs to receive an item, it must be resting on a fixture surface first."
        )
    if allowed_tool_names & GIVE_SPACE_TOOL_NAMES:
        prompt_rules.append(
            "Use give_space at the fixture where that agent is already standing, never after navigating there just to yield. The trigger is that the other agent is about to work at that fixture OR at a fixture paired with it by `parent_fixture`; in both cases the agents communicate first, then the standing agent yields before the other arrives."
        )
        prompt_rules.append(
            "A fixture with a `parent_fixture` in initial_state shares one workspace with that parent, whether it is a cabinet above a counter or an appliance resting on it. If one agent is at either, the other must not navigate into the paired side until the first gives_space from where it stands. When BOTH agents start in that shared workspace, this still applies: before either agent uses one side, the agent that is not about to act must give_space first, at whichever fixture it is standing."
        )
        prompt_rules.append(
            "Do not use give_space while holding an object unless there is no legal way to finish the current placement first."
        )
        prompt_rules.append(
            "After an agent executes give_space, that agent is no longer at the fixture. "
            "Before that agent can interact there again (pick up, place, open, close, or use a control), "
            "it must navigate_to_fixture first. Similarly, the arriving agent must give_space in turn "
            "before the original agent can navigate back."
        )
    if allowed_tool_names & WAIT_TOOL_NAMES:
        # One protocol stated once, as a recipe. This replaced five overlapping
        # prose rules that stated the trigger twice and buried the definition of
        # a conflict last.
        prompt_rules.append(
            "SHARING PROTOCOL. The two agents act at the same time, so whenever they "
            "need the same thing they must hand it over explicitly. A conflict is "
            "exactly one of:\n"
            "  (a) both agents use the SAME object, anywhere;\n"
            "  (b) both agents work at the same TIGHT fixture (cabinet, drawer, "
            "fridge, oven, stove, sink, or a countertop appliance), even on "
            "different objects, because only one robot fits there.\n"
            "Two agents at a roomy counter, island, or dining table using DIFFERENT "
            "objects is NOT a conflict and needs nothing.\n"
            "For every conflict, write these four steps in this order, using the "
            "exact symbolic id X of the thing (never an invented event name):\n"
            "  1. the second agent sends a message naming X, saying it is waiting;\n"
            "  2. the second agent calls wait_for_signal(from=<first agent>, about=X);\n"
            "  3. the first agent finishes with X -- its last use of the object, or "
            "give_space at the fixture;\n"
            "  4. only THEN the first agent sends a message naming X saying it is "
            "done, in a full sentence of at least four words.\n"
            "All four are required. Saying 'I will wait' without step 2 is not "
            "waiting, and sending step 4 before step 3 is a promise, not a release."
        )
        prompt_rules.append(
            "Any message wakes a waiting agent, so if an agent is woken by a message "
            "that is not about what it was waiting for, it must call wait_for_signal "
            "again rather than proceed."
        )
    if allowed_tool_names & INTERACTION_TOOL_NAMES:
        prompt_rules.append(
            "Interaction tools (press_button, press_lever, set_rotary_control) require the agent to be at the target fixture. Navigate to the fixture first."
        )
    for tool_name, tool_spec in allowed_tool_specs.items():
        for arg_group in _canonical_arg_groups_for_prompt(tool_name, tool_spec):
            formatted_group = ", ".join(str(arg_name) for arg_name in arg_group)
            if len(arg_group) == 1:
                prompt_rules.append(
                    f"When using {tool_name}, use {formatted_group} as the canonical anchor arg. Do not include alias alternatives."
                )
            else:
                prompt_rules.append(
                    f"When using {tool_name}, provide exactly one of {formatted_group} and do not include multiple alternatives in the same step."
                )
    if "place_next_to" in allowed_tool_names:
        prompt_rules.append(
            "For place_next_to, prefer reference_object_id for object-relative placements and reference_fixture_id for fixture-relative placements."
        )
        prompt_rules.append(
            "For place_next_to, reference_object_id must name an object from initial_state.objects and reference_fixture_id must name a fixture from initial_state.fixtures. Never put a fixture id in reference_object_id."
        )
    if "place_on_surface" in allowed_tool_names:
        prompt_rules.append(
            "For place_on_surface, use support_id as the fixture anchor and use target_site_id for the exact sub-location such as a rack or burner."
        )
    if "place_in_receptacle" in allowed_tool_names:
        prompt_rules.append(
            "For place_in_receptacle, use receptacle_id as the receptacle anchor and use target_site_id only for the exact interior site."
        )
    if "place_under" in allowed_tool_names:
        prompt_rules.append(
            "For place_under, prefer reference_fixture_id for the fixture and use target_site_id when the exact dispenser or basin site matters."
        )
    # Keep later references aligned with prior FSM effects.
    prompt_rules.append(
        "Keep object locations consistent across steps. After an object moves, later source_id and destination references must match its new location."
    )
    prompt_rules.append(
        "Treat repeated symbolic objects as distinct physical instances. Do not reuse one concrete object to stand in for two different symbolic ids."
    )
    prompt_rules.append(
        "When a partitioned fixture exposes support_sites such as burners, shelves, basins, bowls, slots, trays, or racks, use the concrete source_site_id or target_site_id when the task depends on that exact sub-location."
    )
    prompt_rules.append(
        "If a step specifies a concrete target_site_id or source_site_id, honor that exact site. Do not silently switch to a different site on the same fixture."
    )
    prompt_rules.append(
        "Each id-valued tool arg must be one concrete symbolic id. Do not concatenate fixture ids, site ids, hardware labels, or free text into one field."
    )
    prompt_rules.append(
        "Stop as soon as the goal state is satisfied. Do not add extra task actions afterward."
    )

    for condition in task_preconditions or ():
        condition_kind = condition.get("kind")
        if condition_kind == "fixture_part_state_required_for_pickup":
            prompt_rules.append(
                "Open "
                f"{condition['fixture_id']}.{condition['part_id']} before using "
                f"{condition['tool']} from {condition['source_id']}."
            )
        elif condition_kind == "fixture_part_state_required_for_action":
            arg_name = condition.get("arg_name")
            arg_value = condition.get("arg_value")
            action_scope = ""
            if isinstance(arg_name, str) and isinstance(arg_value, str):
                action_scope = f" when {arg_name}={arg_value}"
            prompt_rules.append(
                "Open "
                f"{condition['fixture_id']}.{condition['part_id']} before using "
                f"{condition['tool']}{action_scope}."
            )
        elif condition_kind == "object_location_required_for_action":
            arg_name = condition.get("arg_name")
            arg_value = condition.get("arg_value")
            action_scope = ""
            if isinstance(arg_name, str) and arg_value is not None:
                action_scope = f" when {arg_name}={arg_value}"
            prompt_rules.append(
                f"Only use {condition['tool']}{action_scope} after "
                f"{condition['object_id']} is already at {condition['required_location']}."
            )

    for effect in task_effects or ():
        if effect.get("kind") != "set_machine_flag_on_action":
            continue
        tool_name = effect.get("tool")
        if not isinstance(tool_name, str):
            continue
        required_object_locations = effect.get("required_object_locations") or ()
        for requirement in required_object_locations:
            if not isinstance(requirement, dict):
                continue
            object_id = requirement.get("object_id")
            location = requirement.get("location")
            if isinstance(object_id, str) and isinstance(location, str):
                prompt_rules.append(
                    f"Only use {tool_name} after {object_id} is already at {location}."
                )
        required_machine_values = effect.get("required_machine_values") or ()
        for requirement in required_machine_values:
            if not isinstance(requirement, dict):
                continue
            machine_path = requirement.get("machine_path")
            value = requirement.get("value")
            if (
                isinstance(machine_path, list)
                and machine_path
                and all(isinstance(part, str) for part in machine_path)
            ):
                machine_path_text = ".".join(machine_path)
                prompt_rules.append(
                    f"Only use {tool_name} after {machine_path_text} is already {value}."
                )

    for rule in extra_rules or ():
        normalized_rule = " ".join(rule.strip().split())
        if normalized_rule:
            prompt_rules.append(normalized_rule)

    deduped_rules: list[str] = []
    seen_rules: set[str] = set()
    for rule in prompt_rules:
        normalized_rule = " ".join(rule.strip().split())
        if not normalized_rule or normalized_rule in seen_rules:
            continue
        seen_rules.add(normalized_rule)
        deduped_rules.append(normalized_rule)
    return deduped_rules


def make_task_prompt_builder(
    *,
    composite_task: str,
    task_goal: str,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, Any],
    non_communicate_tool_names: Sequence[str],
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    task_effects: Sequence[dict[str, Any]] | None = None,
    extra_execution_rules: Sequence[str] | None = None,
    agent_ids: Sequence[str] = ("agent_0", "agent_1"),
) -> TaskPromptBuilder:
    """Builds a reusable task prompt function from the shared prompt template.

    Args:
        composite_task: Task name shown to the model and stored in generated data.
        task_goal: One-sentence goal description for the task.
        initial_state: Initial state presented to the model.
        allowed_tool_specs: Task-specific allowed tools and id constraints.
        non_communicate_tool_names: Non-communication task tools allowed in steps.
        extra_execution_rules: Optional task-specific sequencing rules enforced by validation.
        agent_ids: Ordered agent IDs the task expects the model to simulate.

    Returns:
        A callable that renders the task prompt for a specific variation key.
    """

    agent_id_list_text = _format_agent_id_list(agent_ids)
    agent_count = len(agent_ids)
    _ = non_communicate_tool_names

    def build_prompt(
        variation_key: str,
        task_instance: TaskInstance | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        # Copy the initial state per run so prompt rendering can reflect sampled
        # task instances without mutating the task definition defaults.
        prompt_initial_state = (
            deepcopy(task_instance.initial_state)
            if task_instance is not None
            else deepcopy(initial_state)
        )
        prompt_allowed_tool_specs = (
            deepcopy(task_instance.allowed_tool_specs)
            if task_instance is not None
            and task_instance.allowed_tool_specs is not None
            else deepcopy(allowed_tool_specs)
        )
        prompt_task_goal = (
            task_instance.task_goal
            if task_instance is not None and isinstance(task_instance.task_goal, str)
            else task_goal
        )
        prompt_extra_execution_rules = (
            task_instance.extra_execution_rules
            if task_instance is not None and task_instance.extra_execution_rules
            else tuple(extra_execution_rules or ())
        )
        allowed_tools_text = json.dumps(
            prompt_allowed_tool_specs,
            indent=2,
            sort_keys=True,
        )
        execution_rules_text = "\n".join(
            f"- {rule}"
            for rule in _build_fsm_prompt_rules(
                prompt_allowed_tool_specs,
                task_preconditions=task_preconditions,
                task_effects=task_effects,
                extra_rules=prompt_extra_execution_rules,
            )
        )
        initial_state_text = json.dumps(
            prompt_initial_state,
            indent=2,
            sort_keys=True,
        )
        initial_position_text = _format_initial_agent_positions(
            prompt_initial_state,
            agent_ids,
        )
        prompt = f"""
You are simulating {agent_count} cooperative robot agents in a physical kitchen environment.
Generate a single valid multi-agent task-level trajectory for the composite task {composite_task}.

Important rules:
- Simulate both agents: {agent_id_list_text}.
- Keep track of what object each agent is holding and where the agent's location is at all times.
- Keep track of all agent's locations which can only be at fixture locations. Be sure that the agent is not "teleporting" across the environment to complete tasks; the agent should navigate first via a tool call.
- If agent_A plans to navigate to or use a fixture where agent_B is already positioned, OR a fixture paired with agent_B's fixture by `parent_fixture`, have the agents communicate first about that upcoming action, then have agent_B execute give_space(fixture_id) at the fixture it is standing at, before agent_A acts, so they avoid a location conflict. Do not navigate to a fixture only to call give_space; give_space is only for an agent already there.
- Some fixtures stand on or above another fixture and share its floor space. When a fixture in initial_state.fixtures has a `parent_fixture`, treat that fixture and its parent as ONE shared workspace: if one agent is at either of them, the other agent must not navigate to or use the other until the first agent explicitly gives_space from where it is standing. This applies to cabinets and drawers above a counter and equally to appliances resting on one (toaster oven, coffee machine, and similar).
- This shared-workspace rule applies from step 0, including when BOTH agents start in the same shared workspace. Check the initial agent positions: if both agents begin at the parent fixture (or one at the parent and one at the child), the agent that is not about to act must give_space at the fixture where it stands, before the acting agent navigates to or uses either side. Standing at the parent counter blocks the appliance resting on it, even though they have different fixture ids.
- Prefer finishing a held-object placement before calling give_space. Do not give_space while holding an item unless there is no legal alternative.
- In the initial steps, the agents must coordinate through communication tool calls before any task action. Both agents must communicate during this time.
- Throughout the trajectory, both agents should actively communicate with each other to communicate intentions, plans, and needs, not just in the initial steps.
- Each communicate step sends a message to the other agent in the scene, so args.to must be the exact ID of that other agent.
- For each step, args must contain every required argument for that tool. Only include optional args when they are useful for the placement you are specifying, and do not invent unsupported arg keys.
- In args, use the exact IDs shown in the allowed tools block for this task.
- Keep args as a flat object that contains only that step's tool inputs.
- If an agent is not performing an action, be sure the agent communicates what the agent is waiting for so that no agent is doing nothing.
- Both agents must cooperatively complete the task, a single agent should not do all subtasks.
- Use only the allowed tools for this task.
- Every step must be executable and valid for the current task state.
- Track each agent’s current fixture after every navigation and verify that each non-navigation action matches that current fixture.
- Number steps consecutively starting at 0 with no gaps.
- The reasoning text should explain why the agent is using the tool call from a first-person point-of-view. Each reasoning text must be a single short sentence.
- In reasoning text and communicate.message text, refer to agents using exact IDs like agent_0 and agent_1, not Agent 0 or Agent 1.
- Agents can pass each other freely in the kitchen, including around the island.
- If an agent has no immediate legal task action because it is waiting on the other agent, use communicate to explain the dependency before the other agent proceeds.
- Make this trajectory distinct from previous attempts by following variation key: {variation_key}
- Output JSON only, with no markdown.

Simple execution rules:
{execution_rules_text}

Composite task:
- {composite_task}
- Goal: {prompt_task_goal}

Initial agent positions:
{initial_position_text}

Initial task state:
{initial_state_text}

Allowed tools and exact symbolic arguments for this task:
{allowed_tools_text}

""".strip()
        if retry_feedback is None:
            return prompt
        return f"{prompt}\n\n{retry_feedback.strip()}"

    return build_prompt
