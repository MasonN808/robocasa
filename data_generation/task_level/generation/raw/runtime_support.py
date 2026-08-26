"""Provide shared runtime helpers for validation, sampling, and JSON parsing."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import re
import threading
from typing import Any, Sequence

from data_generation.task_level.generation.raw.config import (
    RETRY_PROGRESS_ERROR_MESSAGE_MAX_LENGTH,
    RuntimeConfig,
    TRAJECTORY_ID_DIGITS,
)
from data_generation.task_level.runtime.client import (
    BATCH_TRAFFIC_TYPE,
    COST_DECIMAL_PLACES,
    GenerationResult,
    GenerationUsage,
    TrajectoryGenerationError,
    _generation_error_status_code,
    build_generation_usage,
)
from data_generation.task_level.sampling import (
    SampledTrajectoryCandidate,
    get_sampling_strategy,
)
from data_generation.task_level.tasks import (
    DuplicateTrajectoryValidationError,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    InsufficientValidUniqueTrajectoriesValidationError,
    ResponseFormatValidationError,
    TaskDefinition,
    ToolArgumentSemanticValidationError,
    TaskValidator,
    TrajectoryValidationError,
    get_task_definition,
    supported_task_names,
)
from data_generation.task_level.tasks.shared.concurrent_fsm import (
    LOCK_STEP,
    ConcurrentTaskValidator,
)
from data_generation.utils import round_cost, stable_json_sha256


class RunExhaustedError(TrajectoryGenerationError):
    """Raised when one run fails permanently but the task can continue."""


class TaskGenerationCancelledError(TrajectoryGenerationError):
    """Raised when another task failure cancels the current task's work."""


def _raise_if_task_cancelled(runtime_config: RuntimeConfig) -> None:
    """Stops cooperative task execution once a sibling task has already failed."""

    cancel_event = runtime_config.task_cancellation_event
    if cancel_event is not None and cancel_event.is_set():
        raise TaskGenerationCancelledError(
            "Task generation cancelled because another task in the same request failed."
        )


def format_trajectory_id(trajectory_index: int) -> str:
    """Formats one persisted trajectory ID with enough padding for large runs."""

    return f"traj_{trajectory_index:0{TRAJECTORY_ID_DIGITS}d}"


def format_trajectory_variation_key(
    trajectory_index: int,
    attempt_index: int,
) -> str:
    """Formats the retry variation key used to diversify model attempts."""

    return (
        f"traj-{trajectory_index:0{TRAJECTORY_ID_DIGITS}d}-attempt-{attempt_index:02d}"
    )


def format_trajectory_progress_label(trajectory_index: int) -> str:
    """Formats the short progress-bar label for one generation run."""

    return (
        format_trajectory_id(trajectory_index).replace("traj", "run").replace("_", " ")
    )


def build_saved_trajectory_metadata(
    *,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> dict[str, Any]:
    """Builds top-level metadata persisted on each saved raw trajectory."""

    return {
        "task": task_definition.composite_task,
    }


def _retry_feedback_step_lines(
    candidate: dict[str, Any] | None,
    *,
    failing_step: int | None,
) -> list[str]:
    """Builds a tiny local counterexample around the failing step when possible."""

    if not isinstance(failing_step, int) or not isinstance(candidate, dict):
        return []
    steps = candidate.get("steps")
    if not isinstance(steps, list) and isinstance(candidate.get("ticks"), list):
        steps = []
        for row in candidate["ticks"]:
            if not isinstance(row, dict):
                continue
            for agent_id in sorted(key for key in row if key != "tick"):
                action = row.get(agent_id)
                if not isinstance(action, dict) or not action.get("tool"):
                    continue
                steps.append(
                    {
                        **action,
                        "agent": agent_id,
                        "step": len(steps),
                    }
                )
    if not isinstance(steps, list):
        return []

    step_by_index = {
        step["step"]: step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("step"), int)
    }
    lines: list[str] = []
    previous_step = step_by_index.get(failing_step - 1)
    if previous_step is not None:
        lines.append(
            f"- step {previous_step['step']}: {json.dumps(previous_step, sort_keys=True)}"
        )
    failing_step_payload = step_by_index.get(failing_step)
    if failing_step_payload is not None:
        lines.append(
            f"- step {failing_step_payload['step']}: {json.dumps(failing_step_payload, sort_keys=True)}"
        )
    return lines


def _retry_feedback_candidate_lines(
    candidate: dict[str, Any] | None,
) -> list[str]:
    """Includes the complete latest rejection so the model can repair what it made."""

    if not isinstance(candidate, dict):
        return []
    return [
        "Latest rejected trajectory (complete):",
        "```json",
        json.dumps(candidate, indent=2, sort_keys=True),
        "```",
    ]


def _semantic_retry_error(error_type: str | None) -> bool:
    """Semantic validators know what failed, but not which valid repair is best."""

    return bool(
        error_type == "TrajectoryValidationError"
        or (isinstance(error_type, str) and "SemanticValidationError" in error_type)
    )


# Targeted repair hints keyed by TrajectoryValidationError subclass name.
# These are appended to the generic repair instructions so the model gets
# error-specific guidance. Add a new entry when an error type starts appearing
# often enough that the generic text is not enough to unblock retries.
_ERROR_TYPE_TARGETED_GUIDANCE: dict[str, list[str]] = {
    "PostGoalActionSemanticValidationError": [
        "- The trajectory's goal state was already satisfied before this step.",
        "- Check `details.goal_satisfied_at_step` to see which earlier step first satisfied the goal.",
        "- The trajectory must END at the step that first satisfies the goal — no further manipulation or placement steps are allowed after that point.",
        "- Only observation tools may appear after the goal is reached; do not pad with extra actions, re-checks, or re-placements.",
        "- Rework the plan so the final task action is exactly the one that satisfies the goal, then stop.",
    ],
    "ResourceConflictSemanticValidationError": [
        "- Drawers, fridges, front-facing appliances, sinks, and stoves may be exclusive and require one current user plus a completed handover. Cabinets and their parent counters are shared workspaces: different-object access may overlap, but two agents must not manipulate the same cabinet door or the same object in one tick.",
        "- Any wait, matching release, and resumed action must occur on distinct ticks in that temporal order; the waiter remains blocked on the release tick.",
    ],
    "WaitSignalSemanticValidationError": [
        "- A wait must precede its matching release, and a resumed action must follow the release; all three occur on distinct ticks.",
        "- The waiter remains blocked on the release tick. Do not wait after give_space.",
    ],
    "CommunicationStepSemanticValidationError": [
        "- communicate.to must be the other agent: agent_0 sends only to agent_1, and agent_1 sends only to agent_0.",
    ],
}


def _compact_retry_constraint(
    error_type: str,
    message: str | None,
    details: dict[str, Any] | None,
) -> str:
    """Turn one failure into a factual observation retained across retries.

    This ledger deliberately avoids prescribing a repair.  A validator can say
    what was invalid, but navigation, ownership, and scheduling often admit
    several valid fixes and only the generator has the complete current plan.
    """

    details = details or {}
    if details.get("missing_ids"):
        required = details.get("required_by_agent") or {}
        assignments = [
            f"{agent} handles {', '.join(map(str, ids))}"
            for agent, ids in required.items() if ids
        ]
        if assignments:
            return "A prior opening proposal omitted required assignments: " + "; ".join(assignments) + "."
        return "A prior opening proposal omitted required IDs: " + ", ".join(
            str(value) for value in details["missing_ids"]
        )
    if error_type == "CommunicationStepSemanticValidationError":
        return "A prior communication addressed its sender rather than the other agent."
    if error_type == "WaitSignalSemanticValidationError":
        return "A prior attempt violated the validator's wait/release ordering."
    if error_type == "ResourceConflictSemanticValidationError":
        return "A prior attempt placed both agents at the same exclusive fixture."
    if error_type == "NavigationSemanticValidationError":
        agent = details.get("agent") or "the agent"
        expected = details.get("expected_location")
        tool = details.get("tool") or "physical call"
        ids = ", ".join(map(str, details.get("actual_ids") or ()))
        return (
            f"A prior attempt called {agent}'s {tool}({ids}) from the wrong location"
            + (f"; the validator expected {expected}." if expected else ".")
        )
    if error_type == "PostGoalActionSemanticValidationError":
        return "A prior attempt continued acting after the FSM goal was first satisfied."
    if error_type == "TaskPreconditionSemanticValidationError":
        held_receptacle = re.search(
            r"receptacle_id=([A-Za-z0-9_]+) because \1 is currently held",
            str(message or ""),
        )
        if held_receptacle:
            receptacle = held_receptacle.group(1)
            return f"A prior attempt used {receptacle} as a receptacle while it was being held."
        return "A prior attempt violated this physical prerequisite: " + str(message or error_type).split("\n")[0]
    pairs = details.get("wait_release_pairs") or []
    unnecessary = [
        pair for pair in pairs if not pair.get("resource_requires_handover", True)
    ]
    if unnecessary:
        pair = unnecessary[0]
        return f"A prior attempt used an unnecessary handover for roomy resource {pair.get('resource')}."
    if details.get("action_issue") == "missing_ids":
        agent = details.get("agent") or "the agent"
        tool = details.get("tool") or "action"
        missing = ", ".join(map(str, details.get("missing_action_ids") or ()))
        return f"A prior assigned {agent} {tool} call omitted required IDs: {missing}."
    if details.get("action_issue") == "duplicate":
        agent = details.get("agent") or "the agent"
        tool = details.get("tool") or "action"
        return f"A prior attempt duplicated {agent}'s assigned {tool} call."
    if "not assigned to it by the generation work partition" in str(message or ""):
        agent = details.get("agent") or "that agent"
        owner = details.get("assigned_agent") or "the assigned agent"
        tool = details.get("tool") or "action"
        ids = ", ".join(map(str, details.get("actual_ids") or ()))
        return f"A prior attempt assigned {agent}'s {tool}({ids}) action to {owner}, violating the sampled work partition."
    return "Prior validator observation: " + str(message or error_type).split("\n")[0][:240]


def _targeted_guidance_lines(error_type: str | None) -> list[str]:
    """Returns error-specific repair hints for the given error class name."""

    if not isinstance(error_type, str):
        return []
    return list(_ERROR_TYPE_TARGETED_GUIDANCE.get(error_type, ()))


def _specific_retry_guidance(
    *,
    error_type: str,
    message: str | None,
    details: dict[str, Any] | None,
    repeated: bool,
    prior_constraints: Sequence[str] = (),
) -> list[str]:
    """Concrete repair text using IDs from the actual failed trajectory."""

    details = details or {}
    lines: list[str] = []
    if error_type == "NavigationSemanticValidationError":
        agent = details.get("agent")
        current = details.get("current_location") or "an unknown location"
        expected = details.get("expected_location")
        tool = details.get("tool")
        ids = ", ".join(map(str, details.get("actual_ids") or ()))
        ownership_correction = (
            f"Remove {agent}'s {tool}({ids});"
            if ids else None
        )
        if tool == "give_space":
            if details.get("partner_needs_location"):
                lines.append(
                    f"- {agent} is now at {current}, but the partner needs {expected}. "
                    f"Move give_space({expected}) to the last tick where {agent} is still "
                    f"at {expected}, immediately before it leaves; remove this later remote call."
                )
            else:
                lines.append(
                    f"- {agent} is at {current}, so remove its unnecessary remote "
                    f"give_space({expected}); the partner has no assigned work there."
                )
        elif expected and not (
            ownership_correction
            and any(
                constraint.startswith(ownership_correction)
                for constraint in prior_constraints
            )
        ):
            lines.append(
                f"- {agent} is at {current}. Immediately before the rejected {tool}, "
                f"insert navigate_to_fixture(fixture_id={expected!r})."
            )
        elif expected:
            lines.append(
                f"- Do not insert navigation for {agent}'s rejected {tool}. "
                "An earlier work-partition correction says this agent must remove that action."
            )
    if "not assigned to it by the generation work partition" in str(message or ""):
        agent = details.get("agent") or "the acting agent"
        owner = details.get("assigned_agent") or "the assigned partner"
        tool = details.get("tool") or "action"
        ids = ", ".join(map(str, details.get("actual_ids") or ()))
        lines.append(
            f"- Remove {agent}'s {tool}({ids}). Keep that action with {owner}; "
            "do not navigate the wrong agent there to make the call executable."
        )
    own_last = details.get("own_last_physical_tick")
    if own_last is not None and "portion_complete message" in str(message or ""):
        agent = details.get("agent") or "the finished agent"
        partner = details.get("partner") or "the partner"
        lines.append(
            f"- {agent}'s physical work ends at tick {own_last}. On its next invoked tick, "
            "send one communicate(coordination_phase='portion_complete') saying only its "
            f"own part is done. If {partner} still has work, its following call must be "
            "wait_for_signal; then use blocked markers until the FSM goal is reached."
        )
    same_tick = details.get("same_tick_wait_release")
    if same_tick:
        pairs = details.get("wait_release_pairs") or []
        if pairs:
            for pair in pairs:
                waiter = pair.get("waiter")
                releaser = pair.get("releaser")
                resource = pair.get("resource")
                if not pair.get("resource_requires_handover", True):
                    lines.append(
                        f"- {resource} is roomy and does not require exclusive access. "
                        f"Remove {waiter}'s wait for {resource} and remove {releaser}'s "
                        "matching release; both agents may continue with their assigned work."
                    )
                else:
                    lines.append(
                        f"- For {resource}, {waiter}'s wait, {releaser}'s matching "
                        "release, and the waiter's resumed action must occur on three "
                        "different ticks, in that order. The waiter remains blocked "
                        "during the release tick."
                    )
        else:
            lines.append(
                f"- Tick {details.get('tick')} makes {', '.join(map(str, same_tick))} wait "
                "and receive its release together. Move the release to a later tick, keep "
                "the waiter blocked in that release tick, and resume it one tick later."
            )
    if details.get("missing_ids"):
        coordinator = details.get("coordinator") or "the coordinator"
        required = details.get("required_by_agent") or {}
        assignments = [
            f"{agent} handles {', '.join(map(str, ids))}"
            for agent, ids in required.items() if ids
        ]
        if assignments:
            lines.append(
                f"- Replace {coordinator}'s tick-0 proposal message with this literal "
                "minimum text: \"" + "; ".join(assignments) + ".\""
            )
    if error_type in {"WaitSignalSemanticValidationError", "DeadlockSemanticValidationError"}:
        lines.append(
            "- Audit each wait: it is valid only if a matching later release arrives, "
            "or the partner eventually satisfies the FSM goal while the waiter remains blocked."
        )
    if repeated and error_type == "ResourceConflictSemanticValidationError":
        text = str(message or "")
        agents = list(dict.fromkeys(re.findall(r"agent_\d+", text)))
        resources = re.findall(r"'([A-Za-z0-9_]+)'", text)
        resource = resources[0] if resources else "the exclusive fixture"
        if len(agents) >= 2:
            lines.append(
                f"- Preserve the sampled action owners. At the failing point, determine from the state which of {agents[0]} and {agents[1]} currently holds {resource}; do not swap their assigned actions.",
            )
    if repeated and details.get("missing_ids"):
        required = details.get("required_by_agent") or {}
        assignments = [
            f"{agent} handles {', '.join(map(str, ids))}"
            for agent, ids in required.items() if ids
        ]
        if assignments:
            lines.append(
                "- Use this minimum opening proposal, without changing ownership: “"
                + "; ".join(assignments) + ".”"
            )
    return lines


def _focused_repair_lines(
    *,
    error_type: str,
    failing_step: int | None,
    details: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    allowed_tool_specs: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Builds validator-guided repair context, with a structural fallback."""

    structural = (
        not isinstance(candidate, dict)
        or "StructureValidationError" in error_type
        or "ResponseFormatValidationError" in error_type
    )
    if structural:
        return [
            "",
            "Repair instructions:",
            "- Regenerate the full trajectory from step 0 using the required schema.",
            "- Do not continue or return a patch to the malformed attempt.",
            "- Keep the whole trajectory symbolically consistent.",
        ]

    lines: list[str] = []
    missing_ids = details.get("missing_ids") if isinstance(details, dict) else None
    if isinstance(missing_ids, list) and missing_ids:
        return [
            "",
            "Focused correction:",
            "- The opening proposal omitted these required IDs: "
            + ", ".join(str(value) for value in missing_ids)
            + ".",
            "- Explicitly assign those IDs in the coordinator's tick-0 proposal.",
            "- Return the COMPLETE corrected trajectory, not a patch or continuation.",
        ]
    if isinstance(details, dict) and details.get("state_before_step") is not None:
        lines.extend(
            [
                "",
                "Symbolic state immediately before the failing step:",
                json.dumps(details["state_before_step"], sort_keys=True),
            ]
        )

    failing_tool = None
    steps = candidate.get("steps")
    if not isinstance(steps, list) and isinstance(candidate.get("ticks"), list):
        steps = []
        for row in candidate["ticks"]:
            if not isinstance(row, dict):
                continue
            for agent_id in sorted(key for key in row if key != "tick"):
                action = row.get(agent_id)
                if isinstance(action, dict) and action.get("tool"):
                    steps.append({**action, "agent": agent_id, "step": len(steps)})
    if isinstance(steps, list) and isinstance(failing_step, int):
        failing_payload = next(
            (
                step
                for step in steps
                if isinstance(step, dict) and step.get("step") == failing_step
            ),
            None,
        )
        if isinstance(failing_payload, dict):
            failing_tool = failing_payload.get("tool")
    if (
        isinstance(failing_tool, str)
        and isinstance(allowed_tool_specs, dict)
        and failing_tool in allowed_tool_specs
    ):
        lines.extend(
            [
                "",
                f"Exact allowed contract for {failing_tool}:",
                json.dumps(allowed_tool_specs[failing_tool], sort_keys=True),
            ]
        )

    correction_lines = [
            "",
            "Focused correction:",
            "- Correct the identified failing step and any prerequisite or later step that must change because of it.",
            "- Regenerate the full trajectory from step 0.",
            "- Return the COMPLETE corrected trajectory, not a patch or continuation.",
    ]
    if isinstance(details, dict) and details.get("same_tick_wait_release"):
        correction_lines.extend(
            [
                "- Keep the task goal and chosen object ownership, but rebuild the timing around the named resource.",
                "- Do not copy the rejected wait/release tick unchanged.",
            ]
        )
    else:
        correction_lines.append(
            "- Preserve valid portions unless changing them is necessary for the correction."
        )
    correction_lines.append(
        "- Use only arguments declared by the selected tool's exact contract."
    )
    lines.extend(correction_lines)
    return lines


def _build_retry_feedback_text(
    exc: TrajectoryValidationError,
    *,
    candidate: dict[str, Any] | None = None,
    allowed_tool_specs: dict[str, dict[str, Any]] | None = None,
    prior_constraints: Sequence[str] = (),
    feedback_style: str = "targeted",
) -> str:
    """Builds the compact retry block appended to the next generation prompt."""

    lines = [
        "Previous attempt failed validation.",
        "",
        "Failure summary:",
        f"- error_type: {exc.error_type}",
    ]
    if isinstance(exc.step, int):
        lines.append(f"- failing_step: {exc.step}")
    if str(exc).strip():
        lines.append(f"- message: {str(exc).strip()}")
    if exc.details:
        lines.append(f"- details: {json.dumps(exc.details, sort_keys=True)}")
    if prior_constraints:
        lines.extend(["", "Earlier validator observations to audit (not repair instructions):"])
        lines.extend(f"- {constraint}" for constraint in dict.fromkeys(prior_constraints))
        repeated = [
            constraint for constraint in dict.fromkeys(prior_constraints)
            if prior_constraints.count(constraint) >= 2
        ]
        if repeated:
            lines.extend([
                "",
                "Repeated-failure escalation:",
                "- The same kind of violation has appeared more than once. Re-check the complete revised trajectory for it before returning.",
            ])
    else:
        repeated = []

    # Repair is much easier when the model can see the complete plan it made.
    # A two-step excerpt hid the ownership and wait/release context responsible
    # for many later failures.
    candidate_lines = _retry_feedback_candidate_lines(candidate)
    if candidate_lines:
        lines.extend(["", *candidate_lines])

    targeted_lines = _targeted_guidance_lines(exc.error_type)
    if feedback_style == "targeted":
        targeted_lines.extend(_specific_retry_guidance(
            error_type=exc.error_type,
            message=str(exc),
            details=exc.details,
            repeated=bool(repeated),
            prior_constraints=prior_constraints,
        ))
    elif _semantic_retry_error(exc.error_type):
        targeted_lines = []
    if targeted_lines:
        lines.extend(["", "Targeted guidance:", *targeted_lines])

    lines.extend(
        _focused_repair_lines(
            error_type=exc.error_type,
            failing_step=exc.step,
            details=exc.details,
            candidate=candidate,
            allowed_tool_specs=allowed_tool_specs,
        )
    )
    return "\n".join(lines)


def _build_retry_feedback_text_from_validation(
    validation: dict[str, Any],
    *,
    candidate: dict[str, Any] | None = None,
    allowed_tool_specs: dict[str, dict[str, Any]] | None = None,
    prior_constraints: Sequence[str] = (),
    feedback_style: str = "targeted",
) -> str:
    """Builds retry feedback directly from a serialized validation payload."""

    error_type = _validation_error_type(validation) or "TrajectoryValidationError"
    step = validation.get("step")
    error_message = validation.get("error")
    error_details = validation.get("error_details")

    lines = [
        "Previous attempt failed validation.",
        "",
        "Failure summary:",
        f"- error_type: {error_type}",
    ]
    if isinstance(step, int):
        lines.append(f"- failing_step: {step}")
    if isinstance(error_message, str) and error_message.strip():
        lines.append(f"- message: {error_message.strip()}")
    if isinstance(error_details, dict) and error_details:
        lines.append(f"- details: {json.dumps(error_details, sort_keys=True)}")
    if prior_constraints:
        lines.extend(["", "Earlier validator observations to audit (not repair instructions):"])
        lines.extend(f"- {constraint}" for constraint in dict.fromkeys(prior_constraints))
        repeated = [
            constraint for constraint in dict.fromkeys(prior_constraints)
            if prior_constraints.count(constraint) >= 2
        ]
        if repeated:
            lines.extend([
                "",
                "Repeated-failure escalation:",
                "- The same kind of violation has appeared more than once. Re-check the complete revised trajectory for it before returning.",
            ])
    else:
        repeated = []

    candidate_lines = _retry_feedback_candidate_lines(candidate)
    if candidate_lines:
        lines.extend(["", *candidate_lines])

    targeted_lines = _targeted_guidance_lines(error_type)
    if feedback_style == "targeted":
        targeted_lines.extend(_specific_retry_guidance(
            error_type=error_type,
            message=error_message if isinstance(error_message, str) else None,
            details=error_details if isinstance(error_details, dict) else None,
            repeated=bool(repeated),
            prior_constraints=prior_constraints,
        ))
    elif _semantic_retry_error(error_type):
        targeted_lines = []
    if targeted_lines:
        lines.extend(["", "Targeted guidance:", *targeted_lines])

    lines.extend(
        _focused_repair_lines(
            error_type=error_type,
            failing_step=step if isinstance(step, int) else None,
            details=error_details if isinstance(error_details, dict) else None,
            candidate=candidate,
            allowed_tool_specs=allowed_tool_specs,
        )
    )
    return "\n".join(lines)


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    if isinstance(exc, TrajectoryGenerationError):
        return True

    status_code = _generation_error_status_code(exc)
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return True

    text = str(exc)
    non_retryable_markers = (
        "400 INVALID_ARGUMENT",
        "403 PERMISSION_DENIED",
        "401 UNAUTHENTICATED",
        "429 RESOURCE_EXHAUSTED",
        "DefaultCredentialsError",
        "credentials were not found",
        "aiplatform.endpoints.predict",
        "API has not been used",
        "SERVICE_DISABLED",
    )
    return any(marker in text for marker in non_retryable_markers)


def _resolve_task_definition_or_raise(composite_task: str) -> TaskDefinition:
    # Keep CLI parsing and generation on the same task registry lookup path.
    task_definition = get_task_definition(composite_task)
    if task_definition is None:
        supported_tasks = ", ".join(supported_task_names())
        raise TrajectoryGenerationError(
            f"Unsupported task '{composite_task}'. "
            f"Available tasks: {supported_tasks}."
        )
    return task_definition


def _resolve_task_definitions_or_raise(
    composite_tasks: tuple[str, ...],
) -> tuple[TaskDefinition, ...]:
    """Resolves every selected task definition in CLI order."""

    return tuple(
        _resolve_task_definition_or_raise(composite_task)
        for composite_task in composite_tasks
    )


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return stable_json_sha256(candidate, default=str)


def _default_traffic_type_for_runtime(runtime_config: RuntimeConfig) -> str:
    if runtime_config.batch_processing:
        return BATCH_TRAFFIC_TYPE
    return "ON_DEMAND"


def _sampling_strategy_for_runtime(runtime_config: RuntimeConfig):
    """Returns the configured sampling strategy for the current runtime."""

    return get_sampling_strategy(getattr(runtime_config, "sampling", "base"))


def _requested_run_indices(runtime_config: RuntimeConfig) -> tuple[int, ...]:
    """Returns the run indices that this invocation should execute."""

    if runtime_config.run_indices:
        return tuple(runtime_config.run_indices)
    return tuple(range(runtime_config.num_runs))


def _requested_run_count(runtime_config: RuntimeConfig) -> int:
    """Returns how many runs this invocation should execute."""

    return len(_requested_run_indices(runtime_config))


def _trajectories_per_run(runtime_config: RuntimeConfig) -> int:
    """Returns how many saved trajectories one successful run should emit."""

    return _sampling_strategy_for_runtime(runtime_config).trajectories_per_run(
        runtime_config
    )


def _expected_saved_trajectory_count(runtime_config: RuntimeConfig) -> int:
    """Returns the expected saved trajectory count for one successful job."""

    return _requested_run_count(runtime_config) * _trajectories_per_run(runtime_config)


def _global_trajectory_index(
    runtime_config: RuntimeConfig,
    *,
    run_index: int,
    candidate_index: int,
) -> int:
    """Builds the flattened saved-trajectory index for one run candidate."""

    return (run_index * _trajectories_per_run(runtime_config)) + candidate_index


def _sampling_metadata_for_candidate(
    *,
    runtime_config: RuntimeConfig,
    sampled_candidate: SampledTrajectoryCandidate,
    candidate_index: int,
    run_index: int,
) -> dict[str, Any] | None:
    """Builds optional saved metadata for sampling strategy outputs."""

    if (
        sampled_candidate.probability is None
        and sampled_candidate.sampling_configuration is None
        and sampled_candidate.sampling_seed is None
        and sampled_candidate.sampling_attempt_number is None
    ):
        return None

    sampling_metadata: dict[str, Any] = {
        "strategy": runtime_config.sampling,
        "candidate_index": candidate_index,
        "run_index": run_index,
    }
    if sampled_candidate.probability is not None:
        sampling_metadata["probability"] = sampled_candidate.probability
    if sampled_candidate.sampling_configuration is not None:
        sampling_metadata["configuration"] = sampled_candidate.sampling_configuration
    if sampled_candidate.sampling_seed is not None:
        sampling_metadata["seed"] = sampled_candidate.sampling_seed
    if sampled_candidate.sampling_attempt_number is not None:
        sampling_metadata["attempt_number"] = (
            sampled_candidate.sampling_attempt_number
        )
    return sampling_metadata


def _build_trajectory_record_from_candidate(
    *,
    trajectory_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    task_instance: Any,
    candidate: dict[str, Any],
    prompt: str,
    raw_output: Any,
    usage: GenerationUsage | None,
    validator: TaskValidator,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
    attempt_number: int,
) -> dict[str, Any]:
    # Reuse the shared sampled-candidate path so base and verbalized modes keep
    # one normalization and validation implementation.
    trajectory_records = _build_trajectory_records_from_sampled_candidates(
        run_index=trajectory_index,
        runtime_config=runtime_config,
        task_definition=task_definition,
        task_instance=task_instance,
        sampled_candidates=[
            SampledTrajectoryCandidate(
                candidate=candidate,
                raw_output=raw_output,
            )
        ],
        prompt=prompt,
        raw_response=raw_output,
        usage=usage,
        validator=validator,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
        attempt_number=attempt_number,
    )
    return trajectory_records[0]


def _exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _validation_error_type(validation: dict[str, Any]) -> str | None:
    error_type = validation.get("error_type")
    if isinstance(error_type, str) and error_type:
        return error_type
    return None


def _validation_error_base_type(validation: dict[str, Any]) -> str | None:
    """Returns the top-level validation family when one is available."""

    error_base_type = validation.get("error_base_type")
    if isinstance(error_base_type, str) and error_base_type:
        return error_base_type
    return None


def _validation_error_summary(validation: dict[str, Any]) -> str | None:
    error_type = _validation_error_type(validation)
    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message:
        if error_type is not None:
            return f"{error_type}: {error_message}"
        return error_message
    return error_type


def _validation_error_progress_summary(validation: dict[str, Any]) -> str | None:
    """Builds a compact validation summary for progress bars."""

    error_type = _validation_error_type(validation)
    step = validation.get("step")
    if error_type is not None:
        if isinstance(step, int):
            return f"{error_type} step={step}"
        return error_type

    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message:
        return error_message.split(":", 1)[0]
    return None


def _truncate_progress_text(text: str, *, max_length: int) -> str:
    """Collapses whitespace and truncates long progress-bar text with an ellipsis."""

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return f"{normalized_text[: max_length - 3].rstrip()}..."


def _validation_error_retry_summary(validation: dict[str, Any]) -> str | None:
    """Builds a retry status summary that includes the failing validation message."""

    progress_summary = _validation_error_progress_summary(validation)
    error_message = validation.get("error")
    if isinstance(error_message, str) and error_message.strip():
        truncated_message = _truncate_progress_text(
            error_message,
            max_length=RETRY_PROGRESS_ERROR_MESSAGE_MAX_LENGTH,
        )
        if progress_summary is not None:
            return f"{progress_summary}: {truncated_message}"
        return truncated_message
    return progress_summary


def _validation_errors_retry_summary(validations: list[dict[str, Any]]) -> str | None:
    """Aggregates one attempt's invalid validations into a compact retry suffix."""

    distinct_summaries: list[str] = []
    for validation in validations:
        retry_summary = _validation_error_retry_summary(validation)
        if retry_summary is None or retry_summary in distinct_summaries:
            continue
        distinct_summaries.append(retry_summary)

    if not distinct_summaries:
        return None
    if len(distinct_summaries) == 1:
        return distinct_summaries[0]

    displayed_summaries = "; ".join(distinct_summaries[:2])
    if len(distinct_summaries) > 2:
        return f"{displayed_summaries}; +{len(distinct_summaries) - 2} more"
    return displayed_summaries


def _unwrap_generation_response(
    raw_response: Any,
) -> tuple[Any, GenerationUsage | None]:
    # Both SDK wrappers and test doubles feed through here, so normalize the
    # payload shape before sampling code looks at it.
    if isinstance(raw_response, GenerationResult):
        return raw_response.payload, raw_response.usage
    return raw_response, None


def _is_tick_format(candidate: dict[str, Any]) -> bool:
    """Tick output is identified by shape, so no flag has to be threaded here."""

    return isinstance(candidate, dict) and isinstance(candidate.get("ticks"), list)


def _flatten_tick_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Turn tick rows into the numbered step list the FSM validates."""

    from tick_format import to_steps

    agent_ids = [
        entry.get("agent")
        for entry in (candidate.get("agents") or [])
        if isinstance(entry, dict) and entry.get("agent")
    ]
    if not agent_ids:
        # A set first: an agent appearing in several rows must still be
        # listed once, or to_steps emits its action once per occurrence.
        agent_ids = sorted({
            key
            for row in candidate["ticks"]
            if isinstance(row, dict)
            for key in row
            if key != "tick"
        })
    flattened = dict(candidate)
    flattened["steps"] = to_steps(candidate["ticks"], agent_ids)
    # Keep the rows: they are what the model actually produced, and the tick
    # numbering is not recoverable from the flat list once waits move.
    flattened["tick_rows"] = candidate["ticks"]
    flattened.pop("ticks", None)
    return flattened


def _derive_coordination(
    candidate: dict[str, Any],
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Fill in the waits and releases before the trajectory is validated.

    The FSM requires a wait wherever two agents share a resource, but the model
    is not asked to produce one -- placement is decidable from the plan, so
    asking is pure waste. It costs tokens, and the FSM's own repair loop
    oscillates on it: the model adds the wait it was told about, that wait is
    then unreleased, and the next attempt removes it again.

    Deriving here, before validation, means a retry is only ever spent on the
    work itself. Measured over the 1560-trajectory subset, this converts 836
    wait-protocol failures into passes and breaks nothing.
    """

    import random

    from insert_waits import insert

    steps = candidate.get("steps") or []
    if not steps:
        return candidate
    # Seeded from the plan so phrasing varies between trajectories but stays
    # reproducible for any one of them.
    seed = hashlib.sha256(
        json.dumps(steps, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    derived, _ = insert(
        deepcopy(steps),
        initial_state.get("fixtures") or {},
        set(initial_state.get("objects") or {}),
        random.Random(int(seed[:16], 16)),
        {
            agent: (state or {}).get("location")
            for agent, state in (initial_state.get("agents") or {}).items()
        },
    )
    updated = dict(candidate)
    updated["steps"] = derived
    return updated


def _validate_candidate(
    candidate: dict[str, Any],
    validator: TaskValidator,
    *,
    enforce_validation: bool,
    enable_static_referential_validation: bool = True,
    initial_state: dict[str, Any] | None = None,
    allowed_tool_specs: dict[str, dict[str, Any]] | None = None,
    derive_coordination: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrote_ticks = _is_tick_format(candidate)
    if wrote_ticks:
        # The concurrent FSM owns the canonical tick representation and does
        # the only permitted flattening after invocation state is validated.
        # ...and judge it with the validator that actually RUNS the plan. The
        # linear one replays the flat list in written order, where two agents
        # never collide and no wait ever blocks, so it cannot see whether the
        # coordination the model placed works. Under it, tick output that
        # deadlocks on the first tick validated clean. Raising here is what
        # makes the generation loop retry, which is the only way the data ends
        # up correct while the model is still allowed to be wrong.
        # LOCK_STEP only: the model writes a grid of equal instants, so that
        # is the regime its plan means. See ConcurrentTaskValidator.__init__.
        validator = ConcurrentTaskValidator(validator, models=(LOCK_STEP,))
    elif derive_coordination and isinstance(initial_state, dict):
        candidate = _derive_coordination(candidate, initial_state)
    try:
        if (
            enforce_validation
            and enable_static_referential_validation
            and isinstance(initial_state, dict)
            and isinstance(allowed_tool_specs, dict)
        ):
            reference_candidate = (
                validator.canonicalize(candidate) if wrote_ticks else candidate
            )
            _validate_candidate_references_without_sim(
                reference_candidate,
                initial_state=initial_state,
                allowed_tool_specs=allowed_tool_specs,
            )
        validation = dict(validator.validate(candidate))
        normalized_candidate = validation.pop("normalized_candidate", candidate)
        if wrote_ticks and "tick_rows" not in normalized_candidate:
            # Normalization rebuilds the record from a fixed field list. The
            # rows have to survive it: they are the schedule, and the flat
            # steps cannot be turned back into them.
            normalized_candidate = dict(normalized_candidate)
            normalized_candidate["tick_rows"] = candidate["tick_rows"]
        if wrote_ticks:
            # The persisted form omits explicit blocked markers. Require that
            # exact canonical representation to pass the same validator too;
            # generation must never save a record postprocessing will reject.
            persisted_validation = dict(validator.validate(normalized_candidate))
            persisted_normalized = persisted_validation.pop(
                "normalized_candidate", normalized_candidate
            )
            validation = persisted_validation
            normalized_candidate = persisted_normalized
        return validation, normalized_candidate
    except TrajectoryValidationError as exc:
        if enforce_validation:
            raise
        # Preserve the invalid trace for inspection when validation is disabled.
        return (
            {
                "is_valid": False,
                "validation_disabled": True,
                "error_type": exc.error_type,
                "error_base_type": exc.error_base_type,
                "error": str(exc),
                "error_details": dict(exc.details) if exc.details else None,
                "step": exc.step if isinstance(exc.step, int) else None,
                "checks": [],
                "final_state": None,
                "signature": _candidate_signature(candidate),
            },
            candidate,
        )


def _declared_tool_arg_names(tool_spec: dict[str, Any]) -> set[str]:
    declared_arg_names = {
        arg_name
        for arg_name in tool_spec.get("tool_args", ())
        if isinstance(arg_name, str)
    }
    declared_arg_names.update(
        arg_name
        for arg_name in tool_spec.get("optional_tool_args", ())
        if isinstance(arg_name, str)
    )
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        declared_arg_names.update(
            arg_name for arg_name in arg_group if isinstance(arg_name, str)
        )
    return declared_arg_names


def _support_site_parent_by_id(initial_state: dict[str, Any]) -> dict[str, str]:
    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return {}

    parent_by_site: dict[str, str] = {}
    for fixture_id, fixture_state in fixtures_by_id.items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if isinstance(support_sites, dict):
            for support_site_id in support_sites:
                if isinstance(support_site_id, str):
                    parent_by_site[support_site_id] = fixture_id
        elif isinstance(support_sites, list):
            for support_site_id in support_sites:
                if isinstance(support_site_id, str):
                    parent_by_site[support_site_id] = fixture_id
    return parent_by_site


def _step_fixture_id(
    args: dict[str, Any],
    support_site_parent_by_id: dict[str, str],
) -> str | None:
    for fixture_arg_name in (
        "target_id",
        "fixture_id",
        "reference_fixture_id",
        "source_id",
        "support_id",
        "receptacle_id",
    ):
        fixture_value = args.get(fixture_arg_name)
        if not isinstance(fixture_value, str):
            continue
        return support_site_parent_by_id.get(fixture_value, fixture_value)
    return None


def _validate_candidate_references_without_sim(
    candidate: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> None:
    """Validate task-local symbolic part/control/site references without sim init.

    This is intentionally stricter than plain FSM replay for simulator-facing id
    fields. It blocks obvious hallucinated ids early in raw generation runs so
    the same protection applies both inside and outside the pipeline.
    """

    steps = candidate.get("steps")
    if not isinstance(steps, list):
        return

    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        fixtures_by_id = {}
    support_site_parent_map = _support_site_parent_by_id(initial_state)
    fixture_ids = {
        fixture_id for fixture_id in fixtures_by_id if isinstance(fixture_id, str)
    }

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        args = step.get("args")
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            continue
        tool_spec = allowed_tool_specs.get(tool_name)
        if not isinstance(tool_spec, dict):
            continue

        step_index = step.get("step")
        step_number = step_index if isinstance(step_index, int) else None
        declared_arg_names = _declared_tool_arg_names(tool_spec)
        if declared_arg_names:
            unexpected_arg_names = sorted(set(args) - declared_arg_names)
            if unexpected_arg_names:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} does not declare args {unexpected_arg_names}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "unexpected_args": unexpected_arg_names,
                        "declared_args": sorted(declared_arg_names),
                    },
                )

        fixture_id = _step_fixture_id(args, support_site_parent_map)
        fixture_state = (
            fixtures_by_id.get(fixture_id) if isinstance(fixture_id, str) else None
        )
        fixture_parts = (
            fixture_state.get("parts", {})
            if isinstance(fixture_state, dict)
            and isinstance(fixture_state.get("parts"), dict)
            else {}
        )
        fixture_controls = (
            fixture_state.get("controls", {})
            if isinstance(fixture_state, dict)
            and isinstance(fixture_state.get("controls"), dict)
            else {}
        )
        fixture_support_sites = (
            fixture_state.get("support_sites", {})
            if isinstance(fixture_state, dict)
            else {}
        )
        fixture_support_site_ids = (
            set(fixture_support_sites)
            if isinstance(fixture_support_sites, dict)
            else {
                site_id for site_id in fixture_support_sites if isinstance(site_id, str)
            }
            if isinstance(fixture_support_sites, list)
            else set()
        )

        part_id = args.get("part_id")
        allowed_part_ids = tool_spec.get("allowed_part_ids")
        if (
            isinstance(part_id, str)
            and isinstance(allowed_part_ids, list)
            and all(isinstance(part, str) for part in allowed_part_ids)
            and part_id not in allowed_part_ids
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {part_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "part_id": part_id,
                    "fixture_id": fixture_id,
                },
            )
        if isinstance(part_id, str) and fixture_parts and part_id not in fixture_parts:
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {part_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "part_id": part_id,
                    "fixture_id": fixture_id,
                },
            )

        control_id = args.get("control_id")
        allowed_control_ids = tool_spec.get("allowed_control_ids")
        if (
            isinstance(control_id, str)
            and isinstance(allowed_control_ids, list)
            and all(isinstance(control, str) for control in allowed_control_ids)
            and control_id not in allowed_control_ids
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {control_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "control_id": control_id,
                    "fixture_id": fixture_id,
                },
            )
        if (
            isinstance(control_id, str)
            and fixture_controls
            and control_id not in fixture_controls
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {control_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "control_id": control_id,
                    "fixture_id": fixture_id,
                },
            )

        for site_arg_name in ("source_site_id", "target_site_id"):
            site_id = args.get(site_arg_name)
            if not isinstance(site_id, str):
                continue
            if site_id in fixture_ids:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses fixture id {site_id!r} as {site_arg_name}; expected a support-site id.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "fixture_id": fixture_id,
                        site_arg_name: site_id,
                    },
                )
            allowed_site_ids = tool_spec.get(f"allowed_{site_arg_name[:-3]}_ids")
            if (
                isinstance(allowed_site_ids, list)
                and all(isinstance(site, str) for site in allowed_site_ids)
                and site_id not in allowed_site_ids
            ):
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses unknown {site_arg_name} {site_id!r}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        site_arg_name: site_id,
                        "allowed_site_ids": allowed_site_ids,
                    },
                )
            if fixture_support_site_ids and site_id not in fixture_support_site_ids:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses unknown {site_arg_name} {site_id!r} for fixture {fixture_id!r}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "fixture_id": fixture_id,
                        site_arg_name: site_id,
                    },
                )


def _validation_error_payload(
    exc: TrajectoryValidationError,
    *,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializes one validation exception into the shared validation shape."""

    payload = {
        "is_valid": False,
        "error_type": exc.error_type,
        "error_base_type": exc.error_base_type,
        "error": str(exc),
        "error_details": dict(exc.details) if exc.details else None,
        "step": exc.step if isinstance(exc.step, int) else None,
        "checks": [],
        "final_state": None,
    }
    if candidate is not None:
        payload["signature"] = _candidate_signature(candidate)
    return payload


def _build_multi_sample_insufficient_results_error(
    *,
    sampling_name: str,
    required_count: int,
    collected_count: int,
    invalid_validations: list[dict[str, Any]],
) -> TrajectoryValidationError:
    """Builds one descriptive insufficiency error for multi-sample retries."""

    duplicate_count = 0
    invalid_count = 0
    for validation in invalid_validations:
        if _validation_error_type(validation) == "DuplicateTrajectoryValidationError":
            duplicate_count += 1
            continue
        invalid_count += 1

    missing_count = max(required_count - collected_count, 0)
    details = {
        "required_valid_unique_trajectories": required_count,
        "collected_valid_unique_trajectories": collected_count,
        "missing_valid_unique_trajectories": missing_count,
        "duplicate_candidate_count": duplicate_count,
        "invalid_candidate_count": invalid_count,
    }
    run_label = (
        "Verbalized run" if sampling_name == "verbalized" else "Multi-sample run"
    )

    if duplicate_count and invalid_count:
        return InsufficientValidUniqueTrajectoriesMixedError(
            f"{run_label} did not produce enough valid unique trajectories because "
            "some candidates failed validation and others duplicated existing "
            "trajectories.",
            details=details,
        )
    if duplicate_count:
        return InsufficientValidUniqueTrajectoriesDuplicateError(
            f"{run_label} did not produce enough valid unique trajectories because "
            "some candidates duplicated existing trajectories.",
            details=details,
        )
    if invalid_count:
        return InsufficientValidUniqueTrajectoriesInvalidError(
            f"{run_label} did not produce enough valid unique trajectories because "
            "some candidates failed validation.",
            details=details,
        )
    return InsufficientValidUniqueTrajectoriesValidationError(
        f"{run_label} did not produce enough valid unique trajectories.",
        details=details,
    )


def _split_integer_total(total: int, parts: int) -> list[int]:
    """Splits one integer total across parts while preserving the sum."""

    base_value, remainder = divmod(total, parts)
    return [base_value + (1 if index < remainder else 0) for index in range(parts)]


def _split_float_total(total: float | None, parts: int) -> list[float | None]:
    """Splits one rounded float total across parts while preserving the sum."""

    if total is None:
        return [None] * parts
    if parts == 1:
        return [round_cost(total, decimal_places=COST_DECIMAL_PLACES)]

    split_values: list[float] = []
    remaining = total
    for index in range(parts):
        parts_left = parts - index
        if parts_left == 1:
            split_values.append(
                round_cost(remaining, decimal_places=COST_DECIMAL_PLACES) or 0.0
            )
            continue
        split_value = (
            round_cost(
                total / parts,
                decimal_places=COST_DECIMAL_PLACES,
            )
            or 0.0
        )
        split_values.append(split_value)
        remaining -= split_value
    return split_values


def _reasoning_token_count(generation_usage: dict[str, Any]) -> int:
    """Read reasoning tokens from usage payloads when splitting shared totals."""

    reasoning_tokens = generation_usage.get("reasoning_tokens")
    if not isinstance(reasoning_tokens, int):
        return 0
    return reasoning_tokens


def _cached_input_token_count(generation_usage: dict[str, Any]) -> int:
    """Read cached input tokens from usage payloads when splitting shared totals."""

    prompt_tokens = generation_usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int):
        prompt_tokens = 0
    cached_input_tokens = generation_usage.get("cached_input_tokens")
    if not isinstance(cached_input_tokens, int):
        return 0
    return min(cached_input_tokens, prompt_tokens)


def _split_generation_usage_across_candidates(
    generation_usage: dict[str, Any],
    *,
    candidate_count: int,
) -> list[dict[str, Any]]:
    """Splits one run's usage metadata across the saved candidates from that run."""

    if candidate_count == 1:
        return [dict(generation_usage)]

    prompt_splits = _split_integer_total(
        generation_usage["prompt_tokens"], candidate_count
    )
    cached_input_splits = _split_integer_total(
        _cached_input_token_count(generation_usage),
        candidate_count,
    )
    output_splits = _split_integer_total(
        generation_usage["output_tokens"], candidate_count
    )
    reasoning_splits = _split_integer_total(
        _reasoning_token_count(generation_usage),
        candidate_count,
    )
    cost_splits = _split_float_total(
        generation_usage.get("observed_cost_usd"),
        candidate_count,
    )

    split_usages: list[dict[str, Any]] = []
    for index in range(candidate_count):
        split_usage = dict(generation_usage)
        split_usage["prompt_tokens"] = prompt_splits[index]
        split_usage["cached_input_tokens"] = cached_input_splits[index]
        split_usage["output_tokens"] = output_splits[index]
        split_usage["reasoning_tokens"] = reasoning_splits[index]
        split_usage["total_tokens"] = (
            split_usage["prompt_tokens"]
            + split_usage["output_tokens"]
            + split_usage["reasoning_tokens"]
        )
        split_usage["observed_cost_usd"] = cost_splits[index]
        split_usages.append(split_usage)
    return split_usages


def _reserve_signature_batch(
    validations: list[dict[str, Any]],
    *,
    disable_validation: bool,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
) -> None:
    """Atomically reserves all candidate signatures produced by one successful run."""

    if disable_validation or seen_signatures is None or seen_signatures_lock is None:
        return

    signatures = [validation["signature"] for validation in validations]
    if len(signatures) != len(set(signatures)):
        raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")

    with seen_signatures_lock:
        if any(signature in seen_signatures for signature in signatures):
            raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")
        seen_signatures.update(signatures)


def _build_trajectory_records_from_sampled_candidates(
    *,
    run_index: int,
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
    task_instance: Any,
    sampled_candidates: list[SampledTrajectoryCandidate],
    prompt: str,
    raw_response: Any,
    usage: GenerationUsage | None,
    validator: TaskValidator,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
    attempt_number: int,
) -> list[dict[str, Any]]:
    """Builds saved trajectory records for all candidates emitted by one run."""

    validations: list[dict[str, Any]] = []
    normalized_candidates: list[dict[str, Any]] = []
    initial_state = (
        task_instance.initial_state
        if isinstance(getattr(task_instance, "initial_state", None), dict)
        else None
    )
    allowed_tool_specs = (
        task_instance.allowed_tool_specs
        if isinstance(getattr(task_instance, "allowed_tool_specs", None), dict)
        else getattr(validator, "allowed_tool_specs", None)
    )
    if not isinstance(allowed_tool_specs, dict):
        allowed_tool_specs = None
    for sampled_candidate in sampled_candidates:
        validation, normalized_candidate = _validate_candidate(
            sampled_candidate.candidate,
            validator,
            enforce_validation=not runtime_config.disable_validation,
            enable_static_referential_validation=(
                runtime_config.enable_static_referential_validation
            ),
            initial_state=initial_state,
            allowed_tool_specs=allowed_tool_specs,
        )
        validations.append(validation)
        normalized_candidates.append(normalized_candidate)

    _reserve_signature_batch(
        validations,
        disable_validation=runtime_config.disable_validation,
        seen_signatures=seen_signatures,
        seen_signatures_lock=seen_signatures_lock,
    )

    # Split one attempt-level usage record across the candidates that will be
    # persisted so downstream outputs stay trajectory-centric.
    shared_usage = _build_shared_generation_usage(
        runtime_config=runtime_config,
        sampled_candidates=sampled_candidates,
        prompt=prompt,
        raw_response=raw_response,
        usage=usage,
        attempt_number=attempt_number,
    )
    split_generation_usages = _split_generation_usage_across_candidates(
        shared_usage,
        candidate_count=len(sampled_candidates),
    )

    trajectory_records: list[dict[str, Any]] = []
    for candidate_index, (
        sampled_candidate,
        validation,
        normalized_candidate,
        generation_usage,
    ) in enumerate(
        zip(
            sampled_candidates,
            validations,
            normalized_candidates,
            split_generation_usages,
        )
    ):
        trajectory_id = format_trajectory_id(
            _global_trajectory_index(
                runtime_config,
                run_index=run_index,
                candidate_index=candidate_index,
            )
        )
        trajectory_record = task_definition.build_trajectory_record(
            candidate=normalized_candidate,
            validation=validation,
            trajectory_id=trajectory_id,
            generation_usage=generation_usage,
            task_instance=task_instance,
        )
        # `build_trajectory_record` copies a fixed field list, so the rows the
        # model actually wrote are dropped unless carried over here. They are
        # not recoverable from the flat steps -- which agent was idle on which
        # tick is exactly what flattening throws away -- and the image pass
        # needs them to keep the coordination schedule intact.
        tick_rows = normalized_candidate.get("tick_rows")
        if isinstance(tick_rows, list):
            trajectory_record["tick_rows"] = tick_rows
        trajectory_record.update(
            build_saved_trajectory_metadata(
                runtime_config=runtime_config,
                task_definition=task_definition,
            )
        )
        trajectory_record["coordinator_id"] = task_instance.coordinator_id
        if task_instance.physical_configuration is not None:
            from data_generation.task_level.scene_sampling import (
                physical_configuration_signature,
            )
            trajectory_record["physical_configuration"] = deepcopy(
                task_instance.physical_configuration
            )
            trajectory_record["physical_configuration_signature"] = (
                physical_configuration_signature(
                    task_instance.physical_configuration
                )
            )
        # Generation-only audit metadata. Preprocessing intentionally does not
        # place this latent diversity scaffold in the model context.
        if task_instance.work_partition is not None:
            trajectory_record["generation_work_partition"] = deepcopy(
                task_instance.work_partition
            )
        sampling_metadata = _sampling_metadata_for_candidate(
            runtime_config=runtime_config,
            sampled_candidate=sampled_candidate,
            candidate_index=candidate_index,
            run_index=run_index,
        )
        if sampling_metadata is not None:
            trajectory_record["sampling_metadata"] = sampling_metadata
        trajectory_record["prompt"] = prompt
        trajectory_record["raw_output"] = sampled_candidate.raw_output
        trajectory_records.append(trajectory_record)
    return trajectory_records


def _build_shared_generation_usage(
    *,
    runtime_config: RuntimeConfig,
    sampled_candidates: list[SampledTrajectoryCandidate],
    prompt: str,
    raw_response: Any,
    usage: GenerationUsage | None,
    attempt_number: int,
) -> dict[str, Any]:
    """Builds one attempt-level usage payload before splitting across candidates."""

    usage_candidate: dict[str, Any]
    if len(sampled_candidates) == 1:
        usage_candidate = sampled_candidates[0].candidate
    elif isinstance(raw_response, dict):
        usage_candidate = raw_response
    else:
        usage_candidate = {
            "responses": [
                sampled_candidate.raw_output for sampled_candidate in sampled_candidates
            ]
        }
    return build_generation_usage(
        model=runtime_config.model,
        prompt=prompt,
        candidate=usage_candidate,
        usage=usage,
        attempt_number=attempt_number,
        default_traffic_type=_default_traffic_type_for_runtime(runtime_config),
    )


def _tool_call_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    return len(steps)


def _maybe_reserve_signature(
    validation: dict[str, Any],
    *,
    disable_validation: bool,
    seen_signatures: set[str] | None,
    seen_signatures_lock: threading.Lock | None,
) -> None:
    if disable_validation or seen_signatures is None or seen_signatures_lock is None:
        return

    signature = validation["signature"]
    # Reserve signatures before persisting outputs so concurrent workers do not
    # save the same normalized trajectory twice.
    # Enforce uniqueness only for validated trajectories we intend to keep.
    with seen_signatures_lock:
        if signature in seen_signatures:
            raise DuplicateTrajectoryValidationError("Duplicate trajectory signature.")
        seen_signatures.add(signature)


def extract_json_candidate(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ResponseFormatValidationError(
            f"Unsupported model response type: {type(raw_response).__name__}"
        )

    # Some SDK/model paths wrap the JSON object in fences or surrounding text.
    stripped = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if not json_match:
            raise ResponseFormatValidationError(
                "Model response did not contain JSON."
            ) from exc
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as inner_exc:
            raise ResponseFormatValidationError(
                "Model response contained invalid JSON."
            ) from inner_exc
