"""Prompt construction helpers for task-level VLM fine-tuning."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable

from data_generation.task_level.tasks.shared.prompting import format_concurrency_facts
from training.bc_task_vlm.communication_profiles import get_communication_profile
from training.bc_task_vlm.schema_utils import compact_json_dumps


DEFAULT_PARTIAL_STEP_INDEX_MODE = "none"
PROMPT_CONTRACT_VERSION = (
    "state_grounded_global_tools_no_index_v9_unambiguous_garnish_cake_goal"
)

PHYSICAL_HANDOFF_RULES = (
    "Physical workspace handoffs:\n"
    "- give_space(X) frees X only after that tool call finishes. The partner "
    "must not enter or use X in the same concurrent tick.\n"
    "- An unblocked partner may enter or use X on the following tick without "
    "calling wait_for_signal and without receiving a release.\n"
    "- Use wait_for_signal only when the agent must actually block before X "
    "becomes free. Once waiting, it remains blocked throughout a later matching "
    "release tick and resumes only on the following tick. Never wait and receive "
    "the matching release in the same tick.\n"
)

WORKSPACE_RULES = (
    "Workspace relationships:\n"
    "- A cabinet and its parent counter are one shared workspace. Navigate to "
    "the parent counter for cabinet work. Cabinet tools may be used from that "
    "counter, and different agents may use different objects there without a "
    "handover. Do not open or close the cabinet during another agent's access "
    "to its contents.\n"
    "- If an agent is at an exclusive child appliance on a parent counter, it "
    "may use that counter and objects on it without navigating again. This does "
    "not move the agent: its location remains the exclusive child appliance.\n"
)

PHYSICAL_GIVE_SPACE_RULE = (
    "Physical workspace handoffs:\n"
    "- give_space(X) frees X only after that tool call finishes. The other "
    "agent may enter or use X starting on the following tick, not during the "
    "same tick.\n"
)

MINIMAL_COMMUNICATION_RULES = (
    "Communication guidance:\n"
    "- Communicate with the other agent to complete the task together.\n"
    "- When you need to wait for X, first communicate a request naming the "
    "exact symbolic ID X. On your next call, use wait_for_signal with about=X. "
    "The wait call is private. A later message with releases=X wakes the "
    "waiting agent.\n"
)

ACTIVE_OBSERVATION_RULES = (
    "Camera observations:\n"
    "- get_image.views must be a non-empty list containing only these exact "
    "names: top_view, room_view, map, wrist, agentview_center, agentview_left, "
    "agentview_right.\n"
    "- Copy these names exactly. Do not invent view names from a direction, "
    "task, scene, object, or fixture. For example, use "
    '`get_image(views=["agentview_center", "wrist"])`. Names such as front, '
    "back, overhead, left, right, counter, and fridge_interior are invalid.\n"
)

SYSTEM_PROMPT = (
    "You are a robot task planner. Predict exactly one next tool call for the "
    "current acting agent. Use the images only as scene context. Do not output "
    "image paths, do not output get_image, and do not describe the images."
)

# Agent-prediction (v2) variant: the model chooses which agent acts next
# instead of being told, and may declare the task finished.
SYSTEM_PROMPT_PREDICT_AGENT = (
    "You are a robot task planner. Decide which agent should act next and "
    'predict exactly one next tool call for it, passing the agent as the "agent" '
    "argument. When the task goal is already satisfied, call task_complete. Use "
    "the images only as scene context. Do not output image paths, do not output "
    "get_image, and do not describe the images."
)

# Partial-observability (v3) variant: active observation with a FIXED caller.
# The agent is told who it is, so it must not choose an acting agent; it only
# decides whether to look first and what tool to call.
SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT = (
    "You are a robot task planner. Predict exactly one next tool call for the "
    "current acting agent. Request the camera views you need before an action "
    "by calling get_image. Use images only as scene context. Do not output "
    "image paths or describe the images."
)

SYSTEM_PROMPT_ACTIVE_OBSERVATION = (
    "You are a robot task planner. Decide which agent should act next and "
    'predict exactly one next tool call for it, passing the agent as the "agent" '
    "argument. Request the camera views needed for the next action by calling "
    "get_image. When the task goal is already satisfied, call task_complete. "
    "Use images only as scene context. Do not output image paths or describe "
    "the images."
)


def _format_allowed_values(tool_spec: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in sorted(tool_spec):
        if not key.startswith("allowed_"):
            continue
        value = tool_spec[key]
        if not isinstance(value, list) or not value:
            continue
        allowed_values = ", ".join(str(item) for item in value)
        chunks.append(f"{key}=[{allowed_values}]")
    return "; ".join(chunks)


def format_allowed_tool_block(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> str:
    """Renders task-local tool constraints for the prompt."""

    lines: list[str] = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        tool_args = ", ".join(tool_spec.get("tool_args", ())) or "no args"
        # The place_* family declares one required arg plus a tool_arg_any_of
        # group. The runtime validator enforces that group, but this block used
        # to drop it, so the prompt advertised place_on_object(object_id) while
        # the runtime rejected any call lacking support_object_id. Models with
        # demonstrations to imitate never noticed (expert targets carry the
        # second argument); untrained models obeyed the signature and were
        # rejected on ~4% of steps.
        for arg_group in tool_spec.get("tool_arg_any_of", ()) or ():
            group = [a for a in arg_group if a]
            if group:
                tool_args = f"{tool_args}, one of ({' | '.join(group)})"
        description = tool_spec.get("description", "").strip()
        line = f"- {tool_name}({tool_args}): {description}"
        allowed_values = _format_allowed_values(tool_spec)
        if allowed_values:
            line = f"{line} Constraints: {allowed_values}."
        lines.append(line)
    return "\n".join(lines)


def format_history_steps(history_steps: Iterable[dict[str, Any]]) -> str:
    """Renders prior symbolic action history compactly.

    A step may carry an "error" key (live-sim rejected attempts); it renders as
    a trailing `FAILED: <reason>` so the model can see why nothing changed.
    Training data never sets it, so v1 prompts are unaffected.
    """

    rendered_steps = []
    for step in history_steps:
        # step=None omits the index entirely (partial-observability
        # step_index_mode="none"): a joint demonstration index leaks how much
        # hidden activity the other agent performed between this agent's turns.
        index_text = "" if step.get("step") is None else f"step={step['step']} "
        line = (
            f"- {index_text}agent={step['agent']} tool={step['tool']} "
            f"args={compact_json_dumps(step['args'])}"
        )
        error_text = step.get("error")
        if error_text:
            line = f"{line} FAILED: {error_text}"
        rendered_steps.append(line)
    if not rendered_steps:
        return "- none"
    return "\n".join(rendered_steps)


def normalize_partial_history_steps(
    history_steps: Iterable[dict[str, Any]],
    *,
    index_mode: str,
) -> list[dict[str, Any]]:
    """Return private history with the requested prompt-visible numbering.

    Both offline SFT construction and closed-loop evaluation must call this
    helper.  Applying the mode only to the ``Next ... index`` line is not
    sufficient: it leaks global concurrency through historical indices and
    produces a context the model never saw during training.
    """

    if index_mode not in {"global", "local", "none"}:
        raise ValueError(
            "partial history index_mode must be one of: global, local, none; "
            f"got {index_mode!r}"
        )
    normalized = [deepcopy(step) for step in history_steps]
    if index_mode == "global":
        return normalized
    for local_index, step in enumerate(normalized):
        step["step"] = local_index if index_mode == "local" else None
    return normalized


def build_user_prompt(
    *,
    composite_task: str,
    task_instruction: str,
    agent_id: str,
    next_step_index: int | None,
    step_index_label: str = "Next global step index",
    observation_views: list[str],
    history_steps: list[dict[str, Any]],
    allowed_tool_specs: dict[str, dict[str, Any]],
    sft_format: str = "tool_call",
    predict_agent: bool = False,
    observation_owner: str | None = None,
    include_observation_owner: bool = False,
    coordinator_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
    communication_mode: str = "full",
) -> str:
    """Builds the text block that accompanies the current image observation.

    With predict_agent (v2), the acting agent is not given: the model chooses
    it, emits it as the "agent" argument, and may call task_complete when the
    goal is already satisfied. Default keeps the v1 prompt byte-identical.
    """

    # "none" (not "unknown") when no images are attached: under partial-v3
    # consume-once semantics that is the normal state after an observation has
    # been spent, and "unknown" would imply views exist but are unidentified.
    observations_text = ", ".join(observation_views) if observation_views else "none"
    if predict_agent:
        agent_rule = (
            '- First decide which agent acts next; pass it as the "agent" argument.\n'
            "- If the task goal is already fully satisfied, call task_complete.\n"
        )
    else:
        agent_rule = (
            "- The acting agent is fixed by the prompt; do not choose actions for the other agent.\n"
        )
    communication_profile = get_communication_profile(communication_mode)
    initial_communication_rule = (
        "- Before ANY non-communicate action, BOTH agents must each have sent "
        "at least one communicate message; task actions are rejected until then.\n"
        if communication_profile.require_initial_communication
        else ""
    )
    if sft_format == "plain":
        output_rules = (
            "Rules:\n"
            "- Predict exactly one next action.\n"
            "- Use symbolic IDs only, never concrete simulator IDs.\n"
            "- Symbolic IDs are dictionary keys in the initial state; copy the "
            "relevant object, fixture, part, or control key verbatim. Use a site "
            "ID only when the state explicitly provides a named site.\n"
            f"{agent_rule}"
            f"{initial_communication_rule}"
            "- The tool must be one of the available tool definitions.\n"
            "- Supply exactly the arguments required by the selected tool.\n"
            "- Prefer the most immediate executable next action.\n\n"
            "Output format:\n"
            '- Return exactly one compact JSON object with keys "tool" and "args".\n'
            + (
                '- Include the acting agent as the "agent" entry inside "args".\n'
                if predict_agent
                else ""
            )
            + "- Do not emit markdown or narrative outside the JSON object."
        )
    elif sft_format == "tool_call":
        output_rules = (
            "The tool schemas are provided separately as function definitions.\n\n"
            "Rules:\n"
            "- Predict exactly one next tool call.\n"
            "- Use symbolic IDs only, never concrete simulator IDs.\n"
            "- Symbolic IDs are dictionary keys in the initial state; copy the "
            "relevant object, fixture, part, or control key verbatim. Use a site "
            "ID only when the state explicitly provides a named site.\n"
            f"{agent_rule}"
            f"{initial_communication_rule}"
            "- The tool must be one of the provided function schemas.\n"
            "- Supply exactly the arguments required by the selected tool.\n"
            "- Prefer the most immediate executable next action.\n"
            "- Do not emit markdown or narrative after the tool call."
        )
    else:
        raise ValueError(f"Unsupported SFT format: {sft_format!r}")

    acting_agent_line = "" if predict_agent else f"Current acting agent: {agent_id}\n"
    participation_line = {
        "full": "",
        "minimal": (
            "You are one of two agents working together to complete the task.\n"
        ),
        "unguided": (
            "You are one of two agents jointly completing the task. "
            "Communication with the other agent is possible.\n"
        ),
        "none": (
            "You are one of two agents jointly completing the task. The other "
            "agent acts independently, but you cannot exchange messages. Choose "
            "actions that contribute to the shared task goal.\n"
        ),
    }[communication_mode]
    coordinator_line = (
        f"Coordinator for this episode: {coordinator_id}\n"
        if coordinator_id and communication_profile.require_opening_protocol
        else ""
    )
    coordinator_rules = (
        "\nCoordinator handshake:\n"
        "- Opening coordination uses two communication ticks; optional ticks "
        "containing only get_image do not count.\n"
        "- Communication tick 1: the coordinator sends one concrete division "
        "using exact symbolic IDs and coordination_phase=propose; the partner "
        "sends coordination_phase=await_plan.\n"
        "- Communication tick 2: the coordinator sends "
        "coordination_phase=await_confirmation; the partner sends "
        "coordination_phase=confirm.\n"
        "- Physical work may begin only on the following tick.\n"
        "- Never claim that the global task is complete; the FSM alone ends the "
        "episode. If your own portion finishes first, send exactly one "
        "coordination_phase=portion_complete message saying only that your part "
        "is done and naming one exact release keyword X, for example `My portion "
        "is done. Release \"X\" if you need me again`. On your very next call, "
        "call wait_for_signal with about=X. The private wait call is not visible "
        "to the partner, so the message must name the same X and say to release "
        "it. Stay blocked until the "
        "partner sends a later matching release.\n"
        if coordinator_id and communication_profile.require_opening_protocol
        else ""
    )
    communication_rules = {
        "full": PHYSICAL_HANDOFF_RULES,
        "minimal": MINIMAL_COMMUNICATION_RULES + PHYSICAL_GIVE_SPACE_RULE,
        "unguided": PHYSICAL_GIVE_SPACE_RULE,
        "none": PHYSICAL_GIVE_SPACE_RULE,
    }[communication_mode]
    # next_step_index=None omits the line (step_index_mode="none");
    # step_index_label lets partial-observability prompts say "local turn
    # index" instead of the leaky joint "global step index".
    if next_step_index is None:
        step_index_line = ""
    else:
        step_index_line = f"{step_index_label}: {next_step_index}\n"
    observation_owner_line = ""
    if include_observation_owner:
        observation_owner_line = (
            f"Active observation owner: {observation_owner or 'none'}\n"
        )
    tool_block = (
        "Available tools:\n"
        f"{format_allowed_tool_block(allowed_tool_specs)}\n\n"
        if sft_format == "plain"
        else ""
    )
    initial_state_block = (
        "Symbolic initial state:\n"
        f"{json.dumps(initial_state, indent=2, sort_keys=True)}\n\n"
        f"{format_concurrency_facts(initial_state)}\n\n"
        if initial_state is not None
        else ""
    )
    active_observation_rules = (
        ACTIVE_OBSERVATION_RULES if "get_image" in allowed_tool_specs else ""
    )
    return (
        f"Task family: {composite_task}\n"
        f"Task instruction: {task_instruction}\n"
        f"{acting_agent_line}"
        f"{participation_line}"
        f"{coordinator_line}"
        f"{observation_owner_line}"
        f"{step_index_line}"
        f"Observation views attached in order: {observations_text}\n\n"
        f"{initial_state_block}"
        "Previous executed symbolic action history:\n"
        f"{format_history_steps(history_steps)}\n\n"
        f"{tool_block}"
        f"{coordinator_rules}"
        f"{active_observation_rules}"
        f"\n{communication_rules}"
        f"{WORKSPACE_RULES}"
        f"{output_rules}"
    )


def build_partial_user_prompt(
    *,
    composite_task: str,
    task_instruction: str,
    agent_id: str,
    history_steps: list[dict[str, Any]],
    observation_views: list[str],
    allowed_tool_specs: dict[str, dict[str, Any]],
    partial_step_index_mode: str = DEFAULT_PARTIAL_STEP_INDEX_MODE,
    global_step_index: int | None = None,
    coordinator_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
    communication_mode: str = "full",
) -> str:
    """Build the canonical private-history prompt used by SFT and live eval.

    Keeping index normalization here prevents the two callers from silently
    drifting.  Production no-index runs use ``partial_step_index_mode=none``.
    """

    prompt_history = normalize_partial_history_steps(
        history_steps,
        index_mode=partial_step_index_mode,
    )
    if partial_step_index_mode == "local":
        next_step_index: int | None = len(prompt_history)
        step_index_label = "Next local agent turn index"
    elif partial_step_index_mode == "none":
        next_step_index = None
        step_index_label = "Next global step index"
    else:
        next_step_index = global_step_index
        step_index_label = "Next global step index"
    return build_user_prompt(
        composite_task=composite_task,
        task_instruction=task_instruction,
        agent_id=agent_id,
        next_step_index=next_step_index,
        step_index_label=step_index_label,
        observation_views=observation_views,
        history_steps=prompt_history,
        allowed_tool_specs=allowed_tool_specs,
        sft_format="tool_call",
        predict_agent=False,
        coordinator_id=coordinator_id,
        initial_state=initial_state,
        communication_mode=communication_mode,
    )


def build_system_message(
    *, predict_agent: bool = False, train_get_image: bool = False
) -> dict[str, Any]:
    """Builds the fixed system instruction for one chat conversation."""

    if train_get_image:
        system_text = (
            SYSTEM_PROMPT_ACTIVE_OBSERVATION
            if predict_agent
            else SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT
        )
    else:
        system_text = SYSTEM_PROMPT_PREDICT_AGENT if predict_agent else SYSTEM_PROMPT
    return {
        "role": "system",
        "content": [{"type": "text", "text": system_text}],
    }


def build_few_shot_block(example_trajectory: dict[str, Any]) -> str:
    """Renders one spec example trajectory as an in-context demonstration.

    Eval-time ablation only: the SFT training prompt never includes this.
    """

    steps = [
        {
            "step": step.get("step"),
            "agent": step.get("agent"),
            "tool": step.get("tool"),
            "args": step.get("args", {}),
        }
        for step in example_trajectory.get("steps", [])
    ]
    return (
        "Example of one complete valid trajectory for this task family "
        "(for format and ordering reference only):\n"
        f"{format_history_steps(steps)}"
    )


def append_few_shot_block(
    messages: list[dict[str, Any]],
    few_shot_block: str,
) -> list[dict[str, Any]]:
    """Returns a copy of chat messages with the example appended to the last user text."""

    updated_messages: list[dict[str, Any]] = []
    last_user_index = max(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    )
    for index, message in enumerate(messages):
        if index != last_user_index:
            updated_messages.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            updated_messages.append(
                {**message, "content": f"{content}\n\n{few_shot_block}"}
            )
            continue
        updated_content = [dict(item) for item in content]
        text_indices = [
            item_index
            for item_index, item in enumerate(updated_content)
            if item.get("type") == "text"
        ]
        target_index = text_indices[-1]
        updated_content[target_index]["text"] = (
            f"{updated_content[target_index]['text']}\n\n{few_shot_block}"
        )
        updated_messages.append({**message, "content": updated_content})
    return updated_messages


def build_user_message(*, user_prompt: str, num_images: int) -> dict[str, Any]:
    """Builds one multimodal user turn with placeholder image slots."""

    user_content = [{"type": "image"} for _ in range(num_images)]
    user_content.append({"type": "text", "text": user_prompt})
    return {
        "role": "user",
        "content": user_content,
    }


def build_assistant_text_message(
    *, target_text: str, reasoning_text: str | None = None
) -> dict[str, Any]:
    """Builds one assistant text message for plain SFT supervision."""

    content = f"<think>{reasoning_text}</think>{target_text}" if reasoning_text else target_text
    return {
        "role": "assistant",
        "content": content,
    }


def build_messages(
    *,
    user_prompt: str,
    num_images: int,
    target_tool_call: dict[str, Any] | None = None,
    target_text: str | None = None,
    predict_agent: bool = False,
    train_get_image: bool = False,
    reasoning_text: str | None = None,
) -> list[dict[str, Any]]:
    """Builds one chat conversation for training or generation.

    With reasoning_text (v1.5 probe), the assistant turn is supervised with a
    `<think>{reasoning_text}</think>` prefix before the target. Orthogonal to
    predict_agent/train_get_image; composes with either.
    """

    if target_tool_call is not None and target_text is not None:
        raise ValueError("Provide either target_tool_call or target_text, not both.")

    messages: list[dict[str, Any]] = [
        build_system_message(
            predict_agent=predict_agent, train_get_image=train_get_image
        ),
        build_user_message(user_prompt=user_prompt, num_images=num_images),
    ]
    if target_text is not None:
        messages.append(
            build_assistant_text_message(
                target_text=target_text, reasoning_text=reasoning_text
            )
        )
    elif target_tool_call is not None:
        from training.bc_task_vlm.tool_calling import build_assistant_tool_call_message

        messages.append(
            build_assistant_tool_call_message(
                tool_name=target_tool_call["name"],
                arguments=target_tool_call["arguments"],
                reasoning_text=reasoning_text,
            )
        )
    return messages
