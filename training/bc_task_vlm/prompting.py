"""Prompt construction helpers for task-level VLM fine-tuning."""

from __future__ import annotations

from typing import Any, Iterable

from training.bc_task_vlm.schema_utils import compact_json_dumps

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
    if sft_format == "plain":
        output_rules = (
            "Rules:\n"
            "- Predict exactly one next action.\n"
            "- Use symbolic IDs only, never concrete simulator IDs.\n"
            f"{agent_rule}"
            "- The tool must be one of the allowed tools listed above.\n"
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
            f"{agent_rule}"
            "- The tool must be one of the allowed tools listed above.\n"
            "- Supply exactly the arguments required by the selected tool.\n"
            "- Prefer the most immediate executable next action.\n"
            "- Do not emit markdown or narrative after the tool call."
        )
    else:
        raise ValueError(f"Unsupported SFT format: {sft_format!r}")

    acting_agent_line = "" if predict_agent else f"Current acting agent: {agent_id}\n"
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
    return (
        f"Task family: {composite_task}\n"
        f"Task instruction: {task_instruction}\n"
        f"{acting_agent_line}"
        f"{observation_owner_line}"
        f"{step_index_line}"
        f"Observation views attached in order: {observations_text}\n\n"
        "Previous executed symbolic action history:\n"
        f"{format_history_steps(history_steps)}\n\n"
        "Available tools for this task:\n"
        f"{format_allowed_tool_block(allowed_tool_specs)}\n\n"
        f"{output_rules}"
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
