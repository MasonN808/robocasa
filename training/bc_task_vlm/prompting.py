"""Prompt construction helpers for task-level VLM fine-tuning."""

from __future__ import annotations

from typing import Any, Iterable

from training.bc_task_vlm.schema_utils import compact_json_dumps

SYSTEM_PROMPT = (
    "You are a robot task planner. Predict exactly one next tool call for the "
    "current acting agent. Use the images only as scene context. Do not output "
    "image paths, do not output get_image, and do not describe the images."
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
        description = tool_spec.get("description", "").strip()
        line = f"- {tool_name}({tool_args}): {description}"
        allowed_values = _format_allowed_values(tool_spec)
        if allowed_values:
            line = f"{line} Constraints: {allowed_values}."
        lines.append(line)
    return "\n".join(lines)


def format_history_steps(history_steps: Iterable[dict[str, Any]]) -> str:
    """Renders prior symbolic action history compactly."""

    rendered_steps = []
    for step in history_steps:
        rendered_steps.append(
            f"- step={step['step']} agent={step['agent']} tool={step['tool']} "
            f"args={compact_json_dumps(step['args'])}"
        )
    if not rendered_steps:
        return "- none"
    return "\n".join(rendered_steps)


def build_user_prompt(
    *,
    composite_task: str,
    task_instruction: str,
    agent_id: str,
    next_step_index: int,
    observation_views: list[str],
    history_steps: list[dict[str, Any]],
    allowed_tool_specs: dict[str, dict[str, Any]],
    sft_format: str = "tool_call",
) -> str:
    """Builds the text block that accompanies the current image observation."""

    observations_text = ", ".join(observation_views) if observation_views else "unknown"
    if sft_format == "plain":
        output_rules = (
            "Rules:\n"
            "- Predict exactly one next action.\n"
            "- Use symbolic IDs only, never concrete simulator IDs.\n"
            "- The acting agent is fixed by the prompt; do not choose actions for the other agent.\n"
            "- The tool must be one of the allowed tools listed above.\n"
            "- Supply exactly the arguments required by the selected tool.\n"
            "- Prefer the most immediate executable next action.\n\n"
            "Output format:\n"
            '- Return exactly one compact JSON object with keys "tool" and "args".\n'
            "- Do not emit markdown or narrative outside the JSON object."
        )
    elif sft_format == "tool_call":
        output_rules = (
            "The tool schemas are provided separately as function definitions.\n\n"
            "Rules:\n"
            "- Predict exactly one next tool call.\n"
            "- Use symbolic IDs only, never concrete simulator IDs.\n"
            "- The acting agent is fixed by the prompt; do not choose actions for the other agent.\n"
            "- The tool must be one of the allowed tools listed above.\n"
            "- Supply exactly the arguments required by the selected tool.\n"
            "- Prefer the most immediate executable next action.\n"
            "- Do not emit markdown or narrative after the tool call."
        )
    else:
        raise ValueError(f"Unsupported SFT format: {sft_format!r}")

    return (
        f"Task family: {composite_task}\n"
        f"Task instruction: {task_instruction}\n"
        f"Current acting agent: {agent_id}\n"
        f"Next global step index: {next_step_index}\n"
        f"Observation views attached in order: {observations_text}\n\n"
        "Previous executed symbolic action history:\n"
        f"{format_history_steps(history_steps)}\n\n"
        "Available tools for this task:\n"
        f"{format_allowed_tool_block(allowed_tool_specs)}\n\n"
        f"{output_rules}"
    )


def build_system_message() -> dict[str, Any]:
    """Builds the fixed system instruction for one chat conversation."""

    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
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


def build_assistant_text_message(*, target_text: str) -> dict[str, Any]:
    """Builds one assistant text message for plain SFT supervision."""

    return {
        "role": "assistant",
        "content": target_text,
    }


def build_messages(
    *,
    user_prompt: str,
    num_images: int,
    target_tool_call: dict[str, Any] | None = None,
    target_text: str | None = None,
) -> list[dict[str, Any]]:
    """Builds one chat conversation for training or generation."""

    if target_tool_call is not None and target_text is not None:
        raise ValueError("Provide either target_tool_call or target_text, not both.")

    messages: list[dict[str, Any]] = [
        build_system_message(),
        build_user_message(user_prompt=user_prompt, num_images=num_images),
    ]
    if target_text is not None:
        messages.append(build_assistant_text_message(target_text=target_text))
    elif target_tool_call is not None:
        from training.bc_task_vlm.tool_calling import build_assistant_tool_call_message

        messages.append(
            build_assistant_tool_call_message(
                tool_name=target_tool_call["name"],
                arguments=target_tool_call["arguments"],
            )
        )
    return messages
