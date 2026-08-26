"""Export exact initial-turn SFT and live-eval prompt examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_generation.task_level.generation.raw.config import RuntimeConfig
from data_generation.task_level.generation.raw.runtime_support import (
    format_trajectory_variation_key,
)
from data_generation.task_level.sampling.structured_random import (
    StructuredRandomSamplingStrategy,
)
from data_generation.task_level.tasks.specs.runtime import SPEC_TASK_REGISTRY
from training.bc_task_vlm.prompting import (
    SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT,
    build_user_prompt,
)
from data_generation.task_level.subatomic_tool_specs import build_model_tool_specs
from training.bc_task_vlm.task_registry import get_task_metadata
from training.bc_task_vlm.tool_calling import build_tool_schemas


def _block(label: str, value: str) -> str:
    return f"### {label}\n\n```text\n{value}\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument(
        "--tasks", default="arrange_bread_bowl,prepare_coffee"
    )
    args = parser.parse_args()

    sections = [
        "# Exact demonstration-generation, SFT, and live-evaluation prompts\n",
        "These examples are rendered through the current production prompt builders "
        "and real rendered trajectory carriers. Tool schemas are supplied separately "
        "to the model and are therefore printed alongside the user prompts.\n",
    ]
    task_sections: list[tuple[str, str, int, int]] = []
    generation_config = RuntimeConfig(
        composite_task="PrepareCoffee",
        num_runs=1,
        model="gemini-3-flash-preview",
        sdk="google-genai",
        project=None,
        location="global",
        sampling="structured_random",
        temperature=0.6,
        max_workers=8,
        max_retries=10,
        random_start_location=True,
        random_access_state=True,
        tick_format=True,
        prompt_style="simplified_v3",
        retry_feedback_style="targeted",
        partition_policy="none",
    )
    generation_definition = SPEC_TASK_REGISTRY["PrepareCoffee"]
    generation_instance = generation_definition.build_task_instance(
        0, generation_config
    )
    generation_strategy = StructuredRandomSamplingStrategy()
    generation_prompt = generation_strategy.build_prompt(
        task_definition=generation_definition,
        runtime_config=generation_config,
        task_instance=generation_instance,
        variation_key=format_trajectory_variation_key(0, 0),
    )
    generation_schema = generation_strategy.response_schema(
        task_definition=generation_definition,
        runtime_config=generation_config,
    )
    generation_section = (
        "## Demonstration-generation request\n\n"
        "This is the exact current production-style first-attempt prompt for "
        "`PrepareCoffee`: structured-random sampling, temperature 0.6, "
        "random starts and access state, tick format, `simplified_v3`, and no "
        "forced partition. Gemini also receives the response schema printed "
        "after the text prompt.\n\n"
        + _block("Generator text prompt", generation_prompt)
        + "### Generator response schema\n\n```json\n"
        + json.dumps(generation_schema, indent=2, sort_keys=True)
        + "\n```\n"
    )
    sections.append(generation_section)
    qwen_tool_wrapper = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{one JSON function schema per available tool}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
    interface_section = (
        "## The Qwen chat template adds the missing output-format instructions\n\n"
        "The earlier view separated the user message and function schemas, but "
        "the model sees both after Qwen's chat template inserts this wrapper "
        "into the system turn. Therefore the tool-call format is present in "
        "the actual tokenized request even though it is absent from the user "
        "message.\n\n"
        + _block("Model-visible tool wrapper", qwen_tool_wrapper)
    )
    sections.append(interface_section)
    for task_name in (x.strip() for x in args.tasks.split(",") if x.strip()):
        metadata = get_task_metadata(task_name)
        trajectory_dir = sorted((args.dataset_root / task_name).glob("traj_*"))[0]
        trajectory = json.loads(
            (trajectory_dir / "original_trajectory.json").read_text()
        )
        render_metadata = json.loads((trajectory_dir / "metadata.json").read_text())
        tool_specs = build_model_tool_specs(include_get_image=True)
        tool_schemas = build_tool_schemas(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=tool_specs,
            include_agent_param=False,
        )
        coordinator = trajectory.get("coordinator_id") or "agent_0"
        acting_agent = coordinator

        def prompt(task_instruction: str) -> str:
            return build_user_prompt(
                composite_task=metadata.composite_task,
                task_instruction=task_instruction,
                agent_id=acting_agent,
                next_step_index=None,
                observation_views=[],
                history_steps=[],
                allowed_tool_specs=tool_specs,
                sft_format="tool_call",
                predict_agent=False,
                coordinator_id=coordinator,
                initial_state=trajectory.get("initial_state"),
            )

        sft_prompt = prompt(metadata.task_goal)
        eval_prompt = prompt(metadata.task_goal)
        if sft_prompt != eval_prompt:
            raise AssertionError(
                f"Initial-turn SFT/live prompt mismatch for {metadata.composite_task}"
            )
        task_body = [
            f"## {metadata.composite_task}\n",
            f"Carrier: `{task_name}/{trajectory_dir.name}`\n",
            _block("System message (both paths)", SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT),
            _block("SFT user message", sft_prompt),
            _block("Live-evaluation user message", eval_prompt),
            "### Function/tool schemas (both paths)\n\n```json\n"
            + json.dumps(tool_schemas, indent=2, sort_keys=True)
            + "\n```\n"
        ,
            "### Saved initial state (included identically in both model prompts)\n\n```json\n"
            + json.dumps(trajectory.get("initial_state"), indent=2, sort_keys=True)
            + "\n```\n"
        ]
        task_text = "\n".join(task_body)
        sections.append(task_text)
        task_sections.append((task_name, task_text, len(sft_prompt), len(eval_prompt)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(args.output)
    if args.artifact_output:
        sources = [
            {
                "id": "prompt-code",
                "label": "Demonstration-generation, SFT, and live-evaluation prompt builders",
                "path": "{data_generation/task_level/{sampling,tasks},training/bc_task_vlm}/{prompting.py,dataset.py,live_sim_eval.py,tool_calling.py}",
                "query": {
                    "language": "python",
                    "description": "Render initial-turn prompts through the current shared helper using each path's actual task-instruction source.",
                    "tables_used": ["prompting.py", "dataset.py", "live_sim_eval.py", "tool_calling.py"],
                    "filters": ["partial history", "no step index", "tool-call format", "initial turn"],
                },
            },
            {
                "id": "rendered-carriers",
                "label": "Rendered ArrangeBreadBowl and PrepareCoffee carriers",
                "path": "tick53x150_state_grounded_cascade_v1_rendered/{arrange_bread_bowl,prepare_coffee}/traj_*/{original_trajectory.json,metadata.json}",
                "query": {
                    "language": "python",
                    "description": "Select the first sorted real carrier for each task and compare metadata.task with original_trajectory.task.",
                    "tables_used": ["original_trajectory.json", "metadata.json"],
                },
            },
            {
                "id": "prompt-lengths",
                "label": "Exact rendered prompt character counts",
                "path": "training/bc_task_vlm/reports/current_prompt_contract_examples/artifact.json",
                "query": {
                    "language": "sql",
                    "engine": "DuckDB",
                    "description": "Materialize the exact Python-rendered character counts used by the comparison chart.",
                    "tables_used": ["rendered_prompt_lengths"],
                    "sql": " UNION ALL ".join(
                        f"SELECT '{task_name}' AS task, '{path}' AS path, {chars} AS characters"
                        for task_name, _task_text, sft_chars, eval_chars in task_sections
                        for path, chars in (("SFT", sft_chars), ("Live evaluation", eval_chars))
                    ),
                },
            },
        ]
        blocks = [
            {"id": "title", "type": "markdown", "body": "# Exact current demonstration, SFT, and evaluation prompts"},
            {
                "id": "summary",
                "type": "markdown",
                "sourceId": "prompt-code",
                "body": "## Technical summary\n\nThe current demonstration generator receives a generation-only protocol, task-specific facts, sampled coordinator, symbolic initial state, tool contracts, and a structured response schema. SFT and live evaluation use the same system message, the same natural-language task instruction, the same symbolic initial state, and the same global tool interface. The script asserts that their initial-turn user messages are byte-identical before producing this artifact.",
            },
            {
                "id": "generation-prompt",
                "type": "markdown",
                "sourceId": "prompt-code",
                "body": generation_section,
            },
            {
                "id": "tool-wrapper",
                "type": "markdown",
                "sourceId": "prompt-code",
                "body": interface_section,
            },
        ]
        for task_name, task_text, _sft_chars, _eval_chars in task_sections:
            blocks.append(
                {
                    "id": f"prompt-{task_name}",
                    "type": "markdown",
                    "sourceId": "rendered-carriers",
                    "body": task_text,
                }
            )
        blocks.insert(
            2,
            {
                "id": "length-heading",
                "type": "markdown",
                "sourceId": "prompt-code",
                "body": "## SFT and evaluation use one prompt contract\n\nFor each carrier below, SFT and live evaluation are rendered with the same task instruction, coordinator, initial state, history, observation-view list, global tool contracts, and no step index. Equal character counts are a compact parity check; the builder additionally compares the complete strings and stops if any byte differs.",
            },
        )
        blocks.insert(3, {"id": "length-chart", "type": "chart", "chartId": "prompt-length"})
        blocks.extend(
            [
                {
                    "id": "interpretation",
                    "type": "markdown",
                    "sourceId": "prompt-code",
                    "body": "## What each model request discloses\n\nSFT and evaluation disclose the model's agent identity, sampled coordinator, task goal, symbolic initial state, global tool descriptions, current history, and attached observation-view order. The initial-state block includes agent locations, held objects, object locations, modeled fixture parts and controls, and modeled machine state. It does not replace visual observation with full simulator geometry or hidden native simulator state.",
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": "## Recommended validation\n\nKeep the existing prompt-contract gate in every preprocessing and evaluation launch. At matched decision points, compare the fully rendered message sequence after history and images are inserted—not only the initial user message shown here. Any future change to task wording, initial-state fields, coordinator display, tools, or chat templates should fail that gate until both SFT and evaluation are updated together.",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Scope and limitations\n\nThese are exact initial-turn examples for two representative tasks. Later turns additionally contain each agent's partial history and visual observations, so this artifact does not claim that an entire multi-turn training example and live rollout are identical merely from the initial-turn check. The demonstration-generation prompt is intentionally different: it asks a stronger model to produce and justify a complete validated trajectory rather than predict one acting agent's next tool call.",
                },
            ]
        )
        artifact = {
            "surface": "report",
            "manifest": {
                "version": 1,
                "surface": "report",
                "title": "Exact current demonstration, SFT, and evaluation prompts",
                "description": "Exact demonstration-generation request plus byte-identical initial-turn SFT and live-evaluation prompts from real task carriers.",
                "generatedAt": "2026-08-20T00:00:00Z",
                "sources": sources,
                "cards": [],
                "charts": [
                    {
                        "id": "prompt-length",
                        "title": "Initial-turn user prompt length",
                        "subtitle": "Exact character counts before chat-template tokenization",
                        "question": "How much text differs between SFT and live evaluation?",
                        "rationale": "Grouped bars compare the two discrete prompt paths for each task without implying a time trend.",
                        "intent": "comparison",
                        "type": "bar",
                        "dataset": "prompt_lengths",
                        "sourceId": "prompt-lengths",
                        "encodings": {
                            "x": {"field": "task_path", "type": "nominal", "label": "Task and prompt path"},
                            "y": {"field": "characters", "type": "quantitative", "label": "Characters"},
                            "tooltip": [
                                {"field": "task", "type": "nominal", "label": "Task"},
                                {"field": "path", "type": "nominal", "label": "Path"},
                                {"field": "characters", "type": "quantitative", "label": "Characters"}
                            ]
                        },
                        "surface": {"palette": {"kind": "single"}, "legend": {"show": False}, "valueLabels": {"show": True}},
                        "layout": "full"
                    }
                ],
                "tables": [],
                "blocks": blocks,
            },
            "snapshot": {
                "version": 1,
                "generatedAt": "2026-08-20T00:00:00Z",
                "status": "ready",
                "datasets": {
                    "prompt_lengths": [
                        row
                        for task_name, _task_text, sft_chars, eval_chars in task_sections
                        for row in (
                            {"task": task_name, "path": "SFT", "task_path": f"{task_name} — SFT", "characters": sft_chars},
                            {"task": task_name, "path": "Live evaluation", "task_path": f"{task_name} — eval", "characters": eval_chars},
                        )
                    ]
                },
            },
            "sources": sources,
        }
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_output.write_text(json.dumps(artifact, indent=2) + "\n")
        print(args.artifact_output)


if __name__ == "__main__":
    main()
