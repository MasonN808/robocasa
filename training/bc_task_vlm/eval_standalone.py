"""Standalone, model-agnostic structured tool-call eval for the task VLM.

Runs the exact per-step scoring used by in-training structured eval
(`score_structured_prediction`) against a fixed eval manifest, with two
generation backends sharing byte-identical prompts:

- ``hf``: a local HF vision-language model (base weights or base + LoRA
  adapter), greedy decoding, optional xgrammar-constrained JSON output.
- ``gemini``: a Vertex AI Gemini model via google-genai, temperature 0,
  optional JSON response schema, threaded with retries and resume.

Outputs per run directory: structured_eval_predictions.jsonl,
structured_eval_metrics.json, eval_config.json, optional prompt_dump.jsonl.

Examples:
    python -m training.bc_task_vlm.eval_standalone --backend gemini \
        --manifest .../eval_manifest_heldout_tasks.json \
        --output-dir training/bc_task_vlm/eval_runs/gemini_zeroshot__heldout_task

    python -m training.bc_task_vlm.eval_standalone --backend hf \
        --model-name-or-path Qwen/Qwen3.6-27B \
        --adapter-path training/bc_task_vlm/runs/qwen36-27b-47task-sft \
        --manifest .../eval_manifest_heldout_trajectories.json \
        --output-dir training/bc_task_vlm/eval_runs/qwen36_sft__heldout_traj
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.bc_task_vlm.dataset import (
    SFT_FORMAT_PLAIN,
    SUPPORTED_SFT_FORMATS,
    build_centralized_examples,
)
from training.bc_task_vlm.evaluation import (
    load_prediction_records,
    metrics_from_prediction_records,
    score_structured_prediction,
)
from training.bc_task_vlm.metrics import build_trajectory_aggregate_metrics
from training.bc_task_vlm.prompting import (
    append_few_shot_block,
    build_few_shot_block,
)
from training.bc_task_vlm.schema_utils import (
    build_single_step_response_schema,
    declared_tool_arg_names,
)
from training.bc_task_vlm.task_registry import AGENT_IDS

PREDICTIONS_FILENAME = "structured_eval_predictions.jsonl"
METRICS_FILENAME = "structured_eval_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "gemini", "vllm"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Overrides the dataset_root recorded in the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sft-format",
        choices=sorted(SUPPORTED_SFT_FORMATS),
        default=SFT_FORMAT_PLAIN,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--dump-prompts",
        type=int,
        default=0,
        help="Write the first N rendered prompts to prompt_dump.jsonl for parity checks.",
    )
    parser.add_argument(
        "--forced-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constrain generation to the plain tool-call JSON schema.",
    )
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Gemini backend: use Vertex's native function-calling API "
        "(FunctionDeclaration tools + forced ANY mode) instead of a "
        "response_schema. The returned function call is scored identically. "
        "Makes Gemini comparable to the hf backend's tool_call format.",
    )
    parser.add_argument(
        "--task-spec-detail",
        action="store_true",
        help="Append the verified task spec's goal, execution rules, initial "
        "object locations, and communication protocol to each prompt.",
    )
    parser.add_argument(
        "--predict-acting-agent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Agent-prediction (v2) eval: the prompt does not name the acting "
        'agent; the model chooses it via a required "agent" argument and may '
        "call task_complete. Requires --no-forced-json (hf) or --native-tools "
        "(gemini) — the plain forced-JSON schema has no agent slot.",
    )
    parser.add_argument(
        "--train-get-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate active-observation (v3) get_image targets.",
    )
    parser.add_argument(
        "--predict-task-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Match the flag the adapter was trained with. "
            "--no-predict-task-complete drops task_complete from the tool set "
            "and from the supervised targets."
        ),
    )
    parser.add_argument(
        "--causal-single-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild v3 examples with the causal single-cache contract.",
    )
    parser.add_argument(
        "--partial-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Partial-observability v1 eval: restrict each example's history "
            "to the acting agent's own actions plus delivered communicate "
            "messages. Mutually exclusive with --predict-acting-agent/"
            "--train-get-image."
        ),
    )
    parser.add_argument(
        "--partial-step-index-mode",
        choices=("global", "local", "none"),
        default="global",
        help=(
            "With --partial-history: how step indices are rendered. 'global' "
            "keeps the joint demonstration index (leaks the other agent's "
            "hidden activity via gaps), 'local' renumbers per agent, 'none' "
            "omits indices from the prompt."
        ),
    )
    parser.add_argument(
        "--partial-observation-mode",
        choices=("cache", "consume-once"),
        default="consume-once",
        help=(
            'With --partial-history: how pixels are supplied. "cache" keeps a persistent per-agent observation; "consume-once" feeds a get_image result to that agent\'s next target tool call, then discards it.'
        ),
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        choices=(0, 1),
        help="Ablation: append the task spec example trajectory to the prompt.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample_ids already present in the output predictions file.",
    )
    parser.add_argument("--example-build-workers", type=int, default=8)

    hf_group = parser.add_argument_group("hf backend")
    hf_group.add_argument("--model-name-or-path", type=str, default=None)
    hf_group.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="LoRA adapter run dir; loaded on top of --model-name-or-path.",
    )
    hf_group.add_argument(
        "--processor-name-or-path",
        type=str,
        default=None,
        help="Defaults to --model-name-or-path (always the BASE model).",
    )
    hf_group.add_argument("--batch-size", type=int, default=4)
    hf_group.add_argument("--max-new-tokens", type=int, default=256)
    hf_group.add_argument("--max-length", type=int, default=16384)
    hf_group.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model's <think> reasoning in the chat template "
        "(hf backend). Incompatible with --forced-json: reasoning needs free "
        "generation, so the tool call is parsed from the post-</think> text. "
        "Uses --thinking-max-new-tokens for the generation budget.",
    )
    hf_group.add_argument(
        "--thinking-max-new-tokens",
        type=int,
        default=2048,
        help="Generation budget when --enable-thinking is set (reasoning + "
        "answer).",
    )
    hf_group.add_argument(
        "--image-resolution",
        type=int,
        default=512,
        help="Square pixel budget applied to the processor, matching the "
        "training pipeline's --image-resolution. Must equal the value the "
        "evaluated checkpoint was trained with.",
    )
    hf_group.add_argument("--num-workers", type=int, default=0)
    hf_group.add_argument("--trust-remote-code", action="store_true")

    vllm_group = parser.add_argument_group("vllm backend")
    vllm_group.add_argument(
        "--vllm-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM API base URL.",
    )
    vllm_group.add_argument(
        "--vllm-model",
        default=None,
        help="Served model or LoRA name used in requests. Defaults to "
        "--adapter-path, else --model-name-or-path.",
    )
    vllm_group.add_argument("--vllm-api-key", default=None)
    vllm_group.add_argument("--vllm-request-timeout", type=float, default=180.0)

    gemini_group = parser.add_argument_group("gemini backend")
    gemini_group.add_argument("--model", type=str, default="gemini-3-flash-preview")
    gemini_group.add_argument("--project", type=str, default=None)
    gemini_group.add_argument("--location", type=str, default=None)
    gemini_group.add_argument("--concurrency", type=int, default=8)
    gemini_group.add_argument("--max-retries", type=int, default=5)
    gemini_group.add_argument("--request-timeout", type=int, default=120)
    gemini_group.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible sampling in the hf backend when "
        "--temperature > 0.",
    )
    gemini_group.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Gemini thinking token budget; 0 disables thinking. -1 leaves the model default.",
    )
    gemini_group.add_argument("--max-output-tokens", type=int, default=256)
    gemini_group.add_argument(
        "--max-cost-usd",
        type=float,
        default=25.0,
        help="Abort scheduling new requests once the observed cost exceeds this.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared example loading
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    sample_id: str
    feature: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        feature = self.feature
        return {
            "sample_id": feature["sample_id"],
            "task_name": feature["task_name"],
            "trajectory_id": feature["trajectory_id"],
            "step_index": feature["step_index"],
            "agent_id": feature["agent_id"],
            "target_payload": feature["target_payload"],
            "target_tool_call": feature["target_tool_call"],
            "target_text": feature["target_text"],
            "allowed_tool_specs": feature["allowed_tool_specs"],
        }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _load_few_shot_blocks(task_names: list[str]) -> dict[str, str]:
    from data_generation.task_level.tasks.specs import load_verified_task_specs
    from training.bc_task_vlm.task_registry import _camel_to_snake_case

    blocks: dict[str, str] = {}
    wanted = set(task_names)
    for spec in load_verified_task_specs():
        task_name = _camel_to_snake_case(spec.composite_task)
        if task_name in wanted and spec.example_trajectory:
            blocks[task_name] = build_few_shot_block(spec.example_trajectory)
    missing = wanted - set(blocks)
    if missing:
        raise ValueError(f"No spec example trajectory for tasks: {sorted(missing)}")
    return blocks


def _load_task_spec_blocks(task_names: list[str]) -> dict[str, str]:
    """Builds per-task prompt blocks with authoritative task-spec details.

    Tests the lack-of-knowledge hypothesis: out-of-the-box models fail largely on
    unstated conventions (goal specifics, execution rules, where objects
    start, the communicate-first protocol) that SFT models absorb from data.
    """

    from data_generation.task_level.tasks.specs import load_verified_task_specs
    from training.bc_task_vlm.task_registry import _camel_to_snake_case

    blocks: dict[str, str] = {}
    wanted = set(task_names)
    for spec in load_verified_task_specs():
        task_name = _camel_to_snake_case(spec.composite_task)
        if task_name not in wanted:
            continue
        lines = [
            "Task specification (authoritative details from the task definition):",
            f"- Goal: {spec.task_goal}",
        ]
        for rule in spec.extra_execution_rules:
            lines.append(f"- Rule: {rule}")
        objects = (spec.initial_state or {}).get("objects", {})
        if objects:
            placements = ", ".join(
                f"{object_id} at {info.get('location', 'unknown')}"
                for object_id, info in objects.items()
            )
            lines.append(f"- Initial object locations: {placements}")
        if "initial_communication" in spec.validator_checks:
            lines.append(
                "- Protocol: demonstrations open with communicate steps in "
                "which each agent announces its plan before any physical "
                "action, and agents announce hand-offs to each other."
            )
        blocks[task_name] = "\n".join(lines)
    missing = wanted - set(blocks)
    if missing:
        raise ValueError(f"No verified task spec for tasks: {sorted(missing)}")
    return blocks


def load_eval_samples(
    *,
    manifest: dict[str, Any],
    dataset_root: Path,
    sft_format: str,
    few_shot: int,
    task_spec_detail: bool,
    max_samples: int | None,
    example_build_workers: int,
    predict_agent: bool = False,
    predict_task_complete: bool = False,
    train_get_image: bool = False,
    causal_single_cache: bool = False,
    partial_history: bool = False,
    partial_step_index_mode: str = "global",
    partial_observation_mode: str = "cache",
) -> list[EvalSample]:
    manifest_samples = manifest["samples"]
    if max_samples is not None:
        manifest_samples = manifest_samples[: max(0, max_samples)]

    task_names = sorted({sample["task_name"] for sample in manifest_samples})
    trajectory_ids_by_task: dict[str, set[str]] = {}
    for sample in manifest_samples:
        trajectory_ids_by_task.setdefault(sample["task_name"], set()).add(
            sample["trajectory_id"]
        )

    examples = build_centralized_examples(
        dataset_root=dataset_root,
        task_names=task_names,
        sft_format=sft_format,
        trajectory_ids_by_task=trajectory_ids_by_task,
        example_build_workers=example_build_workers,
        show_progress=True,
        progress_description="Building eval examples",
        predict_agent=predict_agent,
        predict_task_complete=predict_task_complete,
        train_get_image=train_get_image,
        causal_single_cache=causal_single_cache,
        partial_history=partial_history,
        partial_step_index_mode=partial_step_index_mode,
        partial_observation_mode=partial_observation_mode,
    )
    examples_by_id = {example.sample_id: example for example in examples}

    few_shot_blocks = (
        _load_few_shot_blocks(task_names) if few_shot > 0 else {}
    )
    task_spec_blocks = (
        _load_task_spec_blocks(task_names) if task_spec_detail else {}
    )

    def decorated_sample(example: Any) -> EvalSample:
        feature = example.to_feature_dict()
        if few_shot > 0 or task_spec_detail:
            feature = dict(feature)
            feature["messages"] = copy.deepcopy(feature["messages"])
        if task_spec_detail:
            feature["messages"] = append_few_shot_block(
                feature["messages"],
                task_spec_blocks[feature["task_name"]],
            )
        if few_shot > 0:
            feature["messages"] = append_few_shot_block(
                feature["messages"],
                few_shot_blocks[feature["task_name"]],
            )
        return EvalSample(sample_id=example.sample_id, feature=feature)

    eval_samples: list[EvalSample] = []
    missing_ids: list[str] = []
    for sample in manifest_samples:
        example = examples_by_id.get(sample["sample_id"])
        if example is None:
            missing_ids.append(sample["sample_id"])
            continue
        eval_samples.append(decorated_sample(example))

    if predict_agent:
        # Manifests predate v2 and only list real expert steps; the synthetic
        # terminal task_complete examples (one per selected trajectory) would
        # otherwise be dropped by the manifest filter. Downstream (resume,
        # metrics) keys off this function's output, so appending here is
        # sufficient.
        manifest_id_set = {sample["sample_id"] for sample in manifest_samples}
        terminal_examples = sorted(
            (
                example
                for example in examples
                if example.sample_id not in manifest_id_set
            ),
            key=lambda example: example.sample_id,
        )
        eval_samples.extend(
            decorated_sample(example) for example in terminal_examples
        )

    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} manifest samples missing from dataset root "
            f"{dataset_root} (first: {missing_ids[:5]})"
        )
    return eval_samples


def dump_prompts(
    eval_samples: list[EvalSample],
    *,
    count: int,
    output_dir: Path,
) -> None:
    if count <= 0:
        return
    dump_path = output_dir / "prompt_dump.jsonl"
    with dump_path.open("w", encoding="utf-8") as handle:
        for sample in eval_samples[:count]:
            messages = sample.feature["messages"]
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "system_text": _message_text(messages[0]),
                        "user_text": _message_text(messages[-2])
                        if len(messages) > 2
                        else _message_text(messages[-1]),
                        "image_paths": list(sample.feature["image_paths"]),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    print(f"Wrote {min(count, len(eval_samples))} prompts to {dump_path}")


# ---------------------------------------------------------------------------
# Forced-JSON schema (plain format)
# ---------------------------------------------------------------------------


def build_vertex_function_declarations(
    allowed_tool_specs: dict[str, dict[str, Any]],
    *,
    include_agent_param: bool = False,
) -> list[dict[str, Any]]:
    """Builds one Vertex FunctionDeclaration per allowed tool.

    Reuses the shared response-schema helpers so a tool's argument names and
    types are declared exactly as they are for the forced-JSON path — the only
    difference is the delivery mechanism (native function calling vs a
    response_schema), keeping the two comparable.
    """

    from data_generation.task_level.tasks.shared.schema import (
        _build_symbolic_field_schema,
        _iter_declared_tool_arg_names,
        _resolve_tool_arg_schema_type,
    )

    declarations: list[dict[str, Any]] = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        required_args, _ = declared_tool_arg_names(tool_spec)
        properties: dict[str, Any] = {}
        if include_agent_param:
            properties["agent"] = {
                "type": "STRING",
                "enum": list(AGENT_IDS),
                "description": "The agent that performs this call.",
            }
            required_args = ["agent", *required_args]
        for field_name in _iter_declared_tool_arg_names(tool_spec):
            properties[field_name] = _build_symbolic_field_schema(
                field_name,
                AGENT_IDS,
                _resolve_tool_arg_schema_type(field_name, tool_spec),
            )
            allowed_values_key = f"allowed_{field_name}"
            if allowed_values_key in tool_spec:
                target = properties[field_name]
                if target.get("type") == "ARRAY":
                    target = target["items"]
                target["enum"] = list(tool_spec[allowed_values_key])
        declarations.append(
            {
                "name": tool_name,
                "description": (tool_spec.get("description") or "").strip(),
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": list(required_args),
                },
            }
        )
    return declarations


def build_plain_response_schema(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Narrows the single-step schema to the plain {"tool","args"} shape."""

    step_schema = build_single_step_response_schema(
        agent_ids=AGENT_IDS,
        allowed_tool_specs=allowed_tool_specs,
    )
    step_properties = step_schema["properties"]["steps"]["items"]["properties"]
    return {
        "type": "OBJECT",
        "required": ["tool", "args"],
        "properties": {
            "tool": copy.deepcopy(step_properties["tool"]),
            "args": copy.deepcopy(step_properties["args"]),
        },
    }


_VERTEX_TO_JSON_SCHEMA_TYPES = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
}


def vertex_schema_to_json_schema(schema: Any) -> Any:
    """Converts the Vertex-style schema (uppercase types) to standard JSON schema."""

    if isinstance(schema, list):
        return [vertex_schema_to_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            converted[key] = _VERTEX_TO_JSON_SCHEMA_TYPES.get(value, value.lower())
        elif key == "nullable":
            continue
        else:
            converted[key] = vertex_schema_to_json_schema(value)
    if converted.get("type") == "object" and "additionalProperties" not in converted:
        converted["additionalProperties"] = False
    return converted


# ---------------------------------------------------------------------------
# Record writing / resume
# ---------------------------------------------------------------------------


class RecordWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=True) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def load_completed_sample_ids(predictions_path: Path) -> set[str]:
    """Sample ids that are genuinely done — records whose generation FAILED
    (retries exhausted) are excluded so --resume regenerates them instead of
    treating a transient API/GPU outage as a permanent wrong answer."""

    if not predictions_path.exists():
        return set()
    completed: set[str] = set()
    for record in load_prediction_records(predictions_path):
        parse_error = record.get("parse_error")
        if isinstance(parse_error, str) and parse_error.startswith(
            "generation_failed"
        ):
            continue
        completed.add(record["sample_id"])
    return completed


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------


def run_hf_backend(
    eval_samples: list[EvalSample],
    args: argparse.Namespace,
    writer: RecordWriter,
) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from training.bc_task_vlm.evaluation import VisionGenerationCollator

    try:
        from transformers import AutoModelForImageTextToText as AutoVisionModel
    except ImportError:  # pragma: no cover - older transformers
        from transformers import AutoModelForVision2Seq as AutoVisionModel

    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required for the hf backend.")

    if args.enable_thinking and args.forced_json:
        raise ValueError(
            "--enable-thinking is incompatible with --forced-json: constrained "
            "JSON decoding suppresses the <think> block. Re-run with "
            "--no-forced-json."
        )

    model = AutoVisionModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path))
        model = model.merge_and_unload()
        print(f"Loaded and merged LoRA adapter from {args.adapter_path}")
    model.eval()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = True
    if args.temperature and args.temperature > 0:
        torch.manual_seed(args.seed)

    collator = VisionGenerationCollator(
        processor_name_or_path=args.processor_name_or_path or args.model_name_or_path,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        sft_format=args.sft_format,
        image_resolution=args.image_resolution if args.image_resolution > 0 else None,
        enable_thinking=args.enable_thinking,
    )
    processor = collator.processor
    max_new_tokens = (
        args.thinking_max_new_tokens if args.enable_thinking else args.max_new_tokens
    )

    grammar_compiler = None
    compiled_grammars: dict[str, Any] = {}
    if args.forced_json:
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise RuntimeError(
                "--forced-json for the hf backend needs xgrammar "
                "(`uv pip install xgrammar`). Re-run with --no-forced-json "
                "for unconstrained generation."
            ) from exc
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=getattr(
                model.config,
                "vocab_size",
                None,
            )
            or len(tokenizer),
        )
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

    class _ListDataset(Dataset):
        def __init__(self, features: list[dict[str, Any]]):
            self._features = features

        def __len__(self) -> int:
            return len(self._features)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self._features[index]

    # Group by task so one grammar (per-task tool schema) covers a whole batch.
    samples_by_task: dict[str, list[EvalSample]] = {}
    for sample in eval_samples:
        samples_by_task.setdefault(sample.feature["task_name"], []).append(sample)

    device = next(model.parameters()).device
    total_done = 0
    for task_name, task_samples in samples_by_task.items():
        logits_processor_factory = None
        if grammar_compiler is not None:
            import xgrammar as xgr

            if task_name not in compiled_grammars:
                schema = vertex_schema_to_json_schema(
                    build_plain_response_schema(
                        task_samples[0].feature["allowed_tool_specs"]
                    )
                )
                compiled_grammars[task_name] = grammar_compiler.compile_json_schema(
                    json.dumps(schema)
                )
            compiled = compiled_grammars[task_name]

            def logits_processor_factory(compiled=compiled):
                return xgr.contrib.hf.LogitsProcessor(compiled)

        dataloader = DataLoader(
            _ListDataset([sample.feature for sample in task_samples]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(args.num_workers, 0),
            collate_fn=collator,
        )
        with torch.inference_mode():
            for batch in dataloader:
                sample_metadata = batch.pop("sample_metadata")
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                }
                if args.temperature and args.temperature > 0:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = args.temperature
                else:
                    generate_kwargs["do_sample"] = False
                if logits_processor_factory is not None:
                    generate_kwargs["logits_processor"] = [logits_processor_factory()]
                generated_ids = model.generate(**batch, **generate_kwargs)
                input_width = batch["input_ids"].shape[1]
                trimmed = [output[input_width:] for output in generated_ids]
                decoded_outputs = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                attention_mask = batch.get("attention_mask")
                for index, (metadata, decoded_text) in enumerate(
                    zip(sample_metadata, decoded_outputs, strict=True)
                ):
                    record = score_structured_prediction(
                        decoded_text=decoded_text,
                        metadata=metadata,
                        sft_format=args.sft_format,
                        predict_agent=args.predict_acting_agent,
                    )
                    truncated = False
                    if attention_mask is not None and args.max_length is not None:
                        truncated = bool(
                            int(attention_mask[index].sum().item()) >= args.max_length
                        )
                    record["generation_info"] = {
                        "backend": "hf",
                        "truncated": truncated,
                        "forced_json": bool(args.forced_json),
                        "enable_thinking": bool(args.enable_thinking),
                        "reasoning_chars": len(record.get("reasoning_text") or ""),
                        "temperature": args.temperature,
                    }
                    writer.write(record)
                    total_done += 1
                if total_done % 50 < args.batch_size:
                    print(f"[hf] scored {total_done}/{len(eval_samples)}")
    print(f"[hf] finished {total_done} samples")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

_GEMINI_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def run_vllm_backend(
    eval_samples: list[EvalSample],
    args: argparse.Namespace,
    writer: "RecordWriter",
) -> None:
    """Scores samples against a localhost vLLM OpenAI-compatible server.

    Reuses live_sim_eval.VllmPolicy so off-sim and live-sim share one request
    encoder: identical message/image serialization and identical tool-call
    decoding, which keeps the two eval regimes comparable.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from training.bc_task_vlm.live_sim_eval import VllmPolicy

    if not args.vllm_model:
        args.vllm_model = str(args.adapter_path or args.model_name_or_path or "")
    if not args.vllm_model:
        raise ValueError(
            "--vllm-model (or --adapter-path/--model-name-or-path) is required "
            "for the vllm backend."
        )
    policy = VllmPolicy(args)
    total = len(eval_samples)
    done = 0
    lock = threading.Lock()

    def worker(sample: EvalSample) -> None:
        nonlocal done
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                decoded_text = policy.generate(sample.feature)
                record = score_structured_prediction(
                    decoded_text=decoded_text,
                    metadata=sample.metadata,
                    sft_format=args.sft_format,
                    predict_agent=args.predict_acting_agent,
                )
                record["generation_info"] = {
                    "backend": "vllm",
                    "model": args.vllm_model,
                    "attempts": attempt,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "usage": dict(policy.last_usage or {}),
                }
                with lock:
                    writer.write(record)
                    done += 1
                    if done % 50 == 0 or done == total:
                        print(f"[vllm] {done}/{total}", flush=True)
                return
            except Exception as exc:  # transport/server errors are retryable
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 30))
        record = score_structured_prediction(
            decoded_text="",
            metadata=sample.metadata,
            sft_format=args.sft_format,
            predict_agent=args.predict_acting_agent,
        )
        record["generation_info"] = {
            "backend": "vllm",
            "model": args.vllm_model,
            "attempts": args.max_retries,
            "error": last_error,
        }
        with lock:
            writer.write(record)
            done += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, sample) for sample in eval_samples]
        for future in as_completed(futures):
            future.result()
    print(f"[vllm] finished {done} samples", flush=True)


def run_gemini_backend(
    eval_samples: list[EvalSample],
    args: argparse.Namespace,
    writer: RecordWriter,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google.genai import types as genai_types

    from data_generation.task_level.runtime.client import (
        MODEL_TEXT_PRICING_USD_PER_MILLION,
        build_raw_google_genai_client,
        load_dotenv_file,
    )

    import os

    load_dotenv_file()
    location = (
        args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    )
    client = build_raw_google_genai_client(
        args.project,
        location,
        timeout_sec=args.request_timeout,
    )

    pricing = None
    for pricing_model, tiers in MODEL_TEXT_PRICING_USD_PER_MILLION.items():
        if pricing_model in args.model:
            pricing = tiers.get("ON_DEMAND") or next(iter(tiers.values()), None)
            break
    if pricing is None:
        print(
            f"[gemini] no pricing entry for {args.model}; observed cost will "
            "read 0 and --max-cost-usd cannot trip."
        )

    schemas_by_task: dict[str, dict[str, Any]] = {}
    cost_lock = threading.Lock()
    state = {"cost_usd": 0.0, "done": 0, "aborted": False}

    def build_request(sample: EvalSample) -> tuple[list[Any], Any]:
        feature = sample.feature
        messages = feature["messages"]
        system_text = _message_text(messages[0])
        user_text = _message_text(messages[-2] if len(messages) > 2 else messages[-1])

        parts: list[Any] = []
        for image_path in feature["image_paths"]:
            path = Path(image_path)
            mime_type = _GEMINI_MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg")
            parts.append(
                genai_types.Part.from_bytes(
                    data=path.read_bytes(),
                    mime_type=mime_type,
                )
            )
        parts.append(genai_types.Part.from_text(text=user_text))

        config_kwargs: dict[str, Any] = {
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "system_instruction": system_text,
        }
        if args.thinking_budget >= 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=args.thinking_budget
            )
        task_name = feature["task_name"]
        if args.native_tools:
            # Native function calling: declare the task's tools and force the
            # model to emit one of them (mode=ANY).
            if task_name not in schemas_by_task:
                schemas_by_task[task_name] = [
                    genai_types.FunctionDeclaration(**declaration)
                    for declaration in build_vertex_function_declarations(
                        feature["allowed_tool_specs"],
                        include_agent_param=args.predict_acting_agent,
                    )
                ]
            config_kwargs["tools"] = [
                genai_types.Tool(function_declarations=schemas_by_task[task_name])
            ]
            config_kwargs["tool_config"] = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode="ANY")
            )
        elif args.forced_json:
            if task_name not in schemas_by_task:
                schemas_by_task[task_name] = build_plain_response_schema(
                    feature["allowed_tool_specs"]
                )
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schemas_by_task[task_name]
        return parts, genai_types.GenerateContentConfig(**config_kwargs)

    def decoded_text_from(response: Any) -> str:
        """Returns the model's answer as plain {"tool","args"} JSON text.

        With native tools the answer arrives as a structured function_call
        part; rendering it into the same plain shape lets the existing scorer
        handle both paths identically.
        """

        if args.native_tools:
            for candidate in getattr(response, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in (getattr(content, "parts", None) or []):
                    call = getattr(part, "function_call", None)
                    if call is not None and getattr(call, "name", None):
                        return json.dumps(
                            {"tool": call.name, "args": dict(call.args or {})}
                        )
            return ""
        return getattr(response, "text", None) or ""

    def observed_cost(usage_metadata: Any) -> float:
        if pricing is None or usage_metadata is None:
            return 0.0
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = (getattr(usage_metadata, "candidates_token_count", 0) or 0) + (
            getattr(usage_metadata, "thoughts_token_count", 0) or 0
        )
        return (
            prompt_tokens * float(pricing.get("input", 0.0))
            + output_tokens * float(pricing.get("output", 0.0))
        ) / 1_000_000.0

    def worker(sample: EvalSample) -> None:
        with cost_lock:
            if state["aborted"]:
                return
        parts, config = build_request(sample)
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=parts,
                    config=config,
                )
                decoded_text = decoded_text_from(response)
                usage_metadata = getattr(response, "usage_metadata", None)
                record = score_structured_prediction(
                    decoded_text=decoded_text,
                    metadata=sample.metadata,
                    sft_format=args.sft_format,
                    predict_agent=args.predict_acting_agent,
                )
                cost = observed_cost(usage_metadata)
                record["generation_info"] = {
                    "backend": "gemini",
                    "model": args.model,
                    "attempts": attempt,
                    "forced_json": bool(args.forced_json),
                    "native_tools": bool(args.native_tools),
                    "thinking_budget": args.thinking_budget,
                    "temperature": args.temperature,
                    "prompt_tokens": getattr(
                        usage_metadata, "prompt_token_count", None
                    ),
                    "output_tokens": getattr(
                        usage_metadata, "candidates_token_count", None
                    ),
                    "observed_cost_usd": cost,
                }
                writer.write(record)
                with cost_lock:
                    state["cost_usd"] += cost
                    state["done"] += 1
                    if state["done"] % 25 == 0:
                        print(
                            f"[gemini] scored {state['done']} "
                            f"(observed cost ${state['cost_usd']:.2f})"
                        )
                    if state["cost_usd"] > args.max_cost_usd:
                        state["aborted"] = True
                        print(
                            f"[gemini] cost guard tripped at "
                            f"${state['cost_usd']:.2f} > ${args.max_cost_usd:.2f}; "
                            "not scheduling further work."
                        )
                return
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                last_error = exc
                sleep_seconds = min(2**attempt, 60)
                time.sleep(sleep_seconds)
        record = score_structured_prediction(
            decoded_text="",
            metadata=sample.metadata,
            sft_format=args.sft_format,
        )
        record["parse_error"] = f"generation_failed: {last_error}"
        record["generation_info"] = {
            "backend": "gemini",
            "model": args.model,
            "attempts": args.max_retries,
            "failed": True,
        }
        writer.write(record)
        with cost_lock:
            state["done"] += 1

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, sample) for sample in eval_samples]
        for future in as_completed(futures):
            future.result()

    print(
        f"[gemini] finished {state['done']} samples, "
        f"observed cost ${state['cost_usd']:.2f}"
    )
    if state["aborted"]:
        raise SystemExit(
            "Gemini eval aborted by --max-cost-usd guard; re-run with --resume "
            "and a higher cap to finish."
        )


# ---------------------------------------------------------------------------
# Metrics finalization
# ---------------------------------------------------------------------------


# Flags that change what the model is shown or how it decodes; a resume that
# differs on any of these would silently pool incomparable records.
_PROMPT_DEFINING_CONFIG_KEYS = (
    "backend",
    "model",
    "model_name_or_path",
    "adapter_path",
    "sft_format",
    "task_spec_detail",
    "few_shot",
    "forced_json",
    "native_tools",
    "predict_acting_agent",
    "predict_task_complete",
    "train_get_image",
    "partial_history",
    "partial_step_index_mode",
    "partial_observation_mode",
    "temperature",
    "thinking_budget",
    "enable_thinking",
    "image_resolution",
    "max_length",
)


def _assert_resume_config_matches(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> None:
    """Refuses to resume into an output dir written under different prompt or
    decoding settings, which would pool records from two configurations."""

    config_path = args.output_dir / "eval_config.json"
    if not config_path.exists():
        return
    prior = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = []
    for key in _PROMPT_DEFINING_CONFIG_KEYS:
        if key not in prior:
            continue
        current = getattr(args, key, None)
        current = str(current) if isinstance(current, Path) else current
        if prior[key] != current:
            mismatches.append(f"{key}: prior={prior[key]!r} now={current!r}")
    if prior.get("manifest_split") not in (None, manifest.get("split")):
        mismatches.append(
            f"manifest_split: prior={prior.get('manifest_split')!r} "
            f"now={manifest.get('split')!r}"
        )
    if mismatches:
        raise SystemExit(
            "Refusing to --resume: current settings differ from the run that "
            f"produced {config_path}:\n  " + "\n  ".join(mismatches)
        )


def finalize_metrics(
    *,
    output_dir: Path,
    manifest_sample_ids: list[str],
) -> dict[str, float]:
    predictions_path = output_dir / PREDICTIONS_FILENAME
    raw_records = load_prediction_records(predictions_path)

    order = {sample_id: index for index, sample_id in enumerate(manifest_sample_ids)}
    manifest_id_set = set(manifest_sample_ids)

    # Keep only records for the current manifest, and when a sample_id was
    # written more than once (e.g. a failed generation followed by a
    # successful --resume retry) keep the last write. This makes metrics
    # correspond exactly to the declared eval set regardless of resume history.
    latest_by_id: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        sample_id = record["sample_id"]
        if sample_id in manifest_id_set:
            latest_by_id[sample_id] = record
    records = sorted(
        latest_by_id.values(),
        key=lambda record: order[record["sample_id"]],
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    metrics = metrics_from_prediction_records(records)
    metrics.update(build_trajectory_aggregate_metrics(records))
    (output_dir / METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.train_get_image and not (
        args.predict_acting_agent or args.partial_history
    ):
        raise SystemExit(
            "--train-get-image requires --predict-acting-agent (centralized v3) "
            "or --partial-history (partial-observability v3)."
        )
    if args.causal_single_cache and not (
        args.predict_acting_agent and args.train_get_image
    ):
        raise SystemExit(
            "--causal-single-cache requires --predict-acting-agent "
            "and --train-get-image."
        )
    if args.partial_history and args.predict_acting_agent:
        raise SystemExit(
            "--partial-history requires --no-predict-acting-agent: under "
            "partial observability the caller IS the actor."
        )
    if args.predict_acting_agent and args.forced_json and not args.native_tools:
        raise SystemExit(
            "--predict-acting-agent needs the agent inside the emitted "
            "arguments, which the plain forced-JSON schema cannot express. "
            "Use --no-forced-json (hf backend, native tool calls) or "
            "--native-tools (gemini backend)."
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_root = args.dataset_root or Path(manifest["dataset_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = args.output_dir / PREDICTIONS_FILENAME
    if predictions_path.exists() and not args.resume:
        raise SystemExit(
            f"{predictions_path} already exists; pass --resume to continue it "
            "or choose a fresh --output-dir."
        )

    eval_samples = load_eval_samples(
        manifest=manifest,
        dataset_root=dataset_root,
        sft_format=args.sft_format,
        few_shot=args.few_shot,
        task_spec_detail=args.task_spec_detail,
        max_samples=args.max_samples,
        example_build_workers=args.example_build_workers,
        predict_agent=args.predict_acting_agent,
        predict_task_complete=args.predict_task_complete,
        train_get_image=args.train_get_image,
        causal_single_cache=args.causal_single_cache,
        partial_history=args.partial_history,
        partial_step_index_mode=args.partial_step_index_mode,
        partial_observation_mode=args.partial_observation_mode.replace('-', '_'),
    )
    manifest_sample_ids = [sample.sample_id for sample in eval_samples]
    dump_prompts(eval_samples, count=args.dump_prompts, output_dir=args.output_dir)

    if args.resume:
        _assert_resume_config_matches(args, manifest)
    completed_ids = (
        load_completed_sample_ids(predictions_path) if args.resume else set()
    )
    pending_samples = [
        sample for sample in eval_samples if sample.sample_id not in completed_ids
    ]
    print(
        f"Eval set: {len(eval_samples)} samples "
        f"({len(completed_ids)} already done, {len(pending_samples)} pending)"
    )

    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_payload["manifest_split"] = manifest.get("split")
    config_payload["dataset_root_resolved"] = str(dataset_root)
    config_payload["num_samples"] = len(eval_samples)
    (args.output_dir / "eval_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    writer = RecordWriter(predictions_path)
    try:
        if pending_samples:
            if args.backend == "hf":
                run_hf_backend(pending_samples, args, writer)
            elif args.backend == "vllm":
                run_vllm_backend(pending_samples, args, writer)
            else:
                run_gemini_backend(pending_samples, args, writer)
    finally:
        writer.close()

    metrics = finalize_metrics(
        output_dir=args.output_dir,
        manifest_sample_ids=manifest_sample_ids,
    )
    headline_keys = (
        "structured_eval_num_samples",
        "structured_eval_tool_call_parse_rate",
        "structured_eval_tool_name_accuracy",
        "structured_eval_exact_tool_call_accuracy",
        "structured_eval_trajectory_all_steps_correct_rate",
        "structured_eval_trajectory_mean_correct_prefix_fraction",
    )
    print(json.dumps({key: metrics.get(key) for key in headline_keys}, indent=2))


if __name__ == "__main__":
    main()
