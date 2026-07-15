"""Structured generation evaluation for task-level VLM fine-tuning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

from training.bc_task_vlm.metrics import build_structured_eval_metrics
from training.bc_task_vlm.schema_utils import (
    canonicalize_for_comparison,
    compact_json_dumps,
    parse_first_json_object,
    validate_single_step_payload,
)
from training.bc_task_vlm.task_registry import AGENT_IDS
from training.bc_task_vlm.tool_calling import (
    parse_first_qwen_tool_call,
    tool_call_to_single_step_payload,
)


class VisionGenerationCollator:
    """Collates prompt-only multimodal batches for structured generation eval."""

    def __init__(
        self,
        *,
        processor_name_or_path: str,
        max_length: int | None,
        trust_remote_code: bool,
        sft_format: str,
        image_resolution: int | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
        if image_resolution and image_resolution > 0:
            from training.bc_task_vlm.dataset import _set_processor_image_resolution

            _set_processor_image_resolution(self.processor, image_resolution)
        self.max_length = max_length
        self.sft_format = sft_format
        self.enable_thinking = enable_thinking

    @staticmethod
    def _load_images(image_paths: list[str]):
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        return images

    @staticmethod
    def _message_text_for_plain_fallback(
        template_owner,
        message: dict[str, Any],
    ) -> str:
        tokenizer = getattr(template_owner, "tokenizer", None)
        image_token = (
            getattr(template_owner, "image_token", None)
            or getattr(tokenizer, "image_token", None)
            or "<|image|>"
        )

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()

        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                chunks.append(str(item.get("text", "")).strip())
            elif item_type == "image":
                chunks.append(image_token)
        return "".join(chunks).strip()

    def _apply_plain_fallback_template(
        self,
        template_owner,
        *,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        tokenizer = getattr(template_owner, "tokenizer", None) or template_owner
        bos_token = getattr(tokenizer, "bos_token", None) or ""
        chunks = [bos_token]
        for message in messages:
            role = message.get("role", "user")
            if role == "assistant":
                role = "model"
            elif role == "developer":
                role = "system"
            chunks.append(f"<|turn>{role}\n")
            chunks.append(
                self._message_text_for_plain_fallback(template_owner, message)
            )
            chunks.append("<turn|>\n")
        if add_generation_prompt:
            chunks.append("<|turn>model\n")
        return "".join(chunks)

    def _apply_chat_template(
        self,
        *,
        feature: dict[str, Any],
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": self.enable_thinking,
        }
        if self.sft_format == "tool_call":
            template_kwargs["tools"] = feature["tool_schemas"]

        if self.sft_format == "plain" and not getattr(
            self.processor, "chat_template", None
        ):
            return self._apply_plain_fallback_template(
                self.processor,
                messages=messages,
                add_generation_prompt=add_generation_prompt,
            )

        return self.processor.apply_chat_template(messages, **template_kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = [feature["messages"][:-1] for feature in features]
        images = [self._load_images(feature["image_paths"]) for feature in features]
        prompt_texts = [
            self._apply_chat_template(
                feature=feature,
                messages=message,
                add_generation_prompt=True,
            )
            for message, feature in zip(prompt_messages, features, strict=True)
        ]

        tokenizer_kwargs = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
            tokenizer_kwargs["truncation"] = True

        batch = self.processor(**tokenizer_kwargs)
        batch.pop("token_type_ids", None)
        batch["sample_metadata"] = [
            {
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
            for feature in features
        ]
        return batch


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    tensor_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor_batch[key] = value.to(device)
        else:
            tensor_batch[key] = value
    return tensor_batch


def _limit_dataset_by_trajectories(
    dataset,
    max_trajectories: int | None,
) -> tuple[Any, int | None]:
    if max_trajectories is None:
        return dataset, None

    max_trajectories = max(0, max_trajectories)
    selected_keys: set[tuple[str, str]] = set()
    selected_indices: list[int] = []
    for index in range(len(dataset)):
        feature = dataset[index]
        key = (str(feature["task_name"]), str(feature["trajectory_id"]))
        if key not in selected_keys:
            if len(selected_keys) >= max_trajectories:
                continue
            selected_keys.add(key)
        selected_indices.append(index)

    return torch.utils.data.Subset(dataset, selected_indices), len(selected_keys)


def _parse_plain_json_tool_call(text: str) -> dict[str, Any]:
    payload = parse_first_json_object(text)
    if not isinstance(payload, dict):
        raise ValueError("Plain response JSON must be an object.")
    tool_name = payload.get("tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError('Plain response JSON must contain a non-empty "tool".')
    arguments = payload.get("args")
    if not isinstance(arguments, dict):
        raise ValueError('Plain response JSON must contain object "args".')
    return {"name": tool_name.strip(), "arguments": dict(arguments)}


def _plain_tool_call_to_single_step_payload(
    tool_call: dict[str, Any],
    *,
    step_index: int,
    agent_id: str,
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return validate_single_step_payload(
        {
            "steps": [
                {
                    "step": step_index,
                    "agent": agent_id,
                    "tool": tool_call["name"],
                    "args": dict(tool_call["arguments"]),
                }
            ]
        },
        agent_ids=AGENT_IDS,
        allowed_tool_specs=allowed_tool_specs,
    )


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def split_reasoning(decoded_text: str) -> tuple[str, str]:
    """Separates a thinking model's `<think>...</think>` reasoning from its
    answer. Returns (reasoning_text, answer_text). Handles an unterminated
    <think> (truncated generation) by treating everything after the opening
    tag as reasoning with an empty answer. Text without think tags is all
    answer.
    """

    if "<think>" not in decoded_text:
        return "", decoded_text
    if "</think>" not in decoded_text:
        return decoded_text.split("<think>", 1)[1], ""
    reasoning_parts = _THINK_BLOCK_RE.findall(decoded_text)
    answer_text = _THINK_BLOCK_RE.sub("", decoded_text).strip()
    reasoning_text = "\n".join(reasoning_parts)
    return reasoning_text, answer_text


def score_structured_prediction(
    *,
    decoded_text: str,
    metadata: dict[str, Any],
    sft_format: str,
) -> dict[str, Any]:
    """Parses, validates, and exact-matches one decoded prediction.

    Returns the per-sample record written to structured_eval_predictions.jsonl.
    `metadata` needs: sample_id, task_name, trajectory_id, step_index, agent_id,
    target_payload, target_tool_call, target_text, allowed_tool_specs.
    A thinking model's <think>...</think> block is stripped before parsing so
    reasoning braces cannot be mistaken for the answer JSON.
    """

    reasoning_text, decoded_text = split_reasoning(decoded_text)
    target_payload = metadata["target_payload"]
    parsed_tool_call = None
    normalized_prediction = None
    parse_error = None
    validation_error = None
    exact_tool_match = False
    exact_args_match = False
    exact_tool_call_match = False
    exact_action_match = False

    try:
        if sft_format == "plain":
            parsed_tool_call = _parse_plain_json_tool_call(decoded_text)
        else:
            # Prefer the Qwen-style <tool_call> parser, but accept a plain
            # forced-JSON object as a fallback. This handles cases where the
            # generation backend (hf/gemini) was constrained to emit the
            # plain {"tool","args"} schema (e.g. --forced-json) even when
            # the SFT format used during training was the <tool_call> block.
            try:
                parsed_tool_call = parse_first_qwen_tool_call(decoded_text)
            except Exception as exc:  # pragma: no cover - defensive eval logging
                # If parsing failed because there's no <tool_call> block,
                # try the plain JSON parser before giving up.
                msg = str(exc)
                if "<tool_call>" in msg or "did not contain" in msg:
                    try:
                        parsed_tool_call = _parse_plain_json_tool_call(decoded_text)
                    except Exception as exc2:
                        # Prefer the original error if the fallback also fails.
                        raise exc2
                else:
                    raise
    except Exception as exc:  # pragma: no cover - defensive eval logging
        parse_error = str(exc)

    if parsed_tool_call is not None:
        try:
            if sft_format == "plain":
                normalized_prediction = _plain_tool_call_to_single_step_payload(
                    parsed_tool_call,
                    step_index=metadata["step_index"],
                    agent_id=metadata["agent_id"],
                    allowed_tool_specs=metadata["allowed_tool_specs"],
                )
            else:
                normalized_prediction = tool_call_to_single_step_payload(
                    parsed_tool_call,
                    step_index=metadata["step_index"],
                    agent_id=metadata["agent_id"],
                    agent_ids=AGENT_IDS,
                    allowed_tool_specs=metadata["allowed_tool_specs"],
                )
        except Exception as exc:  # pragma: no cover - defensive eval logging
            validation_error = str(exc)

    if normalized_prediction is not None:
        predicted_step = normalized_prediction["steps"][0]
        target_step = target_payload["steps"][0]
        exact_tool_match = predicted_step["tool"] == target_step["tool"]
        exact_args_match = canonicalize_for_comparison(
            predicted_step["args"]
        ) == canonicalize_for_comparison(target_step["args"])
        exact_tool_call_match = exact_tool_match and exact_args_match
        exact_action_match = canonicalize_for_comparison(
            {
                "step": predicted_step["step"],
                "agent": predicted_step["agent"],
                "tool": predicted_step["tool"],
                "args": predicted_step["args"],
            }
        ) == canonicalize_for_comparison(
            {
                "step": target_step["step"],
                "agent": target_step["agent"],
                "tool": target_step["tool"],
                "args": target_step["args"],
            }
        )

    return {
        "sample_id": metadata["sample_id"],
        "task_name": metadata["task_name"],
        "trajectory_id": metadata["trajectory_id"],
        "step_index": metadata["step_index"],
        "prediction_text": decoded_text,
        "reasoning_text": reasoning_text,
        "target_text": metadata["target_text"],
        "target_tool_call": metadata["target_tool_call"],
        "target_payload": compact_json_dumps(target_payload),
        "parsed_tool_call": parsed_tool_call,
        "parse_error": parse_error,
        "validation_error": validation_error,
        "exact_tool_match": exact_tool_match,
        "exact_args_match": exact_args_match,
        "exact_tool_call_match": exact_tool_call_match,
        "exact_action_step_match": exact_action_match,
    }


def _is_communicate_record(record: dict[str, Any]) -> bool:
    return (record.get("target_tool_call") or {}).get("name") == "communicate"


def metrics_from_prediction_records(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Reduces per-sample prediction records to the structured eval metrics."""

    total_samples = len(records)
    parsed = sum(1 for record in records if record["parse_error"] is None)
    valid = sum(
        1
        for record in records
        if record["parse_error"] is None and record["validation_error"] is None
    )
    metrics = build_structured_eval_metrics(
        total_samples=total_samples,
        parsed_tool_calls=parsed,
        valid_tool_calls=valid,
        exact_tool_matches=sum(bool(r["exact_tool_match"]) for r in records),
        exact_args_matches=sum(bool(r["exact_args_match"]) for r in records),
        exact_tool_call_matches=sum(bool(r["exact_tool_call_match"]) for r in records),
        exact_action_matches=sum(bool(r["exact_action_step_match"]) for r in records),
    )

    # Communication steps carry free-text messages, where exact match is a
    # near-impossible bar for un-finetuned models; report the two step
    # families separately so physical-action competence is visible on its own.
    comm_records = [r for r in records if _is_communicate_record(r)]
    action_records = [r for r in records if not _is_communicate_record(r)]
    if comm_records:
        metrics["structured_eval_comm_step_fraction"] = len(comm_records) / max(
            total_samples, 1
        )
        metrics["structured_eval_comm_tool_selected_rate"] = sum(
            bool(r["exact_tool_match"]) for r in comm_records
        ) / len(comm_records)
        metrics["structured_eval_comm_exact_call_accuracy"] = sum(
            bool(r["exact_tool_call_match"]) for r in comm_records
        ) / len(comm_records)
    if action_records:
        metrics["structured_eval_action_tool_name_accuracy"] = sum(
            bool(r["exact_tool_match"]) for r in action_records
        ) / len(action_records)
        metrics["structured_eval_action_exact_call_accuracy"] = sum(
            bool(r["exact_tool_call_match"]) for r in action_records
        ) / len(action_records)
    return metrics


def load_prediction_records(predictions_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_structured_generation(
    *,
    model,
    eval_dataset,
    processor_name_or_path: str,
    output_dir: Path,
    max_length: int | None,
    max_new_tokens: int,
    batch_size: int,
    num_workers: int,
    trust_remote_code: bool,
    sft_format: str,
    max_samples: int | None = None,
    max_trajectories: int | None = None,
    image_resolution: int | None = None,
) -> dict[str, float]:
    """Runs generation over the validation split and computes structured metrics."""

    if len(eval_dataset) == 0:
        return {}

    if max_samples is not None:
        max_samples = max(0, min(max_samples, len(eval_dataset)))
        dataset = torch.utils.data.Subset(eval_dataset, range(max_samples))
    else:
        dataset = eval_dataset
    dataset, trajectory_count = _limit_dataset_by_trajectories(
        dataset,
        max_trajectories,
    )

    collator = VisionGenerationCollator(
        processor_name_or_path=processor_name_or_path,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        sft_format=sft_format,
        image_resolution=image_resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers, 0),
        collate_fn=collator,
    )

    processor = collator.processor
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "structured_eval_predictions.jsonl"

    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module
    device = next(unwrapped_model.parameters()).device

    records: list[dict[str, Any]] = []

    previous_use_cache = getattr(unwrapped_model.config, "use_cache", None)
    if previous_use_cache is not None:
        unwrapped_model.config.use_cache = True
    unwrapped_model.eval()

    with predictions_path.open("w", encoding="utf-8") as handle:
        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
                desc="Structured validation",
                leave=False,
            ):
                sample_metadata = batch.pop("sample_metadata")
                batch = _move_batch_to_device(batch, device)
                generated_ids = unwrapped_model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                input_width = batch["input_ids"].shape[1]
                trimmed_ids = [output_ids[input_width:] for output_ids in generated_ids]
                decoded_outputs = processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for metadata, decoded_text in zip(
                    sample_metadata,
                    decoded_outputs,
                    strict=True,
                ):
                    record = score_structured_prediction(
                        decoded_text=decoded_text,
                        metadata=metadata,
                        sft_format=sft_format,
                    )
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    if previous_use_cache is not None:
        unwrapped_model.config.use_cache = previous_use_cache

    metrics = metrics_from_prediction_records(records)
    if trajectory_count is not None:
        metrics["structured_eval_num_trajectories"] = float(trajectory_count)
    metrics["structured_eval_num_workers"] = float(max(num_workers, 0))
    metrics_path = output_dir / "structured_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
