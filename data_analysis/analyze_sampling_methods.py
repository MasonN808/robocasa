#!/usr/bin/env python3
"""Analyze raw task-level sampling-method trajectory conversations.

The expected input layout is:

    <input-root>/<sampling-method>/<task>/trajectories/traj_*.json

For each trajectory, the script stitches all ``communicate`` messages into one
conversation string, embeds that string once, and computes exact unique-pair
cosine similarity summaries per task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np


QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_DATA_K3_ROOT = Path("data_generation/task_level/data_k3/raw/sampling_methods")
DEFAULT_DATA_ROOT = Path("data_generation/task_level/data/raw/sampling_methods")
DEFAULT_MAX_TRAJECTORIES_PER_TASK = 30
DEFAULT_EMBEDDING_BATCH_SIZE = 256
EMBEDDING_CACHE_VERSION = 1


@dataclass(frozen=True)
class TrajectoryRecord:
    sampling_method: str
    task: str
    trajectory_id: str
    composite_task: str
    path: Path
    step_count: int
    tool_counts: Counter[str]
    conversation_text: str
    conversation_message_count: int

    @property
    def environment(self) -> str:
        """Backward-compatible alias for older analysis call sites."""

        return self.task


@dataclass(frozen=True)
class EmbeddingItem:
    item_id: str
    sampling_method: str
    task: str
    trajectory_id: str
    text: str


def default_input_root() -> Path:
    if DEFAULT_DATA_K3_ROOT.exists():
        return DEFAULT_DATA_K3_ROOT
    return DEFAULT_DATA_ROOT


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def word_count(text: str) -> int:
    return len(text.split())


def numeric_stats(values: Iterable[int | float]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0, "total": 0, "min": None, "max": None, "avg": None}
    total = float(sum(values))
    return {
        "count": len(values),
        "total": total,
        "min": min(values),
        "max": max(values),
        "avg": total / len(values),
    }


def text_length_stats(texts: Iterable[str]) -> dict[str, Any]:
    texts = [text for text in texts if text]
    return {
        "texts": len(texts),
        "chars": numeric_stats(len(text) for text in texts),
        "words": numeric_stats(word_count(text) for text in texts),
    }


def tool_distribution(tool_counts: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(tool_counts.values())
    return {
        tool: {
            "count": count,
            "fraction": (count / total) if total else 0.0,
        }
        for tool, count in sorted(tool_counts.items())
    }


def discover_trajectory_dirs(input_root: Path) -> list[tuple[str, str, Path]]:
    """Return (sampling_method, task, trajectories_dir) tuples."""

    if (input_root / "trajectories").is_dir():
        return [("dataset", input_root.name, input_root / "trajectories")]

    children = sorted(path for path in input_root.iterdir() if path.is_dir())
    if any((child / "trajectories").is_dir() for child in children):
        return [
            (input_root.name, child.name, child / "trajectories")
            for child in children
            if (child / "trajectories").is_dir()
        ]

    trajectory_dirs: list[tuple[str, str, Path]] = []
    for method_dir in children:
        for task_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            trajectories_dir = task_dir / "trajectories"
            if trajectories_dir.is_dir():
                trajectory_dirs.append(
                    (method_dir.name, task_dir.name, trajectories_dir)
                )
    return trajectory_dirs


def extract_tool_name(step: dict[str, Any]) -> str:
    tool = step.get("tool")
    if isinstance(tool, str) and tool:
        return tool
    tool_call = step.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str) and name:
            return name
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return "<missing>"


def extract_communication_message(step: dict[str, Any]) -> str | None:
    args = step.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("message", "content", "text"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def format_conversation_line(step: dict[str, Any], message: str) -> str:
    agent = step.get("agent")
    args = step.get("args")
    recipient = args.get("to") if isinstance(args, dict) else None
    if isinstance(agent, str) and agent and isinstance(recipient, str) and recipient:
        return f"{agent} -> {recipient}: {message}"
    if isinstance(agent, str) and agent:
        return f"{agent}: {message}"
    return message


def load_records(
    input_root: Path,
    *,
    include_invalid: bool,
    max_trajectories_per_task: int | None,
    load_workers: int = 1,
) -> tuple[list[TrajectoryRecord], list[str]]:
    trajectory_dirs = discover_trajectory_dirs(input_root)
    if load_workers <= 1 or len(trajectory_dirs) <= 1:
        return load_records_from_dirs(
            trajectory_dirs,
            include_invalid=include_invalid,
            max_trajectories_per_task=max_trajectories_per_task,
        )

    records: list[TrajectoryRecord] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=load_workers) as executor:
        futures = [
            executor.submit(
                load_records_from_dir,
                sampling_method,
                task,
                trajectories_dir,
                include_invalid=include_invalid,
                max_trajectories_per_task=max_trajectories_per_task,
            )
            for sampling_method, task, trajectories_dir in trajectory_dirs
        ]
        for future in futures:
            group_records, group_warnings = future.result()
            records.extend(group_records)
            warnings.extend(group_warnings)

    return records, warnings


def load_records_from_dirs(
    trajectory_dirs: list[tuple[str, str, Path]],
    *,
    include_invalid: bool,
    max_trajectories_per_task: int | None,
) -> tuple[list[TrajectoryRecord], list[str]]:
    records: list[TrajectoryRecord] = []
    warnings: list[str] = []

    for sampling_method, task, trajectories_dir in trajectory_dirs:
        group_records, group_warnings = load_records_from_dir(
            sampling_method,
            task,
            trajectories_dir,
            include_invalid=include_invalid,
            max_trajectories_per_task=max_trajectories_per_task,
        )
        records.extend(group_records)
        warnings.extend(group_warnings)

    return records, warnings


def load_records_from_dir(
    sampling_method: str,
    task: str,
    trajectories_dir: Path,
    *,
    include_invalid: bool,
    max_trajectories_per_task: int | None,
) -> tuple[list[TrajectoryRecord], list[str]]:
    records: list[TrajectoryRecord] = []
    warnings: list[str] = []
    paths = sorted(trajectories_dir.glob("traj_*.json"))
    records_loaded_for_task = 0

    for path in paths:
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            warnings.append(f"Skipped invalid JSON {path}: {exc}")
            continue

        if not isinstance(payload, dict):
            warnings.append(f"Skipped non-object trajectory JSON {path}")
            continue

        validation = payload.get("validation")
        if (
            not include_invalid
            and isinstance(validation, dict)
            and validation.get("is_valid") is False
        ):
            continue

        steps = payload.get("steps")
        if not isinstance(steps, list):
            warnings.append(f"Skipped trajectory without steps list {path}")
            continue

        tool_counts: Counter[str] = Counter()
        conversation_lines: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool_name = extract_tool_name(step)
            tool_counts[tool_name] += 1
            if tool_name == "communicate":
                message = extract_communication_message(step)
                if message is not None:
                    conversation_lines.append(format_conversation_line(step, message))

        records.append(
            TrajectoryRecord(
                sampling_method=sampling_method,
                task=task,
                trajectory_id=str(payload.get("trajectory_id") or path.stem),
                composite_task=str(payload.get("composite_task") or ""),
                path=path,
                step_count=sum(tool_counts.values()),
                tool_counts=tool_counts,
                conversation_text="\n".join(conversation_lines),
                conversation_message_count=len(conversation_lines),
            )
        )
        records_loaded_for_task += 1
        if (
            max_trajectories_per_task is not None
            and records_loaded_for_task >= max_trajectories_per_task
        ):
            break

    return records, warnings


def summarize_records(
    records: list[TrajectoryRecord],
    *,
    sampling_method: str | None = None,
    task: str | None = None,
    similarity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    conversation_texts: list[str] = []
    composite_tasks = sorted(
        {record.composite_task for record in records if record.composite_task}
    )

    for record in records:
        tool_counts.update(record.tool_counts)
        conversation_texts.append(record.conversation_text)

    summary: dict[str, Any] = {
        "sampling_method": sampling_method,
        "task": task,
        "environment": task,
        "composite_tasks": composite_tasks,
        "num_trajectories": len(records),
        "num_steps": sum(record.step_count for record in records),
        "tool_call_distribution": tool_distribution(tool_counts),
        "conversation_messages": numeric_stats(
            record.conversation_message_count for record in records
        ),
        "conversation_length": text_length_stats(conversation_texts),
    }
    if similarity is not None:
        summary["trajectory_embedding_similarity"] = similarity
    return summary


def build_trajectory_embedding_items(
    records: list[TrajectoryRecord],
    *,
    instruction: str,
) -> list[EmbeddingItem]:
    def maybe_instruct(text: str) -> str:
        if not instruction:
            return text
        return f"Instruct: {instruction}\nQuery:{text}"

    return [
        EmbeddingItem(
            item_id=f"{record.sampling_method}/{record.task}/{record.trajectory_id}",
            sampling_method=record.sampling_method,
            task=record.task,
            trajectory_id=record.trajectory_id,
            text=maybe_instruct(record.conversation_text),
        )
        for record in records
    ]


def log_timing(message: str) -> None:
    print(message, file=sys.stderr)


def indexed_batches(
    items: list[str], batch_size: int
) -> Iterable[tuple[int, int, list[str]]]:
    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        yield start, end, items[start:end]


def maybe_log_embedding_progress(
    *,
    completed: int,
    total: int,
    batch_index: int,
    progress_interval: int,
    started_at: float,
) -> None:
    if progress_interval <= 0:
        return
    if completed < total and batch_index % progress_interval != 0:
        return
    elapsed = perf_counter() - started_at
    log_timing(
        f"Embedded {completed}/{total} trajectory conversations in {elapsed:.1f}s"
    )


def embed_with_vllm_offline(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    dtype: str | None,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    progress_interval: int,
) -> np.ndarray:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "Embedding provider 'vllm' requires vLLM. Install vllm>=0.8.5 "
            "in the environment where you run embedding analysis."
        ) from exc

    kwargs: dict[str, Any] = {"model": model_name}
    if dtype:
        kwargs["dtype"] = dtype
    if tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = tensor_parallel_size
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    try:
        model = LLM(task="embed", **kwargs)
    except TypeError:
        model = LLM(runner="pooling", **kwargs)

    vectors: list[list[float]] = []
    started_at = perf_counter()
    for batch_index, (_start, end, batch) in enumerate(
        indexed_batches(texts, batch_size),
        start=1,
    ):
        outputs = model.embed(batch)
        vectors.extend(output.outputs.embedding for output in outputs)
        maybe_log_embedding_progress(
            completed=end,
            total=len(texts),
            batch_index=batch_index,
            progress_interval=progress_interval,
            started_at=started_at,
        )
    return np.asarray(vectors, dtype=np.float32)


def embed_with_vllm_openai(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    base_url: str,
    api_key: str | None,
    progress_interval: int,
) -> np.ndarray:
    endpoint = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    vectors: list[list[float]] = []
    started_at = perf_counter()
    for batch_index, (_start, end, batch) in enumerate(
        indexed_batches(texts, batch_size),
        start=1,
    ):
        request_payload = {
            "model": model_name,
            "input": batch,
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM embeddings request failed: {exc} {body}") from exc

        data = response_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(
                f"vLLM embeddings response missing data list: {response_payload}"
            )
        for row in sorted(data, key=lambda item: int(item.get("index", 0))):
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"vLLM embeddings row missing embedding: {row}")
            vectors.append(embedding)
        maybe_log_embedding_progress(
            completed=end,
            total=len(texts),
            batch_index=batch_index,
            progress_interval=progress_interval,
            started_at=started_at,
        )

    return np.asarray(vectors, dtype=np.float32)


def embedding_items_sha256(items: list[EmbeddingItem]) -> str:
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(
            compact_json(
                {
                    "item_id": item.item_id,
                    "sampling_method": item.sampling_method,
                    "task": item.task,
                    "trajectory_id": item.trajectory_id,
                    "text": item.text,
                }
            ).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_embedding_cache_metadata(
    items: list[EmbeddingItem],
    *,
    provider: str,
    model_name: str,
    instruction: str,
    dtype: str | None,
    max_model_len: int | None,
) -> dict[str, Any]:
    return {
        "version": EMBEDDING_CACHE_VERSION,
        "provider": provider,
        "model": model_name,
        "instruction": instruction,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "num_items": len(items),
        "items_sha256": embedding_items_sha256(items),
    }


def embedding_cache_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "trajectory_embeddings.npy",
        output_dir / "trajectory_embeddings_metadata.json",
    )


def load_cached_embeddings(
    output_dir: Path,
    expected_metadata: dict[str, Any],
) -> np.ndarray | None:
    embeddings_path, metadata_path = embedding_cache_paths(output_dir)
    if not embeddings_path.is_file() or not metadata_path.is_file():
        return None
    try:
        existing_metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None
    if existing_metadata != expected_metadata:
        return None
    embeddings = np.load(embeddings_path, allow_pickle=False)
    return np.asarray(embeddings, dtype=np.float32)


def write_cached_embeddings(
    output_dir: Path,
    metadata: dict[str, Any],
    embeddings: np.ndarray,
) -> None:
    embeddings_path, metadata_path = embedding_cache_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def update_similarity_accumulator(
    accumulator: dict[str, Any], values: np.ndarray
) -> None:
    if values.size == 0:
        return
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    accumulator["computed_pairs"] += int(values.size)
    accumulator["sum"] += float(values.sum(dtype=np.float64))
    accumulator["min"] = (
        float(values.min())
        if accumulator["min"] is None
        else min(accumulator["min"], float(values.min()))
    )
    accumulator["max"] = (
        float(values.max())
        if accumulator["max"] is None
        else max(accumulator["max"], float(values.max()))
    )


def empty_similarity_summary(num_items: int) -> dict[str, Any]:
    num_pairs = num_items * (num_items - 1) // 2
    return {
        "num_items": num_items,
        "num_pairs": num_pairs,
        "computed_pairs": 0,
        "exact": True,
        "min": None,
        "max": None,
        "avg": None,
    }


def exact_pairwise_similarity_summary(
    vectors: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    """Compute min/max/avg over every unique unordered pair."""

    count = int(vectors.shape[0])
    num_pairs = count * (count - 1) // 2
    if num_pairs == 0:
        return empty_similarity_summary(count)

    accumulator = {"computed_pairs": 0, "sum": 0.0, "min": None, "max": None}
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        sims = vectors[start:end] @ vectors.T
        for row_offset, row_index in enumerate(range(start, end)):
            update_similarity_accumulator(
                accumulator, sims[row_offset, row_index + 1 :]
            )

    computed_pairs = accumulator["computed_pairs"]
    return {
        "num_items": count,
        "num_pairs": num_pairs,
        "computed_pairs": computed_pairs,
        "exact": computed_pairs == num_pairs,
        "min": accumulator["min"],
        "max": accumulator["max"],
        "avg": accumulator["sum"] / computed_pairs if computed_pairs else None,
    }


def similarity_for_items(
    items: list[EmbeddingItem],
    embeddings_by_id: dict[str, np.ndarray],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    vectors = [
        embeddings_by_id[item.item_id]
        for item in items
        if item.item_id in embeddings_by_id
    ]
    if not vectors:
        return empty_similarity_summary(0)
    return exact_pairwise_similarity_summary(
        np.asarray(vectors, dtype=np.float32),
        chunk_size=chunk_size,
    )


def compute_similarity_maps(
    items: list[EmbeddingItem],
    embeddings: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    embeddings = normalize_embeddings(embeddings)
    if len(items) != len(embeddings):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(embeddings)} vectors for {len(items)} items"
        )
    embeddings_by_id = {
        item.item_id: embedding for item, embedding in zip(items, embeddings)
    }

    by_task: dict[tuple[str, str], list[EmbeddingItem]] = defaultdict(list)
    for item in items:
        by_task[(item.sampling_method, item.task)].append(item)

    return {
        "by_task": {
            key: similarity_for_items(
                task_items,
                embeddings_by_id,
                chunk_size=chunk_size,
            )
            for key, task_items in sorted(by_task.items())
        },
    }


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sampling_method": summary.get("sampling_method"),
        "task": summary.get("task"),
        "environment": summary.get("environment"),
        "composite_tasks": ",".join(summary.get("composite_tasks", [])),
        "num_trajectories": summary.get("num_trajectories"),
        "num_steps": summary.get("num_steps"),
        "tool_call_distribution_json": compact_json(
            summary.get("tool_call_distribution", {})
        ),
    }

    message_stats = summary.get("conversation_messages", {})
    for field in ("total", "min", "max", "avg"):
        row[f"conversation_messages_{field}"] = message_stats.get(field)

    length_stats = summary.get("conversation_length", {})
    row["conversation_length_texts"] = length_stats.get("texts")
    for unit in ("words", "chars"):
        unit_stats = length_stats.get(unit, {})
        for field in ("total", "min", "max", "avg"):
            row[f"conversation_length_{unit}_{field}"] = unit_stats.get(field)

    similarity = summary.get("trajectory_embedding_similarity", {})
    for field in (
        "num_items",
        "num_pairs",
        "computed_pairs",
        "exact",
        "min",
        "max",
        "avg",
    ):
        row[f"trajectory_cosine_{field}"] = similarity.get(field)

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_max_trajectories(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stitched trajectory conversations by task."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root(),
        help=(
            "Sampling methods root. Defaults to data_k3 raw sampling methods if it "
            "exists, otherwise data/raw/sampling_methods."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/sampling_methods"),
        help="Directory for summary JSON and CSV outputs.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include trajectories whose validation.is_valid is false.",
    )
    parser.add_argument(
        "--max-trajectories-per-task",
        type=int,
        default=DEFAULT_MAX_TRAJECTORIES_PER_TASK,
        help=(
            "Maximum sorted trajectories to compare per sampling-method/task group. "
            "Default 30 gives 435 unique pairs for complete groups. Use 0 for all."
        ),
    )
    parser.add_argument(
        "--max-trajectories-per-environment",
        type=int,
        dest="max_trajectories_per_task",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=16,
        help=(
            "Parallel task-directory readers for trajectory JSON loading. Default 16 "
            "keeps the metadata pass fast on large sampling-method datasets."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("none", "vllm", "vllm-openai"),
        default="none",
        help=(
            "'vllm' uses vLLM's Python LLM.embed API. 'vllm-openai' calls a "
            "running vLLM OpenAI-compatible /v1/embeddings endpoint."
        ),
    )
    parser.add_argument("--embedding-model", default=QWEN3_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-instruction",
        default="",
        help=(
            "Optional Qwen-style instruction prefix for each embedded trajectory "
            "conversation. Leave empty to embed raw conversation text."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=(
            "Embedding texts per vLLM call. Default 256 is tuned for a large GPU; "
            "lower it if vLLM runs out of memory."
        ),
    )
    parser.add_argument(
        "--embedding-cache",
        choices=("auto", "refresh", "off"),
        default="auto",
        help=(
            "Cache trajectory embeddings in --output-dir. 'auto' reuses a matching "
            "cache, 'refresh' recomputes and overwrites it, and 'off' disables cache."
        ),
    )
    parser.add_argument(
        "--embedding-progress-interval",
        type=int,
        default=10,
        help="Log embedding progress every N batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000/v1",
        help="Base URL for --embedding-provider vllm-openai.",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="Optional API key for a vLLM server started with --api-key.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Optional dtype forwarded to vLLM offline loading, for example float16.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Optional tensor parallel size for vLLM offline loading.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model length for vLLM offline loading.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional GPU memory utilization for vLLM offline loading.",
    )
    parser.add_argument(
        "--similarity-chunk-size",
        type=int,
        default=512,
        help="Rows per exact cosine matrix multiply chunk.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    total_started_at = perf_counter()
    args = parse_args(argv)
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"No such input root: {args.input_root}")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be a positive integer")
    if args.similarity_chunk_size <= 0:
        raise ValueError("--similarity-chunk-size must be a positive integer")
    if args.load_workers <= 0:
        raise ValueError("--load-workers must be a positive integer")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_trajectories_per_task = parse_max_trajectories(args.max_trajectories_per_task)
    load_started_at = perf_counter()
    records, warnings = load_records(
        args.input_root,
        include_invalid=args.include_invalid,
        max_trajectories_per_task=max_trajectories_per_task,
        load_workers=args.load_workers,
    )
    if not records:
        raise RuntimeError(f"No trajectory records found under {args.input_root}")
    log_timing(
        f"Loaded {len(records)} trajectories in {perf_counter() - load_started_at:.1f}s"
    )

    similarity_maps: dict[str, Any] | None = None
    embedding_info: dict[str, Any] = {
        "provider": args.embedding_provider,
        "unit": "trajectory_conversation",
        "similarity_scope": "sampling_method_task",
        "max_trajectories_per_task": max_trajectories_per_task,
    }
    if args.embedding_provider != "none":
        items = build_trajectory_embedding_items(
            records,
            instruction=args.embedding_instruction,
        )
        texts = [item.text for item in items]
        cache_metadata = build_embedding_cache_metadata(
            items,
            provider=args.embedding_provider,
            model_name=args.embedding_model,
            instruction=args.embedding_instruction,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
        )
        embeddings: np.ndarray | None = None
        if args.embedding_cache == "auto":
            cache_started_at = perf_counter()
            embeddings = load_cached_embeddings(args.output_dir, cache_metadata)
            if embeddings is not None:
                log_timing(
                    "Loaded cached trajectory embeddings "
                    f"from {args.output_dir} in {perf_counter() - cache_started_at:.1f}s"
                )

        if embeddings is None:
            log_timing(
                f"Embedding {len(texts)} trajectory conversations "
                f"with {args.embedding_provider}:{args.embedding_model}"
            )
            embed_started_at = perf_counter()
            if args.embedding_provider == "vllm":
                embeddings = embed_with_vllm_offline(
                    texts,
                    model_name=args.embedding_model,
                    batch_size=args.embedding_batch_size,
                    dtype=args.dtype,
                    tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    progress_interval=args.embedding_progress_interval,
                )
            else:
                embeddings = embed_with_vllm_openai(
                    texts,
                    model_name=args.embedding_model,
                    batch_size=args.embedding_batch_size,
                    base_url=args.vllm_base_url,
                    api_key=args.vllm_api_key,
                    progress_interval=args.embedding_progress_interval,
                )
            log_timing(
                f"Computed embeddings in {perf_counter() - embed_started_at:.1f}s"
            )
            if args.embedding_cache != "off":
                cache_started_at = perf_counter()
                write_cached_embeddings(args.output_dir, cache_metadata, embeddings)
                log_timing(
                    "Wrote trajectory embedding cache "
                    f"in {perf_counter() - cache_started_at:.1f}s"
                )
        else:
            log_timing("Skipping embedding model run because cache matched")

        similarity_started_at = perf_counter()
        similarity_maps = compute_similarity_maps(
            items,
            embeddings,
            chunk_size=args.similarity_chunk_size,
        )
        log_timing(
            "Computed per-task cosine summaries "
            f"in {perf_counter() - similarity_started_at:.1f}s"
        )
        embedding_info.update(
            {
                "model": args.embedding_model,
                "num_text_items": len(items),
                "embedding_dimension": int(embeddings.shape[1])
                if embeddings.ndim == 2
                else None,
                "batch_size": args.embedding_batch_size,
                "cache": args.embedding_cache,
                "cache_metadata": cache_metadata,
            }
        )

    records_by_task: dict[tuple[str, str], list[TrajectoryRecord]] = defaultdict(list)
    records_by_task_all_methods: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    records_by_method: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        records_by_task[(record.sampling_method, record.task)].append(record)
        records_by_task_all_methods[record.task].append(record)
        records_by_method[record.sampling_method].append(record)

    task_summaries = [
        summarize_records(
            group_records,
            sampling_method=key[0],
            task=key[1],
            similarity=(similarity_maps or {}).get("by_task", {}).get(key),
        )
        for key, group_records in sorted(records_by_task.items())
    ]
    task_all_method_summaries = [
        summarize_records(
            group_records,
            sampling_method=None,
            task=task,
        )
        for task, group_records in sorted(records_by_task_all_methods.items())
    ]
    method_summaries = [
        summarize_records(
            group_records,
            sampling_method=method,
            task=None,
        )
        for method, group_records in sorted(records_by_method.items())
    ]
    dataset_summary = summarize_records(
        records,
        sampling_method=None,
        task=None,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "include_invalid": args.include_invalid,
        "load_workers": args.load_workers,
        "embedding": embedding_info,
        "warnings": warnings,
        "dataset": dataset_summary,
        "by_sampling_method": method_summaries,
        "by_task_all_methods": task_all_method_summaries,
        "by_task": task_summaries,
    }

    summary_path = args.output_dir / "sampling_method_analysis.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    task_rows = [flatten_summary(summary) for summary in task_summaries]
    task_all_method_rows = [
        flatten_summary(summary) for summary in task_all_method_summaries
    ]
    write_csv(args.output_dir / "task_metrics.csv", task_rows)
    write_csv(args.output_dir / "task_all_methods_metrics.csv", task_all_method_rows)
    write_csv(
        args.output_dir / "sampling_method_metrics.csv",
        [flatten_summary(summary) for summary in method_summaries],
    )
    write_csv(
        args.output_dir / "dataset_metrics.csv",
        [flatten_summary(dataset_summary)],
    )

    # Legacy aliases for older notebooks that used environment terminology.
    write_csv(args.output_dir / "environment_metrics.csv", task_rows)
    write_csv(
        args.output_dir / "environment_all_methods_metrics.csv",
        task_all_method_rows,
    )

    log_timing(f"Wrote {summary_path}")
    log_timing(
        f"Analyzed {len(records)} trajectories in {perf_counter() - total_started_at:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
