"""Dataset loading and multimodal collation for task-level VLM SFT."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional in lightweight test envs
    Image = None

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional in lightweight test envs

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class so example builders can be imported without torch."""

        pass


try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover - optional in lightweight test envs
    AutoProcessor = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional in lightweight test envs
    tqdm = None

from training.bc_task_vlm.prompting import (
    build_assistant_text_message,
    build_messages,
    build_system_message,
    build_user_message,
    build_user_prompt,
)
from training.bc_task_vlm.schema_utils import (
    build_single_step_response_schema,
    compact_json_dumps,
    validate_single_step_payload,
)
from training.bc_task_vlm.task_registry import AGENT_IDS, get_task_metadata
from training.bc_task_vlm.tool_calling import (
    build_assistant_tool_call_message,
    build_tool_schemas,
)

_TRAJECTORY_METADATA_FILENAMES = (
    "original_trajectory.json",
    "plan.json",
    "metadata.json",
)
SFT_FORMAT_PLAIN = "plain"
SFT_FORMAT_TOOL_CALL = "tool_call"
SUPPORTED_SFT_FORMATS = (SFT_FORMAT_PLAIN, SFT_FORMAT_TOOL_CALL)
_EXAMPLE_CACHE_FORMAT_VERSION = 1
_EXAMPLE_CACHE_DEPENDENCY_PATHS = (
    Path(__file__).resolve(),
    Path(__file__).with_name("prompting.py").resolve(),
    Path(__file__).with_name("schema_utils.py").resolve(),
    Path(__file__).with_name("tool_calling.py").resolve(),
    Path(__file__).with_name("task_registry.py").resolve(),
)
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _missing_dependency_error(module_name: str) -> ImportError:
    return ImportError(
        f"Missing optional dependency {module_name!r}. "
        f"Install training/bc_task_vlm/requirements.txt first."
    )


def _require_dependency(dependency: Any, module_name: str) -> Any:
    if dependency is None:
        raise _missing_dependency_error(module_name)
    return dependency


@dataclass(frozen=True)
class CentralizedExample:
    """One supervised next-step prediction example in centralized mode."""

    sample_id: str
    task_name: str
    composite_task: str
    trajectory_id: str
    step_index: int
    agent_id: str
    task_instruction: str
    observation_views: list[str]
    image_paths: list[str]
    history_steps: list[dict[str, Any]]
    allowed_tool_specs: dict[str, dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    response_schema: dict[str, Any]
    target_payload: dict[str, Any]
    target_tool_call: dict[str, Any]
    target_text: str
    messages: list[dict[str, Any]]

    def to_manifest_entry(self) -> dict[str, Any]:
        """Returns the compact sample metadata persisted with the run."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "num_images": len(self.image_paths),
            "num_supervised_actions": 1,
            "observation_views": list(self.observation_views),
        }

    def to_feature_dict(self) -> dict[str, Any]:
        """Builds the trainer-facing feature dict for one centralized example."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "composite_task": self.composite_task,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "observation_views": list(self.observation_views),
            "image_paths": list(self.image_paths),
            "allowed_tool_specs": self.allowed_tool_specs,
            "tool_schemas": self.tool_schemas,
            "response_schema": self.response_schema,
            "target_payload": self.target_payload,
            "target_tool_call": self.target_tool_call,
            "target_text": self.target_text,
            "messages": self.messages,
        }


@dataclass(frozen=True)
class DecentralizedExample:
    """One supervised per-agent trajectory conversation in decentralized mode."""

    sample_id: str
    task_name: str
    composite_task: str
    trajectory_id: str
    agent_id: str
    task_instruction: str
    target_step_indices: list[int]
    image_paths: list[str]
    allowed_tool_specs: dict[str, dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    messages: list[dict[str, Any]]

    @property
    def num_target_steps(self) -> int:
        return len(self.target_step_indices)

    def to_manifest_entry(self) -> dict[str, Any]:
        """Returns the compact sample metadata persisted with the run."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "trajectory_id": self.trajectory_id,
            "agent_id": self.agent_id,
            "num_images": len(self.image_paths),
            "num_supervised_actions": self.num_target_steps,
            "target_step_indices": list(self.target_step_indices),
        }

    def to_feature_dict(self) -> dict[str, Any]:
        """Builds the trainer-facing feature dict for one decentralized example."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "composite_task": self.composite_task,
            "trajectory_id": self.trajectory_id,
            "agent_id": self.agent_id,
            "target_step_indices": list(self.target_step_indices),
            "num_target_steps": self.num_target_steps,
            "image_paths": list(self.image_paths),
            "allowed_tool_specs": self.allowed_tool_specs,
            "tool_schemas": self.tool_schemas,
            "messages": self.messages,
        }


ManifestExample = CentralizedExample | DecentralizedExample


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_trajectory_dirs(
    task_root: Path,
    *,
    show_progress: bool,
    progress_description: str | None,
    include_trajectory_ids: set[str] | None = None,
):
    trajectory_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    if include_trajectory_ids is not None:
        trajectory_dirs = [
            path for path in trajectory_dirs if path.name in include_trajectory_ids
        ]
    if not show_progress or tqdm is None:
        return trajectory_dirs
    return tqdm(
        trajectory_dirs,
        desc=progress_description or task_root.name,
        dynamic_ncols=True,
        leave=False,
        unit="traj",
    )


def list_task_trajectory_ids(*, dataset_root: Path, task_name: str) -> list[str]:
    """Returns sorted trajectory directory names for one dataset task."""

    task_metadata = get_task_metadata(task_name)
    task_root = dataset_root / task_metadata.dataset_name
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_root}")
    return sorted(path.name for path in task_root.iterdir() if path.is_dir())


def _normalize_history_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step["step"],
        "agent": step["agent"],
        "tool": step["tool"],
        "args": dict(step["args"]),
        "reasoning": step["reasoning"],
    }


def _find_latest_same_agent_observation(
    plan_steps: list[dict[str, Any]],
    *,
    before_index: int,
    agent_id: str,
) -> tuple[list[str], list[str]]:
    for step in reversed(plan_steps[:before_index]):
        if step.get("tool") != "get_image":
            continue
        metadata = step.get("metadata", {})
        if metadata.get("source_agent") != agent_id:
            continue
        args = step.get("args", {})
        image_paths = list(args.get("image_paths", ()))
        views = list(args.get("views", ()))
        if not image_paths:
            continue
        return image_paths, views
    raise ValueError(
        f"Missing preceding observation image for agent {agent_id!r} before step {before_index}."
    )


def _validate_alignment(
    *,
    task_name: str,
    trajectory_id: str,
    raw_steps: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
    executed_steps: list[dict[str, Any]],
) -> None:
    if not (len(raw_steps) == len(plan_steps) == len(executed_steps)):
        raise ValueError(
            f"Step count mismatch for {task_name}/{trajectory_id}: "
            f"raw={len(raw_steps)} plan={len(plan_steps)} executed={len(executed_steps)}."
        )
    for index, (raw_step, plan_step, executed_step) in enumerate(
        zip(raw_steps, plan_steps, executed_steps, strict=True)
    ):
        if raw_step.get("step") != index:
            raise ValueError(
                f"Unexpected raw step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {raw_step.get('step')}."
            )
        plan_index = plan_step.get("metadata", {}).get("step_index")
        if plan_index != index:
            raise ValueError(
                f"Unexpected plan step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {plan_index}."
            )
        executed_index = executed_step.get("step_index")
        if executed_index != index:
            raise ValueError(
                f"Unexpected executed step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {executed_index}."
            )
        if raw_step.get("tool") != plan_step.get("tool"):
            raise ValueError(
                f"Tool mismatch at {task_name}/{trajectory_id} step {index}: "
                f"raw={raw_step.get('tool')} plan={plan_step.get('tool')}."
            )


def _ensure_image_paths_exist(
    *,
    task_name: str,
    trajectory_id: str,
    image_paths: list[str],
) -> None:
    validate_paths = os.environ.get("ROBOCASA_VALIDATE_IMAGE_PATHS", "true")
    if validate_paths.strip().lower() in _FALSE_ENV_VALUES:
        return

    missing_images = [
        image_path for image_path in image_paths if not Path(image_path).exists()
    ]
    if not missing_images:
        return
    missing = ", ".join(missing_images)
    raise FileNotFoundError(
        f"Missing rendered images for {task_name}/{trajectory_id}: {missing}"
    )


def _record_latest_observation(
    *,
    latest_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    plan_step: dict[str, Any],
    task_name: str,
    trajectory_id: str,
) -> None:
    if plan_step.get("tool") != "get_image":
        return

    metadata = plan_step.get("metadata", {})
    source_agent = metadata.get("source_agent")
    if source_agent is None:
        return

    args = plan_step.get("args", {})
    image_paths = list(args.get("image_paths", ()))
    views = list(args.get("views", ()))
    if not image_paths:
        return

    _ensure_image_paths_exist(
        task_name=task_name,
        trajectory_id=trajectory_id,
        image_paths=image_paths,
    )
    latest_observations_by_agent[source_agent] = (image_paths, views)


def _require_latest_observation(
    *,
    latest_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    agent_id: str,
    before_index: int,
) -> tuple[list[str], list[str]]:
    try:
        return latest_observations_by_agent[agent_id]
    except KeyError as exc:
        raise ValueError(
            f"Missing preceding observation image for agent {agent_id!r} "
            f"before step {before_index}."
        ) from exc


def _validate_sft_format(sft_format: str) -> str:
    if sft_format not in SUPPORTED_SFT_FORMATS:
        raise ValueError(f"Unsupported SFT format: {sft_format!r}")
    return sft_format


def _raw_step_payload(
    raw_step: dict[str, Any],
    *,
    include_reasoning: bool = True,
) -> dict[str, Any]:
    step = {
        "step": raw_step["step"],
        "agent": raw_step["agent"],
        "tool": raw_step["tool"],
        "args": dict(raw_step["args"]),
    }
    if include_reasoning:
        step["reasoning"] = raw_step["reasoning"]
    return {"steps": [step]}


def _plain_target_text(raw_step: dict[str, Any]) -> str:
    return compact_json_dumps(
        {
            "tool": raw_step["tool"],
            "args": dict(raw_step["args"]),
        }
    )


def _build_target_metadata(
    *,
    raw_step: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
    sft_format: str = SFT_FORMAT_TOOL_CALL,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    sft_format = _validate_sft_format(sft_format)
    if sft_format == SFT_FORMAT_PLAIN:
        target_payload = _raw_step_payload(raw_step, include_reasoning=False)
        target_tool_call = {
            "name": raw_step["tool"],
            "arguments": dict(raw_step["args"]),
        }
        return target_payload, target_tool_call, _plain_target_text(raw_step)

    target_payload = validate_single_step_payload(
        _raw_step_payload(raw_step),
        agent_ids=AGENT_IDS,
        allowed_tool_specs=allowed_tool_specs,
    )
    target_tool_call = {
        "name": target_payload["steps"][0]["tool"],
        "arguments": dict(target_payload["steps"][0]["args"]),
    }
    target_text = compact_json_dumps(target_tool_call)
    return target_payload, target_tool_call, target_text


def build_centralized_examples(
    *,
    dataset_root: Path,
    task_names: list[str],
    show_progress: bool = False,
    progress_description: str | None = None,
    sft_format: str = SFT_FORMAT_TOOL_CALL,
    trajectory_ids_by_task: dict[str, set[str]] | None = None,
) -> list[CentralizedExample]:
    """Builds one SFT example per successful non-image action step."""

    sft_format = _validate_sft_format(sft_format)
    examples: list[CentralizedExample] = []

    for task_name in task_names:
        task_metadata = get_task_metadata(task_name)
        task_root = dataset_root / task_metadata.dataset_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_root}")

        if sft_format == SFT_FORMAT_TOOL_CALL:
            response_schema = build_single_step_response_schema(
                agent_ids=AGENT_IDS,
                allowed_tool_specs=task_metadata.allowed_tool_specs,
            )
        else:
            response_schema = {}
        tool_schemas = build_tool_schemas(
            agent_ids=AGENT_IDS,
            allowed_tool_specs=task_metadata.allowed_tool_specs,
        )
        include_trajectory_ids = None
        if trajectory_ids_by_task is not None:
            include_trajectory_ids = trajectory_ids_by_task.get(
                task_metadata.dataset_name, set()
            )

        for trajectory_dir in _iter_trajectory_dirs(
            task_root,
            show_progress=show_progress,
            progress_description=progress_description or task_metadata.dataset_name,
            include_trajectory_ids=include_trajectory_ids,
        ):
            original_trajectory = _load_json(
                trajectory_dir / "original_trajectory.json"
            )
            plan_steps = _load_json(trajectory_dir / "plan.json")
            metadata = _load_json(trajectory_dir / "metadata.json")

            raw_steps = list(original_trajectory["steps"])
            executed_steps = list(metadata["steps"])
            trajectory_id = str(original_trajectory["trajectory_id"])

            _validate_alignment(
                task_name=task_name,
                trajectory_id=trajectory_id,
                raw_steps=raw_steps,
                plan_steps=plan_steps,
                executed_steps=executed_steps,
            )

            history_steps: list[dict[str, Any]] = []
            latest_observations_by_agent: dict[str, tuple[list[str], list[str]]] = {}
            for raw_step, plan_step, executed_step in zip(
                raw_steps,
                plan_steps,
                executed_steps,
                strict=True,
            ):
                _record_latest_observation(
                    latest_observations_by_agent=latest_observations_by_agent,
                    plan_step=plan_step,
                    task_name=task_name,
                    trajectory_id=trajectory_id,
                )
                if raw_step["tool"] == "get_image":
                    continue

                if not executed_step.get("success", False):
                    continue

                image_paths, observation_views = _require_latest_observation(
                    latest_observations_by_agent=latest_observations_by_agent,
                    before_index=raw_step["step"],
                    agent_id=raw_step["agent"],
                )

                target_payload, target_tool_call, target_text = _build_target_metadata(
                    raw_step=raw_step,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                    sft_format=sft_format,
                )
                user_prompt = build_user_prompt(
                    composite_task=task_metadata.composite_task,
                    task_instruction=metadata["task"],
                    agent_id=raw_step["agent"],
                    next_step_index=raw_step["step"],
                    observation_views=observation_views,
                    history_steps=history_steps,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                    sft_format=sft_format,
                )
                message_kwargs: dict[str, Any]
                if sft_format == SFT_FORMAT_PLAIN:
                    message_kwargs = {"target_text": target_text}
                else:
                    message_kwargs = {"target_tool_call": target_tool_call}
                sample_id = f"{task_metadata.dataset_name}/{trajectory_id}/step_{raw_step['step']:06d}"
                examples.append(
                    CentralizedExample(
                        sample_id=sample_id,
                        task_name=task_metadata.dataset_name,
                        composite_task=task_metadata.composite_task,
                        trajectory_id=trajectory_id,
                        step_index=raw_step["step"],
                        agent_id=raw_step["agent"],
                        task_instruction=metadata["task"],
                        observation_views=observation_views,
                        image_paths=list(image_paths),
                        history_steps=list(history_steps),
                        allowed_tool_specs=task_metadata.allowed_tool_specs,
                        tool_schemas=tool_schemas,
                        response_schema=response_schema,
                        target_payload=target_payload,
                        target_tool_call=target_tool_call,
                        target_text=target_text,
                        messages=build_messages(
                            user_prompt=user_prompt,
                            num_images=len(image_paths),
                            **message_kwargs,
                        ),
                    )
                )
                history_steps.append(_normalize_history_step(raw_step))

    return examples


def build_decentralized_examples(
    *,
    dataset_root: Path,
    task_names: list[str],
    show_progress: bool = False,
    progress_description: str | None = None,
    sft_format: str = SFT_FORMAT_TOOL_CALL,
    trajectory_ids_by_task: dict[str, set[str]] | None = None,
) -> list[DecentralizedExample]:
    """Builds one conversation per `(trajectory, agent)` training example."""

    sft_format = _validate_sft_format(sft_format)
    examples: list[DecentralizedExample] = []

    for task_name in task_names:
        task_metadata = get_task_metadata(task_name)
        task_root = dataset_root / task_metadata.dataset_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_root}")

        tool_schemas = build_tool_schemas(
            agent_ids=AGENT_IDS,
            allowed_tool_specs=task_metadata.allowed_tool_specs,
        )
        include_trajectory_ids = None
        if trajectory_ids_by_task is not None:
            include_trajectory_ids = trajectory_ids_by_task.get(
                task_metadata.dataset_name, set()
            )

        for trajectory_dir in _iter_trajectory_dirs(
            task_root,
            show_progress=show_progress,
            progress_description=progress_description or task_metadata.dataset_name,
            include_trajectory_ids=include_trajectory_ids,
        ):
            original_trajectory = _load_json(
                trajectory_dir / "original_trajectory.json"
            )
            plan_steps = _load_json(trajectory_dir / "plan.json")
            metadata = _load_json(trajectory_dir / "metadata.json")

            raw_steps = list(original_trajectory["steps"])
            executed_steps = list(metadata["steps"])
            trajectory_id = str(original_trajectory["trajectory_id"])

            _validate_alignment(
                task_name=task_name,
                trajectory_id=trajectory_id,
                raw_steps=raw_steps,
                plan_steps=plan_steps,
                executed_steps=executed_steps,
            )

            history_steps: list[dict[str, Any]] = []
            messages_by_agent: dict[str, list[dict[str, Any]]] = {
                agent_id: [build_system_message()] for agent_id in AGENT_IDS
            }
            image_paths_by_agent: dict[str, list[str]] = {
                agent_id: [] for agent_id in AGENT_IDS
            }
            target_step_indices_by_agent: dict[str, list[int]] = {
                agent_id: [] for agent_id in AGENT_IDS
            }
            latest_observations_by_agent: dict[str, tuple[list[str], list[str]]] = {}

            for raw_step, plan_step, executed_step in zip(
                raw_steps,
                plan_steps,
                executed_steps,
                strict=True,
            ):
                _record_latest_observation(
                    latest_observations_by_agent=latest_observations_by_agent,
                    plan_step=plan_step,
                    task_name=task_name,
                    trajectory_id=trajectory_id,
                )
                if raw_step["tool"] == "get_image":
                    continue

                if not executed_step.get("success", False):
                    continue

                image_paths, observation_views = _require_latest_observation(
                    latest_observations_by_agent=latest_observations_by_agent,
                    before_index=raw_step["step"],
                    agent_id=raw_step["agent"],
                )

                _, target_tool_call, target_text = _build_target_metadata(
                    raw_step=raw_step,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                    sft_format=sft_format,
                )
                user_prompt = build_user_prompt(
                    composite_task=task_metadata.composite_task,
                    task_instruction=metadata["task"],
                    agent_id=raw_step["agent"],
                    next_step_index=raw_step["step"],
                    observation_views=observation_views,
                    history_steps=history_steps,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                    sft_format=sft_format,
                )
                acting_agent = raw_step["agent"]
                messages_by_agent[acting_agent].append(
                    build_user_message(
                        user_prompt=user_prompt,
                        num_images=len(image_paths),
                    )
                )
                if sft_format == SFT_FORMAT_PLAIN:
                    messages_by_agent[acting_agent].append(
                        build_assistant_text_message(target_text=target_text)
                    )
                else:
                    messages_by_agent[acting_agent].append(
                        build_assistant_tool_call_message(
                            tool_name=target_tool_call["name"],
                            arguments=target_tool_call["arguments"],
                        )
                    )
                image_paths_by_agent[acting_agent].extend(image_paths)
                target_step_indices_by_agent[acting_agent].append(raw_step["step"])
                history_steps.append(_normalize_history_step(raw_step))

            for agent_id in AGENT_IDS:
                target_step_indices = target_step_indices_by_agent[agent_id]
                if not target_step_indices:
                    continue

                sample_id = f"{task_metadata.dataset_name}/{trajectory_id}/{agent_id}"
                examples.append(
                    DecentralizedExample(
                        sample_id=sample_id,
                        task_name=task_metadata.dataset_name,
                        composite_task=task_metadata.composite_task,
                        trajectory_id=trajectory_id,
                        agent_id=agent_id,
                        task_instruction=metadata["task"],
                        target_step_indices=list(target_step_indices),
                        image_paths=list(image_paths_by_agent[agent_id]),
                        allowed_tool_specs=task_metadata.allowed_tool_specs,
                        tool_schemas=tool_schemas,
                        messages=list(messages_by_agent[agent_id]),
                    )
                )

    return examples


def _file_signature(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {
        "mtime_ns": stat_result.st_mtime_ns,
        "size": stat_result.st_size,
    }


def build_example_cache_fingerprint(
    *,
    dataset_root: Path,
    task_name: str,
    granularity: str,
    sft_format: str = SFT_FORMAT_TOOL_CALL,
) -> dict[str, Any]:
    """Builds a fingerprint that invalidates cached task examples when inputs change."""

    sft_format = _validate_sft_format(sft_format)
    task_metadata = get_task_metadata(task_name)
    task_root = dataset_root / task_metadata.dataset_name
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_root}")

    trajectory_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    return {
        "cache_format_version": _EXAMPLE_CACHE_FORMAT_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "task_name": task_metadata.dataset_name,
        "granularity": granularity,
        "sft_format": sft_format,
        "agent_ids": list(AGENT_IDS),
        "task_metadata": {
            "dataset_name": task_metadata.dataset_name,
            "composite_task": task_metadata.composite_task,
            "allowed_tool_specs": task_metadata.allowed_tool_specs,
        },
        "builder_dependencies": {
            dependency_path.name: _file_signature(dependency_path)
            for dependency_path in _EXAMPLE_CACHE_DEPENDENCY_PATHS
        },
        "trajectories": [
            {
                "trajectory_id": trajectory_dir.name,
                "files": {
                    file_name: _file_signature(trajectory_dir / file_name)
                    for file_name in _TRAJECTORY_METADATA_FILENAMES
                },
            }
            for trajectory_dir in trajectory_dirs
        ],
    }


def build_example_cache_path(
    *,
    cache_dir: Path,
    dataset_root: Path,
    task_name: str,
    granularity: str,
) -> Path:
    """Returns the on-disk cache path for one `(dataset, task, granularity)` tuple."""

    dataset_key = sha256(str(dataset_root.resolve()).encode("utf-8")).hexdigest()[:16]
    normalized_task_name = get_task_metadata(task_name).dataset_name
    safe_task_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in normalized_task_name
    )
    return cache_dir / dataset_key / granularity / f"{safe_task_name}.json"


def _example_type_for_granularity(
    granularity: str,
) -> type[CentralizedExample] | type[DecentralizedExample]:
    if granularity == "centralized":
        return CentralizedExample
    if granularity == "decentralized":
        return DecentralizedExample
    raise ValueError(f"Unsupported granularity: {granularity}")


def load_examples_from_cache(
    *,
    cache_path: Path,
    expected_fingerprint: dict[str, Any] | None,
    granularity: str,
) -> list[ManifestExample] | None:
    """Loads cached task examples, optionally requiring a fingerprint match."""

    try:
        payload = _load_json(cache_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    if payload.get("cache_format_version") != _EXAMPLE_CACHE_FORMAT_VERSION:
        return None
    if (
        expected_fingerprint is not None
        and payload.get("fingerprint") != expected_fingerprint
    ):
        return None

    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        return None

    example_type = _example_type_for_granularity(granularity)
    try:
        return [example_type(**raw_example) for raw_example in raw_examples]
    except TypeError:
        return None


def save_examples_to_cache(
    *,
    cache_path: Path,
    fingerprint: dict[str, Any],
    examples: list[ManifestExample],
) -> None:
    """Persists task examples to a deterministic JSON cache file."""

    payload = {
        "cache_format_version": _EXAMPLE_CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "examples": [asdict(example) for example in examples],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(cache_path)


class CentralizedDataset(Dataset):
    """Thin PyTorch dataset wrapper around centralized training examples."""

    def __init__(self, examples: list[CentralizedExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index].to_feature_dict()


class DecentralizedDataset(Dataset):
    """Thin PyTorch dataset wrapper around decentralized training examples."""

    def __init__(self, examples: list[DecentralizedExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index].to_feature_dict()


def _num_supervised_actions(example: ManifestExample) -> int:
    if isinstance(example, DecentralizedExample):
        return example.num_target_steps
    return 1


def build_split_manifest(
    *,
    dataset_root: Path,
    train_examples: list[ManifestExample],
    val_examples: list[ManifestExample],
) -> dict[str, Any]:
    """Builds a compact manifest describing the train/validation split."""

    def summarize_split(examples: list[ManifestExample]) -> dict[str, Any]:
        counts_by_task: dict[str, dict[str, Any]] = {}
        for example in examples:
            entry = counts_by_task.setdefault(
                example.task_name,
                {
                    "num_samples": 0,
                    "num_supervised_actions": 0,
                    "trajectory_ids": set(),
                },
            )
            entry["num_samples"] += 1
            entry["num_supervised_actions"] += _num_supervised_actions(example)
            entry["trajectory_ids"].add(example.trajectory_id)

        return {
            "num_samples": len(examples),
            "num_supervised_actions": sum(
                _num_supervised_actions(example) for example in examples
            ),
            "tasks": {
                task_name: {
                    "num_samples": task_summary["num_samples"],
                    "num_supervised_actions": task_summary["num_supervised_actions"],
                    "num_trajectories": len(task_summary["trajectory_ids"]),
                }
                for task_name, task_summary in sorted(counts_by_task.items())
            },
            "samples": [example.to_manifest_entry() for example in examples],
        }

    return {
        "dataset_root": str(dataset_root.resolve()),
        "train": summarize_split(train_examples),
        "validation": summarize_split(val_examples),
    }


class LazyVisionSFTCollator:
    """Collates multimodal SFT batches and masks loss to assistant tokens only."""

    def __init__(
        self,
        *,
        processor_name_or_path: str,
        max_length: int | None,
        max_images_per_sample: int | None = None,
        max_history_steps_per_prompt: int | None = None,
        supervise_last_assistant_turn_only: bool = False,
        trust_remote_code: bool,
        sft_format: str = SFT_FORMAT_TOOL_CALL,
    ) -> None:
        self.processor_name_or_path = processor_name_or_path
        self.max_length = max_length
        self.max_images_per_sample = (
            max_images_per_sample
            if max_images_per_sample and max_images_per_sample > 0
            else None
        )
        self.max_history_steps_per_prompt = (
            max_history_steps_per_prompt
            if max_history_steps_per_prompt and max_history_steps_per_prompt > 0
            else None
        )
        self.supervise_last_assistant_turn_only = supervise_last_assistant_turn_only
        self.trust_remote_code = trust_remote_code
        self.sft_format = _validate_sft_format(sft_format)
        self._processor = None

    def _get_processor(self):
        processor_cls = _require_dependency(AutoProcessor, "transformers")
        if self._processor is None:
            processor = processor_cls.from_pretrained(
                self.processor_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "right"
            self._processor = processor
        return self._processor

    @staticmethod
    def _load_images(image_paths: list[str]) -> list[Any]:
        image_module = _require_dependency(Image, "Pillow")
        images: list[Any] = []
        for image_path in image_paths:
            with image_module.open(image_path) as image:
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
        audio_token = (
            getattr(template_owner, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
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
            elif item_type == "audio":
                chunks.append(audio_token)
        return "".join(chunks).strip()

    def _apply_plain_fallback_template(
        self,
        template_owner,
        *,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
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
        template_owner,
        *,
        feature: dict[str, Any],
        messages: list[dict[str, Any]],
        **kwargs,
    ):
        template_kwargs = dict(kwargs)
        if self.sft_format == SFT_FORMAT_TOOL_CALL:
            template_kwargs["tools"] = feature["tool_schemas"]

        if self.sft_format == SFT_FORMAT_PLAIN and not getattr(
            template_owner, "chat_template", None
        ):
            if template_kwargs.get("tokenize"):
                raise TypeError("Plain fallback chat template only renders text.")
            return self._apply_plain_fallback_template(
                template_owner,
                messages=messages,
                add_generation_prompt=bool(
                    template_kwargs.get("add_generation_prompt", False)
                ),
            )

        return template_owner.apply_chat_template(messages, **template_kwargs)

    def _tokenize_texts(
        self,
        processor,
        *,
        texts: list[str],
        images_by_sample: list[list[Any]],
    ):
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if any(images_by_sample):
            processor_kwargs["images"] = images_by_sample
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length
            processor_kwargs["truncation"] = True

        batch = processor(**processor_kwargs)
        batch.pop("token_type_ids", None)
        return batch

    @staticmethod
    def _count_images_in_message(message: dict[str, Any]) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0
        return sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        )

    def _trim_feature_to_image_budget(self, feature: dict[str, Any]) -> dict[str, Any]:
        if self.max_images_per_sample is None:
            return feature

        image_paths = list(feature["image_paths"])
        if len(image_paths) <= self.max_images_per_sample:
            return feature

        messages = list(feature["messages"])
        leading_message_count = 0
        while (
            leading_message_count < len(messages)
            and messages[leading_message_count].get("role") in {"system", "developer"}
            and self._count_images_in_message(messages[leading_message_count]) == 0
        ):
            leading_message_count += 1

        selected_start = len(messages)
        selected_image_count = 0
        for message_index in range(len(messages) - 1, leading_message_count - 1, -1):
            message_image_count = self._count_images_in_message(messages[message_index])
            if (
                message_image_count > 0
                and selected_image_count + message_image_count
                > self.max_images_per_sample
            ):
                break
            selected_image_count += message_image_count
            selected_start = message_index

        while (
            selected_start < len(messages)
            and messages[selected_start].get("role") == "assistant"
        ):
            selected_start += 1

        selected_image_count = sum(
            self._count_images_in_message(message)
            for message in messages[selected_start:]
        )
        if selected_start >= len(messages) or selected_image_count == 0:
            return feature

        dropped_image_count = sum(
            self._count_images_in_message(message)
            for message in messages[:selected_start]
        )
        selected_image_paths = image_paths[
            dropped_image_count : dropped_image_count + selected_image_count
        ]
        if len(selected_image_paths) != selected_image_count:
            return feature

        trimmed_feature = dict(feature)
        trimmed_feature["messages"] = (
            messages[:leading_message_count] + messages[selected_start:]
        )
        trimmed_feature["image_paths"] = selected_image_paths
        return trimmed_feature

    def _trim_prompt_history_text(self, text: str) -> str:
        if self.max_history_steps_per_prompt is None:
            return text

        history_header = "Previous executed symbolic action history:\n"
        tools_header = "\n\nAvailable tools for this task:\n"
        history_start = text.find(history_header)
        if history_start < 0:
            return text
        history_body_start = history_start + len(history_header)
        history_end = text.find(tools_header, history_body_start)
        if history_end < 0:
            return text

        history_lines = [
            line
            for line in text[history_body_start:history_end].splitlines()
            if line.strip()
        ]
        if len(history_lines) <= self.max_history_steps_per_prompt:
            return text

        kept_history_lines = history_lines[-self.max_history_steps_per_prompt :]
        omitted_count = len(history_lines) - len(kept_history_lines)
        trimmed_history = "\n".join(
            [f"- {omitted_count} earlier steps omitted"] + kept_history_lines
        )
        return text[:history_body_start] + trimmed_history + text[history_end:]

    def _trim_feature_prompt_history(
        self,
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        if self.max_history_steps_per_prompt is None:
            return feature

        messages: list[dict[str, Any]] = []
        changed = False
        for message in feature["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                messages.append(message)
                continue

            trimmed_content: list[Any] = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    trimmed_text = self._trim_prompt_history_text(item["text"])
                    if trimmed_text != item["text"]:
                        changed = True
                        trimmed_item = dict(item)
                        trimmed_item["text"] = trimmed_text
                        trimmed_content.append(trimmed_item)
                    else:
                        trimmed_content.append(item)
                else:
                    trimmed_content.append(item)

            if trimmed_content != content:
                trimmed_message = dict(message)
                trimmed_message["content"] = trimmed_content
                messages.append(trimmed_message)
            else:
                messages.append(message)

        if not changed:
            return feature
        trimmed_feature = dict(feature)
        trimmed_feature["messages"] = messages
        return trimmed_feature

    def _trim_feature_to_last_assistant_turn(
        self,
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.supervise_last_assistant_turn_only:
            return feature

        messages = list(feature["messages"])
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        if len(assistant_indices) <= 1:
            return feature

        final_assistant_index = assistant_indices[-1]
        user_index = None
        for index in range(final_assistant_index - 1, -1, -1):
            if messages[index].get("role") == "user":
                user_index = index
                break
        if user_index is None:
            return feature

        leading_messages = [
            message
            for message in messages[:user_index]
            if message.get("role") in {"system", "developer"}
            and self._count_images_in_message(message) == 0
        ]
        kept_messages = (
            leading_messages + messages[user_index : final_assistant_index + 1]
        )

        image_paths = list(feature["image_paths"])
        dropped_image_count = sum(
            self._count_images_in_message(message) for message in messages[:user_index]
        )
        kept_image_count = sum(
            self._count_images_in_message(message) for message in kept_messages
        )
        kept_image_paths = image_paths[
            dropped_image_count : dropped_image_count + kept_image_count
        ]
        if len(kept_image_paths) != kept_image_count:
            return feature

        trimmed_feature = dict(feature)
        trimmed_feature["messages"] = kept_messages
        trimmed_feature["image_paths"] = kept_image_paths
        return trimmed_feature

    @staticmethod
    def _has_single_final_assistant_message(messages: list[dict[str, Any]]) -> bool:
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        return len(assistant_indices) == 1 and assistant_indices[0] == len(messages) - 1

    @staticmethod
    def _normalize_template_mask(mask: Any) -> list[bool] | None:
        if mask is None:
            return None
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if isinstance(mask, list) and mask and isinstance(mask[0], list):
            mask = mask[0]
        if not isinstance(mask, list):
            return None
        return [bool(value) for value in mask]

    def _build_assistant_mask_from_template(
        self,
        *,
        processor,
        feature: dict[str, Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        torch_module = _require_dependency(torch, "torch")
        template_owners = (processor, getattr(processor, "tokenizer", None))
        for template_owner in template_owners:
            if template_owner is None or not hasattr(
                template_owner, "apply_chat_template"
            ):
                continue

            try:
                template_output = self._apply_chat_template(
                    template_owner,
                    feature=feature,
                    messages=feature["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
            except TypeError:
                continue
            except Exception:
                continue

            candidate_mask = None
            for key in (
                "assistant_masks",
                "assistant_mask",
                "assistant_tokens_mask",
                "assistant_token_mask",
            ):
                try:
                    candidate_mask = template_output[key]
                except Exception:
                    continue
                if candidate_mask is not None:
                    break

            normalized_mask = self._normalize_template_mask(candidate_mask)
            if normalized_mask is None or len(normalized_mask) != full_sequence_length:
                continue

            assistant_mask = torch_module.zeros(sequence_width, dtype=torch_module.bool)
            assistant_mask[:full_sequence_length] = torch_module.tensor(
                normalized_mask,
                dtype=torch_module.bool,
            )
            return assistant_mask
        return None

    def _build_assistant_mask_from_prefixes(
        self,
        *,
        processor,
        feature: dict[str, Any],
        images: list[Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        torch_module = _require_dependency(torch, "torch")
        assistant_mask = torch_module.zeros(sequence_width, dtype=torch_module.bool)

        prefix_messages: list[dict[str, Any]] = []
        prefix_image_count = 0
        previous_length = 0
        for message in feature["messages"]:
            prefix_messages.append(message)
            prefix_image_count += self._count_images_in_message(message)
            prefix_text = self._apply_chat_template(
                processor,
                feature=feature,
                messages=prefix_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_batch = self._tokenize_texts(
                processor,
                texts=[prefix_text],
                images_by_sample=[images[:prefix_image_count]],
            )
            current_length = min(
                full_sequence_length,
                int(prefix_batch["attention_mask"][0].sum().item()),
            )
            if message.get("role") == "assistant" and current_length > previous_length:
                assistant_mask[previous_length:current_length] = True
            previous_length = current_length
            if previous_length >= full_sequence_length:
                break

        return assistant_mask

    def _build_assistant_mask(
        self,
        *,
        processor,
        feature: dict[str, Any],
        images: list[Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        assistant_mask = self._build_assistant_mask_from_template(
            processor=processor,
            feature=feature,
            full_sequence_length=full_sequence_length,
            sequence_width=sequence_width,
        )
        if assistant_mask is not None:
            return assistant_mask

        return self._build_assistant_mask_from_prefixes(
            processor=processor,
            feature=feature,
            images=images,
            full_sequence_length=full_sequence_length,
            sequence_width=sequence_width,
        )

    @staticmethod
    def _raise_if_any_empty_labels(labels, features: list[dict[str, Any]]) -> None:
        supervised_counts = (labels != -100).sum(dim=1).tolist()
        empty_sample_ids = [
            str(feature.get("sample_id", row_index))
            for row_index, count in enumerate(supervised_counts)
            if int(count) == 0
        ]
        if empty_sample_ids:
            sample_text = ", ".join(empty_sample_ids[:8])
            raise ValueError(
                "Collator produced no supervised assistant tokens after "
                f"truncation for samples: {sample_text}. Reduce "
                "--max-history-steps-per-prompt, reduce --max-images-per-sample, "
                "or increase --max-length."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        torch_module = _require_dependency(torch, "torch")
        processor = self._get_processor()
        features = [self._trim_feature_to_image_budget(feature) for feature in features]
        features = [
            self._trim_feature_to_last_assistant_turn(feature) for feature in features
        ]
        features = [self._trim_feature_prompt_history(feature) for feature in features]
        messages = [feature["messages"] for feature in features]
        images = [self._load_images(feature["image_paths"]) for feature in features]

        full_texts = [
            self._apply_chat_template(
                processor,
                feature=feature,
                messages=message,
                tokenize=False,
                add_generation_prompt=False,
            )
            for message, feature in zip(messages, features, strict=True)
        ]
        batch = self._tokenize_texts(
            processor,
            texts=full_texts,
            images_by_sample=images,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        if all(
            self._has_single_final_assistant_message(feature["messages"])
            for feature in features
        ):
            prompt_messages = [message[:-1] for message in messages]
            prompt_texts = [
                self._apply_chat_template(
                    processor,
                    feature=feature,
                    messages=message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for message, feature in zip(prompt_messages, features, strict=True)
            ]
            prompt_batch = self._tokenize_texts(
                processor,
                texts=prompt_texts,
                images_by_sample=images,
            )
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                labels[row_index, :prompt_length] = -100
            self._raise_if_any_empty_labels(labels, features)
            batch["labels"] = labels
            return batch

        sequence_width = batch["input_ids"].shape[1]
        for row_index, feature in enumerate(features):
            full_sequence_length = int(batch["attention_mask"][row_index].sum().item())
            assistant_mask = self._build_assistant_mask(
                processor=processor,
                feature=feature,
                images=images[row_index],
                full_sequence_length=full_sequence_length,
                sequence_width=sequence_width,
            )
            valid_positions = assistant_mask & batch["attention_mask"][row_index].bool()
            labels[row_index, ~valid_positions] = -100

        self._raise_if_any_empty_labels(labels, features)
        batch["labels"] = labels
        return batch
