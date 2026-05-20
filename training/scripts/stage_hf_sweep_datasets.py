#!/usr/bin/env python3
"""Stage Hugging Face RoboCasa sweep datasets for task-VLM SFT.

The SFT trainer consumes the local rendered dataset layout:

  <root>/<task>/traj_XXXXXX/{original_trajectory.json,plan.json,metadata.json}

This helper converts one or more Hugging Face sweep datasets produced by
scripts/sweep_trajectories.py back into that layout, including local image files.
It supports both trajectory-row datasets and step-row datasets with uploaded JSON
sidecars.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.bc_task_vlm.task_registry import get_task_metadata, resolve_task_name

IMAGE_COLUMNS = (
    "room_view",
    "top_view",
    "map",
    "agentview_center",
    "agentview_left",
    "agentview_right",
    "wrist",
)

REQUIRED_STAGE_FILES = (
    "original_trajectory.json",
    "adapted_trajectory.json",
    "plan.json",
    "metadata.json",
    "trajectory_execution_metadata.json",
)


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    tool: str
    args: dict[str, Any]
    robot_idx: int
    success: bool
    images_by_view: dict[str, Any]


@dataclass(frozen=True)
class EpisodeStagingJob:
    repo_id: str
    output_root: Path
    task_dir: str
    trajectory_id: str
    source_label: str
    task_instruction: str
    records: list[StepRecord]
    original_trajectory: dict[str, Any]
    execution_metadata: dict[str, Any] | None


@dataclass(frozen=True)
class EpisodeStagingResult:
    repo_id: str
    task_dir: str


@dataclass(frozen=True)
class EpisodeSource:
    source_row: dict[str, Any]
    row_indices: tuple[int, ...]
    episode_index: int


@dataclass
class RepoStageState:
    repo_id: str
    split: str
    revision: str | None
    sidecar_root: Path | None
    dataset: Any
    row_granularity: str
    episodes: Iterator[EpisodeSource]
    submitted: int = 0
    staged: int = 0
    skipped_existing: int = 0
    exhausted: bool = False
    finished_logged: bool = False
    tasks: set[str] | None = None

    def __post_init__(self) -> None:
        if self.tasks is None:
            self.tasks = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        required=True,
        help="Hugging Face dataset repo id. Repeat once per source dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination rendered-dataset root for the SFT trainer.",
    )
    parser.add_argument("--split", default="train", help="HF dataset split to stage.")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional shared HF revision for all dataset repos.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward trust_remote_code to datasets.load_dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Episode staging workers. Values above 1 parallelize image and JSON "
            "writes across trajectories and repos."
        ),
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=1,
        help=(
            "Repository loading workers. Values above 1 parallelize "
            "datasets.load_dataset() downloads and Arrow cache generation across "
            "source repos."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help=(
            "Maximum submitted episode jobs waiting or running at once. Defaults "
            "to 2 * --workers to bound image memory."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress after this many completed episodes per repo.",
    )
    parser.add_argument(
        "--resume",
        "--resume-existing",
        dest="resume_existing",
        action="store_true",
        help=(
            "Resume an interrupted staging run by skipping complete existing "
            "trajectory directories in --output-root."
        ),
    )
    parser.add_argument(
        "--resume-validation",
        choices=("validated", "unchecked"),
        default="validated",
        help=(
            "When resuming, validate existing JSON files and staged image paths "
            "before skipping them, or only check that required files exist."
        ),
    )
    return parser.parse_args()


def _load_dataset(
    repo_id: str, *, split: str, revision: str | None, trust_remote_code: bool
):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Missing optional dependency 'datasets'. Install "
            "training/bc_task_vlm/requirements.txt before staging HF datasets."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    return load_dataset(repo_id, **kwargs)


def _snapshot_dataset_sidecars(repo_id: str, *, revision: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Missing optional dependency 'huggingface_hub'. Install "
            "training/bc_task_vlm/requirements.txt before staging HF datasets."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "allow_patterns": [
            "**/original_trajectory.json",
            "**/adapted_trajectory.json",
            "**/trajectory_execution_metadata.json",
            "sweep_summary.json",
        ],
    }
    if revision is not None:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def _parse_json_value(value: Any, *, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return value


def _repo_tail(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def _task_dir_from_repo(repo_id: str) -> str:
    tail = _repo_tail(repo_id)
    marker = "_full_run_"
    if marker in tail:
        return tail.split(marker, 1)[1]
    return tail


def _task_dir_for_row(row: dict[str, Any], repo_id: str) -> str:
    raw_task_dir = str(row.get("task_dir") or "").strip()
    if raw_task_dir:
        return resolve_task_name(raw_task_dir)
    return resolve_task_name(_task_dir_from_repo(repo_id))


def _episode_id_for_row(row: dict[str, Any], fallback_index: int) -> str:
    episode_id = str(row.get("episode_id") or "").strip()
    if episode_id:
        return episode_id
    return f"episode_{fallback_index:06d}"


def _row_images_by_view(
    row: dict[str, Any], *, sequence_index: int | None = None
) -> dict[str, Any]:
    images: dict[str, Any] = {}
    for view_name in IMAGE_COLUMNS:
        value = row.get(view_name)
        if sequence_index is not None and isinstance(value, (list, tuple)):
            value = value[sequence_index] if sequence_index < len(value) else None
        if value is not None:
            images[view_name] = value
    return images


def _rows_from_trajectory_row(row: dict[str, Any]) -> list[StepRecord]:
    step_indices = list(row["step_index"])
    tools = list(row["tool_name"])
    tool_args = list(row["tool_args"])
    robot_indices = list(row["robot_idx"])
    successes = list(row["success"])
    records: list[StepRecord] = []
    for index, step_index in enumerate(step_indices):
        records.append(
            StepRecord(
                step_index=int(step_index),
                tool=str(tools[index]),
                args=dict(_parse_json_value(tool_args[index], default={}) or {}),
                robot_idx=int(robot_indices[index]),
                success=bool(successes[index]),
                images_by_view=_row_images_by_view(row, sequence_index=index),
            )
        )
    return records


def _rows_from_step_rows(rows: Iterable[dict[str, Any]]) -> list[StepRecord]:
    records: list[StepRecord] = []
    for row in sorted(rows, key=lambda item: int(item.get("step_index", 0))):
        records.append(
            StepRecord(
                step_index=int(row.get("step_index", 0)),
                tool=str(row.get("tool_name", "")),
                args=dict(_parse_json_value(row.get("tool_args"), default={}) or {}),
                robot_idx=int(row.get("robot_idx", 0)),
                success=bool(row.get("success", False)),
                images_by_view=_row_images_by_view(row),
            )
        )
    return records


def _sidecar_json(
    *,
    row: dict[str, Any],
    field_name: str,
    sidecar_root: Path | None,
) -> Any | None:
    inline_value = _parse_json_value(row.get(field_name), default=None)
    if inline_value is not None:
        return inline_value
    path_field = f"{field_name}_path"
    relpath = str(row.get(path_field) or "").strip()
    if not relpath or sidecar_root is None:
        return None
    sidecar_path = sidecar_root / relpath
    if not sidecar_path.is_file():
        return None
    return _parse_json_value(sidecar_path.read_text(encoding="utf-8"), default=None)


def _agent_id(robot_idx: int) -> str:
    return f"agent_{robot_idx}"


def _image_extension(view_name: str) -> str:
    return ".png" if view_name == "map" else ".jpg"


def _save_image(image: Any, destination: Path, *, view_name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, dict):
        image_path = image.get("path")
        if image_path:
            from PIL import Image

            with Image.open(image_path) as opened:
                opened.convert("RGB").save(destination)
            return
        image_bytes = image.get("bytes")
        if image_bytes:
            from io import BytesIO
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as opened:
                opened.convert("RGB").save(destination)
            return
    if hasattr(image, "save"):
        if view_name == "map":
            image.save(destination)
        else:
            image.convert("RGB").save(destination, quality=95)
        return
    raise TypeError(f"Unsupported image value for {destination}: {type(image)!r}")


def _views_for_get_image(
    *,
    record: StepRecord,
    raw_step: dict[str, Any] | None,
) -> list[str]:
    raw_args = dict(raw_step.get("args", {})) if raw_step is not None else {}
    views = list(record.args.get("views") or raw_args.get("views") or ())
    if views:
        return [str(view) for view in views if str(view) in IMAGE_COLUMNS]
    return [
        view_name for view_name in IMAGE_COLUMNS if view_name in record.images_by_view
    ]


def _normalize_raw_steps(
    original_trajectory: dict[str, Any],
    *,
    trajectory_id: str,
    records: list[StepRecord],
) -> list[dict[str, Any]]:
    raw_steps = list(original_trajectory.get("steps") or [])
    records_by_index = {record.step_index: record for record in records}
    for index, raw_step in enumerate(raw_steps):
        record = records_by_index.get(index)
        raw_step["step"] = index
        raw_step.setdefault("agent", _agent_id(record.robot_idx if record else 0))
        raw_step.setdefault("tool", record.tool if record else "")
        raw_step.setdefault("args", dict(record.args) if record else {})
        raw_step.setdefault("reasoning", "")
    original_trajectory["trajectory_id"] = trajectory_id
    return raw_steps


def _stage_episode(
    *,
    output_root: Path,
    task_dir: str,
    trajectory_id: str,
    source_label: str,
    task_instruction: str,
    records: list[StepRecord],
    original_trajectory: dict[str, Any],
    execution_metadata: dict[str, Any] | None,
) -> None:
    task_metadata = get_task_metadata(task_dir)
    trajectory_dir = output_root / task_dir / trajectory_id
    image_dir = trajectory_dir / "images" / trajectory_id
    raw_steps = _normalize_raw_steps(
        original_trajectory,
        trajectory_id=trajectory_id,
        records=records,
    )
    raw_steps_by_index = {int(step["step"]): step for step in raw_steps}

    plan_steps: list[dict[str, Any]] = []
    metadata_steps: list[dict[str, Any]] = []
    for record in records:
        raw_step = raw_steps_by_index.get(record.step_index)
        source_agent = (
            str(raw_step.get("agent"))
            if raw_step is not None and raw_step.get("agent")
            else _agent_id(record.robot_idx)
        )
        args = dict(record.args)
        image_paths: list[str] = []
        relative_image_paths: list[str] = []
        views: list[str] = []

        if record.tool == "get_image":
            views = _views_for_get_image(record=record, raw_step=raw_step)
            staged_views: list[str] = []
            for view_name in views:
                image = record.images_by_view.get(view_name)
                if image is None:
                    continue
                relpath = (
                    Path("images")
                    / trajectory_id
                    / f"{record.step_index}_{view_name}_{source_agent}{_image_extension(view_name)}"
                )
                destination = trajectory_dir / relpath
                _save_image(image, destination, view_name=view_name)
                image_paths.append(str(destination.resolve()))
                relative_image_paths.append(str(relpath))
                staged_views.append(view_name)
            if not image_paths:
                raise ValueError(
                    f"{source_label} {trajectory_id} step {record.step_index} "
                    "is get_image but has no staged images."
                )
            views = staged_views
            args["views"] = views
            args["image_paths"] = image_paths

        reasoning = str(raw_step.get("reasoning", "")) if raw_step else ""
        plan_step = {
            "tool": record.tool,
            "robot_idx": record.robot_idx,
            "args": args,
            "metadata": {
                "step_index": record.step_index,
                "source_agent": source_agent,
                "reasoning": reasoning,
                "image_path": None,
                "image_paths": relative_image_paths or None,
            },
        }
        metadata_step = {
            "step_index": record.step_index,
            "tool": record.tool,
            "robot_idx": record.robot_idx,
            "args": args,
            "success": record.success,
        }
        if image_paths:
            metadata_step["image_paths"] = image_paths
        plan_steps.append(plan_step)
        metadata_steps.append(metadata_step)

    metadata_payload = dict(execution_metadata or {})
    metadata_payload["task"] = (
        task_instruction or metadata_payload.get("task") or task_metadata.task_goal
    )
    metadata_payload["steps"] = metadata_steps
    metadata_payload.setdefault("source_hf_dataset", source_label)

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    (trajectory_dir / "original_trajectory.json").write_text(
        json.dumps(original_trajectory, indent=2),
        encoding="utf-8",
    )
    (trajectory_dir / "plan.json").write_text(
        json.dumps(plan_steps, indent=2),
        encoding="utf-8",
    )
    (trajectory_dir / "metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )
    (trajectory_dir / "trajectory_execution_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )
    (trajectory_dir / "adapted_trajectory.json").write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "composite_task": task_metadata.composite_task,
                "tool_calls": plan_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not image_dir.is_dir():
        image_dir.mkdir(parents=True, exist_ok=True)


def _metadata_scan_dataset(ds):
    image_columns = [
        column_name
        for column_name in IMAGE_COLUMNS
        if column_name in set(ds.column_names)
    ]
    if not image_columns:
        return ds
    remove_columns = getattr(ds, "remove_columns", None)
    if remove_columns is None:
        return ds
    return remove_columns(image_columns)


def _iter_episode_sources(ds) -> Iterator[EpisodeSource]:
    column_names = set(ds.column_names)
    scan_ds = _metadata_scan_dataset(ds)
    if "adapted_trajectory" in column_names or "original_trajectory" in column_names:
        for index, row in enumerate(scan_ds):
            yield EpisodeSource(
                source_row=row,
                row_indices=(index,),
                episode_index=index,
            )
        return

    grouped_rows: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(scan_ds):
        grouped_rows[_episode_id_for_row(row, index)].append((index, row))
    for index, (_episode_id, rows) in enumerate(sorted(grouped_rows.items())):
        yield EpisodeSource(
            source_row=rows[0][1],
            row_indices=tuple(row_index for row_index, _row in rows),
            episode_index=index,
        )


def _row_granularity_for_dataset(ds) -> str:
    column_names = set(ds.column_names)
    if "adapted_trajectory" in column_names or "original_trajectory" in column_names:
        return "trajectory"
    return "step"


def _records_for_episode_source(
    *,
    dataset,
    row_granularity: str,
    source: EpisodeSource,
) -> list[StepRecord]:
    if row_granularity == "trajectory":
        return _rows_from_trajectory_row(dataset[source.row_indices[0]])
    rows = [dataset[row_index] for row_index in source.row_indices]
    return _rows_from_step_rows(rows)


def _load_repo_state(
    *,
    repo_id: str,
    split: str,
    revision: str | None,
    trust_remote_code: bool,
) -> RepoStageState:
    print(f"Loading {repo_id} split={split}", flush=True)
    ds = _load_dataset(
        repo_id,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    sidecar_root = None
    if "original_trajectory" not in set(ds.column_names):
        print(f"Downloading JSON sidecars for {repo_id}", flush=True)
        sidecar_root = _snapshot_dataset_sidecars(repo_id, revision=revision)

    return RepoStageState(
        repo_id=repo_id,
        split=split,
        revision=revision,
        sidecar_root=sidecar_root,
        dataset=ds,
        row_granularity=_row_granularity_for_dataset(ds),
        episodes=iter(_iter_episode_sources(ds)),
    )


def _next_task_trajectory_id(
    *,
    source_row: dict[str, Any],
    repo_id: str,
    counters_by_task: dict[str, int],
) -> tuple[str, str]:
    task_dir = _task_dir_for_row(source_row, repo_id)
    trajectory_index = counters_by_task[task_dir]
    counters_by_task[task_dir] += 1
    return task_dir, f"traj_{trajectory_index:06d}"


def _load_json_for_resume(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _existing_trajectory_is_complete(
    trajectory_dir: Path,
    *,
    validate: bool,
) -> bool:
    if not trajectory_dir.is_dir():
        return False
    for file_name in REQUIRED_STAGE_FILES:
        if not (trajectory_dir / file_name).is_file():
            return False
    if not validate:
        return True

    parsed_json: dict[str, Any] = {}
    for file_name in REQUIRED_STAGE_FILES:
        parsed_value = _load_json_for_resume(trajectory_dir / file_name)
        if parsed_value is None:
            return False
        parsed_json[file_name] = parsed_value

    image_dir = trajectory_dir / "images" / trajectory_dir.name
    if not image_dir.is_dir():
        return False

    plan_steps = parsed_json.get("plan.json")
    if not isinstance(plan_steps, list):
        return False
    for step in plan_steps:
        if not isinstance(step, dict):
            return False
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            continue
        image_paths = metadata.get("image_paths")
        if image_paths is None:
            continue
        if not isinstance(image_paths, list):
            return False
        for image_relpath in image_paths:
            if not isinstance(image_relpath, str) or not image_relpath:
                return False
            if not (trajectory_dir / image_relpath).is_file():
                return False

    return True


def _load_repo_states(
    *,
    repo_ids: list[str],
    split: str,
    revision: str | None,
    trust_remote_code: bool,
    load_workers: int,
) -> list[RepoStageState]:
    if load_workers == 1 or len(repo_ids) == 1:
        return [
            _load_repo_state(
                repo_id=repo_id,
                split=split,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            for repo_id in repo_ids
        ]

    print(
        f"Loading {len(repo_ids)} repos with {load_workers} load workers",
        flush=True,
    )
    states: list[RepoStageState | None] = [None] * len(repo_ids)
    with ThreadPoolExecutor(max_workers=load_workers) as executor:
        futures_by_index = {
            executor.submit(
                _load_repo_state,
                repo_id=repo_id,
                split=split,
                revision=revision,
                trust_remote_code=trust_remote_code,
            ): index
            for index, repo_id in enumerate(repo_ids)
        }
        try:
            for future in as_completed(futures_by_index):
                index = futures_by_index[future]
                states[index] = future.result()
        except BaseException:
            for future in futures_by_index:
                future.cancel()
            raise

    return [state for state in states if state is not None]


def _prepare_episode_staging_job(
    *,
    output_root: Path,
    repo_id: str,
    source_row: dict[str, Any],
    records: list[StepRecord],
    sidecar_root: Path | None,
    episode_index: int,
    task_dir: str,
    trajectory_id: str,
) -> EpisodeStagingJob:
    task_metadata = get_task_metadata(task_dir)
    source_label = f"{repo_id}:{_episode_id_for_row(source_row, episode_index)}"
    original_trajectory = _sidecar_json(
        row=source_row,
        field_name="original_trajectory",
        sidecar_root=sidecar_root,
    )
    if original_trajectory is None:
        raise ValueError(
            f"{source_label} has no original_trajectory JSON. Republish the "
            "dataset with --row-granularity trajectory or upload the "
            "original_trajectory.json sidecars."
        )
    execution_metadata = _sidecar_json(
        row=source_row,
        field_name="execution_metadata",
        sidecar_root=sidecar_root,
    )
    task_instruction = str(
        (execution_metadata or {}).get("task")
        or source_row.get("task")
        or task_metadata.task_goal
    )

    return EpisodeStagingJob(
        repo_id=repo_id,
        output_root=output_root,
        task_dir=task_dir,
        trajectory_id=trajectory_id,
        source_label=source_label,
        task_instruction=task_instruction,
        records=records,
        original_trajectory=original_trajectory,
        execution_metadata=execution_metadata,
    )


def _stage_episode_job(job: EpisodeStagingJob) -> EpisodeStagingResult:
    _stage_episode(
        output_root=job.output_root,
        task_dir=job.task_dir,
        trajectory_id=job.trajectory_id,
        source_label=job.source_label,
        task_instruction=job.task_instruction,
        records=job.records,
        original_trajectory=job.original_trajectory,
        execution_metadata=job.execution_metadata,
    )
    return EpisodeStagingResult(repo_id=job.repo_id, task_dir=job.task_dir)


def _finish_completed_jobs(
    *,
    done: set[Future[EpisodeStagingResult]],
    futures_by_state: dict[Future[EpisodeStagingResult], RepoStageState],
    progress_interval: int,
) -> None:
    for future in done:
        state = futures_by_state.pop(future)
        result = future.result()
        state.staged += 1
        state.tasks.add(result.task_dir)
        if progress_interval > 0 and state.staged % progress_interval == 0:
            print(f"  staged {state.staged} episodes from {state.repo_id}", flush=True)


def _log_finished_repos(states: list[RepoStageState]) -> None:
    for state in states:
        if state.finished_logged or not state.exhausted:
            continue
        if state.staged < state.submitted:
            continue
        if state.skipped_existing:
            print(
                f"Finished {state.repo_id}: staged {state.staged} new episodes, "
                f"skipped {state.skipped_existing} existing episodes",
                flush=True,
            )
        else:
            print(
                f"Finished {state.repo_id}: staged {state.staged} episodes",
                flush=True,
            )
        state.finished_logged = True


def stage_repos(
    *,
    repo_ids: list[str],
    output_root: Path,
    split: str,
    revision: str | None,
    trust_remote_code: bool,
    counters_by_task: dict[str, int],
    load_workers: int,
    workers: int,
    max_in_flight: int,
    progress_interval: int,
    resume_existing: bool = False,
    resume_validation: str = "validated",
) -> list[dict[str, Any]]:
    states = _load_repo_states(
        repo_ids=repo_ids,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
        load_workers=load_workers,
    )

    futures_by_state: dict[Future[EpisodeStagingResult], RepoStageState] = {}
    next_state_index = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            while True:
                active_states = [state for state in states if not state.exhausted]
                made_submission = False
                while active_states and len(futures_by_state) < max_in_flight:
                    state = active_states[next_state_index % len(active_states)]
                    next_state_index += 1
                    try:
                        source = next(state.episodes)
                    except StopIteration:
                        state.exhausted = True
                        active_states = [
                            candidate for candidate in states if not candidate.exhausted
                        ]
                        next_state_index = 0
                        continue

                    task_dir, trajectory_id = _next_task_trajectory_id(
                        source_row=source.source_row,
                        repo_id=state.repo_id,
                        counters_by_task=counters_by_task,
                    )
                    state.tasks.add(task_dir)
                    if resume_existing and _existing_trajectory_is_complete(
                        output_root / task_dir / trajectory_id,
                        validate=resume_validation == "validated",
                    ):
                        state.skipped_existing += 1
                        if (
                            progress_interval > 0
                            and state.skipped_existing % progress_interval == 0
                        ):
                            print(
                                "  skipped "
                                f"{state.skipped_existing} existing episodes from "
                                f"{state.repo_id}",
                                flush=True,
                            )
                        continue

                    records = _records_for_episode_source(
                        dataset=state.dataset,
                        row_granularity=state.row_granularity,
                        source=source,
                    )
                    job = _prepare_episode_staging_job(
                        output_root=output_root,
                        repo_id=state.repo_id,
                        source_row=source.source_row,
                        records=records,
                        sidecar_root=state.sidecar_root,
                        episode_index=source.episode_index,
                        task_dir=task_dir,
                        trajectory_id=trajectory_id,
                    )
                    future = executor.submit(_stage_episode_job, job)
                    futures_by_state[future] = state
                    state.submitted += 1
                    made_submission = True

                _log_finished_repos(states)
                if not futures_by_state and not any(
                    not state.exhausted for state in states
                ):
                    break
                if futures_by_state:
                    done, _pending = wait(
                        futures_by_state,
                        return_when=FIRST_COMPLETED,
                    )
                    _finish_completed_jobs(
                        done=done,
                        futures_by_state=futures_by_state,
                        progress_interval=progress_interval,
                    )
                    _log_finished_repos(states)
                elif not made_submission:
                    break
        except BaseException:
            for future in futures_by_state:
                future.cancel()
            raise

    return [
        {
            "repo_id": state.repo_id,
            "split": state.split,
            "revision": state.revision,
            "num_episodes": state.staged + state.skipped_existing,
            "num_staged_new_episodes": state.staged,
            "num_skipped_existing": state.skipped_existing,
            "tasks": sorted(state.tasks),
        }
        for state in states
    ]


def stage_repo(
    *,
    repo_id: str,
    output_root: Path,
    split: str,
    revision: str | None,
    trust_remote_code: bool,
    counters_by_task: dict[str, int],
    load_workers: int = 1,
    workers: int = 1,
    max_in_flight: int | None = None,
    progress_interval: int = 25,
    resume_existing: bool = False,
    resume_validation: str = "validated",
) -> dict[str, Any]:
    max_in_flight = max_in_flight or max(1, workers * 2)
    return stage_repos(
        repo_ids=[repo_id],
        output_root=output_root,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
        counters_by_task=counters_by_task,
        load_workers=load_workers,
        workers=workers,
        max_in_flight=max_in_flight,
        progress_interval=progress_interval,
        resume_existing=resume_existing,
        resume_validation=resume_validation,
    )[0]


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.load_workers < 1:
        raise ValueError("--load-workers must be at least 1.")
    if args.max_in_flight is not None and args.max_in_flight < 1:
        raise ValueError("--max-in-flight must be at least 1.")
    max_in_flight = args.max_in_flight or max(1, args.workers * 2)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    counters_by_task: dict[str, int] = defaultdict(int)
    staged_sources = stage_repos(
        repo_ids=args.repo_id,
        output_root=output_root,
        split=args.split,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        counters_by_task=counters_by_task,
        load_workers=args.load_workers,
        workers=args.workers,
        max_in_flight=max_in_flight,
        progress_interval=args.progress_interval,
        resume_existing=args.resume_existing,
        resume_validation=args.resume_validation,
    )

    manifest = {
        "output_root": str(output_root),
        "sources": staged_sources,
        "task_episode_counts": dict(sorted(counters_by_task.items())),
    }
    (output_root / "hf_stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Staged HF sweep datasets at {output_root}", flush=True)
    print("Task episode counts:", dict(sorted(counters_by_task.items())), flush=True)


if __name__ == "__main__":
    main()
