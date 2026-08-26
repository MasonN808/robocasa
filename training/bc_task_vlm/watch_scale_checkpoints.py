"""Watch scale-training checkpoints and submit fixed-cohort evaluations once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SCALES = (30, 60, 90, 120, 150)
SPLITS = ("train_task_types", "heldout_task_types")
TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "submissions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _job_state(job_id: str) -> str:
    queued = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=False, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if queued:
        return queued[0].strip().upper()
    accounted = subprocess.run(
        ["sacct", "-X", "-j", job_id, "-n", "-o", "State"],
        check=False, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return accounted[0].strip().split()[0].upper() if accounted else "UNKNOWN"


def _complete_checkpoints(run_dir: Path) -> list[tuple[Path, float, str]]:
    checkpoints = []
    now = time.time()
    for checkpoint in run_dir.glob("checkpoint-*"):
        state_path = checkpoint / "trainer_state.json"
        adapter_path = checkpoint / "adapter_model.safetensors"
        config_path = checkpoint / "adapter_config.json"
        if not all(path.is_file() and path.stat().st_size > 0 for path in (state_path, adapter_path, config_path)):
            continue
        # trainer_state.json is written at the end of the Transformers
        # checkpoint transaction. A small age gate also protects NFS readers.
        if now - state_path.stat().st_mtime < 20:
            continue
        try:
            trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
            epoch = float(trainer_state["epoch"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        boundary = round(epoch * 2) / 2
        if boundary not in (0.5, 1.0, 1.5, 2.0) or abs(epoch - boundary) > 0.03:
            continue
        checkpoints.append((checkpoint.resolve(), epoch, str(boundary).replace(".", "p")))
    return sorted(checkpoints, key=lambda row: row[1])


def _submit_eval(*, scale: int, checkpoint: Path, epoch_tag: str, split: str) -> str:
    short_split = "train47" if split == "train_task_types" else "held6"
    run_name = f"sft_scale{scale}_ep{epoch_tag}_fixed10_promptv9"
    exports = {
        "MANIFEST": "training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json",
        "DATASET_ROOT": "/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered",
        "SCENE_COMPATIBILITY_CACHE": "training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json",
        "EPISODES_PER_TASK": "10",
        "EVALUATION_SEED": "20260817",
        "SCENE_SAMPLING_SEED": "20260819",
        "WORKERS": "4",
        "MODEL": "Qwen/Qwen3-VL-8B-Instruct",
        "ADAPTER": str(checkpoint),
        "COMMUNICATION_MODE": "full",
        "COHORT_SPLIT": split,
        "RUN_NAME": run_name,
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    result = subprocess.run(
        [
            "sbatch", "--parsable",
            f"--job-name=sft-s{scale}-e{epoch_tag}-{short_split}",
            f"--export={export_arg}",
            "fixed_live_sim_open_weight_vllm.sbatch",
        ],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip().split(";", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-array-job", default="440098")
    parser.add_argument(
        "--watch-config",
        type=Path,
        help="Optional JSON containing per-scale run_dir, training_job_id, and target_epochs.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-runtime-hours", type=float, default=23.0)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("training/bc_task_vlm/eval_runs/fixed_live_sim_scale_checkpoint_watch/state.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    state = _load_state(args.state)
    submissions = state.setdefault("submissions", {})
    if args.watch_config:
        raw_specs = json.loads(args.watch_config.read_text(encoding="utf-8"))["scales"]
        specs = [
            {
                "scale": int(row["scale"]),
                "run_dir": Path(row["run_dir"]),
                "training_job_id": str(row["training_job_id"]),
                "target_epochs": float(row["target_epochs"]),
            }
            for row in raw_specs
        ]
    else:
        specs = [
            {
                "scale": scale,
                "run_dir": Path(
                    f"training/bc_task_vlm/runs/qwen3vl8b-tick150-scale{scale}-ep2-halfckpt-noindex-promptv9"
                ),
                "training_job_id": f"{args.training_array_job}_{index}",
                "target_epochs": 2.0,
            }
            for index, scale in enumerate(SCALES)
        ]
    expected_submissions = sum(
        int(spec["target_epochs"] * 2) * len(SPLITS) for spec in specs
    )
    terminal_seen: set[int] = set()
    while True:
        for spec in specs:
            scale = spec["scale"]
            run_dir = spec["run_dir"]
            for checkpoint, epoch, epoch_tag in _complete_checkpoints(run_dir):
                if epoch > spec["target_epochs"] + 0.03:
                    continue
                for split in SPLITS:
                    key = f"scale{scale}/{checkpoint.name}/{split}"
                    if key in submissions:
                        continue
                    try:
                        job_id = _submit_eval(
                            scale=scale, checkpoint=checkpoint, epoch_tag=epoch_tag, split=split
                        )
                    except subprocess.CalledProcessError as exc:
                        detail = (exc.stderr or exc.stdout or str(exc)).strip()
                        print(f"submission failed for {key}; will retry: {detail}", flush=True)
                        continue
                    submissions[key] = {
                        "scale": scale,
                        "epoch": epoch,
                        "checkpoint": str(checkpoint),
                        "split": split,
                        "job_id": job_id,
                        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    _save_state(args.state, state)
                    print(f"submitted {key}: job {job_id}", flush=True)
            training_state = _job_state(spec["training_job_id"])
            print(f"scale={scale} training_state={training_state}", flush=True)
            if training_state in TERMINAL_STATES:
                terminal_seen.add(scale)

        if len(submissions) == expected_submissions:
            print(f"all {expected_submissions} checkpoint evaluations submitted", flush=True)
            return 0
        if len(terminal_seen) == len(specs):
            print(
                f"all training jobs terminal, but only {len(submissions)}/{expected_submissions} evaluations submitted",
                flush=True,
            )
            return 2
        if time.monotonic() - started >= args.max_runtime_hours * 3600:
            print("watcher segment reached runtime limit; request successor", flush=True)
            return 75
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
