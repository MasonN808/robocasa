"""Batch low-level VLA probing over a trajectory-root folder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_evals.low_level.eval_trajectory_tools import (
    _agent_to_robot_idx,
    _complete_scene_metadata,
    _execute_synthetic_step,
    _load_adapter_and_client,
    _prepare_held_objects_for_probe,
    _write_probe_outputs,
    run_vla_probe,
)
from model_evals.low_level.state_snapshot import capture_executor_state, restore_executor_state
from model_evals.low_level.tool_categories import expand_tool_filter, support_tier
from model_evals.low_level.tool_prompts import prompt_for_tool_call
from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["dry", "gwp", "pi05", "rldx", "gr00t_n1_5"], default="dry")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--trajectory_root", required=True)
    parser.add_argument("--trajectory_indices", type=int, nargs="+", default=[0])
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--exclude_tasks", nargs="+", default=[])
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_jobs", type=int, default=None)
    parser.add_argument("--missing_policy", choices=["skip", "error"], default="skip")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--scene_sampling", choices=["official", "trajectory"], default="official", help="Scene selection mode for child trajectory runs. official samples layout/style like RoboCasa eval split resets.")
    parser.add_argument("--split", choices=["pretrain", "target", "all"], default="pretrain", help="Scene split for --scene_sampling official.")
    parser.add_argument("--scene_seed", type=int, default=7, help="RNG seed for official-style scene sampling, matching the official eval default.")
    parser.add_argument("--tools", nargs="+", default=["all_physical"])
    parser.add_argument("--skip_tools", nargs="+", default=[])
    parser.add_argument("--max_segments_per_trajectory", type=int, default=None)
    parser.add_argument("--max_steps_per_probe", type=int, default=240)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--cameras", default="default")
    parser.add_argument("--render_width", type=int, default=256)
    parser.add_argument("--render_height", type=int, default=256)
    parser.add_argument("--gl_backend", default="osmesa")
    parser.add_argument("--dry_run", action="store_true", help="Alias for --backend dry.")
    parser.add_argument("--stop_on_success", action="store_true", help="Stop each low-level probe as soon as its predicate becomes true. Default runs to the full horizon and checks final success.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child runs.")
    return parser.parse_args()


def _task_name_from_dir(task_dir: Path) -> str:
    # Trajectories usually carry composite_task; this is only a fallback.
    return "".join(part.capitalize() for part in task_dir.name.split("_"))


def _discover_jobs(args) -> list[dict]:
    root = Path(args.trajectory_root)
    if not root.exists():
        raise FileNotFoundError(f"trajectory_root does not exist: {root}")

    excluded = set(args.exclude_tasks or [])
    if args.tasks and args.tasks != ["all"]:
        task_dirs = [root / task for task in args.tasks]
    else:
        task_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    task_dirs = [path for path in task_dirs if path.name not in excluded]
    if args.max_tasks is not None:
        task_dirs = task_dirs[: args.max_tasks]

    jobs = []
    task_episode_counts = defaultdict(int)
    for task_dir in task_dirs:
        traj_dir = task_dir / "trajectories"
        for traj_idx in args.trajectory_indices:
            traj_path = traj_dir / f"traj_{traj_idx:06d}.json"
            if not traj_path.exists():
                if args.missing_policy == "error":
                    raise FileNotFoundError(f"missing trajectory: {traj_path}")
                continue
            payload = json.loads(traj_path.read_text())
            task_name = payload.get("composite_task") or _task_name_from_dir(task_dir)
            for trial_idx in range(args.num_trials):
                episode_index = task_episode_counts[task_name]
                task_episode_counts[task_name] += 1
                jobs.append(
                    {
                        "task_dir": task_dir.name,
                        "task_name": task_name,
                        "trajectory_index": traj_idx,
                        "trajectory_path": str(traj_path),
                        "trajectory_id": payload.get("trajectory_id", traj_path.stem),
                        "trial_index": trial_idx,
                        "task_episode_index": episode_index,
                        "scene": None,
                    }
                )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return [job for i, job in enumerate(jobs) if i % args.num_workers == args.worker_id]


def _capture_env_rng_state(executor):
    rng = getattr(executor.env, "rng", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is None:
        return None
    return deepcopy(bit_generator.state)


def _restore_env_rng_state(executor, state) -> None:
    if state is None:
        return
    rng = getattr(executor.env, "rng", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None:
        bit_generator.state = deepcopy(state)


def _failure_result(job: dict, destination: Path, exc: BaseException, stage: str) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    (destination / "failure.json").write_text(json.dumps(payload, indent=2))
    result = dict(job)
    result.update(
        {
            "returncode": 1,
            "stage": stage,
            "error": str(exc),
            "output_dir": str(destination),
            "success_rate": 0.0,
            "num_segments": 0,
            "num_successes": 0,
            "segments": [],
        }
    )
    return result


def _run_job_on_executor(args, job: dict, destination: Path, executor, adapter, client) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        trajectory_path = Path(job["trajectory_path"])
        trajectory = json.loads(trajectory_path.read_text())
        scene_metadata = _complete_scene_metadata(
            executor,
            {
                "sampling": "official_persistent_reset",
                "split": args.split,
                "scene_seed": int(args.scene_seed),
                "episode_index": int(job.get("task_episode_index", 0)),
                "seed": int(args.scene_seed),
            },
            int(job.get("task_episode_index", 0)),
        )
        if adapter is not None and getattr(adapter, "on_episode_start", None) is not None:
            adapter.on_episode_start(job["task_name"], int(job.get("task_episode_index", 0)))

        adapted = TrajectoryAdapter(executor=executor).adapt(trajectory, output_dir=destination)
        load_result = executor.load_initial_state(adapted.get("initial_state"))
        (destination / "input_trajectory.json").write_text(json.dumps(trajectory, indent=2))
        (destination / "adapted_trajectory.json").write_text(json.dumps(adapted, indent=2, default=str))
        (destination / "initial_state_load.json").write_text(json.dumps(load_result, indent=2, default=str))
        (destination / "scene_metadata.json").write_text(json.dumps(scene_metadata, indent=2, default=str))

        selected_tools = expand_tool_filter(args.tools) - set(args.skip_tools or [])
        results = []
        probed = 0
        original_steps = trajectory.get("steps", [])
        adapted_steps = adapted.get("tool_calls", [])
        for step_pos, step in enumerate(adapted_steps):
            original_step = original_steps[step_pos] if step_pos < len(original_steps) else step
            tool = step.get("tool")
            if tool not in selected_tools:
                continue
            if args.max_segments_per_trajectory is not None and probed >= args.max_segments_per_trajectory:
                break

            step_index = int(original_step.get("step", step_pos))
            robot_idx = int(step.get("robot_idx", _agent_to_robot_idx(original_step.get("agent", "agent_0"))))
            prompt = prompt_for_tool_call(original_step)
            agent_label = original_step.get("agent", f"agent_{robot_idx}")
            probe_name = f"step_{step_index:03d}_{tool}_{agent_label}"
            probe_dir = destination / probe_name

            held_objects_before_probe = _prepare_held_objects_for_probe(executor)
            snapshot = capture_executor_state(executor)
            try:
                probe_result = run_vla_probe(
                    executor=executor,
                    step=step,
                    prompt=prompt,
                    adapter=adapter,
                    client=client,
                    backend=args.backend,
                    robot_idx=robot_idx,
                    max_steps=args.max_steps_per_probe,
                    replan_steps=args.replan_steps,
                    expected_action_chunk=args.action_chunk,
                    cameras=args.cameras,
                    stop_on_success=args.stop_on_success,
                )
            except Exception as exc:
                probe_result = {
                    "success": False,
                    "predicate": {"success": False, "metrics": {}, "reason": f"probe error: {exc}"},
                    "initial_predicate": {"success": False, "metrics": {}, "reason": f"probe error: {exc}"},
                    "first_success_step": None,
                    "num_env_steps": 0,
                    "actions": np.zeros((0, 12), dtype=np.float32),
                    "frames_by_label": {},
                }
            finally:
                restore_executor_state(executor, snapshot)

            synthetic_success = False
            synthetic_details = {}
            try:
                synthetic_result = _execute_synthetic_step(executor, step, robot_idx)
                synthetic_success = bool(synthetic_result.success)
                synthetic_details = synthetic_result.details
            except Exception as exc:
                synthetic_details = {"error": str(exc), "traceback": traceback.format_exc()}

            stats = {
                "task": job["task_name"],
                "trajectory_id": trajectory.get("trajectory_id", trajectory_path.stem),
                "trajectory_path": str(trajectory_path),
                "scene": scene_metadata,
                "layout": scene_metadata["layout"],
                "style": scene_metadata["style"],
                "seed": scene_metadata["seed"],
                "step_index": step_index,
                "agent": original_step.get("agent"),
                "robot_idx": robot_idx,
                "tool": tool,
                "support_tier": support_tier(tool),
                "args": original_step.get("args") or {},
                "adapted_args": step.get("args") or {},
                "prompt": prompt,
                "backend": args.backend,
                "max_steps_per_probe": args.max_steps_per_probe,
                "replan_steps": args.replan_steps,
                "held_objects_before_probe": held_objects_before_probe,
                "success": bool(probe_result["success"]),
                "predicate": probe_result["predicate"],
                "initial_predicate": probe_result.get("initial_predicate"),
                "first_success_step": probe_result.get("first_success_step"),
                "stop_on_success": args.stop_on_success,
                "num_env_steps": int(probe_result["num_env_steps"]),
                "actions_shape": list(np.asarray(probe_result["actions"]).shape),
                "synthetic_advance_success": synthetic_success,
                "synthetic_advance_details": synthetic_details,
            }
            _write_probe_outputs(probe_dir, probe_result, stats)
            results.append(stats)
            probed += 1

        summary = {
            "task": job["task_name"],
            "trajectory_id": trajectory.get("trajectory_id", trajectory_path.stem),
            "backend": args.backend,
            "scene": scene_metadata,
            "layout": scene_metadata["layout"],
            "style": scene_metadata["style"],
            "seed": scene_metadata["seed"],
            "num_segments": len(results),
            "num_successes": sum(1 for r in results if r["success"]),
            "success_rate": (sum(1 for r in results if r["success"]) / len(results)) if results else 0.0,
            "segments": results,
        }
        (destination / "summary.json").write_text(json.dumps(summary, indent=2))
        job_result = dict(job)
        job_result.update(
            {
                "returncode": 0,
                "scene": scene_metadata,
                "output_dir": str(destination),
                "success_rate": summary["success_rate"],
                "num_segments": summary["num_segments"],
                "num_successes": summary["num_successes"],
                "segments": results,
            }
        )
        return job_result
    except Exception as exc:
        return _failure_result(job, destination, exc, "job")


def _run_official_persistent(args, jobs: list[dict], out_root: Path) -> list[dict]:
    adapter, client = _load_adapter_and_client(args.backend, args.host, args.port)
    results = []
    executor = None
    current_task = None
    current_episode_index = None
    reset_rng_state = None
    try:
        for idx, job in enumerate(jobs):
            destination = out_root / job["task_dir"] / job["trajectory_id"] / f"trial_{job['trial_index']:03d}"
            print(
                f"[{idx + 1}/{len(jobs)}] {job['task_name']} {job['trajectory_id']} trial {job['trial_index']} -> {destination}",
                flush=True,
            )
            if current_task != job["task_name"]:
                if executor is not None:
                    executor.close()
                current_task = job["task_name"]
                current_episode_index = 0
                reset_rng_state = None
                try:
                    executor = SimToolExecutor(
                        task_name=current_task,
                        robots=args.robots,
                        seed=args.scene_seed,
                        split=args.split,
                        render_width=args.render_width,
                        render_height=args.render_height,
                        gl_backend=args.gl_backend,
                    )
                    reset_rng_state = _capture_env_rng_state(executor)
                except Exception as exc:
                    result = _failure_result(job, destination, exc, "executor_init")
                    results.append(result)
                    print(json.dumps({"returncode": result.get("returncode"), "error": result.get("error")}), flush=True)
                    current_task = None
                    current_episode_index = None
                    reset_rng_state = None
                    executor = None
                    continue

            if executor is None:
                result = _failure_result(job, destination, RuntimeError("executor unavailable"), "executor_init")
                results.append(result)
                continue

            target_episode_index = int(job.get("task_episode_index", 0))
            try:
                if current_episode_index is None:
                    current_episode_index = 0
                if target_episode_index < current_episode_index:
                    raise RuntimeError(
                        f"Cannot rewind persistent env from episode {current_episode_index} "
                        f"to {target_episode_index}"
                    )
                while current_episode_index < target_episode_index:
                    _restore_env_rng_state(executor, reset_rng_state)
                    executor.reset_scene()
                    reset_rng_state = _capture_env_rng_state(executor)
                    current_episode_index += 1
            except Exception as exc:
                result = _failure_result(job, destination, exc, "reset_scene")
                results.append(result)
                print(json.dumps({"returncode": result.get("returncode"), "error": result.get("error")}), flush=True)
                if executor is not None:
                    executor.close()
                executor = None
                current_task = None
                current_episode_index = None
                reset_rng_state = None
                continue

            result = _run_job_on_executor(args, job, destination, executor, adapter, client)
            results.append(result)
            print(
                json.dumps(
                    {
                        "returncode": result.get("returncode"),
                        "stage": result.get("stage"),
                        "scene": result.get("scene"),
                        "num_segments": result.get("num_segments"),
                        "success_rate": result.get("success_rate"),
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.close()
        if client is not None:
            client.close()
    return results


def _run_single_job(args, job: dict, destination: Path) -> dict:
    script = Path(__file__).with_name("eval_trajectory_tools.py")
    with tempfile.TemporaryDirectory(prefix="robocasa_low_level_job_") as tmp:
        cmd = [
            args.python,
            str(script),
            "--backend",
            args.backend,
            "--host",
            args.host,
            "--trajectory_json",
            job["trajectory_path"],
            "--task",
            job["task_name"],
            "--log_dir",
            tmp,
            "--scene_sampling",
            args.scene_sampling,
            "--split",
            args.split,
            "--scene_seed",
            str(args.scene_seed),
            "--scene_episode_index",
            str(job.get("task_episode_index", 0)),
            "--tools",
            *args.tools,
            "--max_steps_per_probe",
            str(args.max_steps_per_probe),
            "--replan_steps",
            str(args.replan_steps),
            "--cameras",
            args.cameras,
            "--render_width",
            str(args.render_width),
            "--render_height",
            str(args.render_height),
            "--gl_backend",
            args.gl_backend,
        ]
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        if args.action_chunk is not None:
            cmd.extend(["--action_chunk", str(args.action_chunk)])
        if args.skip_tools:
            cmd.extend(["--skip_tools", *args.skip_tools])
        if args.max_segments_per_trajectory is not None:
            cmd.extend(["--max_segments", str(args.max_segments_per_trajectory)])
        if args.dry_run:
            cmd.append("--dry_run")
        if args.stop_on_success:
            cmd.append("--stop_on_success")

        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
        job_result = dict(job)
        job_result["returncode"] = proc.returncode
        job_result["stdout"] = proc.stdout[-4000:]
        job_result["stderr"] = proc.stderr[-4000:]

        if proc.returncode != 0:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "stdout.txt").write_text(proc.stdout)
            (destination / "stderr.txt").write_text(proc.stderr)
            job_result["success_rate"] = 0.0
            job_result["num_segments"] = 0
            job_result["num_successes"] = 0
            return job_result

        summaries = sorted(Path(tmp).glob("**/summary.json"))
        if not summaries:
            raise RuntimeError(f"single run succeeded but no summary.json was found under {tmp}")
        summary_path = summaries[-1]
        run_dir = summary_path.parent
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(run_dir, destination)
        summary = json.loads((destination / "summary.json").read_text())
        job_result.update(
            {
                "output_dir": str(destination),
                "success_rate": summary.get("success_rate", 0.0),
                "num_segments": summary.get("num_segments", 0),
                "num_successes": summary.get("num_successes", 0),
                "segments": summary.get("segments", []),
            }
        )
        return job_result


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _summarize(batch_results: list[dict], args, jobs_total: int) -> dict:
    total_segments = sum(int(result.get("num_segments", 0)) for result in batch_results)
    total_successes = sum(int(result.get("num_successes", 0)) for result in batch_results)
    by_tool = defaultdict(lambda: {"n": 0, "successes": 0})
    by_task = defaultdict(lambda: {"n": 0, "successes": 0})
    by_tier = defaultdict(lambda: {"n": 0, "successes": 0})
    failures = Counter()

    for result in batch_results:
        for segment in result.get("segments", []) or []:
            success = bool(segment.get("success"))
            tool = str(segment.get("tool"))
            task = str(segment.get("task"))
            tier = str(segment.get("support_tier"))
            for bucket, key in ((by_tool, tool), (by_task, task), (by_tier, tier)):
                bucket[key]["n"] += 1
                bucket[key]["successes"] += int(success)
            if not success:
                reason = ((segment.get("predicate") or {}).get("reason")) or "unknown"
                failures[str(reason)] += 1

    def finalize(bucket):
        return {
            key: {
                "n": value["n"],
                "successes": value["successes"],
                "success_rate": value["successes"] / value["n"] if value["n"] else 0.0,
            }
            for key, value in sorted(bucket.items())
        }

    return {
        "backend": args.backend,
        "trajectory_root": args.trajectory_root,
        "scene_sampling": args.scene_sampling,
        "split": args.split,
        "scene_seed": args.scene_seed,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "jobs_run": len(batch_results),
        "jobs_total_this_worker": jobs_total,
        "job_failures": sum(1 for result in batch_results if result.get("returncode") != 0),
        "num_segments": total_segments,
        "num_successes": total_successes,
        "success_rate": total_successes / total_segments if total_segments else 0.0,
        "by_tool": finalize(by_tool),
        "by_task": finalize(by_task),
        "by_support_tier": finalize(by_tier),
        "failure_reasons": dict(failures.most_common()),
    }


def main():
    args = parse_args()
    if args.dry_run:
        args.backend = "dry"
    out_root = Path(args.log_dir) / f"batch_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_worker{args.worker_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = _discover_jobs(args)
    (out_root / "jobs.json").write_text(json.dumps(jobs, indent=2))
    if args.scene_sampling == "official":
        results = _run_official_persistent(args, jobs, out_root)
    else:
        results = []
        for idx, job in enumerate(jobs):
            destination = (
                out_root
                / job["task_dir"]
                / job["trajectory_id"]
                / f"trial_{job['trial_index']:03d}"
            )
            print(
                f"[{idx + 1}/{len(jobs)}] {job['task_name']} {job['trajectory_id']} trial {job['trial_index']} -> {destination}",
                flush=True,
            )
            result = _run_single_job(args, job, destination)
            results.append(result)
            print(
                json.dumps(
                    {
                        "returncode": result.get("returncode"),
                        "scene": result.get("scene"),
                        "num_segments": result.get("num_segments"),
                        "success_rate": result.get("success_rate"),
                    }
                ),
                flush=True,
            )

    summary = _summarize(results, args, len(jobs))
    (out_root / "batch_results.json").write_text(json.dumps(results, indent=2))
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2))

    segment_records = []
    for result in results:
        for segment in result.get("segments", []) or []:
            record = {k: v for k, v in result.items() if k != "segments"}
            record.update(segment)
            segment_records.append(record)
    _write_jsonl(out_root / "batch_segments.jsonl", segment_records)

    print(json.dumps(summary, indent=2))
    print(f"Batch log dir: {out_root}")


if __name__ == "__main__":
    main()
