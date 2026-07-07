"""Batch VLA segment collection over a trajectory-root folder."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
    parser.add_argument("--segments_dir", default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--scene_sampling", choices=["official", "trajectory"], default="official")
    parser.add_argument("--split", choices=["pretrain", "target", "all"], default="pretrain")
    parser.add_argument("--scene_seed", type=int, default=7)
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
    parser.add_argument("--save_policy", choices=["all", "success_only", "manual_review"], default="success_only")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_success", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _task_name_from_dir(task_dir: Path) -> str:
    return "".join(part.capitalize() for part in task_dir.name.split("_"))


def _discover_jobs(args) -> list[dict]:
    root = Path(args.trajectory_root)
    if args.tasks and args.tasks != ["all"]:
        task_dirs = [root / task for task in args.tasks]
    else:
        task_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    task_dirs = [path for path in task_dirs if path.name not in set(args.exclude_tasks or [])]
    if args.max_tasks is not None:
        task_dirs = task_dirs[: args.max_tasks]

    jobs = []
    task_episode_counts = {}
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
                episode_index = task_episode_counts.get(task_name, 0)
                task_episode_counts[task_name] = episode_index + 1
                jobs.append(
                    {
                        "task_dir": task_dir.name,
                        "task_name": task_name,
                        "trajectory_index": traj_idx,
                        "trajectory_path": str(traj_path),
                        "trajectory_id": payload.get("trajectory_id", traj_path.stem),
                        "trial_index": trial_idx,
                        "task_episode_index": episode_index,
                    }
                )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return [job for i, job in enumerate(jobs) if i % args.num_workers == args.worker_id]


def main():
    args = parse_args()
    out_root = Path(args.log_dir) / f"collection_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_worker{args.worker_id}"
    segments_root = Path(args.segments_dir) if args.segments_dir else out_root / "segments"
    out_root.mkdir(parents=True, exist_ok=True)
    segments_root.mkdir(parents=True, exist_ok=True)

    jobs = _discover_jobs(args)
    (out_root / "jobs.json").write_text(json.dumps(jobs, indent=2))
    script = Path(__file__).with_name("collect_vla.py")
    results = []
    for idx, job in enumerate(jobs):
        job_dir = out_root / job["task_dir"] / job["trajectory_id"] / f"trial_{job['trial_index']:03d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(script),
            "--backend", args.backend,
            "--host", args.host,
            "--trajectory_json", job["trajectory_path"],
            "--task", job["task_name"],
            "--log_dir", str(job_dir),
            "--segments_dir", str(segments_root),
            "--robots", str(args.robots),
            "--scene_sampling", args.scene_sampling,
            "--split", args.split,
            "--scene_seed", str(args.scene_seed),
            "--scene_episode_index", str(job["task_episode_index"]),
            "--tools", *args.tools,
            "--max_steps_per_probe", str(args.max_steps_per_probe),
            "--replan_steps", str(args.replan_steps),
            "--cameras", args.cameras,
            "--render_width", str(args.render_width),
            "--render_height", str(args.render_height),
            "--gl_backend", args.gl_backend,
            "--save_policy", args.save_policy,
        ]
        if args.port is not None:
            cmd.extend(["--port", str(args.port)])
        if args.skip_tools:
            cmd.extend(["--skip_tools", *args.skip_tools])
        if args.max_segments_per_trajectory is not None:
            cmd.extend(["--max_segments", str(args.max_segments_per_trajectory)])
        if args.action_chunk is not None:
            cmd.extend(["--action_chunk", str(args.action_chunk)])
        if args.dry_run:
            cmd.append("--dry_run")
        if args.stop_on_success:
            cmd.append("--stop_on_success")

        print(f"[{idx + 1}/{len(jobs)}] {job['task_name']} {job['trajectory_id']} trial {job['trial_index']}", flush=True)
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)
        (job_dir / "stdout.txt").write_text(proc.stdout)
        (job_dir / "stderr.txt").write_text(proc.stderr)
        result = dict(job)
        result.update({"returncode": proc.returncode, "job_dir": str(job_dir)})
        summaries = sorted(job_dir.glob("**/collection_summary.json"))
        if summaries:
            summary = json.loads(summaries[-1].read_text())
            result.update({k: v for k, v in summary.items() if k != "segments"})
            result["segments"] = summary.get("segments", [])
        elif proc.returncode != 0:
            result["error"] = proc.stderr[-2000:]
        results.append(result)
        print(json.dumps({"returncode": proc.returncode, "num_accepted": result.get("num_accepted", 0)}), flush=True)

    total_segments = sum(int(r.get("num_segments", 0)) for r in results)
    total_accepted = sum(int(r.get("num_accepted", 0)) for r in results)
    total_successes = sum(int(r.get("num_successes", 0)) for r in results)
    summary = {
        "backend": args.backend,
        "trajectory_root": args.trajectory_root,
        "jobs_run": len(results),
        "job_failures": sum(1 for r in results if r.get("returncode") != 0),
        "segments_root": str(segments_root),
        "num_segments": total_segments,
        "num_successes": total_successes,
        "num_accepted": total_accepted,
        "save_policy": args.save_policy,
    }
    (out_root / "batch_results.json").write_text(json.dumps(results, indent=2, default=str))
    (out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Collection log dir: {out_root}")
    print(f"Segments dir: {segments_root}")


if __name__ == "__main__":
    main()
