#!/usr/bin/env bash
#SBATCH --job-name=struct-rand-probe
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Measures actual structured_random generation throughput AND failure rate at
# one or more --max-workers concurrency levels, so the full 500k-trajectory
# run can be sized from the real knee of the curve instead of a linear
# extrapolation off a smaller baseline.
#
# An earlier 30/60/120-worker sweep showed per-worker throughput collapsing
# ~9x versus the 8-worker production baseline (2.19 traj/min/worker), with
# total throughput plateauing around ~15-20 traj/min regardless of worker
# count -- a real ceiling, hit well below 30 workers. failure_rate barely
# moved, which means whatever is throttling this (almost certainly the
# google-genai SDK's own transport-level retry/backoff on 429s, invisible in
# this repo's code) shows up as stalled elapsed time, not failed runs. So the
# default sweep here now brackets 4-24 workers to find where the real knee
# is, instead of the 30-120 range that turned out to already be past it.
#
# Every invocation writes to a fresh timestamped directory per level (never
# resumed) -- reusing a run dir across script runs let an already-complete
# level report "done" in a few seconds and get logged as bogus throughput.
#
# Sized to cost only a few dollars: --num-runs per level scales with worker
# count (TARGET_MINUTES of sustained load), not a flat number, so low worker
# counts aren't wastefully over-requested just to give the highest level
# enough runway. Prints a projected spend before doing any generation and
# requires CONFIRM_SPEND=yes to actually proceed.
#
# Usage:
#   CONFIRM_SPEND=yes sbatch scripts/full_traj_gen/probe_structured_random_rate.sh
#   CONFIRM_SPEND=yes WORKER_COUNTS="4 8 12 16 24" TARGET_MINUTES=1.5 \
#     sbatch scripts/full_traj_gen/probe_structured_random_rate.sh

set -euo pipefail

mkdir -p slurm_logs

repo_root="${ROBOCASA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_root="$(cd "${repo_root}" && pwd)"
if [[ ! -f "${repo_root}/data_generation/task_level/generation/raw/cli.py" ]]; then
  echo "Could not resolve RoboCasa repository root from: ${repo_root}" >&2
  echo "Set ROBOCASA_REPO_ROOT to the repo root." >&2
  exit 1
fi
default_python_bin="${repo_root}/.venv_d/bin/python"
if [[ ! -x "${default_python_bin}" ]]; then
  default_python_bin="${repo_root}/.venv/bin/python"
fi
if [[ ! -x "${default_python_bin}" ]]; then
  echo "No .venv_d or .venv found under ${repo_root}; set PYTHON_BIN explicitly." >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-${default_python_bin}}"
probe_root="${PROBE_ROOT:-${repo_root}/data_generation/task_level/data_probe/structured_random_rate}"
model="${MODEL:-gemini-3-flash-preview}"
sdk="${SDK:-${ROBOCASA_RAW_SDK:-google-genai}}"
location="${GOOGLE_CLOUD_LOCATION:-global}"
temperature="${TEMPERATURE:-0.6}"
thinking_level="${THINKING_LEVEL:-low}"
max_retries="${MAX_RETRIES:-5}"
random_start_location="${RANDOM_START_LOCATION:-true}"

# One known-good task (no open generation bugs), kept off the full-dataset
# output path so this never touches production structured_random data.
read -r -a probe_tasks <<< "${PROBE_TASKS:-HotDogSetup}"
read -r -a worker_counts <<< "${WORKER_COUNTS:-4 8 12 16 24 32 36 40 45}"

# --num-runs per level = workers * measured_traj_per_min_per_worker *
# TARGET_MINUTES, so every level sustains roughly the same wall-clock load
# instead of a flat count that over-requests at low worker counts. The
# per-worker rate is the empirical baseline from the 8-worker production run
# (struct-rand-gen-18419675: 1531 traj / 87.6 min / 8 workers = 2.19/min).
baseline_traj_per_min_per_worker="2.19"
target_minutes="${TARGET_MINUTES:-1.0}"
# Empirical average cost per saved trajectory from that same baseline run
# (cost_summary.total_cost_usd / num_trajectories = 8.7171 / 1531).
avg_cost_per_trajectory_usd="0.0057"

num_tasks="${#probe_tasks[@]}"
total_planned_runs=0
declare -A num_runs_by_workers
for workers in "${worker_counts[@]}"; do
  runs_per_task="$(python3 -c "
import math
print(max(math.ceil(${workers} * ${baseline_traj_per_min_per_worker} * ${target_minutes}), 1))
")"
  num_runs_by_workers["${workers}"]="${runs_per_task}"
  total_planned_runs=$(( total_planned_runs + runs_per_task * num_tasks ))
done

projected_cost_usd="$(python3 -c "print(f'{${total_planned_runs} * ${avg_cost_per_trajectory_usd}:.2f}')")"

printf 'Probing structured_random throughput for worker counts: %s\n' "${worker_counts[*]}"
printf 'Tasks: %s | target load per level: %s min | model: %s\n' "${probe_tasks[*]}" "${target_minutes}" "${model}"
for workers in "${worker_counts[@]}"; do
  printf '  %3s workers -> %s runs/task\n' "${workers}" "${num_runs_by_workers[${workers}]}"
done
printf 'Projected total: %s requested runs, ~$%s USD (at $%s/trajectory observed baseline rate)\n' \
  "${total_planned_runs}" "${projected_cost_usd}" "${avg_cost_per_trajectory_usd}"

if [[ "${CONFIRM_SPEND:-}" != "yes" ]]; then
  echo "Set CONFIRM_SPEND=yes to actually run this probe and spend the amount above." >&2
  exit 1
fi

mkdir -p "${probe_root}"

results_path="${probe_root}/results.tsv"
if [[ ! -f "${results_path}" ]]; then
  printf 'workers\telapsed_seconds\tnum_trajectories\ttotal_requested_runs\tfailed_runs\tfailure_rate\ttraj_per_min\ttraj_per_min_per_worker\ttotal_cost_usd\n' > "${results_path}"
fi

probe_run_stamp="$(date +%Y%m%dT%H%M%S)"

for workers in "${worker_counts[@]}"; do
  num_runs_probe="${num_runs_by_workers[${workers}]}"
  # Always a fresh directory per invocation (never resumed): reusing a run
  # dir across script runs meant an already-complete level would just report
  # "already done" in a few seconds and get logged as bogus throughput.
  # Probes are cheap enough now that re-measuring cleanly beats reusing.
  run_dir="${probe_root}/w${workers}_${probe_run_stamp}"
  summary_path="${run_dir}/summary.json"

  args=(
    -m data_generation.task_level.generation.raw.cli
    --tasks "${probe_tasks[@]}"
    --num-runs "${num_runs_probe}"
    --random-start-location "${random_start_location}"
    --sampling structured_random
    --temperature "${temperature}"
    --model "${model}"
    --sdk "${sdk}"
    --location "${location}"
    --thinking-level "${thinking_level}"
    --max-workers "${workers}"
    --max-retries "${max_retries}"
    --enable-validation
  )

  mkdir -p "${run_dir}"
  printf 'Starting probe at %s workers in: %s\n' "${workers}" "${run_dir}"
  args+=(--summary-path "${summary_path}")

  printf 'Running:'
  printf ' %q' "${python_bin}" "${args[@]}"
  printf '\n'

  start_epoch="$(date +%s)"
  set +e
  (cd "${repo_root}" && "${python_bin}" "${args[@]}")
  cli_exit_code=$?
  set -e
  end_epoch="$(date +%s)"
  elapsed_seconds=$(( end_epoch - start_epoch ))

  if [[ ${cli_exit_code} -ne 0 ]]; then
    printf 'Probe run at %s workers exited %s (check for rate-limit errors in %s)\n' \
      "${workers}" "${cli_exit_code}" "${run_dir}" >&2
  fi

  "${python_bin}" - "${summary_path}" "${workers}" "${elapsed_seconds}" "${results_path}" <<'PY'
import json
import sys

summary_path, workers, elapsed_seconds, results_path = sys.argv[1:]
elapsed_seconds = int(elapsed_seconds)
workers = int(workers)

with open(summary_path) as f:
    summary = json.load(f)

num_trajectories = summary["num_trajectories"]
# Single-task runs write "num_runs"; multi-task summaries write
# "total_requested_runs" instead. Support both schemas.
total_requested_runs = summary.get("total_requested_runs", summary.get("num_runs"))
total_cost_usd = summary["cost_summary"]["total_cost_usd"]
failed_runs = max(total_requested_runs - num_trajectories, 0)
failure_rate = failed_runs / total_requested_runs if total_requested_runs else 0.0
traj_per_min = (num_trajectories / elapsed_seconds) * 60 if elapsed_seconds else 0.0
traj_per_min_per_worker = traj_per_min / workers if workers else 0.0

with open(results_path, "a") as f:
    f.write(
        f"{workers}\t{elapsed_seconds}\t{num_trajectories}\t{total_requested_runs}\t"
        f"{failed_runs}\t{failure_rate:.3f}\t"
        f"{traj_per_min:.2f}\t{traj_per_min_per_worker:.3f}\t{total_cost_usd:.4f}\n"
    )
PY
done

printf '\nProbe results written to: %s\n' "${results_path}"
column -t "${results_path}"
