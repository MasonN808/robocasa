#!/usr/bin/env bash
#SBATCH --job-name=struct-rand-parallelize-probe
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Confirms that the throughput/reliability seen in the proven
# scripts/sbatch/sbatch_trajectory_generation_full.sh run (155,854/156,000
# trajectories, 4h18m, 604.6 traj/min, --parallelize-tasks --max-workers 16,
# --sampling verbalized) also holds for --sampling structured_random.
#
# Our earlier single-task probe (probe_structured_random_rate.sh) only ever
# varied --max-workers inside one task's ThreadPoolExecutor and found
# throughput collapsing past ~8 workers with a 429 at just 12. That result
# doesn't generalize here: --parallelize-tasks spawns one thread PER TASK
# (ThreadPoolExecutor(max_workers=len(indexed_tasks)) at cli.py:438), each
# running its own --max-workers pool, so the historical run's real
# concurrency was up to 52 * 16 = 832 -- and it finished with a 0.09%
# failure rate. The leading theory: single-task bursts fire all N workers'
# first request at once (thundering herd), while task-parallel submission is
# naturally staggered by real generation latency, avoiding short-term burst
# throttling.
#
# --num-runs defaults to 30 to exactly reproduce the task/run mix of the
# very first structured_random baseline (52 tasks x 30 runs = 1560
# trajectories, $8.72 actual cost at --max-workers 8, no --parallelize-tasks,
# 87.6 min) -- so this is a clean apples-to-apples comparison on the same
# workload, varying only the concurrency strategy.
#
# gemini-3-flash-preview is only reachable via the "global" endpoint for this
# project -- confirmed directly with single minimal generate_content calls
# per region (not a paid batch): every named Vertex AI region (us-central1,
# us-east1/4/5, us-south1, us-west1/4) returns 404 "your project does not
# have access to it", while "global" succeeds. So --location is fixed to
# GOOGLE_CLOUD_LOCATION (default "global") here, not swept.
#
# Usage:
#   CONFIRM_SPEND=yes sbatch scripts/full_traj_gen/probe_parallelize_tasks_structured_random.sh
#   CONFIRM_SPEND=yes NUM_RUNS=100 sbatch scripts/full_traj_gen/probe_parallelize_tasks_structured_random.sh

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
sampling="${SAMPLING:-structured_random}"
probe_root="${PROBE_ROOT:-${repo_root}/data_generation/task_level/data_probe/${sampling}_parallelize}"
model="${MODEL:-gemini-3-flash-preview}"
sdk="${SDK:-${ROBOCASA_RAW_SDK:-google-genai}}"
location="${GOOGLE_CLOUD_LOCATION:-global}"
temperature="${TEMPERATURE:-0.6}"
thinking_level="${THINKING_LEVEL:-low}"
max_workers="${MAX_WORKERS:-16}"
max_retries="${MAX_RETRIES:-5}"
random_start_location="${RANDOM_START_LOCATION:-true}"
num_runs="${NUM_RUNS:-30}"
avg_cost_per_trajectory_usd="0.0057"

num_tasks=52
projected_trajectories=$(( num_tasks * num_runs ))
projected_cost_usd="$(python3 -c "print(f'{${projected_trajectories} * ${avg_cost_per_trajectory_usd}:.2f}')")"

run_stamp="$(date +%Y%m%dT%H%M%S)"
run_dir="${probe_root}/${run_stamp}"
summary_path="${run_dir}/summary.json"

printf 'Confirming --parallelize-tasks throughput for %s.\n' "${sampling}"
printf 'Tasks: verified (%s) | num-runs/task: %s | max-workers/task: %s | model: %s | location: %s\n' \
  "${num_tasks}" "${num_runs}" "${max_workers}" "${model}" "${location}"
printf 'Projected: %s trajectories, ~$%s USD (at $%s/trajectory observed baseline rate)\n' \
  "${projected_trajectories}" "${projected_cost_usd}" "${avg_cost_per_trajectory_usd}"

if [[ "${CONFIRM_SPEND:-}" != "yes" ]]; then
  echo "Set CONFIRM_SPEND=yes to actually run this and spend the amount above." >&2
  exit 1
fi

mkdir -p "${run_dir}"

args=(
  -m data_generation.task_level.generation.raw.cli
  --tasks verified
  --num-runs "${num_runs}"
  --random-start-location "${random_start_location}"
  --parallelize-tasks
  --sampling "${sampling}"
  --temperature "${temperature}"
  --model "${model}"
  --sdk "${sdk}"
  --location "${location}"
  --thinking-level "${thinking_level}"
  --max-workers "${max_workers}"
  --max-retries "${max_retries}"
  --enable-validation
  --summary-path "${summary_path}"
)

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
  printf 'Run exited %s (check %s for details -- may include per-task failures)\n' \
    "${cli_exit_code}" "${run_dir}" >&2
fi

"${python_bin}" - "${summary_path}" "${elapsed_seconds}" <<'PY'
import json
import sys

summary_path, elapsed_seconds = sys.argv[1:]
elapsed_seconds = int(elapsed_seconds)

with open(summary_path) as f:
    summary = json.load(f)

num_trajectories = summary["num_trajectories"]
total_requested_runs = summary.get("total_requested_runs", summary.get("num_runs"))
total_cost_usd = summary["cost_summary"]["total_cost_usd"]
failed_runs = max(total_requested_runs - num_trajectories, 0)
failure_rate = failed_runs / total_requested_runs if total_requested_runs else 0.0
traj_per_min = (num_trajectories / elapsed_seconds) * 60 if elapsed_seconds else 0.0

print(f"elapsed_seconds:     {elapsed_seconds}")
print(f"num_trajectories:    {num_trajectories}")
print(f"total_requested:     {total_requested_runs}")
print(f"failed_runs:         {failed_runs}")
print(f"failure_rate:        {failure_rate:.4f}")
print(f"traj_per_min:        {traj_per_min:.2f}")
print(f"total_cost_usd:      {total_cost_usd:.4f}")
PY
