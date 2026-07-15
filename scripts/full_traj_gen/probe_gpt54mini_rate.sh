#!/usr/bin/env bash
#SBATCH --job-name=gpt54mini-probe
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Probes structured_random generation throughput on GPT-5.4-Mini via Azure
# OpenAI, as an alternative to gemini-3-flash-preview -- a separate provider
# with its own quota system, unaffected by Google's current Dynamic Shared
# Quota crunch (see probe_parallelize_tasks_structured_random.sh history).
#
# There is one real prior data point for this exact model/sampling combo:
# data_analysis/model_sampling_comparison/raw/gpt_5_4_mini_azure/
# sampling_methods/structured_random/ (run 2026-05-25, via
# generate_model_sampling_comparison_delta_cpu.sh) -- 1560/1560 trajectories
# (100% success) at --max-workers 4, no --parallelize-tasks, in ~70 minutes
# (22.3 traj/min). That run existed purely to feed the diversity analysis,
# not to find a throughput ceiling, so its scale (30 runs/task) isn't a
# useful default here.
#
# This script's defaults instead target the config we'd actually use to
# reach 500k trajectories: --parallelize-tasks on with --max-workers 16,
# mirroring the pattern that (historically, in April) got gemini-3-flash to
# 604.6 traj/min via the same one-thread-per-task fan-out (cli.py:438). We
# don't yet know if Azure's quota scales the same way under that pattern --
# that's what this probe is for. --num-runs defaults to 100/task (5200
# trajectories total), deliberately larger than the old diversity-analysis
# scale, to give a real read at this concurrency without committing to the
# full run. For reference, reaching 500k across the 52 verified tasks needs
# --num-runs ~9616/task -- scale NUM_RUNS toward that once this config is
# confirmed to hold up.
#
# Cost: this repo's cost_summary pricing table has no entry for azure-openai
# models (cost_summary.total_cost_usd comes back null for gpt-5.4-mini --
# confirmed from the historical run above), so no dollar estimate is printed
# or gated here -- only projected trajectory/token counts. Actual $ cost
# depends on your Azure/OpenAI contract, not this script.
#
# Requires (add to .env, not currently set there):
#   AZURE_OPENAI_ENDPOINT / AI_FOUNDRY_PROJECT_ENDPOINT
#   AZURE_OPENAI_API_KEY  / AI_FOUNDRY_API_KEY
# The script checks these are non-empty (never prints their values) and
# fails fast with a clear message if missing, instead of a deep SDK error.
#
# Usage:
#   CONFIRM_RUN=yes sbatch scripts/full_traj_gen/probe_gpt54mini_rate.sh
#   CONFIRM_RUN=yes NUM_RUNS=300 MAX_WORKERS=32 \
#     sbatch scripts/full_traj_gen/probe_gpt54mini_rate.sh
#   CONFIRM_RUN=yes PARALLELIZE_TASKS=false MAX_WORKERS=4 NUM_RUNS=30 \
#     sbatch scripts/full_traj_gen/probe_gpt54mini_rate.sh  # reproduce the old baseline exactly

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

# Load .env the same way the CLI does, so env-based endpoint/key checks
# below see whatever is actually configured there.
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${repo_root}/.env"
  set +a
fi

has_endpoint=0
if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" || -n "${AI_FOUNDRY_PROJECT_ENDPOINT:-}" ]]; then
  has_endpoint=1
fi
has_key=0
if [[ -n "${AZURE_OPENAI_API_KEY:-}" || -n "${AI_FOUNDRY_API_KEY:-}" ]]; then
  has_key=1
fi
if [[ "${has_endpoint}" -ne 1 || "${has_key}" -ne 1 ]]; then
  echo "Azure/AI Foundry credentials are not configured -- refusing to start." >&2
  echo "Add to .env:" >&2
  echo "  AZURE_OPENAI_ENDPOINT=<your endpoint>   (or AI_FOUNDRY_PROJECT_ENDPOINT)" >&2
  echo "  AZURE_OPENAI_API_KEY=<your key>         (or AI_FOUNDRY_API_KEY)" >&2
  exit 1
fi

sampling="${SAMPLING:-structured_random}"
probe_root="${PROBE_ROOT:-${repo_root}/data_generation/task_level/data_probe/gpt54mini_${sampling}}"
model="${MODEL:-gpt-5.4-mini}"
sdk="azure-openai"
location="${AZURE_LOCATION:-global}"
temperature="${TEMPERATURE:-1}"
thinking_level="${THINKING_LEVEL:-low}"
max_workers="${MAX_WORKERS:-16}"
max_retries="${MAX_RETRIES:-5}"
random_start_location="${RANDOM_START_LOCATION:-true}"
num_runs="${NUM_RUNS:-100}"
parallelize_tasks="${PARALLELIZE_TASKS:-true}"

num_tasks=52
projected_trajectories=$(( num_tasks * num_runs ))
num_runs_for_500k=$(( (500000 + num_tasks - 1) / num_tasks ))

run_stamp="$(date +%Y%m%dT%H%M%S)"
run_dir="${probe_root}/${run_stamp}"
summary_path="${run_dir}/summary.json"

printf 'Probing structured_random throughput for GPT-5.4-Mini (azure-openai) at a config aimed at the 500k target.\n'
printf 'Tasks: verified (%s) | num-runs/task: %s | max-workers: %s | parallelize-tasks: %s | model: %s\n' \
  "${num_tasks}" "${num_runs}" "${max_workers}" "${parallelize_tasks}" "${model}"
printf 'Projected: %s trajectories requested this probe. Reaching 500k would need --num-runs ~%s/task at this config.\n' \
  "${projected_trajectories}" "${num_runs_for_500k}"
printf 'No $ estimate: azure-openai has no pricing table entry in this repo (cost_summary comes back null); actual cost depends on your Azure contract.\n'

if [[ "${CONFIRM_RUN:-}" != "yes" ]]; then
  echo "Set CONFIRM_RUN=yes to actually run this." >&2
  exit 1
fi

mkdir -p "${run_dir}"

args=(
  -m data_generation.task_level.generation.raw.cli
  --tasks verified
  --num-runs "${num_runs}"
  --random-start-location "${random_start_location}"
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

if [[ "${parallelize_tasks}" == "true" ]]; then
  args+=(--parallelize-tasks)
fi

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
failed_runs = max(total_requested_runs - num_trajectories, 0)
failure_rate = failed_runs / total_requested_runs if total_requested_runs else 0.0
traj_per_min = (num_trajectories / elapsed_seconds) * 60 if elapsed_seconds else 0.0

cost = summary.get("cost_summary", {})
total_tokens = cost.get("total_tokens")
total_cost_usd = cost.get("total_cost_usd")

print(f"elapsed_seconds:     {elapsed_seconds}")
print(f"num_trajectories:    {num_trajectories}")
print(f"total_requested:     {total_requested_runs}")
print(f"failed_runs:         {failed_runs}")
print(f"failure_rate:        {failure_rate:.4f}")
print(f"traj_per_min:        {traj_per_min:.2f}")
print(f"total_tokens:        {total_tokens}")
print(f"total_cost_usd:      {total_cost_usd!r} (null if no pricing table entry for this model)")
PY
