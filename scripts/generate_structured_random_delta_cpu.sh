#!/usr/bin/env bash
#SBATCH --job-name=struct-rand-gen
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python}"
data_root="${ROBOCASA_TASK_LEVEL_DATA_ROOT:-$PWD/data_generation/task_level/data_k3}"
method_dir="$data_root/raw/sampling_methods/structured_random"
summary_path="$method_dir/summary.json"

args=(
  -m data_generation.task_level.generation.raw.cli
  --tasks "${TASKS:-verified}"
  --num-runs "${NUM_RUNS:-30}"
  --random-start-location "${RANDOM_START_LOCATION:-true}"
  --sampling structured_random
  --temperature "${TEMPERATURE:-0.6}"
  --model "${MODEL:-gemini-3-flash-preview}"
  --sdk "${SDK:-${ROBOCASA_RAW_SDK:-google-genai}}"
  --location "${GOOGLE_CLOUD_LOCATION:-global}"
  --thinking-level "${THINKING_LEVEL:-low}"
  --max-workers "${MAX_WORKERS:-4}"
  --max-retries "${MAX_RETRIES:-5}"
  --enable-validation
)

if [[ "${PARALLELIZE_TASKS:-false}" == "true" ]]; then
  args+=(--parallelize-tasks)
fi

if [[ -f "$summary_path" ]]; then
  printf 'Resuming structured_random outputs in: %s\n' "$method_dir"
  args+=(--resume "$method_dir")
else
  mkdir -p "$method_dir"
  printf 'Writing structured_random outputs to: %s\n' "$method_dir"
  args+=(--summary-path "$summary_path")
fi

printf 'Running:'
printf ' %q' "$python_bin" "${args[@]}"
printf '\n'

"$python_bin" "${args[@]}"
