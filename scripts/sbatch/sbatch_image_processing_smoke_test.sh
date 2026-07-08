#!/usr/bin/env bash
#SBATCH --job-name=img-smoke
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/slurm-%j.out

set -euo pipefail

venv_path="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/.venv"
if [[ -d "$venv_path" ]]; then
  source "$venv_path/bin/activate"
else
  echo "Warning: venv not found at $venv_path" >&2
fi

usage() {
  cat <<'EOF'
Usage: sbatch scripts/sbatch/sbatch_image_processing_smoke_test.sh <run_timestamp> [sweep_cli_args...]

Runs a quick smoke test for task-level image generation on 1 GPU.

Example:
  sbatch scripts/sbatch_image_processing_smoke_test.sh 20260430T030150Z_full_run --tasks add_lemon_to_fish --indices 0 1 2 3 4

Additional args are forwarded to:
  bash scripts/generate_and_insert_images.sh <run_timestamp> ...
EOF
}

export MUJOCO_GL="egl"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

export MUJOCO_GL="egl"
run_timestamp="$1"
shift

start_time=$(date +%s)

# Running on just a single GPU (gpu-ids 0) to verify EGL/resources work properly.
bash scripts/generate_and_insert_images.sh "$run_timestamp" \
  --workers 15 \
  --gpu-ids 0 \
  --procs-per-gpu 15 \
  --gl-backend egl \
  "$@"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

task_name=""
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--tasks" ]]; then
    next_index=$((i + 1))
    task_name="${!next_index}"
    break
  fi
done

if [[ -n "$task_name" ]]; then
  task_dir="data_generation/task_level/data/image/${run_timestamp}/${task_name}"
  if [[ -d "$task_dir" ]]; then
    traj_count=$(find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "traj_*" | wc -l)
    if [[ "$traj_count" -gt 0 ]]; then
      rate=$(python - <<EOF
import math
elapsed = ${elapsed}
count = ${traj_count}
rate = count / elapsed if elapsed > 0 else 0.0
print(f"{rate:.4f}")
EOF
)
      per_traj=$(python - <<EOF
elapsed = ${elapsed}
count = ${traj_count}
sec = elapsed / count if count > 0 else 0.0
print(f"{sec:.2f}")
EOF
)
      echo "Timing: ${elapsed}s total for ${traj_count} trajs."
      echo "Rate: ${rate} traj/s (${per_traj}s per traj)."
    else
      echo "Timing: ${elapsed}s total. No traj_* directories found for rate estimate."
    fi
  else
    echo "Timing: ${elapsed}s total. No task output dir found for rate estimate."
  fi
else
  echo "Timing: ${elapsed}s total. Provide --tasks to estimate per-traj rate."
fi
