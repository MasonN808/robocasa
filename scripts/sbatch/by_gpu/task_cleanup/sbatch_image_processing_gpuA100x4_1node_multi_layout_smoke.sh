#!/usr/bin/env bash
#SBATCH --job-name=img-multi-layout-smoke
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougol@umass.edu
#SBATCH --time=08:00:00
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
Usage: sbatch scripts/sbatch/by_gpu/task_cleanup/sbatch_image_processing_gpuA100x4_1node_multi_layout_smoke.sh <run_timestamp> [sweep_cli_args...]

Smoke test for the full task cleanup pipeline across all tasks, using multiple
layouts/styles/seeds (3 indices x 3 layouts x 2 styles x 2 seeds = 36 trajs per task).
Writes to a separate output dir and HF repo prefix so real datasets are untouched.

Example:
  sbatch scripts/sbatch/by_gpu/task_cleanup/sbatch_image_processing_gpuA100x4_1node_multi_layout_smoke.sh 20260430T030150Z_full_run

Optional environment overrides:
  WORKERS_PER_GPU=8
  GPUS_PER_NODE=4
  KEEP_FRACTION=0.25
  HF_REPO_PREFIX=DorianAtSchool/robocasa_20260430T030150Z_full_run_multi_layout_smoke
  HF_TOKEN=hf_...
  OUTPUT_TIMESTAMP=20260430T030150Z_full_run_multi_layout_smoke

Additional args are forwarded to generate_and_insert_images.sh and override the
default smoke indices/layouts/styles/seeds.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

run_timestamp="$1"
shift

export MUJOCO_GL="egl"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before running this script}"

workers_per_gpu="${WORKERS_PER_GPU:-8}"
gpus_per_node="${GPUS_PER_NODE:-4}"
keep_fraction="${KEEP_FRACTION:-0.25}"

if ! [[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS_PER_GPU must be a positive integer." >&2
  exit 1
fi
if ! [[ "$gpus_per_node" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer." >&2
  exit 1
fi

workers=$((workers_per_gpu * gpus_per_node))
gpu_ids=()
procs_per_gpu=()
for ((gpu_id = 0; gpu_id < gpus_per_node; gpu_id += 1)); do
  gpu_ids+=("$gpu_id")
  procs_per_gpu+=("$workers_per_gpu")
done

data_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/data_generation/task_level/data"
output_dir="$data_root/image/${OUTPUT_TIMESTAMP:-${run_timestamp}_multi_layout_smoke}"
tasks_dir="$data_root/pre_image/$run_timestamp"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-DorianAtSchool/robocasa_${run_timestamp}_multi_layout_smoke}"

if [[ ! -d "$tasks_dir" ]]; then
  echo "Timestamp directory not found under pre_image: $tasks_dir" >&2
  exit 1
fi

mapfile -t tasks < <(find "$tasks_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

echo "Found ${#tasks[@]} tasks to process for timestamp $run_timestamp"
echo "Output dir: $output_dir"
echo "HF prefix:  $HF_REPO_PREFIX"

index=0
for task in "${tasks[@]}"; do
  index=$((index + 1))
  echo "=================================="
  echo "Task ${index}/${#tasks[@]}: $task"
  echo "=================================="

  task_image_dir="$output_dir/$task"
  task_summary_path="$task_image_dir/sweep_summary.json"

  start_time=$(date +%s)
  bash "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/generate_and_insert_images.sh" "$run_timestamp" \
    --tasks "$task" \
    --summary-path "$task_summary_path" \
    --workers "$workers" \
    --gpu-ids "${gpu_ids[@]}" \
    --procs-per-gpu "${procs_per_gpu[@]}" \
    --gl-backend egl \
    --indices 0 1 2 \
    --layouts 11 42 56 \
    --styles 34 42 \
    --seeds 42 99 \
    "$@"
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))

  traj_count=$(find "$task_image_dir" -mindepth 1 -maxdepth 1 -type d -name "traj_*" | wc -l)
  if [[ "$traj_count" -gt 0 ]]; then
    rate=$(python - <<EOF
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
    echo "Timing: ${elapsed}s for ${traj_count} trajs (${per_traj}s per traj)."
    echo "Rate: ${rate} traj/s."
  else
    echo "Timing: ${elapsed}s. No traj_* directories found for rate estimate."
  fi

  python "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/push_sweep_to_hub.py" \
    --sweep-dir "$task_image_dir" \
    --repo-id "${HF_REPO_PREFIX}_${task}"

  mapfile -t traj_dirs < <(find "$task_image_dir" -mindepth 1 -maxdepth 1 -type d -name "traj_*" -printf "%f\n" | sort)
  total_dirs=${#traj_dirs[@]}
  if [[ $total_dirs -gt 0 ]]; then
    keep_count=$(python - <<EOF
import math
fraction = float("$keep_fraction")
print(max(1, math.ceil($total_dirs * fraction)))
EOF
)
    remove_count=$((total_dirs - keep_count))
    echo "Cleanup: keeping $keep_count of $total_dirs trajs; removing $remove_count."
    for ((i=keep_count; i<total_dirs; i++)); do
      rm -rf "$task_image_dir/${traj_dirs[$i]}"
    done
  fi

done

echo "All tasks rendered, pushed, and partially cleaned."
