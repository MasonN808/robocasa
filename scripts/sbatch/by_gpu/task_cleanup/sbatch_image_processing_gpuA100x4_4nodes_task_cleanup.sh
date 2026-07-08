#!/usr/bin/env bash
#SBATCH --job-name=image-processing-clean
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougol@umass.edu
#SBATCH --time=24:00:00
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
Usage: sbatch scripts/sbatch/sbatch_image_processing_gpuA100x4_4nodes_task_cleanup.sh <run_timestamp> [sweep_cli_args...]

Runs task-level image generation across 4 Delta gpuA100x4 nodes.
Processes one task at a time, pushes each task to HuggingFace, then
removes ~75% of local trajectories to save space.

Example:
  sbatch scripts/sbatch/sbatch_image_processing_gpuA100x4_4nodes_task_cleanup.sh 20260401T000000Z

Optional environment overrides:
  WORKERS_PER_GPU=8
  GPUS_PER_NODE=4
  KEEP_FRACTION=0.25
  HF_REPO_PREFIX=DorianAtSchool/robocasa_20260401T000000Z
  HF_TOKEN=hf_...
  OUTPUT_TIMESTAMP=20260401T000000Z_smoke

Additional args are forwarded to:
  bash scripts/generate_and_insert_images.sh <run_timestamp> ...
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
data_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/data_generation/task_level/data"
output_dir="$data_root/image/${OUTPUT_TIMESTAMP:-$run_timestamp}"
tasks_dir="$data_root/pre_image/$run_timestamp"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-DorianAtSchool/robocasa_${run_timestamp}}"

if [[ ! -d "$tasks_dir" ]]; then
  echo "Timestamp directory not found under pre_image: $tasks_dir" >&2
  exit 1
fi

node_tasks="${SLURM_NTASKS:-${SLURM_NNODES:-0}}"
if ! [[ "$node_tasks" =~ ^[1-9][0-9]*$ ]]; then
  echo "Unable to resolve the allocated Slurm task count." >&2
  exit 1
fi

gpu_ids_csv="$(IFS=,; echo "${gpu_ids[*]}")"
procs_per_gpu_csv="$(IFS=,; echo "${procs_per_gpu[*]}")"

mapfile -t tasks < <(find "$tasks_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

echo "Found ${#tasks[@]} tasks to process for timestamp $run_timestamp"

index=0
for task in "${tasks[@]}"; do
  index=$((index + 1))
  echo "=================================="
  echo "Task ${index}/${#tasks[@]}: $task"
  echo "=================================="

  task_image_dir="$output_dir/$task"
  task_shard_dir="$output_dir/.slurm_shards/$task"
  mkdir -p "$task_shard_dir"

  start_time=$(date +%s)
  if ! srun --ntasks="$node_tasks" --ntasks-per-node=1 bash -lc '
    set -euo pipefail
    repo_root="$1"
    run_timestamp="$2"
    task_name="$3"
    task_shard_dir="$4"
    workers="$5"
    gpu_ids_csv="$6"
    procs_per_gpu_csv="$7"
    shift 7

    IFS="," read -r -a gpu_ids <<< "$gpu_ids_csv"
    IFS="," read -r -a procs_per_gpu <<< "$procs_per_gpu_csv"

    log_path="$task_shard_dir/shard${SLURM_PROCID}.log"
    summary_path="$task_shard_dir/sweep_summary_shard${SLURM_PROCID}.json"

    bash "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/generate_and_insert_images.sh" "$run_timestamp" \
      --tasks "$task_name" \
      --summary-path "$summary_path" \
      --workers "$workers" \
      --gpu-ids "${gpu_ids[@]}" \
      --procs-per-gpu "${procs_per_gpu[@]}" \
      --gl-backend egl \
      --num-shards "${SLURM_NTASKS}" \
      --shard-index "${SLURM_PROCID}" \
      "$@" \
      >"$log_path" 2>&1
  ' bash "$repo_root" "$run_timestamp" "$task" "$task_shard_dir" "$workers" "$gpu_ids_csv" "$procs_per_gpu_csv" "$@"; then
    echo "One or more shards failed for $task. Inspect $task_shard_dir" >&2
    exit 1
  fi

  summary_paths=()
  for ((rank = 0; rank < node_tasks; rank += 1)); do
    summary_paths+=("$task_shard_dir/sweep_summary_shard${rank}.json")
  done

  python "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/merge_sweep_summaries.py" \
    --output "$task_image_dir/sweep_summary.json" \
    "${summary_paths[@]}"

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
