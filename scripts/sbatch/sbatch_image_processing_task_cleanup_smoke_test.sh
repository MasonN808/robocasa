#!/usr/bin/env bash
#SBATCH --job-name=img-cleanup-smoke
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
Usage: sbatch scripts/sbatch/sbatch_image_processing_task_cleanup_smoke_test.sh <run_timestamp> [--tasks TASK] [--indices I...]

Smoke test for the full task cleanup pipeline on 1 GPU:
  1. Render images for a small number of trajectories
  2. Push the task dataset to HuggingFace
  3. Run local cleanup (keep 25% of trajs)

Example:
  sbatch scripts/sbatch/sbatch_image_processing_task_cleanup_smoke_test.sh 20260430T030150Z_full_run --tasks add_lemon_to_fish --indices 0 1 2 3 4

Optional environment overrides:
  HF_REPO_PREFIX=DorianAtSchool/robocasa_20260430T030150Z_full_run
  OUTPUT_TIMESTAMP=20260430T030150Z_full_run_smoke
  KEEP_FRACTION=0.25

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

keep_fraction="${KEEP_FRACTION:-0.25}"
data_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/data_generation/task_level/data"
output_dir="$data_root/image/${OUTPUT_TIMESTAMP:-$run_timestamp}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-DorianAtSchool/robocasa_${run_timestamp}}"

# Extract the --tasks value from forwarded args so we know which dir to inspect.
task_name=""
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--tasks" ]]; then
    next=$((i + 1))
    task_name="${!next:-}"
    break
  fi
done

echo "=== Smoke test: generate ==="
start_time=$(date +%s)
summary_path_args=()
if [[ -n "$task_name" ]]; then
  summary_path_args=(--summary-path "$output_dir/$task_name/sweep_summary.json")
fi
bash "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/generate_and_insert_images.sh" "$run_timestamp" \
  --workers 15 \
  --gpu-ids 0 \
  --procs-per-gpu 15 \
  --gl-backend egl \
  "${summary_path_args[@]}" \
  "$@"
end_time=$(date +%s)
elapsed=$((end_time - start_time))

if [[ -n "$task_name" ]]; then
  task_image_dir="$output_dir/$task_name"
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
    echo "Timing: ${elapsed}s. No traj_* directories found."
  fi

  echo "=== Smoke test: push to HuggingFace ==="
  python "/work/hdd/bgjs/dbenhamougoldfajn/robocasa/scripts/push_sweep_to_hub.py" \
    --sweep-dir "$task_image_dir" \
    --repo-id "${HF_REPO_PREFIX}_${task_name}"

  echo "=== Smoke test: cleanup (keep_fraction=${keep_fraction}) ==="
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
    for ((i = keep_count; i < total_dirs; i++)); do
      rm -rf "$task_image_dir/${traj_dirs[$i]}"
    done
  fi
else
  echo "Timing: ${elapsed}s. Pass --tasks to run push+cleanup steps."
fi

echo "=== Smoke test complete ==="
