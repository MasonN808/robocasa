#!/usr/bin/env bash
#SBATCH --job-name=tick-render70sr
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_structured_random_low_image
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render70_structured_random_concurrent
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
shards=8
shard="${SLURM_ARRAY_TASK_ID:-0}"

cd "${repo}"
export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mapfile -t all_tasks < <(find "${input_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
tasks=()
for index in "${!all_tasks[@]}"; do
  if (( index % shards == shard )); then
    tasks+=("${all_tasks[index]}")
  fi
done
echo "render shard=${shard}/${shards} tasks=${#tasks[@]} ${tasks[*]}"
mkdir -p "${output_root}"

"${python_bin}" scripts/sweep_trajectories.py \
  --input-dir "${input_root}" \
  --output-dir "${output_root}" \
  --tasks "${tasks[@]}" \
  --step-order concurrent \
  --gpu-ids 0 --gl-backend egl

echo "render shard=${shard} complete trajectories=$(find "${output_root}" -mindepth 2 -maxdepth 2 -type d -name 'traj_*' | wc -l) images=$(find "${output_root}" -type f -name '*.jpg' | wc -l)"
