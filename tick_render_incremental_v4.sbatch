#!/usr/bin/env bash
#SBATCH --job-name=tick-render-incr-v4
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional_image_affected
output=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v4_transactional
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mapfile -t all < <(find "${input}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
tasks=(); for i in "${!all[@]}"; do (( i % 8 == SLURM_ARRAY_TASK_ID )) && tasks+=("${all[i]}"); done
if (( ${#tasks[@]} )); then
  "${py}" scripts/sweep_trajectories.py --input-dir "${input}" --output-dir "${output}" \
    --tasks "${tasks[@]}" --step-order concurrent --gpu-ids 0 --gl-backend egl
fi
for task in "${tasks[@]}"; do
  [[ "$(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name trajectory_execution_metadata.json -type f | wc -l)" -eq 100 ]]
  [[ "$(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name plan.json -type f | wc -l)" -eq 100 ]]
done
