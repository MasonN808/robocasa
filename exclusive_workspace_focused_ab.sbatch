#!/usr/bin/env bash
#SBATCH --job-name=excl-ws-ab
#SBATCH --partition=rtx-devel
#SBATCH --array=0-149%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
output_root=${OUTPUT_ROOT:-training/bc_task_vlm/eval_runs/exclusive_workspace_directional_ab_v4}

pairs=(
  "11 14" "11 28" "11 34" "11 46" "11 58"
  "15 14" "15 28" "15 34" "15 46" "15 58"
  "18 14" "18 28" "18 34" "18 46" "18 58"
  "40 14" "40 28" "40 34" "40 46" "40 58"
  "50 14" "50 28" "50 34" "50 46" "50 58"
)
seeds=(42 99 7)
arm_index=$((SLURM_ARRAY_TASK_ID % 2))
scene_index=$((SLURM_ARRAY_TASK_ID / 2))
pair_index=$((scene_index / 3))
seed_index=$((scene_index % 3))
read -r layout style <<< "${pairs[$pair_index]}"
seed=${seeds[$seed_index]}
if [[ ${arm_index} -eq 0 ]]; then
  export ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES=0
  arm=baseline
else
  export ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES=1
  arm=reserved
fi
out="${output_root}/${arm}_layout_${layout}_style_${style}_seed_${seed}"

exec "${python_bin}" -m training.bc_task_vlm.audit_access_blocking \
  --dataset-root "${dataset_root}" --output-dir "${out}" \
  --tasks arrange_bread_bowl,sweeten_coffee \
  --layout "${layout}" --style "${style}" --seed "${seed}" \
  --no-save-views --gl-backend egl
