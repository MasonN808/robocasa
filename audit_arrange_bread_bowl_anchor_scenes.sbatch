#!/usr/bin/env bash
#SBATCH --job-name=bread-anchor-audit
#SBATCH --partition=rtx-batch
#SBATCH --array=0-35%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache

python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
dataset_root=${DATASET_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3}
output_root=${OUTPUT_ROOT:-training/bc_task_vlm/eval_runs/arrange_bread_bowl_anchor_scene_audit}

# These are all non-excluded scene pairs used by the pipeline's preferred
# RoboCasa scene inventory for ArrangeBreadBowl, plus the two explicitly
# supported fallback pairs. Each is checked at all three normalization seeds.
scene_pairs=(
  "11 14" "11 28" "11 34" "11 46" "11 58"
  "15 14" "15 28" "15 34" "15 46" "15 58"
  "42 34" "56 34"
)
seeds=(42 99 7)

pair_index=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
read -r layout style <<< "${scene_pairs[$pair_index]}"
seed=${seeds[$seed_index]}
output_dir="${output_root}/layout_${layout}_style_${style}_seed_${seed}"

exec "${python_bin}" -m training.bc_task_vlm.audit_access_blocking \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --tasks arrange_bread_bowl \
  --layout "${layout}" \
  --style "${style}" \
  --seed "${seed}" \
  --gl-backend egl
