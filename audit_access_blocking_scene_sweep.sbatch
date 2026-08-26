#!/usr/bin/env bash
#SBATCH --job-name=access-sweep
#SBATCH --partition=rtx-batch
#SBATCH --array=0-149%12
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
dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
output_root=training/bc_task_vlm/eval_runs/access_blocking_scene_sweep_v2_task_supported

# RoboCasa's 25 standard 5x5 scene pairs. Every pair is tested at three seeds
# and two exact repeats: 75 unique scenes, 150 independent shards. The audit
# itself now applies each task class's EXCLUDE_LAYOUTS / EXCLUDE_STYLES before
# initialization, exactly as Kitchen does when sampling scenes.
pairs=(
  "11 14" "11 28" "11 34" "11 46" "11 58"
  "15 14" "15 28" "15 34" "15 46" "15 58"
  "18 14" "18 28" "18 34" "18 46" "18 58"
  "40 14" "40 28" "40 34" "40 46" "40 58"
  "50 14" "50 28" "50 34" "50 46" "50 58"
)
seeds=(42 99 7)
trial=$((SLURM_ARRAY_TASK_ID % 2))
unique_index=$((SLURM_ARRAY_TASK_ID / 2))
pair_index=$((unique_index / 3))
seed_index=$((unique_index % 3))
read -r layout style <<< "${pairs[$pair_index]}"
seed=${seeds[$seed_index]}
out="${output_root}/layout_${layout}_style_${style}_seed_${seed}_trial_${trial}"

exec "${python_bin}" -m training.bc_task_vlm.audit_access_blocking \
  --dataset-root "${dataset_root}" --output-dir "${out}" \
  --layout "${layout}" --style "${style}" --seed "${seed}" --trial "${trial}" \
  --no-save-views --gl-backend egl
