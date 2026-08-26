#!/usr/bin/env bash
#SBATCH --job-name=scene-init-diag
#SBATCH --partition=b200-batch,rtx-batch
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/robocasa_numba_${SLURM_JOB_ID}
tasks=(GarnishCake MeatSkewerAssembly)
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.diagnose_task_scene_initialization \
  --task "${tasks[$SLURM_ARRAY_TASK_ID]}" --layout 15 --style 14 --seed 42
