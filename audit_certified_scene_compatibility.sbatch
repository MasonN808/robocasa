#!/usr/bin/env bash
#SBATCH --job-name=scene-compat-v1
#SBATCH --partition=rtx-batch
#SBATCH --array=0-59%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/robocasa_numba_${SLURM_JOB_ID}
layouts=(11 15 40 50); styles=(14 28 34 46 58); seeds=(42 99 7)
li=$((SLURM_ARRAY_TASK_ID/15)); rem=$((SLURM_ARRAY_TASK_ID%15)); si=$((rem/3)); di=$((rem%3))
layout=${layouts[$li]}; style=${styles[$si]}; seed=${seeds[$di]}
out=training/bc_task_vlm/eval_runs/certified_scene_compatibility_v1/layout_${layout}_style_${style}_seed_${seed}.json
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python -m training.bc_task_vlm.audit_certified_scene_compatibility --layout "$layout" --style "$style" --seed "$seed" --output "$out"
