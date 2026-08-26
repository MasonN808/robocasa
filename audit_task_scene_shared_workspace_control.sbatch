#!/usr/bin/env bash
#SBATCH --job-name=shared-control
#SBATCH --partition=b200-batch,rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/robocasa_numba_${SLURM_JOB_ID}
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.audit_task_scene_shared_workspaces \
  --layout 18 --style 34 --seed 42 \
  --output training/bc_task_vlm/eval_runs/task_scene_shared_workspace_control/layout_18_style_34_seed_42.json
