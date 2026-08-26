#!/usr/bin/env bash
#SBATCH --job-name=render-prod-cascade-smoke
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec "${python_bin}" scripts/sweep_trajectories.py \
  --input-dir training/bc_task_vlm/eval_runs/production_cascade_smoke_image \
  --output-dir training/bc_task_vlm/eval_runs/production_cascade_smoke_rendered \
  --workers 4 --gl-backend egl --gpu-ids 0 \
  --scene-compatibility-cache training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json \
  --scene-sampling-seed 20260819
