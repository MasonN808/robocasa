#!/usr/bin/env bash
#SBATCH --job-name=render-tick150-v1-smoke
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
image_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_image
smoke_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_render_smoke
cd "${repo}"
export PYTHONPATH="${repo}" PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec "${python_bin}" scripts/sweep_trajectories.py \
  --input-dir "${image_root}" --output-dir "${smoke_root}" \
  --indices 0 --workers 4 --gl-backend egl --gpu-ids 0 \
  --scene-compatibility-cache training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json \
  --scene-sampling-seed 20260819
