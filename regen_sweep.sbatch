#!/bin/bash
#SBATCH --job-name=regen-sweep
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-sweep-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-sweep-%j.err

# Phase 4's sweep wrapper never forwards --gl-backend/--gpu-ids, so it falls back
# to osmesa, which this env cannot load. Drive the sweep directly on EGL instead;
# the pre-image stage it depends on already completed.

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

RUN=/work/umass/shlomo_umass/dbenhamougol_umass/regen_parent_fixture/run1
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

"$PY" scripts/sweep_trajectories.py \
  --input-dir "$RUN/phase4/pre_image" \
  --output-dir "$RUN/phase4/sweeps" \
  --workers 4 \
  --gl-backend egl \
  --gpu-ids 0

echo "SWEEP DONE"
