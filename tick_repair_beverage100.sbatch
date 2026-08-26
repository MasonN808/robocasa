#!/usr/bin/env bash
#SBATCH --job-name=repair-beverage100
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
raw=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_beverage_replacement/beverage_organization
processed=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_beverage_replacement_image/beverage_organization
rendered=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render70_structured_random_concurrent
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

cd "${repo}"
export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

"${python_bin}" -m data_generation.task_level.generation.image.cli \
  --dataset "${raw}/summary.json" \
  --output-dataset "${processed}/summary.json" \
  --revalidate --disable-progress --workers 8

[[ -f "${processed}/trajectories/traj_000070.json" ]]
"${python_bin}" scripts/sweep_trajectories.py \
  --input-dir /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_beverage_replacement_image \
  --output-dir "${rendered}" \
  --tasks beverage_organization \
  --indices 70 \
  --step-order concurrent \
  --gpu-ids 0 --gl-backend egl

[[ -f "${rendered}/beverage_organization/traj_000070/original_trajectory.json" ]]
echo "beverage supplement repaired: $(find "${rendered}/beverage_organization" -mindepth 1 -maxdepth 1 -type d -name 'traj_*' | wc -l) rendered trajectories"
