#!/usr/bin/env bash
#SBATCH --job-name=steak-pick-fix
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_${SLURM_JOB_ID}
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
"${python_bin}" -m pytest -q tests/test_sim_tool_executor.py \
  -k pickup_from_countertop_appliance_matches_non_front_navigation
exec "${python_bin}" -m training.bc_task_vlm.dagger.probe_serve_steak_pickup \
  --output training/bc_task_vlm/eval_runs/dagger_43_10_s30_iterative_smoke_v2/serve_steak_pickup_fix_probe.json
