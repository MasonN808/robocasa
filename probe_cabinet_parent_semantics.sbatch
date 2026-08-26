#!/usr/bin/env bash
#SBATCH --job-name=cabinet-parent-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba-cache-${SLURM_JOB_ID}
mkdir -p "${NUMBA_CACHE_DIR}"
"${python_bin}" -m training.bc_task_vlm.probe_cabinet_parent_semantics
