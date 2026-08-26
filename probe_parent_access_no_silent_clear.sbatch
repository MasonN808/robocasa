#!/usr/bin/env bash
#SBATCH --job-name=parent-no-silent-clear
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DISABLE_SILENT_BLOCKER_CLEARING=1
export NUMBA_CACHE_DIR=/tmp/numba-cache-${SLURM_JOB_ID}
mkdir -p "${NUMBA_CACHE_DIR}"
"${python_bin}" -m training.bc_task_vlm.probe_parent_access_occupied
