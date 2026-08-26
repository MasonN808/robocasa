#!/usr/bin/env bash
#SBATCH --job-name=verify-typed-v2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
canary_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53_all_configs_canary_typed_v2
cd "${repo}"
export NUMBA_CACHE_DIR="/tmp/robocasa_numba_cache_${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

"${python_bin}" -m training.bc_task_vlm.verify_all_configs_canary \
  --manifest training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json \
  --canary-root "${canary_root}" \
  --output training/bc_task_vlm/reports/new_access_state_generation_preflight/all_configs_canary_typed_v2_verification.json
