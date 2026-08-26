#!/usr/bin/env bash
#SBATCH --job-name=verify-cascade-824
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
"${python_bin}" -m training.bc_task_vlm.verify_cascade_canary \
  --manifest training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json \
  --canary-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_all_config_cascade_canary \
  --output training/bc_task_vlm/reports/new_access_state_generation_preflight/all_config_cascade_canary_verification.json
