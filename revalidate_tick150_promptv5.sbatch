#!/usr/bin/env bash
#SBATCH --job-name=revalidate-tick150-promptv5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.revalidate_trajectory_pool \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered \
  --output-dir training/bc_task_vlm/eval_runs/tick150_promptv5_pool_revalidation \
  --workers 24
