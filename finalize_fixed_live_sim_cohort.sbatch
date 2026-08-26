#!/usr/bin/env bash
#SBATCH --job-name=freeze-cohort
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
exec "${PYTHON_BIN}" -m training.bc_task_vlm.fixed_live_sim_cohort \
  --candidate-manifest training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/candidates.json \
  --oracle-results training/bc_task_vlm/eval_runs/fixed_live_sim/oracle_v1/train_task_types/aggregate/live_sim_trajectories.jsonl \
  --oracle-results training/bc_task_vlm/eval_runs/fixed_live_sim/oracle_v1/heldout_task_types/aggregate/live_sim_trajectories.jsonl \
  --output training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/frozen.json
