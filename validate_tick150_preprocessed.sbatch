#!/usr/bin/env bash
#SBATCH --job-name=validate-tick150
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python training/bc_task_vlm/validate_tick100_preprocessed.py --artifact /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick100_from150_train47_toolcall_idxnone --expected-train-tasks 47 --trajectories-per-task 100 --validation-trajectories-per-task 10
