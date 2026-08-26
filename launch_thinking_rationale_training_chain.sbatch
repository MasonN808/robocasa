#!/usr/bin/env bash
#SBATCH --job-name=launch-think-rat-train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
smoke=$(sbatch --parsable train_tick150_thinking_rationale_smoke.sbatch)
full=$(sbatch --parsable --dependency="afterok:${smoke}" train_tick150_thinking_rationale_scale30_full.sbatch)
evals=$(sbatch --parsable --dependency="afterok:${full}" launch_thinking_rationale_scale30_evals.sbatch)
printf 'training_smoke=%s\nfull_training=%s\neval_launcher=%s\n' "${smoke}" "${full}" "${evals}"
