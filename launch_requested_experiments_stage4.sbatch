#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
training=$(sbatch --parsable --job-name=train-nrs-t43-s150 --export=ALL,SCALE=150 train_tick150_noreason_train43_scale.sbatch); training=${training%%;*}
next=$(sbatch --parsable --dependency=afterok:${training} launch_requested_experiments_stage5.sbatch); next=${next%%;*}
echo "noreason150=${training} next=${next}"
