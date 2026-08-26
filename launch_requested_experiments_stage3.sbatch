#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
ids=()
for scale in 60 90 120; do
  id=$(sbatch --parsable --job-name=train-nrs-t43-s${scale} --export=ALL,SCALE=${scale} train_tick150_noreason_train43_scale.sbatch)
  ids+=("${id%%;*}")
done
dependency=$(IFS=:; echo "${ids[*]}")
next=$(sbatch --parsable --dependency=afterok:${dependency} launch_requested_experiments_stage4.sbatch); next=${next%%;*}
echo "noreason_trainings=${ids[*]} next=${next}"
