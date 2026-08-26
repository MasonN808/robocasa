#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage6
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
ids=()
for scale in 30 60; do
  id=$(sbatch --parsable --job-name=train-thinkrat-t43-s${scale} --export=ALL,SCALE=${scale} train_tick150_thinking_rationale_train43_scale.sbatch)
  ids+=("${id%%;*}")
done
dependency=$(IFS=:; echo "${ids[*]}")
next=$(sbatch --parsable --dependency=afterok:${dependency} launch_requested_experiments_stage7.sbatch); next=${next%%;*}
echo "thinking43_trainings=${ids[*]} next=${next}"
