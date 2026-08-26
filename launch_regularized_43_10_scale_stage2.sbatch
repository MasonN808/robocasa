#!/usr/bin/env bash
#SBATCH --job-name=launch-reg43-s2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
source slurm_submit_with_qos_retry.sh

ids=()
for scale in 120 150; do
  id=$(submit_with_qos_retry --job-name=train-reg43-s${scale} --export=ALL,SCALE=${scale} train_tick150_noreason_train43_regularized_scale.sbatch)
  ids+=("${id}")
done
thinking=$(submit_with_qos_retry train_tick150_thinking_rationale_train43_regularized_scale30.sbatch)
ids+=("${thinking}")
dependency=$(IFS=:; echo "${ids[*]}")
next=$(submit_with_qos_retry --dependency=afterok:${dependency} launch_regularized_43_10_scale_stage3.sbatch)
echo "regularized_trainings_120_150=${ids[0]} ${ids[1]} thinking_rationale_30=${thinking} next=${next}"
