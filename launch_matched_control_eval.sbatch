#!/usr/bin/env bash
#SBATCH --job-name=launch-control-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
: "${RUN_DIR:?}" "${RUN_PREFIX:?}" "${MODEL:?}"
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
manifest=$PWD/training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
for attempt in $(seq 1 60); do
  if output=$(sbatch --parsable --job-name="eval-${RUN_PREFIX}" --export="ALL,RUN_DIR=${RUN_DIR},RUN_PREFIX=${RUN_PREFIX},MODEL=${MODEL},MANIFEST=${manifest}" eval_one_epoch_run.sbatch 2>&1); then
    echo "submitted_eval=${output%%;*}"
    exit 0
  fi
  echo "eval submission attempt ${attempt} failed: ${output}" >&2
  sleep 60
done
echo "Could not submit evaluation after 60 attempts" >&2
exit 1
