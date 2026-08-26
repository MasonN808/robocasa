#!/usr/bin/env bash
#SBATCH --job-name=launch-reg43-s3
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

manifest=training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
ids=()
for scale in 30 60 90; do
  run_dir=$PWD/training/bc_task_vlm/runs/qwen3vl8b-tick150-s${scale}-noreason-train43-held10-ep1-halfckpt-lr1e4-wd001-noindex-promptv9
  id=$(submit_with_qos_retry --job-name=eval-reg43-s${scale} --export=ALL,RUN_DIR=${run_dir},RUN_PREFIX=sft_s${scale}_noreason_train43_held10_lr1e4_wd001,MODEL=Qwen/Qwen3-VL-8B-Instruct,MANIFEST=${manifest} eval_half_full_run.sbatch)
  ids+=("${id}")
done
dependency=$(IFS=:; echo "${ids[*]}")
next=$(submit_with_qos_retry --dependency=afterany:${dependency} launch_regularized_43_10_scale_stage4.sbatch)
echo "regularized_evals_30_60_90=${ids[*]} next=${next}"
