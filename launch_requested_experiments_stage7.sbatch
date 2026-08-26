#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage7
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
manifest=training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
for scale in 30 60; do
  run_dir=$PWD/training/bc_task_vlm/runs/qwen3vl8b-thinking-tick150-s${scale}-rationale-train43-held10-ep1-halfckpt-noindex-promptv9
  sbatch --job-name=eval-thinkrat-t43-s${scale} --export=ALL,RUN_DIR=${run_dir},RUN_PREFIX=thinking_rationale_s${scale}_train43_held10,MODEL=Qwen/Qwen3-VL-8B-Thinking,MANIFEST=${manifest} eval_half_full_run.sbatch
done
