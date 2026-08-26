#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
manifest=training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
ids=()
for scale in 60 90 120 150; do
  run_dir=$PWD/training/bc_task_vlm/runs/qwen3vl8b-tick150-s${scale}-noreason-train43-held10-ep1-halfckpt-noindex-promptv9
  id=$(sbatch --parsable --job-name=eval-nrs-t43-s${scale} --export=ALL,RUN_DIR=${run_dir},RUN_PREFIX=sft_s${scale}_noreason_train43_held10,MODEL=Qwen/Qwen3-VL-8B-Instruct,MANIFEST=${manifest} eval_half_full_run.sbatch)
  ids+=("${id%%;*}")
done
dependency=$(IFS=:; echo "${ids[*]}")
next=$(sbatch --parsable --dependency=afterany:${dependency} launch_requested_experiments_stage6.sbatch); next=${next%%;*}
echo "noreason_evals=${ids[*]} next=${next}"
