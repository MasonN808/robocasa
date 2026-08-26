#!/usr/bin/env bash
#SBATCH --job-name=launch-exp-stage2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
run_dir=$PWD/training/bc_task_vlm/runs/qwen3vl8b-thinking-tick150-s30-rationale-ep1-halfckpt-noindex-promptv9
manifest=training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json
evals=$(sbatch --parsable --export=ALL,RUN_DIR=${run_dir},RUN_PREFIX=qwen3vl8b_thinking_rationale_s30,MODEL=Qwen/Qwen3-VL-8B-Thinking,MANIFEST=${manifest} eval_half_full_run.sbatch); evals=${evals%%;*}
next=$(sbatch --parsable --dependency=afterany:${evals} launch_requested_experiments_stage3.sbatch); next=${next%%;*}
echo "thinking47_evals=${evals} next=${next}"
