#!/usr/bin/env bash
#SBATCH --job-name=launch-nrs-t43-evals
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
run_dir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration/training/bc_task_vlm/runs/qwen3vl8b-tick150-s30-noreason-train43-held10-ep1-noindex-promptv9
adapter=$(find "${run_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -t- -k2,2n | tail -n 1)
test -n "${adapter}"
adapter="${run_dir}/${adapter}"
common=ALL,MANIFEST=training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json,DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered,SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json,EPISODES_PER_TASK=10,EVALUATION_SEED=20260817,SCENE_SAMPLING_SEED=20260819,WORKERS=4,MODEL=Qwen/Qwen3-VL-8B-Instruct,ADAPTER=${adapter},COMMUNICATION_MODE=full,RUN_NAME=sft_s30_noreason_train43_held10_ep1_fixed10_promptv9
train_eval=$(sbatch --parsable --job-name=nrs-t43-train-eval --export="${common},COHORT_SPLIT=train_task_types" fixed_live_sim_open_weight_vllm.sbatch)
held_eval=$(sbatch --parsable --job-name=nrs-h10-held-eval --export="${common},COHORT_SPLIT=heldout_task_types" fixed_live_sim_open_weight_vllm.sbatch)
echo "adapter=${adapter} train_eval=${train_eval%%;*} held_eval=${held_eval%%;*}"
