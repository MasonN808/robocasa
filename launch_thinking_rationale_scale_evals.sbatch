#!/usr/bin/env bash
#SBATCH --job-name=launch-thinkrat-evals
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

: "${SCALE:?Set SCALE}"
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
run_dir=${repo}/training/bc_task_vlm/runs/qwen3vl8b-thinking-tick150-s${SCALE}-rationale-train43-held10-ep1-halfckpt-lr1e4-wd001-noindex-promptv9
cd "${repo}"
mapfile -t checkpoints < <(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -t- -k2,2n)
[[ "${#checkpoints[@]}" -eq 2 ]] || { echo "Expected two checkpoints in ${run_dir}, found ${#checkpoints[@]}" >&2; exit 1; }
for index in 0 1; do
  checkpoint=${checkpoints[$index]}
  epoch_label=$([[ "${index}" -eq 0 ]] && echo 0p5 || echo 1p0)
  for split in train_task_types heldout_task_types; do
    sbatch --parsable --job-name="thinkrat-s${SCALE}-${epoch_label}-${split%%_*}" --time=12:00:00 \
      --export=ALL,MANIFEST=training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json,COHORT_SPLIT=${split},RUN_NAME=thinking_rationale_s${SCALE}_train43_held10_lr1e4_wd001_ep${epoch_label}_fixed10_promptv9,DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered,MODEL=Qwen/Qwen3-VL-8B-Thinking,ADAPTER=${run_dir}/${checkpoint},SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json,EPISODES_PER_TASK=10,WORKERS=4,ENABLE_THINKING=true,MAX_NEW_TOKENS=2048,VLLM_TEMPERATURE=0.6,VLLM_TOP_P=0.95,VLLM_TOP_K=20,VLLM_REASONING_PARSER=qwen3,VLLM_TOOL_CALL_PARSER=qwen3_xml \
      fixed_live_sim_open_weight_vllm.sbatch
  done
done
