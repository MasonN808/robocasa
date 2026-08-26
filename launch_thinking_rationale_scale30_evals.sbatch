#!/usr/bin/env bash
#SBATCH --job-name=launch-think-rat-evals
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
run_dir=${repo}/training/bc_task_vlm/runs/qwen3vl8b-thinking-tick150-s30-rationale-ep1-halfckpt-noindex-promptv9
for checkpoint in 356 712; do
  adapter=${run_dir}/checkpoint-${checkpoint}
  if [[ "${checkpoint}" == 356 ]]; then epoch_label=0p5; else epoch_label=1p0; fi
  [[ -d "${adapter}" ]] || { echo "Missing ${adapter}" >&2; exit 1; }
  for split in train_task_types heldout_task_types; do
    sbatch --job-name="thinkrat-${checkpoint}-${split%%_*}" --time=20:00:00 \
      --export=ALL,MANIFEST=training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json,COHORT_SPLIT=${split},RUN_NAME=qwen3vl8b_thinking_rationale_s30_ep${epoch_label}_fixed10_promptv9,DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered,MODEL=Qwen/Qwen3-VL-8B-Thinking,ADAPTER=${adapter},SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json,EPISODES_PER_TASK=10,WORKERS=4,ENABLE_THINKING=true,MAX_NEW_TOKENS=2048,VLLM_TEMPERATURE=0.6,VLLM_TOP_P=0.95,VLLM_TOP_K=20,VLLM_REASONING_PARSER=qwen3,VLLM_TOOL_CALL_PARSER=qwen3_xml \
      fixed_live_sim_open_weight_vllm.sbatch
  done
done
