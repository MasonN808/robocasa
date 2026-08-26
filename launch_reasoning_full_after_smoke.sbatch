#!/usr/bin/env bash
#SBATCH --job-name=launch-reason-after-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

submit_until_accepted() {
  local output status
  while true; do
    set +e
    output=$(sbatch --parsable "$@" 2>&1)
    status=$?
    set -e
    if [[ ${status} -eq 0 ]]; then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    echo "submission unavailable; retrying in 5 minutes: ${output}" >&2
    sleep 300
  done
}

full_job=$(submit_until_accepted train_tick150_reasoning_scale30_full.sbatch)
echo "submitted reasoning full training ${full_job}"

common_export=ALL,MANIFEST=training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json,DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered,SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json,EPISODES_PER_TASK=10,EVALUATION_SEED=20260817,SCENE_SAMPLING_SEED=20260819,WORKERS=4,MODEL=Qwen/Qwen3-VL-8B-Instruct,ADAPTER=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration/training/bc_task_vlm/runs/qwen3vl8b-tick150-scale30-reasoning-ep1-halfckpt-noindex-promptv9/checkpoint-712,COMMUNICATION_MODE=full,RUN_NAME=sft_scale30_reasoning_ep1_fixed10_promptv9
train_eval=$(submit_until_accepted --dependency="afterok:${full_job}" --job-name=reason-s30-e1-train47 --export="${common_export},COHORT_SPLIT=train_task_types" fixed_live_sim_open_weight_vllm.sbatch)
held_eval=$(submit_until_accepted --dependency="afterok:${full_job}" --job-name=reason-s30-e1-held6 --export="${common_export},COHORT_SPLIT=heldout_task_types" fixed_live_sim_open_weight_vllm.sbatch)
echo "queued reasoning evaluations train=${train_eval} heldout=${held_eval}"
