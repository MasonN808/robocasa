#!/usr/bin/env bash
#SBATCH --job-name=launch-think-comm8192
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

: "${MODE_INDEX:?Set MODE_INDEX to 1, 2, or 3}"
modes=(full minimal unguided none)
(( MODE_INDEX >= 1 && MODE_INDEX < ${#modes[@]} )) || { echo "Invalid MODE_INDEX=${MODE_INDEX}" >&2; exit 2; }
mode=${modes[MODE_INDEX]}
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
manifest=${repo}/training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
common="ALL,MODEL=Qwen/Qwen3-VL-8B-Thinking,WORKERS=4,COMMUNICATION_MODE=${mode},MANIFEST=${manifest},DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered,SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json,EPISODES_PER_TASK=10,EVALUATION_SEED=20260817,SCENE_SAMPLING_SEED=20260819,ENABLE_THINKING=true,MAX_NEW_TOKENS=8192,VLLM_TEMPERATURE=0.6,VLLM_TOP_P=0.95,VLLM_TOP_K=20,VLLM_REASONING_PARSER=qwen3,VLLM_TOOL_CALL_PARSER=qwen3_xml,VLLM_MAX_MODEL_LEN=16384"

submit() {
  local split=$1
  local output
  for attempt in $(seq 1 60); do
    if output=$(sbatch --parsable --job-name="think8192-${mode}-${split}" --export="${common},COHORT_SPLIT=${split},RUN_NAME=ootb_qwen3vl8b_thinking_43_10_comm_${mode}_budget8192" fixed_live_sim_open_weight_vllm.sbatch 2>&1); then
      echo "${output%%;*}"
      return 0
    fi
    echo "submission attempt ${attempt} failed: ${output}" >&2
    sleep 60
  done
  return 1
}

train_job=$(submit train_task_types)
held_job=$(submit heldout_task_types)
echo "mode=${mode} train_job=${train_job} held_job=${held_job}"
next_index=$((MODE_INDEX + 1))
if (( next_index < ${#modes[@]} )); then
  next=$(sbatch --parsable --dependency="afterany:${train_job}:${held_job}" --export="ALL,MODE_INDEX=${next_index}" launch_ootb_thinking_comm8192_stage.sbatch)
  echo "next_launcher=${next%%;*} next_mode=${modes[next_index]}"
fi
