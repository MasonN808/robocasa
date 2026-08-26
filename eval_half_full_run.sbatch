#!/usr/bin/env bash
#SBATCH --job-name=eval-half-full
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR}"
: "${RUN_PREFIX:?Set RUN_PREFIX}"
: "${MODEL:?Set MODEL}"
: "${MANIFEST:?Set MANIFEST}"
mapfile -t checkpoints < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -t- -k2,2n)
[[ ${#checkpoints[@]} -eq 2 ]] || { echo "Expected 2 checkpoints in ${RUN_DIR}, found ${#checkpoints[@]}" >&2; exit 1; }
checkpoint_idx=$((SLURM_ARRAY_TASK_ID / 2))
split_idx=$((SLURM_ARRAY_TASK_ID % 2))
splits=(train_task_types heldout_task_types)
epoch_tags=(0p5 1p0)
export ADAPTER=${checkpoints[checkpoint_idx]}
export COHORT_SPLIT=${splits[split_idx]}
export RUN_NAME=${RUN_PREFIX}_ep${epoch_tags[checkpoint_idx]}_fixed10_promptv9
export DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json
export EPISODES_PER_TASK=10 EVALUATION_SEED=20260817 SCENE_SAMPLING_SEED=20260819 WORKERS=4
export COMMUNICATION_MODE=full
if [[ "${MODEL}" == *Thinking* ]]; then
  export ENABLE_THINKING=true MAX_NEW_TOKENS=2048 VLLM_TEMPERATURE=0.6 VLLM_TOP_P=0.95 VLLM_TOP_K=20
  export VLLM_REASONING_PARSER=qwen3 VLLM_TOOL_CALL_PARSER=qwen3_xml
fi
exec bash fixed_live_sim_open_weight_vllm.sbatch
