#!/usr/bin/env bash
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online

# --- multi-GPU + FlashAttention ---
export NUM_PROCESSES=2
export PER_DEVICE_BATCH_SIZE=8
export GRAD_ACCUM=1
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity (throwaway) ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v3-wandbcheck
export WANDB_RUN_NAME=qwen3vl-8b-v3-wandbcheck

# --- schedule ---
export NUM_EPOCHS=3
export EVAL_STEPS=2000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2
export EVAL_MAX_SAMPLES=512
export EVAL_GENERATION_MAX_SAMPLES=0
export MAX_STEPS=20

# --- assemble args ---
source training/scripts/local/_launch_args.sh
launch_args+=(
  --predict-acting-agent
  --train-get-image
  --ddp-timeout-seconds 1800
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
)
[[ -n "${MAX_STEPS:-}" ]] && launch_args+=(--max-steps "${MAX_STEPS}")

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
