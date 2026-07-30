#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v3-active-observation
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# --- multi-GPU + FlashAttention (B200 = 192GB, batch 8 fits) ---
export NUM_PROCESSES=8
export PER_DEVICE_BATCH_SIZE=8
export GRAD_ACCUM=1
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v3-active-observation
export WANDB_RUN_NAME=qwen3vl-8b-v3-active-observation

# --- schedule (full 3-epoch run; checkpoints every 2000 steps) ---
export NUM_EPOCHS=3
export EVAL_STEPS=2000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2
export EVAL_MAX_SAMPLES=512
export EVAL_GENERATION_MAX_SAMPLES=0

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
[[ -n "${RESUME_FROM_CHECKPOINT:-}" ]] && launch_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
