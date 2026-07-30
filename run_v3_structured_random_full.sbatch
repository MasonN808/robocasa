#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v3-structured-random
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

# v3 (active-observation: --predict-acting-agent --train-get-image) SFT,
# trained on structured_random instead of the original v3-active-observation
# run's data source. Mirrors run_v3_full.sbatch's recipe exactly except:
#   - DATA_ROOT points at the 21-traj/task capped structured_random training
#     root (matching v1's scale), not robocasa_agentsft_subset.
#   - --no-gradient-checkpointing: structured_random's batch composition (a
#     much higher share of no-image communicate/give_space steps) triggers a
#     DDP "Expected to mark a variable ready only once" crash when
#     ddp_find_unused_parameters=True (hardcoded in
#     _build_training_arguments) is combined with gradient checkpointing.
#     Confirmed on v1/v1.5 structured_random overnight; disabling checkpointing
#     was the fix that actually worked (non-reentrant checkpointing was NOT
#     sufficient). See feedback_gradient_checkpointing_ddp_fix memory.
#   - unique --training-samples-cache-dir: the default is repo-global and
#     races under concurrent training jobs.
#   sbatch run_v3_structured_random_full.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_train_21
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# --- multi-GPU + FlashAttention (B200 = 192GB, batch 8 fits without
# checkpointing based on last night's structured_random single-GPU headroom;
# monitored closely for OOM) ---
export NUM_PROCESSES=8
export PER_DEVICE_BATCH_SIZE=8
export GRAD_ACCUM=1
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v3-structured-random
export WANDB_RUN_NAME=qwen3vl-8b-v3-structured-random

# --- schedule (same as run_v3_full.sbatch) ---
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
  --no-gradient-checkpointing
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${WANDB_RUN_NAME}"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
