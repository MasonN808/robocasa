#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v3_5-verbalized-sampling
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

# v3.5 = v3 (active-observation: --predict-acting-agent --train-get-image)
# + --train-reasoning, on verbalized_sampling (robocasa_local_train_subset_21,
# v1's actual training data). Centralized (this worktree has no
# partial-observability support -- see feedback memory). This combination
# has never been run before; composability was verified read-only
# (dataset.py's train_reasoning and train_get_image/predict_agent code paths
# are independent/additive, no mutual-exclusivity guard, both data sources
# have full metadata.reasoning coverage) but not exercised, so treat this as
# genuinely untested until it completes.
#
# Gradient checkpointing left ON, matching run_v3_verbalized_sampling_full.sbatch
# (this data source's DDP behavior is proven fine with it).
#   sbatch run_v3_5_verbalized_sampling_full.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
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
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v3_5-verbalized-sampling
export WANDB_RUN_NAME=qwen3vl-8b-v3_5-verbalized-sampling

# --- schedule ---
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
  --train-reasoning
  --ddp-timeout-seconds 1800
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${WANDB_RUN_NAME}"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
